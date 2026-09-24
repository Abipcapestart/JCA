"""
A6  Query Planning      — deterministic templates + ONE vocabulary LLM call
A7  Source-routed retrieval — APIs for structured sources, search for the rest
A8  Evidence Extraction — one typed contract shared by every source class

Design decisions this module encodes, each traceable to a measured failure:

* Queries are composed by CODE from a typed vocabulary object. The vocabulary
  call cannot name a drug, so query planning structurally cannot decide the
  answer.
* Every per-state guideline search runs TWICE: drug-anchored, and a LANDSCAPE
  pass that deliberately omits the drug. A guideline listing the complete
  treatment-line landscape rarely ranks for a query anchored to one drug, which
  is the confirmed cause of missed guideline comparators.
* Guideline domains are filtered by therapeutic area. Without it ~89% of the
  domains offered per state are the wrong specialty.
* Trials and literature go to APIs. A trial's comparator is a typed field.
"""

from __future__ import annotations

import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_LOGGER = logging.getLogger(__name__)

from .. import config as C
from ..providers.llm import LLMClient
from ..providers.registries import (LiteratureClient, PublicationRecord,
                                    TrialRecord, TrialRegistryClient,
                                    _tokens, classify_multi_intervention_arm)
from ..providers.search import SearchHit, SearchProvider, select_balanced
from ..schema import (Comparator, EvidenceRecord, FINDING_COMPARATOR, FINDING_OUTCOME,
                      Intervention, LeakageGuard, LicensedIndicationRecord,
                      OutcomeMention, PASS_COMPARATOR_ANCHORED, PASS_COMPARATOR_FOLLOWUP,
                      PASS_DRUG_ANCHORED, PASS_LANDSCAPE, PASS_OUTCOME_REQUIREMENT,
                      PASS_REFINEMENT, Population, PopulationContext, QueryPlanItem,
                      QueryVocabulary, RetrievedDocument, ScopeBoundary, SourceClassAttempt)
from ..sources.workbook import SourceInventory, normalise_domain
from .a09_a13_validation import _coerce_component_names

# Languages worth localising HTA queries into, keyed by Member State. Only the
# states whose bodies routinely publish in their own language.
STATE_LANGUAGE: Dict[str, str] = {
    "Germany": "de", "Austria": "de", "France": "fr", "Belgium": "fr",
    "Luxembourg": "fr", "Italy": "it", "Spain": "es", "Portugal": "pt",
    "Netherlands": "nl", "Denmark": "da", "Sweden": "sv", "Finland": "fi",
    "Poland": "pl", "Czech Republic": "cs", "Slovakia": "sk", "Hungary": "hu",
    "Romania": "ro", "Bulgaria": "bg", "Greece": "el", "Croatia": "hr",
    "Slovenia": "sl", "Estonia": "et", "Latvia": "lv", "Lithuania": "lt",
}


# ===========================================================================
# A6 — Query planning
# ===========================================================================

# Batch size for the per-language portion of a06.query_vocabulary. Requesting
# all ~23 EU languages' localised_assessment_terms in one call was hitting
# query_vocabulary_max_tokens and truncating mid-response on every real run --
# even after that cap was raised once already (1500 -> 4000), because a full
# multilingual term set is naturally larger than any single-shot cap comfortably
# handles. Splitting into batches keeps each response small regardless of how
# many languages exist, rather than chasing an ever-larger cap.
_VOCAB_LANGUAGE_BATCH_SIZE = 6


def build_vocabulary(population: Population, areas: Sequence[str],
                     llm: Optional[LLMClient] = None) -> QueryVocabulary:
    """One or more `a06.query_vocabulary` LLM calls producing terminology only.

    The output type has no comparator field. That is the guard: a component
    that cannot represent a comparator cannot leak one into retrieval.

    Batched by language (see _VOCAB_LANGUAGE_BATCH_SIZE) rather than one call
    naming every EU language at once, so the per-call response stays well
    under the token cap regardless of how many languages are configured. The
    language-independent fields (synonyms, abbreviations, disease-class terms,
    outcome-requirement terms) are asked for in every batch and unioned across
    batches -- redundant, but each batch's copy is a small fraction of that
    batch's own output, and unioning is more robust than trusting only the
    first batch's answer.
    """
    indication = population.value("indication_disease")
    vocab = QueryVocabulary(indication_synonyms=[indication] if indication else [])
    if llm is None or not indication:
        return vocab

    languages = sorted({lang for lang in STATE_LANGUAGE.values()})
    subtype = population.value("disease_subtype_histology") or "(not provided)"
    area_text = ", ".join(areas) or "(unresolved)"

    synonyms: List[str] = []
    abbreviations: List[str] = []
    disease_class_terms: List[str] = []
    outcome_requirement_terms: List[str] = []
    soc_candidates: List[str] = []
    localised: Dict[str, List[str]] = {}

    batches = ([languages[i:i + _VOCAB_LANGUAGE_BATCH_SIZE]
               for i in range(0, len(languages), _VOCAB_LANGUAGE_BATCH_SIZE)]
              or [[]])  # still make one call (for the language-independent terms) if no languages

    def _call_batch(batch_languages: List[str]) -> Dict[str, Any]:
        payload = (f"INDICATION: {indication}\n"
                   f"DISEASE SUBTYPE: {subtype}\n"
                   f"THERAPEUTIC AREA(S): {area_text}\n"
                   f"LANGUAGES: {', '.join(batch_languages)}\n")
        return llm.call_json("a06.query_vocabulary", payload,
                             max_tokens=C.LLM.query_vocabulary_max_tokens, default={}) or {}

    # These batches are independent calls with nothing for one to wait on
    # from another -- the split that fixed the truncation bug (23 languages
    # -> ~4 batches of 6) turned 1 slow, truncating call into 4 small,
    # non-truncating ones, but running them one after another still pays 4
    # full round trips in serial, once per population, every run.
    with ThreadPoolExecutor(max_workers=min(4, len(batches))) as pool:
        parsed_batches = list(pool.map(_call_batch, batches))

    for parsed in parsed_batches:
        if not isinstance(parsed, dict):
            # The model didn't follow the "return a JSON object" instruction (e.g. it
            # returned a bare array). Treat it the same as an unparseable response
            # rather than letting the shape mismatch crash the run.
            continue
        synonyms.extend(parsed.get("indication_synonyms") or [])
        abbreviations.extend(parsed.get("indication_abbreviations") or [])
        disease_class_terms.extend(parsed.get("disease_class_terms") or [])
        outcome_requirement_terms.extend(parsed.get("outcome_requirement_terms") or [])
        soc_candidates.extend(parsed.get("standard_of_care_candidates") or [])
        batch_localised = parsed.get("localised_assessment_terms") or {}
        for k, v in batch_localised.items():
            if isinstance(v, list):
                localised.setdefault(k, []).extend(v)

    vocab.indication_synonyms = _dedup([indication] + synonyms)
    vocab.indication_abbreviations = _dedup(abbreviations)
    vocab.disease_class_terms = _dedup(disease_class_terms)
    vocab.outcome_requirement_terms = _dedup(outcome_requirement_terms)
    vocab.localised_assessment_terms = {k: _dedup(v) for k, v in localised.items()}
    # Capped, disease-specific shortlist -- NOT the full INN vocabulary. a08
    # extraction is already ~88% of a run's LLM cost, so this stays bounded
    # regardless of how many candidates the model names across batches.
    vocab.standard_of_care_candidates = _dedup(soc_candidates)[
        :C.RETRIEVAL.max_comparator_candidates]

    # Safety net. If the model named something that looks like a drug, drop it —
    # the vocabulary must not carry an answer. Deliberately does NOT touch
    # standard_of_care_candidates: naming a treatment there is correct, not a
    # leak -- that field exists specifically so retrieval can search by name.
    vocab = _strip_possible_drug_names(vocab)
    return vocab


_DRUGLIKE = re.compile(r"\b\w+(?:mab|nib|tinib|ciclib|zumab|ximab|umab|parib|"
                       r"platin|rubicin|taxel|mustine|tecan)\b", re.IGNORECASE)


def _strip_possible_drug_names(v: QueryVocabulary) -> QueryVocabulary:
    def clean(items: List[str]) -> List[str]:
        return [i for i in items if not _DRUGLIKE.search(i or "")]
    v.indication_synonyms = clean(v.indication_synonyms)
    v.disease_class_terms = clean(v.disease_class_terms)
    v.outcome_requirement_terms = clean(v.outcome_requirement_terms)
    return v


def _dedup(items: Iterable[str]) -> List[str]:
    out, seen = [], set()
    for i in items or []:
        s = str(i or "").strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out


def _build_anchored_disease_query(vocab: QueryVocabulary, condition: str) -> str:
    """Force the core disease term as a required match (PubMed field-tag
    syntax), ANDed with the broader synonym/class-term list -- rather than
    OR'ing everything together, where one loosely-related term alone can
    satisfy the whole query, or falling back to an entire unstructured
    sentence as a single search term. Confirmed real gap: the disease-only
    PubMed queries (guideline-restricted and landscape passes) OR'd every
    term with no requirement that the actual disease name appear in a match,
    so unrelated documents (gout, UTI, Cushing's syndrome) matched purely on
    generic co-occurring words like "systemic," "therapy," "progression"."""
    anchor_candidates = (vocab.disease_class_terms or []) + (vocab.indication_synonyms or [])
    anchor = min(anchor_candidates, key=len) if anchor_candidates else condition
    anchor_term = f'"{anchor}"[Title/Abstract]' if " " in anchor else f"{anchor}[Title/Abstract]"

    other_terms = _dedup([t for t in anchor_candidates if t != anchor])
    if not other_terms:
        return anchor_term
    other_query = " OR ".join(f'"{s}"' if " " in s else s for s in other_terms)
    return f"{anchor_term} AND ({other_query})"


def _build_comparator_anchored_query(candidate: str, disease_query: str) -> str:
    """Anchor on the CANDIDATE's own name (PubMed field-tag syntax, same
    technique _build_anchored_disease_query uses to anchor on the disease),
    ANDed with the same disease term set every other PubMed pass already
    builds -- finds a comparator's own literature by name instead of hoping
    it ranks inside a disease-only search. Confirmed real gap: two GT
    comparators (Everolimus, Bevacizumab+chemo) had zero hits anywhere in
    retrieval because no PubMed query ever named a candidate comparator."""
    term = f'"{candidate}"[Title/Abstract]' if " " in candidate else f"{candidate}[Title/Abstract]"
    return f"{term} AND ({disease_query})"


def pubmed_comparator_followup(candidates: Sequence[str], literature: Optional[LiteratureClient],
                               vocab: QueryVocabulary, condition: str,
                               drug: str = "") -> List[PublicationRecord]:
    """Version B's PubMed half: one name-anchored search per candidate that
    organically surfaced from real round-1 evidence with only a thin
    mention. Reuses _build_comparator_anchored_query() and the same
    disease-term-set construction retrieve_structured()'s own queries use,
    so this follow-up round is built exactly the same way as the rest of the
    PubMed passes, just called a second time with a different candidate
    source (thin_mention names, not standard_of_care_candidates)."""
    if literature is None or not C.RETRIEVAL.enable_comparator_followup_round or not candidates:
        return []
    drug_tokens = _tokens(drug) if drug else set()
    bounded = [c for c in candidates if _tokens(c) != drug_tokens][
        :C.RETRIEVAL.max_comparator_candidates]
    if not bounded:
        return []
    synonym_terms = _dedup([condition] + (vocab.indication_synonyms or []))
    disease_query = " OR ".join(f'"{s}"' if " " in s else s for s in synonym_terms)
    pubs: List[PublicationRecord] = []
    for candidate in bounded:
        query = _build_comparator_anchored_query(candidate, disease_query)
        pubs += literature.search(query, max_results=10)
    return pubs


def plan_queries(population: Population, intervention: Intervention,
                 areas: Sequence[str], vocab: QueryVocabulary,
                 inventory: SourceInventory,
                 guard: Optional[LeakageGuard] = None) -> List[QueryPlanItem]:
    """Compose the full query plan deterministically.

    Templates, not a model. Reviewable, diffable, testable, free.
    """
    drug = intervention.product_name
    indication = population.value("indication_disease")
    syn = vocab.indication_synonyms or [indication]
    primary = syn[0] if syn else indication
    plan: List[QueryPlanItem] = []

    # HTA entries with no member_state -- e.g. INAHTA's international HTA
    # database -- are excluded from the per-state call below (it passes
    # include_shared=False so one shared entry doesn't get offered to all 27
    # states as if it were each one's national body). A state with no
    # curated national source still deserves this over the fully open web.
    general_hta_domains = sorted({e.domain for e in inventory.entries
                                  if e.source_class == C.SRC_HTA_REGULATORY
                                  and e.member_state is None})

    # -- Tier 1, EU-wide: the regulatory record -----------------------------
    ema_domains = inventory.domains(C.SRC_DRUG_LABEL)
    if ema_domains:
        plan.append(QueryPlanItem(
            query=f"{drug} EPAR summary of product characteristics "
                  f"{_line_hint(population, intervention)}".strip(),
            source_class=C.SRC_DRUG_LABEL, member_state=C.EU_WIDE,
            pass_type=PASS_DRUG_ANCHORED, domains=ema_domains, max_urls=3,
            note="EU-wide regulatory record; read in full."))

    # -- Tier 1, EU-wide: the drug's own JCA report --------------------------
    # Deliberate, per explicit SME decision (2026-09-23): comparators from the
    # drug's own official JCA report, clearly labeled as such via
    # SRC_EU_JCA_REPORT rather than blended silently into another Tier 1
    # class. Gated on the same flag the LeakageGuard itself checks, so the
    # plan and the actual fetch-time block/allow decision never disagree.
    if guard is not None and guard.allow_jca_reports:
        plan.append(QueryPlanItem(
            query=f"{drug} joint clinical assessment report",
            source_class=C.SRC_EU_JCA_REPORT, member_state=C.EU_WIDE,
            pass_type=PASS_DRUG_ANCHORED, domains=["health.ec.europa.eu"], max_urls=2,
            note="Deliberate JCA-report pass, no longer evaluation-only per SME decision."))

    # -- Tier 1, per Member State ------------------------------------------
    for state in C.EU_27_MEMBER_STATES:
        lang = STATE_LANGUAGE.get(state, "en")
        hta_domains = inventory.domains(C.SRC_HTA_REGULATORY, member_state=state,
                                        include_shared=False)
        # THE AREA FILTER. Without it this list is every specialty's societies.
        guide_domains = inventory.domains(C.SRC_CLINICAL_GUIDELINE, member_state=state,
                                          areas=areas)

        if hta_domains:
            terms = vocab.localised_assessment_terms.get(lang) or ["assessment", "appraisal"]
            plan.append(QueryPlanItem(
                query=f"{drug} {primary} {terms[0]}",
                source_class=C.SRC_HTA_REGULATORY, member_state=state,
                pass_type=PASS_DRUG_ANCHORED, domains=hta_domains,
                language=lang, note="National HTA body, drug-anchored."))
        else:
            plan.append(QueryPlanItem(
                query=f"{state} health technology assessment {drug} {primary}",
                source_class=C.SRC_HTA_REGULATORY, member_state=state,
                pass_type=PASS_DRUG_ANCHORED, domains=general_hta_domains, max_urls=2,
                language=lang,
                note=("No curated national HTA source for this state; searched against "
                      "the international HTA database instead of the open web."
                      if general_hta_domains else
                      "No curated HTA source for this state; unscoped fallback so the "
                      "state is genuinely searched rather than silently skipped.")))

        if guide_domains:
            plan.append(QueryPlanItem(
                query=f"{drug} {primary}",
                source_class=C.SRC_CLINICAL_GUIDELINE, member_state=state,
                pass_type=PASS_DRUG_ANCHORED, domains=guide_domains, language=lang,
                note="Area-filtered guideline sources, drug-anchored."))
            # The landscape pass. Deliberately omits the drug.
            plan.append(QueryPlanItem(
                query=f"{primary} treatment guideline recommended options "
                      f"{_line_hint(population, intervention)}".strip(),
                source_class=C.SRC_CLINICAL_GUIDELINE, member_state=state,
                pass_type=PASS_LANDSCAPE, domains=guide_domains, language=lang,
                note=("NOT drug-anchored: finds the complete treatment-line landscape a "
                      "guideline states, which a drug-anchored query systematically misses.")))

    # -- Tier 2: conference evidence (no API exists) ------------------------
    # Area-filtered for the same reason the guideline domains are (line 145):
    # without it, an oncology-only curated congress list gets searched for
    # every therapeutic area, cardiovascular and infectious disease included.
    conf_domains = inventory.domains(C.SRC_CONFERENCE, areas=areas) or []
    if conf_domains:
        plan.append(QueryPlanItem(
            query=f"{drug} {primary} abstract",
            source_class=C.SRC_CONFERENCE, member_state=C.GENERAL_EVIDENCE,
            pass_type=PASS_DRUG_ANCHORED, domains=conf_domains,
            note="Kept separate from peer-reviewed evidence, per SME Agent 8."))

    # -- Outcome REQUIREMENTS: a different intent from outcome results ------
    req_terms = vocab.outcome_requirement_terms or ["required outcomes", "outcome measures"]
    guide_any = inventory.domains(C.SRC_CLINICAL_GUIDELINE, areas=areas)
    if guide_any:
        plan.append(QueryPlanItem(
            query=f"{primary} {req_terms[0]} health technology assessment",
            source_class=C.SRC_CLINICAL_GUIDELINE, member_state=C.GENERAL_EVIDENCE,
            pass_type=PASS_OUTCOME_REQUIREMENT, domains=guide_any, max_urls=3,
            note=("Outcomes an assessment REQUIRES, which is a different object from "
                  "outcomes a trial happens to report.")))

    # -- Comparator-name-anchored: search for a candidate comparator BY NAME -
    # Additive only -- runs once at GENERAL_EVIDENCE scope, NOT inside the
    # 27-member-state loop above, to keep cost bounded regardless of
    # vocabulary size (see C.RETRIEVAL.max_comparator_candidates). Existing
    # drug-anchored and disease-only landscape passes are untouched; this
    # closes the real gap those two still had: a comparator that never ranks
    # inside a disease-only search never got a query naming it at all.
    # Confirmed real gap: two GT comparators (Everolimus, Bevacizumab+chemo)
    # had zero hits anywhere in retrieval across multiple runs.
    drug_tokens = _tokens(drug)
    comparator_candidates = [c for c in (vocab.standard_of_care_candidates or [])
                             if _tokens(c) != drug_tokens][:C.RETRIEVAL.max_comparator_candidates]
    if guide_any:
        for candidate in comparator_candidates:
            plan.append(QueryPlanItem(
                query=f'"{candidate}" {primary}',
                source_class=C.SRC_CLINICAL_GUIDELINE, member_state=C.GENERAL_EVIDENCE,
                pass_type=PASS_COMPARATOR_ANCHORED, domains=guide_any, max_urls=2,
                note=(f"Comparator-name-anchored: searches for {candidate!r} by name, "
                      "not just disease terms a landscape query hopes it ranks under.")))

    # -- Tier 3 -------------------------------------------------------------
    if C.ENABLE_TIER_3_GENERAL_WEB:
        plan.append(QueryPlanItem(
            query=f"{drug} {primary} standard of care comparator "
                  f"{_line_hint(population, intervention)}".strip(),
            source_class=C.SRC_GENERAL_WEB, member_state=C.GENERAL_EVIDENCE,
            pass_type=PASS_DRUG_ANCHORED, domains=[], max_urls=2,
            note="Tier 3. Disabled by default: MAI-34392 excludes non-peer-reviewed web."))

    return plan


def _line_hint(population: Population, intervention: Intervention) -> str:
    """A line-of-therapy hint for the landscape query, only when the user
    actually stated one. Never invented."""
    for value in (intervention.value("line_of_therapy"),
                  population.value("prior_therapy_line")):
        if value:
            return value
    return ""


def plan_refinement(gaps: Sequence[Tuple[str, str]], population: Population,
                    intervention: Intervention, vocab: QueryVocabulary,
                    inventory: SourceInventory,
                    areas: Sequence[str]) -> List[QueryPlanItem]:
    """One bounded refinement round, fired only where coverage actually failed.

    Broadens terms. Never introduces a candidate comparator name.
    """
    if not C.RETRIEVAL.enable_refinement_round:
        return []
    drug = intervention.product_name
    broad = (vocab.disease_class_terms or vocab.indication_synonyms
             or [population.value("indication_disease")])[0]
    out: List[QueryPlanItem] = []
    for state, source_class in gaps:
        domains = (inventory.domains(source_class, member_state=state, areas=areas)
                   if source_class == C.SRC_CLINICAL_GUIDELINE
                   else inventory.domains(source_class, member_state=state))
        out.append(QueryPlanItem(
            query=f"{broad} {drug}" if source_class != C.SRC_CLINICAL_GUIDELINE else broad,
            source_class=source_class, member_state=state,
            pass_type=PASS_REFINEMENT, domains=domains, max_urls=2,
            note="Refinement round: broadened terms after a coverage gap."))
    return out


def plan_comparator_followup(candidates: Sequence[str], population: Population,
                             intervention: Intervention, areas: Sequence[str],
                             inventory: SourceInventory) -> List[QueryPlanItem]:
    """Version B: ONE additional, name-targeted query per candidate that
    organically surfaced from real round-1 evidence with only a thin mention
    (comparator.thin_mention) -- not a guessed candidate (that's
    standard_of_care_candidates/PASS_COMPARATOR_ANCHORED, a separate, earlier
    pass). Mirrors plan_refinement()'s shape: gated, bounded, additive-only,
    runs once at GENERAL_EVIDENCE scope, never inside the 27-member-state
    loop, so cost stays bounded regardless of how many thin mentions a real
    run surfaces (see C.RETRIEVAL.max_comparator_candidates)."""
    if not C.RETRIEVAL.enable_comparator_followup_round or not candidates:
        return []
    drug = intervention.product_name
    indication = population.value("indication_disease")
    drug_tokens = _tokens(drug)
    bounded = [c for c in candidates if _tokens(c) != drug_tokens][
        :C.RETRIEVAL.max_comparator_candidates]
    guide_domains = inventory.domains(C.SRC_CLINICAL_GUIDELINE, areas=areas)
    out: List[QueryPlanItem] = []
    if guide_domains:
        for candidate in bounded:
            out.append(QueryPlanItem(
                query=f'"{candidate}" {indication}',
                source_class=C.SRC_CLINICAL_GUIDELINE, member_state=C.GENERAL_EVIDENCE,
                pass_type=PASS_COMPARATOR_FOLLOWUP, domains=guide_domains, max_urls=2,
                note=(f"Follow-up: {candidate!r} surfaced from real evidence with only a "
                      "thin mention -- searching for its own dedicated evidence by name.")))
    return out


# ===========================================================================
# A7 — Retrieval
# ===========================================================================

@dataclass
class RetrievalResult:
    documents: List[RetrievedDocument] = field(default_factory=list)
    trials: List[TrialRecord] = field(default_factory=list)
    publications: List[PublicationRecord] = field(default_factory=list)
    attempts: List[SourceClassAttempt] = field(default_factory=list)
    blocked_by_guard: List[str] = field(default_factory=list)


def execute_plan(plan: Sequence[QueryPlanItem], search: SearchProvider,
                 guard: LeakageGuard, inventory: SourceInventory,
                 areas: Sequence[str],
                 max_workers: int = C.RETRIEVAL.max_workers) -> RetrievalResult:
    """Run the plan. One unit of work per plan item; states are independent."""
    result = RetrievalResult()

    # Direct-extract any curated entry that is already a document URL — no
    # search needed, and no ranking luck involved.
    for entry in inventory.document_urls():
        if entry.therapeutic_area and areas and entry.therapeutic_area not in set(areas):
            continue
        if guard.is_blocked(entry.url):
            guard.record(entry.url)
            result.blocked_by_guard.append(entry.url)
            continue
        doc = search.fetch(entry.url, full_document=True)
        doc.source_class = entry.source_class
        doc.member_state = entry.member_state or C.EU_WIDE
        doc.organization = entry.organization
        if doc.ok:
            result.documents.append(doc)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_execute_item, item, search, guard): item for item in plan}
        for fut in as_completed(futures):
            item = futures[fut]
            try:
                docs, attempt, blocked = fut.result()
            except Exception as exc:  # noqa: BLE001 — one item must not lose the rest
                result.attempts.append(SourceClassAttempt(
                    member_state=item.member_state, source_class=item.source_class,
                    attempted=True, status=C.EV_RETRIEVAL_FAILED,
                    detail=f"{type(exc).__name__}: {exc}"))
                continue
            result.documents.extend(docs)
            result.attempts.append(attempt)
            result.blocked_by_guard.extend(blocked)

    result.documents = _dedupe_documents(result.documents, inventory)
    return result


def _dedupe_documents(documents: List[RetrievedDocument],
                      inventory: SourceInventory) -> List[RetrievedDocument]:
    """One document, one extraction.

    Several plan items legitimately surface the same URL — an unscoped fallback
    for a state with no curated source, a landscape pass and a drug-anchored
    pass over the same guideline domain. Extracting it more than once multiplies
    LLM cost and inflates every downstream count, which is how one document ends
    up looking like corroboration across several Member States.

    Member-State attribution is UNIONED rather than dropped: the same guideline
    really can be the source for several states, and losing that would be worse
    than the duplication.
    """
    # Domain -> the source class the curated list says it is. An unscoped
    # fallback search must not stamp its own class onto a document that the
    # source list identifies as something else.
    class_by_domain: Dict[str, str] = {}
    for e in inventory.entries:
        class_by_domain.setdefault(e.domain, e.source_class)

    merged: Dict[str, RetrievedDocument] = {}
    states: Dict[str, List[str]] = {}
    for doc in documents:
        key = doc.resolved_url or doc.url
        true_class = class_by_domain.get(normalise_domain(key))
        if true_class and true_class != doc.source_class:
            doc.source_class = true_class
        if key not in merged:
            merged[key] = doc
            states[key] = []
        if doc.member_state in C.EU_27_MEMBER_STATES:
            if doc.member_state not in states[key]:
                states[key].append(doc.member_state)

    out = []
    for key, doc in merged.items():
        attributed = states.get(key) or []
        if attributed:
            doc.member_state = attributed[0]
            # Keep the full attribution so consolidation can credit every state
            # this document genuinely supports.
            doc.__dict__["attributed_states"] = attributed
        out.append(doc)
    return out


def _execute_item(item: QueryPlanItem, search: SearchProvider, guard: LeakageGuard
                  ) -> Tuple[List[RetrievedDocument], SourceClassAttempt, List[str]]:
    attempt = SourceClassAttempt(member_state=item.member_state,
                                 source_class=item.source_class, attempted=True)
    blocked: List[str] = []

    if not item.domains and item.source_class in (C.SRC_HTA_REGULATORY,):
        # An unscoped fallback is still a genuine search — recorded as such so
        # "no curated source" is never confused with "not searched".
        attempt.detail = "no curated domains; unscoped fallback search"

    hits: List[SearchHit] = search.search(item.query, item.domains,
                                          C.RETRIEVAL.search_max_results)
    if not hits:
        failed = any(c.error for c in search.call_log[-1:])
        attempt.status = C.EV_RETRIEVAL_FAILED if failed else C.EV_NONE
        return [], attempt, blocked

    # One group per curated domain, not one "primary" bucket for all of
    # them -- select_balanced()'s own anti-starvation guarantee (at least one
    # slot per group that returned something) only activates with more than
    # one group. Collapsing a multi-domain query's curated list into a
    # single group meant Tavily's relevance ranking alone decided which
    # domains got a slot, letting one well-ranked domain crowd out the
    # others entirely within the same query -- confirmed never triggered
    # in production despite the function existing specifically for this.
    domain_groups = {d: [d] for d in item.domains} if item.domains else {}
    urls = select_balanced(hits, item.max_urls, domain_groups)
    docs: List[RetrievedDocument] = []
    for url in urls:
        if guard.is_blocked(url):
            guard.record(url)
            blocked.append(url)
            continue
        full = item.source_class in C.FULL_DOCUMENT_SOURCE_CLASSES
        doc = search.fetch(url, query=item.query, full_document=full)
        doc.source_class = item.source_class
        doc.member_state = item.member_state
        if doc.ok:
            docs.append(doc)

    attempt.documents_retrieved = len(docs)
    attempt.status = C.EV_FOUND if docs else (
        C.EV_SOURCE_INACCESSIBLE if urls else C.EV_NONE)
    return docs, attempt, blocked


def retrieve_structured(population: Population, intervention: Intervention,
                        indication: LicensedIndicationRecord,
                        trials: Optional[TrialRegistryClient],
                        literature: Optional[LiteratureClient],
                        vocab: QueryVocabulary) -> RetrievalResult:
    """Structured sources. These bypass LLM extraction for the fields the API
    already types."""
    out = RetrievalResult()
    condition = population.value("indication_disease")
    drug = intervention.product_name

    if trials is not None:
        try:
            out.trials = trials.search(condition=condition, intervention=drug,
                                       identifiers=indication.pivotal_trials,
                                       max_results=20)
            out.attempts.append(SourceClassAttempt(
                member_state=C.GENERAL_EVIDENCE, source_class=C.SRC_TRIAL_REGISTRY,
                attempted=True, documents_retrieved=len(out.trials),
                status=C.EV_FOUND if out.trials else C.EV_NONE))
        except Exception as exc:  # noqa: BLE001
            out.attempts.append(SourceClassAttempt(
                member_state=C.GENERAL_EVIDENCE, source_class=C.SRC_TRIAL_REGISTRY,
                attempted=True, status=C.EV_RETRIEVAL_FAILED,
                detail=f"{type(exc).__name__}: {exc}"))

    if literature is not None:
        try:
            # Use every synonym PubMed can search on, not just the raw
            # indication string (query 1) or one lucky surviving element
            # (query 2's old `[0]` pick) -- PubMed/Entrez's term syntax
            # supports OR directly in the query string (already used one
            # line below for publication-type filters), so this is a
            # query-string change only, no client change needed. Confirmed
            # real gap: a query-vocabulary fix that successfully returns
            # good alternate phrasings had nowhere for most of them to go --
            # the main drug+disease query never referenced them at all.
            synonym_terms = _dedup([condition] + (vocab.indication_synonyms or []))
            disease_query = " OR ".join(f'"{s}"' if " " in s else s for s in synonym_terms)
            main_query = (f"{drug} AND ({disease_query})" if len(synonym_terms) > 1
                         else f"{drug} {condition}")
            pubs = literature.search(main_query, max_results=15)
            # Guidelines as a retrievable CLASS — the query shape a web search
            # cannot express, and the one that finds a landscape guideline.
            broad_query = _build_anchored_disease_query(vocab, condition)
            pubs += literature.search(broad_query,
                                      publication_types=["Practice Guideline", "Guideline"],
                                      max_results=15)
            # Genuine disease-LANDSCAPE pass: same disease-only terms as
            # above, but with NO publication-type restriction -- mirrors
            # plan_queries()'s clinical_guideline landscape pass, which
            # deliberately omits the drug to find the complete treatment-line
            # landscape a drug-anchored query systematically misses. Confirmed
            # real gap: restricting the only disease-only query to "Practice
            # Guideline"/"Guideline" publication types meant a comparator's
            # own regular clinical study or review (never tagged that way by
            # PubMed) was unreachable by either query, however good the
            # synonym list became -- exactly why a real comparator kept not
            # surfacing even after fetch volume and synonym coverage improved.
            pubs += literature.search(broad_query, max_results=15)
            # Comparator-name-anchored pass: search for a candidate BY NAME,
            # ANDed against the same disease term set every other pass uses
            # -- additive only, and bounded to a small disease-specific
            # shortlist (see C.RETRIEVAL.max_comparator_candidates) rather
            # than the full INN vocabulary.
            drug_tokens = _tokens(drug)
            comparator_candidates = [
                c for c in (vocab.standard_of_care_candidates or [])
                if _tokens(c) != drug_tokens][:C.RETRIEVAL.max_comparator_candidates]
            comparator_queries = [_build_comparator_anchored_query(c, disease_query)
                                  for c in comparator_candidates]
            for q in comparator_queries:
                pubs += literature.search(q, max_results=10)
            seen, deduped = set(), []
            for p in pubs:
                if p.pmid and p.pmid not in seen:
                    seen.add(p.pmid)
                    deduped.append(p)
            out.publications = deduped
            # Surfaced so a reviewer can see the ACTUAL query strings used,
            # not just the result count -- PubMed/CT.gov never appear in the
            # "A6 Query Plan" export sheet at all (they bypass plan_queries()
            # entirely, by design), so this was the one retrieval pass with
            # no visibility anywhere into what was actually searched for.
            query_detail = (
                f"[1 drug-anchored] {main_query} | "
                f"[2 guideline-restricted: Practice Guideline/Guideline] {broad_query} | "
                f"[3 landscape, unrestricted] {broad_query}")
            if comparator_queries:
                query_detail += (
                    f" | [4 comparator-anchored: {'; '.join(comparator_candidates)}] "
                    + " ; ".join(comparator_queries))
            out.attempts.append(SourceClassAttempt(
                member_state=C.GENERAL_EVIDENCE, source_class=C.SRC_PUBMED,
                attempted=True, documents_retrieved=len(deduped),
                status=C.EV_FOUND if deduped else C.EV_NONE,
                detail=query_detail))
        except Exception as exc:  # noqa: BLE001
            out.attempts.append(SourceClassAttempt(
                member_state=C.GENERAL_EVIDENCE, source_class=C.SRC_PUBMED,
                attempted=True, status=C.EV_RETRIEVAL_FAILED,
                detail=f"{type(exc).__name__}: {exc}"))
    return out


def documents_from_publications(publications: Sequence[PublicationRecord]
                                ) -> List[RetrievedDocument]:
    """PubMed abstracts, framed as documents for the shared extraction contract.

    Only the abstract is available -- PubMedClient never fetches full text.
    Without this, a publication found by retrieve_structured() has nowhere to
    go: it is real Tier 2 evidence that would otherwise be fetched and then
    silently discarded.
    """
    out: List[RetrievedDocument] = []
    for p in publications:
        text = (p.abstract or "").strip()
        if not text:
            continue  # nothing for the LLM to extract from
        out.append(RetrievedDocument(
            url=p.url, resolved_url=p.url, text=text, ok=True,
            status=C.EV_FOUND, method="pubmed_api", title=p.title,
            published_date=p.year, source_class=C.SRC_PUBMED,
            member_state=C.GENERAL_EVIDENCE, organization=p.journal,
            publication_types=list(p.publication_types)))
    return out


# ===========================================================================
# A8 — Evidence extraction
# ===========================================================================

_seq = {"n": 0}
_seq_lock = threading.Lock()


def _next_id(prefix: str) -> str:
    # extract_from_documents() calls this from inside a ThreadPoolExecutor --
    # += on a shared counter is a read-modify-write race without the lock,
    # and two records can otherwise end up sharing the same finding_id.
    with _seq_lock:
        _seq["n"] += 1
        n = _seq["n"]
    return f"{prefix}-{n:05d}"


def extract_from_documents(documents: Sequence[RetrievedDocument],
                           population: Population, intervention: Intervention,
                           llm: LLMClient,
                           max_workers: int = C.RETRIEVAL.max_workers
                           ) -> List[EvidenceRecord]:
    """One typed contract for every source class.

    The SME writes seven retrieval agents, but they record the same finding
    shape. Seven prompts would be seven places for one rule to drift.

    Parallelizes at the (document, chunk) level, not per-document -- a real
    run showed extraction dominating wall-clock time (59% of a 44-minute
    run) specifically because a document needing several chunks (chunking
    fix, elsewhere in this file) processed them SEQUENTIALLY inside its own
    worker thread: one worker stuck on an 8-chunk PDF blocked that slot for
    a long stretch while the pool's other workers cycled through simple
    1-chunk documents, so the effective concurrency achieved (~2.7x) fell
    well short of max_workers (8x). Flattening every chunk across every
    document into ONE pool means a chunk-heavy document's calls interleave
    fairly with everyone else's instead of monopolizing a worker.
    """
    if not documents:
        return []
    work_items: List[Tuple[RetrievedDocument, str, int, int]] = []
    for d in documents:
        chunks = _chunk_document_text(d.text)
        for i, chunk_text in enumerate(chunks):
            work_items.append((d, chunk_text, i, len(chunks)))

    by_doc_id: Dict[int, RetrievedDocument] = {id(d): d for d in documents}
    parsed_by_doc: Dict[int, List[Dict[str, Any]]] = {id(d): [] for d in documents}
    seen_quotes_by_doc: Dict[int, set] = {id(d): set() for d in documents}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_doc_id = {
            pool.submit(_extract_chunk_items, d, chunk_text, i, total,
                       population, intervention, llm): id(d)
            for d, chunk_text, i, total in work_items}
        for fut in as_completed(future_to_doc_id):
            doc_id = future_to_doc_id[fut]
            try:
                items = fut.result()
            except Exception as exc:  # noqa: BLE001 — one chunk must not lose the rest
                # Previously silent -- zero trace anywhere that a document's
                # extraction crashed outright (e.g. an LLM error surviving
                # call()'s own retries) versus genuinely having no evidence.
                doc = by_doc_id[doc_id]
                _LOGGER.warning(
                    "a08.extraction: document %s raised %s: %s -- chunk skipped",
                    doc.resolved_url or doc.url, type(exc).__name__, exc)
                continue
            seen = seen_quotes_by_doc[doc_id]
            for item in items:
                # A sentence in the chunk-overlap window can be extracted
                # twice -- once per chunk it appears in. Its evidence_quote
                # is a verbatim span, so an exact repeat is the same claim.
                quote = (item.get("evidence_quote") or "").strip()
                if quote:
                    if quote in seen:
                        continue
                    seen.add(quote)
                parsed_by_doc[doc_id].append(item)

    records: List[EvidenceRecord] = []
    for d in documents:
        records.extend(_records_from_parsed_items(parsed_by_doc[id(d)], d))
    return records


def _extract_chunk_items(doc: RetrievedDocument, chunk_text: str, chunk_index: int,
                         total_chunks: int, population: Population,
                         intervention: Intervention, llm: LLMClient
                         ) -> List[Dict[str, Any]]:
    """One a08.extraction call for one chunk. Returns raw parsed dict items
    (not yet EvidenceRecords) -- cross-chunk dedup/record-building happens in
    the caller, since it needs every chunk's output for the same document."""
    payload = (
        f"REQUESTED INTERVENTION: {intervention.product_name}\n"
        f"REQUESTED POPULATION: {population.value('indication_disease')}\n"
        f"SOURCE CLASS: {doc.source_class}\n"
        f"SOURCE URL: {doc.resolved_url or doc.url}\n"
        + (f"NOTE: this is section {chunk_index + 1} of {total_chunks} of a longer "
           f"document; extract only what THIS section states.\n" if total_chunks > 1 else "")
        + f"\nDOCUMENT TEXT:\n{chunk_text}")
    parsed = llm.call_json("a08.extraction", payload,
                           max_tokens=C.LLM.extraction_max_tokens, default=[])
    if not isinstance(parsed, list):
        # This took manual cross-tabulation across hundreds of raw LLM calls
        # to even discover once -- log it so a chunk-level truncation/parse
        # failure is visible without that. One bad chunk must not zero out
        # what the OTHER chunks of this same document already extracted
        # successfully, so the caller continues rather than aborting.
        _LOGGER.warning(
            "a08.extraction: chunk %d/%d of %s returned no usable records "
            "(empty or unparseable response)", chunk_index + 1, total_chunks,
            doc.resolved_url or doc.url)
        return []
    return [item for item in parsed if isinstance(item, dict)]


# A document's real content can need more OUTPUT tokens to describe than any
# single extraction call comfortably produces, even when the INPUT easily
# fits the context window -- confirmed on a real 76KB clinical-guideline PDF
# (GT's own cited source for TPCV, Carboplatin+Vincristine, and Vinblastine)
# whose response hit extraction_max_tokens and was cut off mid-JSON, so
# extraction silently returned nothing usable from it. Chunking the input
# keeps each call's own output small regardless of how content-dense the
# source is, rather than truncating input at a fixed length (the prior
# doc.text[:120000]) and hoping the remaining content's findings still fit
# one response. The overlap keeps a comparator/outcome sentence that happens
# to straddle a chunk boundary readable in at least one chunk; _MAX_CHUNKS
# bounds worst-case latency/cost per document.
_EXTRACTION_CHUNK_CHARS = 30000
_EXTRACTION_CHUNK_OVERLAP_CHARS = 1500
_EXTRACTION_MAX_CHUNKS = 8


def _chunk_document_text(text: str) -> List[str]:
    """Prefer breaking at a paragraph/section boundary (a blank line) near
    the target chunk size, rather than a hard character cutoff -- reduces
    (does not eliminate; that's what the overlap is for) the odds of cutting
    a recommendation-table row or a sentence exactly at the boundary, which
    can leave one chunk with too little context to extract it correctly."""
    if len(text) <= _EXTRACTION_CHUNK_CHARS:
        return [text]
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n and len(chunks) < _EXTRACTION_MAX_CHUNKS:
        end = min(start + _EXTRACTION_CHUNK_CHARS, n)
        if end < n:
            search_from = start + int(_EXTRACTION_CHUNK_CHARS * 0.8)
            break_at = text.rfind("\n\n", search_from, end)
            if break_at != -1 and break_at > search_from:
                end = break_at
        chunks.append(text[start:end])
        if end >= n:
            break
        start = max(end - _EXTRACTION_CHUNK_OVERLAP_CHARS, start + 1)  # always progress
    return chunks


def _extract_one(doc: RetrievedDocument, population: Population,
                 intervention: Intervention, llm: LLMClient) -> List[EvidenceRecord]:
    """Sequential, single-document extraction (chunks processed one at a
    time). Kept for standalone/direct use; `extract_from_documents()` uses
    the same underlying `_extract_chunk_items`/`_records_from_parsed_items`
    but parallelizes chunks across ALL documents at once instead of looping
    sequentially per document -- see that function's docstring."""
    parsed_items: List[Dict[str, Any]] = []
    seen_quotes: set = set()
    chunks = _chunk_document_text(doc.text)
    for i, chunk_text in enumerate(chunks):
        items = _extract_chunk_items(doc, chunk_text, i, len(chunks),
                                     population, intervention, llm)
        for item in items:
            # A sentence in the chunk-overlap window can be extracted twice --
            # once per chunk it appears in. Its evidence_quote is a verbatim
            # span, so an exact repeat is the same claim, not a new one.
            quote = (item.get("evidence_quote") or "").strip()
            if quote:
                if quote in seen_quotes:
                    continue
                seen_quotes.add(quote)
            parsed_items.append(item)
    return _records_from_parsed_items(parsed_items, doc)


def _records_from_parsed_items(parsed_items: List[Dict[str, Any]],
                               doc: RetrievedDocument) -> List[EvidenceRecord]:
    """Build EvidenceRecords from one document's already-deduped, merged raw
    parsed items (across however many chunks that document needed)."""
    out: List[EvidenceRecord] = []
    tier = C.TIER_OF_SOURCE_CLASS.get(doc.source_class, 3)
    attributed = list(getattr(doc, "attributed_states", None)
                      or doc.__dict__.get("attributed_states") or [])
    for item in parsed_items:
        ftype = item.get("finding_type")
        if ftype not in (FINDING_COMPARATOR, FINDING_OUTCOME):
            continue

        # The source's own stated Member State wins; otherwise the document's.
        state = C.canonicalize_member_state(item.get("member_state", "")) or doc.member_state
        pc = item.get("population_context") or {}
        rec = EvidenceRecord(
            finding_id=_next_id("cmp" if ftype == FINDING_COMPARATOR else "out"),
            finding_type=ftype,
            subject_drug=(item.get("subject_drug") or "").strip(),
            source_id=doc.source_id,
            source_url=doc.resolved_url or doc.url,
            source_class=doc.source_class,
            tier=tier,
            member_state=state,
            organization=doc.organization,
            document_title=doc.title,
            document_date=doc.published_date,
            language=doc.language,
            retrieval_method=doc.method,
            population_context=PopulationContext(
                disease=pc.get("disease", ""),
                subtype_histology=pc.get("subtype_histology", ""),
                stage=pc.get("stage", ""),
                biomarker=pc.get("biomarker", ""),
                line_of_therapy=pc.get("line_of_therapy", ""),
                prior_therapy=pc.get("prior_therapy", ""),
                treatment_setting_intent=pc.get("treatment_setting_intent", ""),
                age_band=pc.get("age_band", ""),
                other=pc.get("other", ""),
                verbatim=pc.get("verbatim", "")),
            recommendation_strength=item.get("recommendation_strength") or C.REC_NOT_STATED,
            evidence_quote=item.get("evidence_quote", "") or "",
            evidence_locator=item.get("evidence_locator", "") or "",
            general_evidence_flag=(tier in (2, 3)
                                   and state in (C.EU_WIDE, C.GENERAL_EVIDENCE)))
        if doc.source_class == C.SRC_CONFERENCE:
            rec.evidence_status = "conference_abstract"
            # Gated the same way evidence_status is: the field is meaningful
            # only for conference sources, so it's never populated from
            # whatever the LLM happened to return for any other source class.
            rec.data_maturity = item.get("data_maturity", "") or ""

        # Deterministic peer-review classification — never asked of the LLM.
        # PubMed: real NCBI publication_types metadata decides preprint vs
        # peer-reviewed. General web: no such metadata exists, so it is always
        # "unconfirmed". Every other source class (HTA/regulatory, drug label,
        # guideline, trial registry, conference) is left "" — peer-review
        # status is not a meaningful concept for those source types.
        if doc.source_class == C.SRC_PUBMED:
            types_lower = [t.lower() for t in doc.publication_types]
            url_lower = (doc.resolved_url or doc.url or "").lower()
            if any("preprint" in t for t in types_lower):
                rec.peer_review_status = C.PEER_REVIEW_PREPRINT
            elif doc.publication_types:
                rec.peer_review_status = C.PEER_REVIEW_CONFIRMED
            elif any(d in url_lower for d in C.PREPRINT_SERVER_DOMAINS):
                # publication_types is only populated on the direct PubMed-API
                # path (documents_from_publications). A pubmed-tagged document
                # reaching here some other way (e.g. a future curated pubmed
                # domain fetched via generic web search) has none -- fall back
                # to a domain check rather than defaulting straight to
                # "unconfirmed" when the URL itself names a known preprint host.
                rec.peer_review_status = C.PEER_REVIEW_PREPRINT
            else:
                rec.peer_review_status = C.PEER_REVIEW_UNCONFIRMED
        elif doc.source_class == C.SRC_GENERAL_WEB:
            rec.peer_review_status = C.PEER_REVIEW_UNCONFIRMED

        if ftype == FINDING_COMPARATOR:
            cmp_block = item.get("comparator") or {}
            as_stated = (cmp_block.get("as_stated") or "").strip()
            if not as_stated:
                continue
            rec.comparator = Comparator(
                as_stated=as_stated,
                role=cmp_block.get("role") or C.ROLE_UNCLEAR,
                is_combination=bool(cmp_block.get("is_combination")),
                components=_coerce_component_names(cmp_block.get("components")),
                comparator_scenario=cmp_block.get("comparator_scenario") or "",
                retain_all_status=cmp_block.get("retain_all_status") or C.RETAIN_UNCONFIRMED,
                thin_mention=bool(cmp_block.get("thin_mention")))
            # INVARIANT 1, enforced at the boundary: extraction never sets a
            # comparator class, whatever the model returned.
            rec.comparator.class_mechanism = ""
            rec.comparator.class_source = ""
        else:
            ob = item.get("outcome") or {}
            measure = (ob.get("measure") or "").strip()
            if not measure:
                continue
            rec.outcome = OutcomeMention(
                measure=measure,
                result=ob.get("result", "") or "",
                unit=ob.get("unit", "") or "",
                instrument=ob.get("instrument", "") or "",
                requirement_type=ob.get("requirement_type", "") or "",
                is_requirement=bool(ob.get("is_requirement")))
        out.append(rec)

    # One document can legitimately be the curated source for several Member
    # States (a shared pan-European guideline is the common case). When the
    # SOURCE itself names no state, credit every state whose curated list points
    # at this document — but only for comparators, since outcomes are one
    # EU-wide list and per-state duplication would be meaningless there.
    if len(attributed) > 1:
        extra: List[EvidenceRecord] = []
        for rec in out:
            if rec.finding_type != FINDING_COMPARATOR:
                continue
            if (item_state := rec.member_state) not in attributed:
                continue
            for state in attributed:
                if state == item_state:
                    continue
                clone = EvidenceRecord(**{**rec.__dict__,
                                          "finding_id": _next_id("cmp"),
                                          "member_state": state})
                extra.append(clone)
        out.extend(extra)
    return out


# ClinicalTrials.gov itself puts this literal prefix in outcomesModule
# measure text for multi-arm trials -- "Arm 1: Overall response rate",
# "Arm 1 and 3: ...", "Arm 1, Arm 2 and Arm 3: ...". It is a mechanical
# registry artifact, not a clinical distinction, and it is exactly why
# "Overall response rate" fails the deliberately-exact catalog match in
# a14_a18_consolidation._catalog_key(). A regex catches every observed
# variant reliably at every batch size; a prompt instruction does not.
_ARM_PREFIX = re.compile(r"^Arm\s+\d+(?:\s*(?:,|and)\s*(?:Arm\s+)?\d+)*\s*:\s*",
                         re.IGNORECASE)


def _strip_arm_prefix(measure: str) -> str:
    return _ARM_PREFIX.sub("", measure).strip()


def records_from_trials(trials: Sequence[TrialRecord], intervention: Intervention
                        ) -> List[EvidenceRecord]:
    """Typed trial arms straight into evidence records — no LLM.

    `armGroups[].interventions[]` already answers "what was this compared
    against". Paying a model to re-derive it from HTML is the most expensive
    possible way to read a database column.
    """
    drug = intervention.product_name
    drug_tokens = _tokens(drug)
    out: List[EvidenceRecord] = []
    for trial in trials:
        for arm in trial.comparator_arms(drug):
            role = (C.ROLE_ACTIVE_COMPARATOR
                   if "COMPARATOR" in (arm.arm_type or "").upper()
                   else C.ROLE_UNCLEAR)
            shared_quote = (f"Arm '{arm.label}' ({arm.arm_type}): "
                           f"{', '.join(arm.interventions)}. {arm.description}").strip()[:1000]

            # A "Standard of Care"/"Investigator's Choice" arm lists several
            # ALTERNATIVE single-agent options a patient could receive one
            # of, not one combination regimen given together -- CT.gov
            # represents both identically as a flat interventions list, so
            # "more than one entry" alone can't tell them apart. Confirmed
            # real bundling defect: an SoC arm listing ["Lurbinectedin",
            # "Topotecan", "Amrubicin"] was being joined into one fake
            # three-drug "comparator" instead of three real, distinct ones.
            # (name, is_combination, components, comparator_scenario)
            candidates: List[Tuple[str, bool, List[str], str]] = []
            classification = ("combination" if len(arm.interventions) <= 1
                             else classify_multi_intervention_arm(arm))
            if classification == "combination":
                name = ", ".join(arm.interventions) or arm.label
                if name:
                    candidates.append((name, len(arm.interventions) > 1,
                                       list(arm.interventions), ""))
            else:
                # "choice"/SoC: an explicit alternatives marker was found.
                # "ambiguous": no marker either way -- don't guess a
                # combination or a choice; splitting is the more useful
                # default for comparator scoping (an uninterpretable joint
                # name helps no one), flagged via comparator_scenario rather
                # than asserted confidently.
                scenario = "at_least_one" if classification == "choice" else "individualised"
                for iv in (arm.interventions or [arm.label]):
                    if iv:
                        candidates.append((iv, False, [], scenario))

            for name, is_combo, components, scenario in candidates:
                # Re-check identity against the subject drug for EACH
                # split-out candidate, not just the joined string -- a choice
                # arm can legitimately list the subject drug itself as one of
                # its options ("continue Tovorafenib, or switch to
                # Topotecan"), which comparator_arms()'s own exact-match
                # check (run against the JOINED interventions list) cannot
                # catch once the arm is split apart.
                if drug_tokens and _tokens(name) == drug_tokens:
                    continue
                rec = EvidenceRecord(
                    finding_id=_next_id("cmp"),
                    finding_type=FINDING_COMPARATOR,
                    subject_drug=drug,
                    source_id="trial-" + trial.identifier,
                    source_url=trial.url,
                    source_class=C.SRC_TRIAL_REGISTRY,
                    tier=C.TIER_OF_SOURCE_CLASS[C.SRC_TRIAL_REGISTRY],
                    member_state=C.GENERAL_EVIDENCE,
                    organization=trial.registry,
                    document_title=trial.title,
                    retrieval_method="registry_api",
                    general_evidence_flag=True,
                    comparator=Comparator(
                        as_stated=name, role=role, is_combination=is_combo,
                        components=components, comparator_scenario=scenario),
                    population_context=PopulationContext(
                        disease=", ".join(trial.conditions),
                        verbatim=(arm.description or "")[:500]),
                    evidence_quote=shared_quote,
                    evidence_locator=f"{trial.identifier} armGroups",
                    # Structured API fields are self-grounding: the value IS the
                    # field, not an inference from prose.
                    grounded=True,
                    trial_status=trial.status)
                out.append(rec)

        for raw_measure in list(trial.primary_outcomes) + list(trial.secondary_outcomes):
            if not raw_measure:
                continue
            measure = _strip_arm_prefix(raw_measure)
            out.append(EvidenceRecord(
                finding_id=_next_id("out"),
                finding_type=FINDING_OUTCOME,
                subject_drug=drug,
                source_id="trial-" + trial.identifier,
                source_url=trial.url,
                source_class=C.SRC_TRIAL_REGISTRY,
                tier=C.TIER_OF_SOURCE_CLASS[C.SRC_TRIAL_REGISTRY],
                member_state=C.GENERAL_EVIDENCE,
                document_title=trial.title,
                retrieval_method="registry_api",
                general_evidence_flag=True,
                outcome=OutcomeMention(measure=measure, is_requirement=False),
                population_context=PopulationContext(disease=", ".join(trial.conditions)),
                # Raw, un-stripped text -- the audit trail keeps the true
                # original wording even though `measure` above is cleaned.
                evidence_quote=f"{trial.identifier} outcome measure: {raw_measure}",
                evidence_locator=f"{trial.identifier} outcomesModule",
                grounded=True,
                trial_status=trial.status))
    return out
