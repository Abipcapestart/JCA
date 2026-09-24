"""
A9   Grounding             — does the value trace to the retrieved text?
A10  Claim Validation      — does the RE-FETCHED source support the claim here?
A11  Comparator Identity   — substance identity, and the comparator's OWN class
A12  Scope Adjudication    — separated from candidate generation, on purpose
A13  Per-State Assignment  — a verdict for all 27, not just states with findings

A9 and A10 are separate because they fail differently. Grounding catches
fabrication; claim validation catches "right drug, wrong line of therapy". The
legacy implementation fused them and did neither: it compared a value against a
±140-character window of already-fetched text and called that validation.
"""

from __future__ import annotations

import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .. import config as C
from ..providers.llm import LLMClient
from ..providers.search import SearchProvider
from ..schema import (Comparator, EvidenceCitation, EvidenceRecord, FINDING_COMPARATOR,
                      Intervention, MemberStateVerdict, RetrievedDocument,
                      ScopeAdjudication, ScopeBoundary, ValidationOutcome)

# ===========================================================================
# A9 — Grounding
# ===========================================================================

GROUNDING_OVERLAP_THRESHOLD = 0.62
_WORD = re.compile(r"[a-z0-9]+")
_NUM = re.compile(r"\d[\d.,]*")


def _normalise(text: str) -> str:
    """NFKC so a ligature in a PDF ('eﬀective') matches its typed form, then
    strip combining diacritics so an accented spelling from a French-language
    source ('Tovorafénib') tokenizes identically to the plain INN
    ('Tovorafenib'). Confirmed missing this step in a real production run:
    every French HAS-sourced claim about the subject drug was being rejected
    as WRONG_SUBJECT_DRUG because the accent alone split the word in two."""
    t = unicodedata.normalize("NFKC", text or "").lower()
    t = unicodedata.normalize("NFKD", t)
    return "".join(ch for ch in t if not unicodedata.combining(ch))


def _tokens(text: str) -> List[str]:
    return [w for w in _WORD.findall(_normalise(text)) if len(w) > 2]


def is_grounded(value: str, document_text: str) -> Tuple[bool, str]:
    """Exact substring, else a word-overlap floor.

    Deliberately a cheap check. Its job is to catch invention, not to judge
    meaning — that is A10's job, and conflating the two is what made the legacy
    validator both permissive and blind.
    """
    if not value:
        return False, "empty value"
    doc = _normalise(document_text)
    if not doc:
        return False, "no document text"
    if _normalise(value) in doc:
        return True, "exact"
    vt = _tokens(value)
    if not vt:
        return False, "no comparable tokens"
    present = sum(1 for t in vt if t in doc)
    ratio = present / len(vt)
    if ratio >= GROUNDING_OVERLAP_THRESHOLD:
        return True, f"overlap {ratio:.2f}"
    return False, f"overlap {ratio:.2f} below {GROUNDING_OVERLAP_THRESHOLD}"


def best_quote_window(value: str, document_text: str, pad: int = 200) -> str:
    """Pick the densest window around the value.

    Anchors on NUMERALS first and weights them, because a quote that shares no
    distinctive token with the value is worse than no quote — it looks like
    traceability and is not.
    """
    doc = document_text or ""
    low = _normalise(doc)
    anchors: List[int] = []
    for num in _NUM.findall(value or "")[:4]:
        idx = low.find(num.lower())
        if idx >= 0:
            anchors.append(idx)
    for tok in _tokens(value)[:8]:
        idx = low.find(tok)
        if idx >= 0:
            anchors.append(idx)
    if not anchors:
        return ""
    value_tokens = set(_tokens(value))
    value_nums = set(n.lower() for n in _NUM.findall(value or ""))
    best, best_score = "", 0.0
    for a in anchors:
        start, end = max(0, a - pad), min(len(doc), a + pad)
        window = doc[start:end]
        wl = _normalise(window)
        score = (sum(3 for n in value_nums if n in wl)
                 + sum(1 for t in value_tokens if t in wl))
        if score > best_score:
            best, best_score = window.strip(), score
    return best if best_score > 0 else ""


def ground_records(records: Sequence[EvidenceRecord],
                   documents: Sequence[RetrievedDocument]) -> List[EvidenceRecord]:
    """Ground each record against the document it came from."""
    by_id = {d.source_id: d for d in documents}
    for rec in records:
        if rec.grounded:
            continue   # structured API records are self-grounding
        doc = by_id.get(rec.source_id)
        if doc is None or not doc.text:
            rec.grounded = False
            rec.grounding_note = "source document not available for grounding"
            continue
        value = (rec.comparator.as_stated if rec.comparator
                 else (rec.outcome.measure if rec.outcome else ""))
        ok_value, note_value = is_grounded(value, doc.text)
        ok_quote, note_quote = (is_grounded(rec.evidence_quote, doc.text)
                                if rec.evidence_quote else (False, "no quote"))
        rec.grounded = ok_value and ok_quote
        rec.grounding_note = f"value: {note_value}; quote: {note_quote}"
        if not rec.evidence_quote or not ok_quote:
            better = best_quote_window(value, doc.text)
            if better:
                rec.evidence_quote = better[:1200]
                rec.grounded = ok_value
                rec.grounding_note = f"value: {note_value}; quote: recovered from source"
    return list(records)


# ===========================================================================
# A10 — Claim validation (with a real re-fetch)
# ===========================================================================

def validate_claims(records: Sequence[EvidenceRecord], population_indication: str,
                    intervention: Intervention, search: SearchProvider,
                    llm: LLMClient, max_workers: int = C.RETRIEVAL.max_workers,
                    documents: Optional[Sequence[RetrievedDocument]] = None
                    ) -> List[EvidenceRecord]:
    """Re-open each cited source and confirm the exact claim, in context.

    The SME is explicit: "Open the cited source directly — do not evaluate a
    finding based on its stated citation alone." Batched per source, which the
    SME permits, with the same standard applied regardless of batch size.

    `documents`, when given, is the set of documents A7 already fetched this
    pass (see orchestrator._run_retrieval_pass). Two distinct uses:

    1. PubMed abstracts in there (method == "pubmed_api") came from NCBI's
       own E-utilities API, not a Tavily scrape -- confirmed in production,
       Tavily's `fetch()` fails on every single pubmed.ncbi.nlm.nih.gov URL
       ("extract returned no results"), so every PubMed-sourced claim was
       being lost as SOURCE_INACCESSIBLE despite the actual abstract text
       already sitting in hand. These skip the re-fetch attempt entirely and
       go straight to validation against that text -- a re-fetch here has a
       measured 0% success rate, so attempting it first only wastes time.
    2. For every OTHER source, a genuine independent re-fetch is still
       attempted first, per the SME's explicit requirement ("open the cited
       source directly -- do not evaluate a finding based on its stated
       citation alone"). Only if that live re-fetch itself fails (network
       error, page taken down, Tavily extract failure) does validation fall
       back to the copy A7 already fetched, rather than losing the claim
       outright -- a live source that is merely flaky at this moment still
       gets judged against real text instead of disappearing as
       SOURCE_INACCESSIBLE. `ValidationOutcome.refetched` records which of
       the two actually happened for that record.
    """
    drug = intervention.product_name
    already_fetched: Dict[str, RetrievedDocument] = {
        (d.resolved_url or d.url): d for d in (documents or [])
        if d.method == "pubmed_api" and d.ok and d.text
    }
    fallback_docs: Dict[str, RetrievedDocument] = {
        (d.resolved_url or d.url): d for d in (documents or [])
        if d.ok and d.text
    }

    # Deterministic pre-filter: subject-drug and role. Both are hard rules that
    # do not need a model, and both are cheaper caught here.
    survivors: List[EvidenceRecord] = []
    for rec in records:
        if not rec.grounded:
            rec.validation = ValidationOutcome(
                verdict=C.V_NOT_SUPPORTED,
                reason="not grounded in the retrieved source text")
            continue
        if rec.retrieval_method == "registry_api":
            # A structured API record is self-evidencing for the fields the API
            # types: the comparator IS `armGroups[].interventions[]`, not an
            # inference from prose, so there is no citation to re-open and
            # nothing a re-read could contradict. It still goes through scope
            # adjudication (A12), which is where a registry arm from the wrong
            # population gets caught.
            rec.validation = ValidationOutcome(
                verdict=C.V_SUPPORTED, refetched=False, attempts=0,
                reason=("structured registry field; self-evidencing, so no document "
                        "re-fetch applies. Scope relevance is still adjudicated."))
            continue
        if rec.comparator is not None:
            # role == unclear means extraction itself wasn't sure how this
            # mention relates to the subject drug -- hard-rejecting it here
            # makes that uncertainty final before the LLM in a10 ever sees
            # the full re-fetched document text. Confirmed in a real
            # production run: a role-ambiguous mention of the correct
            # comparator was deterministically thrown out as
            # WRONG_SUBJECT_DRUG on a false-positive subject-drug match.
            # Confidently role-tagged records are unaffected -- this only
            # defers the ambiguous case to a10's real judgment.
            #
            # For a comparator claim, subject_drug naming the COMPARATOR
            # itself (not the requested intervention) is the normal, correct
            # shape of this data -- not a mismatch. The SME's a10 prompt says
            # so explicitly ("ABSOLUTE RULE FOR COMPARATOR CLAIMS" in
            # prompts.registry._CLAIM_VALIDATION), but that instruction only
            # reaches the LLM's judgment -- it can't override a deterministic
            # rejection that happens before the LLM is ever called. Confirmed
            # in production: this pre-filter was hard-rejecting nearly every
            # confidently-role-tagged comparator claim, since subject_drug
            # naming the comparator always differs from the requested drug by
            # construction. Only fast-reject here when subject_drug is a
            # genuine THIRD drug -- neither the requested intervention nor
            # the comparator itself -- mirroring the prompt's own carve-out.
            if (_is_other_drug(rec.subject_drug, drug)
                    and _is_other_drug(rec.subject_drug, rec.comparator.as_stated)
                    and rec.comparator.role != C.ROLE_UNCLEAR):
                rec.validation = ValidationOutcome(
                    verdict=C.V_WRONG_SUBJECT_DRUG,
                    reason=(f"claim is about {rec.subject_drug!r}, neither the requested "
                            f"intervention {drug!r} nor the comparator "
                            f"{rec.comparator.as_stated!r}"))
                continue
            if rec.comparator.role in (C.ROLE_INTERVENTION_ARM,):
                rec.validation = ValidationOutcome(
                    verdict=C.V_WRONG_INTERVENTION,
                    reason="this is the intervention's own arm, not a comparator")
                continue
            if _same_substance(rec.comparator.as_stated, drug):
                rec.validation = ValidationOutcome(
                    verdict=C.V_WRONG_INTERVENTION,
                    reason="a comparator cannot be the intervention being assessed")
                continue
        survivors.append(rec)

    by_source: Dict[str, List[EvidenceRecord]] = {}
    for rec in survivors:
        by_source.setdefault(rec.source_url, []).append(rec)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        # Fetch each cited URL exactly once (a genuine re-fetch, per the
        # SME's explicit requirement) and THEN split its claims into bounded
        # sub-batches -- so one dense source's claims don't all have to fit
        # in a single a10.claim_validation response. Splitting into
        # sub-batches without also splitting the fetch out this way would
        # re-fetch the same document once per sub-batch. A source already
        # fetched via a real API this pass (PubMed) skips this network
        # re-fetch entirely and goes straight into the same sub-batch
        # dispatch below, using the document already in hand.
        validate_futures = []
        for url, batch in list(by_source.items()):
            cached_doc = already_fetched.get(url)
            if cached_doc is not None:
                del by_source[url]
                for start in range(0, len(batch), _MAX_CLAIMS_PER_VALIDATION_BATCH):
                    sub = batch[start:start + _MAX_CLAIMS_PER_VALIDATION_BATCH]
                    validate_futures.append(pool.submit(
                        _validate_batch, url, sub, cached_doc, population_indication,
                        drug, llm, refetched=False))
        fetch_futures = {pool.submit(search.fetch, url, full_document=True,
                                     use_cache=False): (url, batch)
                         for url, batch in by_source.items()}
        for fut in as_completed(fetch_futures):
            url, batch = fetch_futures[fut]
            try:
                doc = fut.result()
            except Exception as exc:  # noqa: BLE001
                doc = None
            if doc is None or not doc.ok or not doc.text:
                detail = (doc.error or doc.status) if doc is not None else "fetch failed"
                # The live re-fetch itself failed -- before giving up on this
                # claim entirely, fall back to the copy A7 already fetched
                # this pass, if one exists. This is a fallback for a live
                # source that is flaky RIGHT NOW, not a substitute for the
                # independent re-fetch attempted above -- that attempt always
                # happens first.
                doc = fallback_docs.get(url)
                refetched = False
                if doc is None:
                    for rec in batch:
                        rec.validation = ValidationOutcome(
                            verdict=C.V_SOURCE_INACCESSIBLE, refetched=True, attempts=1,
                            reason=f"source could not be re-opened: {detail}")
                    continue
            else:
                refetched = True
            for start in range(0, len(batch), _MAX_CLAIMS_PER_VALIDATION_BATCH):
                sub = batch[start:start + _MAX_CLAIMS_PER_VALIDATION_BATCH]
                validate_futures.append(pool.submit(
                    _validate_batch, url, sub, doc, population_indication, drug, llm,
                    refetched=refetched))
        for fut in as_completed(validate_futures):
            try:
                fut.result()
            except Exception:  # noqa: BLE001 — a batch failure must fail closed,
                continue        # which _validate_batch already does per record
    return list(records)


# Bounds how many claims from one re-fetched source go into a single
# a10.claim_validation call. A source discussing many claims at once (a
# national HTA transcript naming the subject drug plus several comparators)
# was putting a dozen+ claims in one unbounded batch, needing a reasoned
# verdict for every one inside a single token ceiling -- confirmed real
# defect: when that response truncated, EVERY claim in the batch, including
# genuinely on-target ones, failed closed. Same philosophy as
# _EXTRACTION_CHUNK_CHARS for a08.extraction: bound the batch directly so
# the token cap is never the only thing standing between a dense source and
# a mass failed-closed loss.
_MAX_CLAIMS_PER_VALIDATION_BATCH = 10


def _validate_batch(url: str, batch: List[EvidenceRecord], doc: RetrievedDocument,
                    indication: str, drug: str, llm: LLMClient,
                    refetched: bool = True) -> None:
    lines = []
    for i, rec in enumerate(batch, start=1):
        if rec.comparator:
            claim = (f"COMPARATOR {rec.comparator.as_stated!r} "
                     f"(role: {rec.comparator.role}) for subject drug "
                     f"{rec.subject_drug or drug!r}")
        else:
            o = rec.outcome
            claim = (f"OUTCOME {o.measure!r}"
                     + (f" = {o.result!r}" if o and o.result else "")
                     + (" [stated as a REQUIRED scope outcome]" if o and o.is_requirement else ""))
        lines.append(f"{i}. {claim}\n   population context stated by the source: "
                     f"{rec.population_context.summary() or '(none stated)'}\n"
                     f"   quoted evidence: {rec.evidence_quote[:400]}")

    doc_label = "RE-FETCHED SOURCE DOCUMENT" if refetched else "SOURCE DOCUMENT (already fetched)"
    payload = (f"REQUESTED POPULATION: {indication}\n"
               f"REQUESTED INTERVENTION: {drug}\n\n"
               f"CLAIMS TO VALIDATE:\n" + "\n".join(lines)
               + f"\n\n{doc_label} ({url}):\n{doc.text[:100000]}")

    parsed = llm.call_json("a10.claim_validation", payload,
                           max_tokens=C.LLM.validation_max_tokens, default=None)
    results = (parsed or {}).get("results") if isinstance(parsed, dict) else None
    if not results:
        # Fail CLOSED. An unparseable response must never pass a finding
        # through unchecked.
        for rec in batch:
            rec.validation = ValidationOutcome(
                verdict=C.V_NOT_SUPPORTED, refetched=refetched, attempts=1,
                reason="validator response could not be parsed; failed closed")
        return

    seen = set()
    for item in results:
        idx = item.get("index")
        if not isinstance(idx, int) or not (1 <= idx <= len(batch)):
            continue
        seen.add(idx)
        rec = batch[idx - 1]
        verdict = item.get("verdict", C.V_NOT_SUPPORTED)
        rec.validation = ValidationOutcome(
            verdict=verdict, reason=item.get("reason", ""), refetched=refetched, attempts=1)
    for i, rec in enumerate(batch, start=1):
        if i not in seen:
            rec.validation = ValidationOutcome(
                verdict=C.V_NOT_SUPPORTED, refetched=refetched, attempts=1,
                reason="validator did not return a verdict for this claim; failed closed")


_STOP = {"plus", "and", "with", "alone", "monotherapy", "therapy", "treatment",
         "regimen", "combination", "based", "chemotherapy"}


def _substance_tokens(text: str) -> set:
    return {t for t in _WORD.findall(_normalise(text))
            if len(t) > 3 and t not in _STOP}


def _same_substance(a: str, b: str) -> bool:
    ta, tb = _substance_tokens(a), _substance_tokens(b)
    return bool(ta) and ta == tb


def _is_other_drug(subject: str, requested: str) -> bool:
    """True when the record's subject drug is clearly a DIFFERENT drug.

    Silence is not a mismatch — an empty subject_drug means the extractor could
    not tell, which is handled by the LLM validator with the document in hand.
    """
    if not subject:
        return False
    ts, tr = _substance_tokens(subject), _substance_tokens(requested)
    if not ts or not tr:
        return False
    return not (ts & tr)


# ===========================================================================
# A11 — Comparator identity
# ===========================================================================

def _load_vocab() -> Dict[str, Any]:
    import os
    path = os.path.join(C.DATA_DIR, "inn_atc_seed.json")
    if not os.path.exists(path):
        return {"substances": []}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_VOCAB = _load_vocab()
_BY_ALIAS: Dict[str, Dict[str, Any]] = {}
for _s in _VOCAB.get("substances", []):
    for _alias in [_s["inn"]] + _s.get("brand_names", []) + _s.get("aliases", []):
        _BY_ALIAS[_normalise(_alias)] = _s


def _coerce_component_names(raw: Any) -> List[str]:
    """`components` is documented to the model as a bare string list, but the
    schema example shows an empty array with no worked item -- and the
    sibling top-level field `brand_names` makes it a reasonable model
    inference that each entry should be a rich per-component object
    (inn/display_name/brand_names/class_mechanism/atc_code) instead, exactly
    like the item it sits inside. A real production run hit this: Claude
    returned that richer shape for several genuine combination regimens
    (e.g. "Dabrafenib + Trametinib"), and an un-coerced dict reaching
    identity_key()'s unicodedata.normalize() crashed the whole run with a
    TypeError, downstream of any single record. Accept either shape.

    `inn` is preferred over `display_name`/`name` on purpose: this same run
    showed the LLM populating a clean inn ("dabrafenib") alongside a
    brand-annotated display_name ("Dabrafenib (SPEXOTRAS)") for one source's
    mention of a combination three OTHER sources named without the brand --
    preferring display_name would have kept that one mention from merging
    into the same identity_key() group as the rest, undercounting
    cross-source confirmation for no reason a reviewer would want. `name` is
    the per-component key in the schema fixed for Fix 4 (each component now
    resolves its own inn/atc_code/class_mechanism); `display_name` is kept as
    a fallback for the older flat shape.
    """
    out: List[str] = []
    for c in raw or []:
        if isinstance(c, str):
            name = c.strip()
        elif isinstance(c, dict):
            name = str(c.get("inn") or c.get("name") or c.get("display_name") or "").strip()
        else:
            name = str(c or "").strip()
        if name:
            out.append(name)
    return out


def _cluster_key(name: str) -> frozenset:
    """Canonical token set for near-duplicate clustering: strip brand-name
    parentheticals first ("Dabrafenib (Finlee)" -> "Dabrafenib"), then reuse
    the existing stopword-filtered substance tokenizer -- punctuation, word
    order, and connector words ("plus"/"and"/"with") already fall away there,
    which is exactly what's needed to recognise "Dabrafenib + Trametinib" and
    "Trametinib (Spexotras) + Dabrafenib" as the same candidate."""
    stripped = re.sub(r"\([^)]*\)", " ", name)
    return frozenset(_substance_tokens(stripped))


def precluster_candidates(distinct: List[str]) -> Dict[str, List[str]]:
    """Group near-duplicate candidate strings by identical canonical token
    set, BEFORE any batching happens. Returns {representative: [all variants
    including representative]}, representative = the shortest (usually
    cleanest, un-annotated) variant.

    A11's own prompt-level judgment already merges near-duplicate wordings
    within a single batch (confirmed working in production: 4 worded
    variants of "Dabrafenib + Trametinib" collapsed into one identity). But
    A11 has zero visibility ACROSS batches -- once raw candidate volume
    exceeds batch_size, two variants of the same regimen can land in
    different batches and never be shown to the model together, no matter
    how good the prompt is. This collapses them deterministically first, so
    a duplicate pair can never be split by a batch boundary regardless of how
    many raw candidates exist. Conservative by design: exact canonical-token-
    set equality only, so "Carboplatin + Vincristine" and "Carboplatin +
    Vincristine + Etoposide" (genuinely different regimens sharing tokens)
    correctly stay separate rather than being fuzzy-merged.
    """
    clusters: Dict[frozenset, List[str]] = {}
    for name in distinct:
        key = _cluster_key(name)
        clusters.setdefault(key, []).append(name)
    return {min(variants, key=len): variants for variants in clusters.values()}


def resolve_identities(records: Sequence[EvidenceRecord], llm: Optional[LLMClient],
                       batch_size: int = 40
                       ) -> Tuple[Dict[str, Comparator], List[Dict[str, str]]]:
    """Resolve every distinct comparator string to a substance identity.

    Vocabulary lookup FIRST. A curated INN/ATC table is deterministic, free and
    auditable; the model is for the residue only. The legacy implementation had
    a five-entry hardcoded brand table and asked an LLM for everything else,
    including the class — which is how a comparator inherited the intervention's
    mechanism.
    """
    distinct: List[str] = []
    seen = set()
    for rec in records:
        if rec.comparator and rec.comparator.as_stated:
            key = _normalise(rec.comparator.as_stated)
            if key not in seen:
                seen.add(key)
                distinct.append(rec.comparator.as_stated)

    # Deterministic pre-clustering -- see precluster_candidates(). Only each
    # cluster's REPRESENTATIVE goes through vocabulary lookup / batching /
    # A11 below; every other variant's identity is propagated from its
    # representative's result at the end, never resolved independently.
    clusters = precluster_candidates(distinct)
    representatives = list(clusters.keys())

    resolved: Dict[str, Comparator] = {}
    unresolved: List[str] = []
    excluded: List[Dict[str, str]] = []

    for name in representatives:
        entry = _BY_ALIAS.get(_normalise(name))
        if entry:
            resolved[_normalise(name)] = Comparator(
                as_stated=name, inn=entry["inn"], atc_code=entry.get("atc", ""),
                class_mechanism=entry.get("class_mechanism", ""),
                class_source="atc_vocabulary",
                is_combination=False, components=[entry["inn"]])
        else:
            unresolved.append(name)

    if unresolved and llm is not None:
        for start in range(0, len(unresolved), batch_size):
            batch = unresolved[start:start + batch_size]
            payload = "\n".join(f"{i + 1}. {n}" for i, n in enumerate(batch))
            parsed = llm.call_json("a11.comparator_identity", payload,
                                   max_tokens=C.LLM.comparator_identity_max_tokens,
                                   default=None)
            if not parsed:
                continue
            for item in parsed.get("items", []) or []:
                idx = item.get("index")
                if not isinstance(idx, int) or not (1 <= idx <= len(batch)):
                    continue
                name = batch[idx - 1]
                is_cat = bool(item.get("is_category"))
                is_combo = bool(item.get("is_combination"))
                components_raw = item.get("components") or []
                # Fix 4: the schema now lets a combination resolve EACH
                # component's own inn/class_mechanism instead of one flat
                # value for the whole candidate -- that flat value was
                # empty for every real combination, so class_source came out
                # "unresolved" for all of them. Roll the per-component
                # identities up into the same flat Comparator fields here,
                # once, so every downstream consumer (build_comparator's
                # export columns) sees a resolved value with no separate
                # change needed there.
                comp_inns = [c.get("inn", "").strip() for c in components_raw
                            if isinstance(c, dict) and c.get("inn")]
                comp_classes = [c.get("class_mechanism", "").strip() for c in components_raw
                               if isinstance(c, dict) and c.get("class_mechanism")]
                inn = ("+".join(comp_inns) if is_combo and comp_inns
                      else (item.get("inn") or "").strip())
                class_mechanism = (" + ".join(comp_classes) if is_combo and comp_classes
                                  else (item.get("class_mechanism") or "").strip())
                display = (item.get("display_name") or inn or name).strip()
                comp = Comparator(
                    as_stated=name,
                    inn=inn if not is_cat else "",
                    atc_code=item.get("atc_code", "") or "",
                    class_mechanism=class_mechanism,
                    class_source="source_stated" if class_mechanism else "unresolved",
                    is_combination=is_combo,
                    components=_coerce_component_names(components_raw))
                comp.as_stated = display or name
                resolved[_normalise(name)] = comp
            for item in parsed.get("excluded", []) or []:
                idx = item.get("index")
                if isinstance(idx, int) and 1 <= idx <= len(batch):
                    name = batch[idx - 1]
                    resolved.pop(_normalise(name), None)
                    excluded.append({"value": name,
                                     "reason": (item.get("reason") or "").strip()})

    # Anything still unresolved keeps its own wording. Under-resolution is
    # recoverable; inventing an identity is not. But a name A11 explicitly
    # EXCLUDED (e.g. "surgery and chemotherapy" -- not a resolvable substance
    # identity at all) must not be resurrected here just because `distinct`
    # still lists it -- that silently undid the exclusion in production,
    # letting a candidate A11 said to drop reach A12 adjudication anyway.
    excluded_norm = {_normalise(e["value"]) for e in excluded}
    for name in representatives:
        if _normalise(name) in excluded_norm:
            continue
        resolved.setdefault(_normalise(name), Comparator(
            as_stated=name, inn="", class_mechanism="", class_source="unresolved"))

    # Propagate each representative's resolved identity (or exclusion) onto
    # every near-duplicate variant clustered under it -- a variant never goes
    # through vocabulary lookup or A11 itself, only its representative does.
    excluded_by_norm = {_normalise(e["value"]): e for e in excluded}
    final_excluded: List[Dict[str, str]] = []
    for rep, variants in clusters.items():
        rep_norm = _normalise(rep)
        rep_exclusion = excluded_by_norm.get(rep_norm)
        if rep_exclusion is not None:
            for variant in variants:
                final_excluded.append({"value": variant, "reason": rep_exclusion["reason"]})
            continue
        rep_identity = resolved[rep_norm]
        for variant in variants:
            resolved[_normalise(variant)] = rep_identity
    return resolved, final_excluded


def filter_a11_excluded(records: Sequence[EvidenceRecord],
                        excluded: Sequence[Dict[str, str]]) -> List[EvidenceRecord]:
    """Drop any record whose comparator A11 explicitly excluded (not a
    resolvable substance identity at all) before it reaches A14 grouping /
    A12 adjudication. Must run right after resolve_identities(); without it,
    an excluded candidate still forms its own identity group and gets
    adjudicated (usually to `uncertain`, since it has no real identity)."""
    if not excluded:
        return list(records)
    excluded_norm = {_normalise(e["value"]) for e in excluded}
    return [r for r in records
           if not (r.comparator and _normalise(r.comparator.as_stated) in excluded_norm)]


def _copy_identity_fields(target: Comparator, source: Comparator) -> None:
    """Copy one resolved identity's fields onto another comparator object.

    Shared by apply_identities() (propagating A11's per-string resolution
    onto every precluster variant) and apply_identity_merges() (propagating
    the A11b audit's cross-identity merge decision onto every record of the
    "loser" identity) -- factored out so the two merge paths can never drift
    apart on which fields get copied.
    """
    target.inn = source.inn
    target.atc_code = source.atc_code
    target.class_mechanism = source.class_mechanism
    target.class_source = source.class_source
    target.is_combination = source.is_combination or target.is_combination
    if source.components:
        target.components = source.components
    # Propagate the cluster's canonical display wording too, not just its
    # substance identity -- without this, a category/generic comparator
    # with no INN (e.g. "standard of care") still fragments at A14's
    # identity_key() fallback, because that key uses as_stated when inn
    # is empty, and as_stated was never unified across variants here.
    # Preserve the original wording first (build_comparator() needs it
    # for aliases_merged) -- once overwritten below it's gone for good.
    if not target.raw_as_stated:
        target.raw_as_stated = target.as_stated
    target.as_stated = source.as_stated or target.as_stated


def apply_identities(records: Sequence[EvidenceRecord],
                     identities: Dict[str, Comparator]) -> None:
    for rec in records:
        if not rec.comparator:
            continue
        ident = identities.get(_normalise(rec.comparator.as_stated))
        if not ident:
            continue
        _copy_identity_fields(rec.comparator, ident)


def identity_key(comp: Comparator) -> str:
    """Merge key. INN when known; otherwise the normalised stated wording.

    Combinations merge on their COMPLETE component set — two regimens sharing
    one ingredient are two regimens, never a third synthesised one.
    """
    if comp.is_combination and comp.components:
        return "+".join(sorted(_normalise(c) for c in comp.components))
    if comp.inn:
        return _normalise(comp.inn)
    return _normalise(comp.as_stated)


def audit_resolved_identities(identities: Dict[str, Comparator],
                              llm: Optional[LLMClient]
                              ) -> List[Tuple[str, str, str]]:
    """A11b — a SECOND, narrower LLM call over A11's own already-resolved
    output, mirroring A12's "generate broadly, then a separate precise pass"
    split. resolve_identities()'s single call does both generation and
    deduplication over a growing list; precluster_candidates() is purely
    literal-token-based and can NEVER catch a cross-language duplicate
    ("chemioterapia"/"chimiothérapie"/"karboplatyna i winkrystyna" naming the
    same regimen in three languages) -- only a second, focused look at the
    DISTINCT resolved identities themselves can.

    Returns (loser_key, canonical_key, reason) tuples, keyed by
    identity_key(), for apply_identity_merges() to apply -- reason is the
    LLM's own one-sentence justification, carried through purely for Excel
    auditability. Deliberately NOT batched: the input is already the
    distinct, deduped identity set (typically well under 40), a different
    shape of problem from A11's own unresolved-string batches.
    """
    if llm is None:
        return []
    distinct: Dict[str, Comparator] = {}
    for comp in identities.values():
        distinct.setdefault(identity_key(comp), comp)
    keys = list(distinct.keys())
    if len(keys) < 2:
        return []

    lines = []
    for i, key in enumerate(keys, start=1):
        comp = distinct[key]
        lines.append(f"{i}. display_name={comp.as_stated!r}, inn={comp.inn!r}, "
                     f"atc_code={comp.atc_code!r}, class_mechanism={comp.class_mechanism!r}, "
                     f"is_combination={comp.is_combination}, components={comp.components!r}")
    payload = "\n".join(lines)
    parsed = llm.call_json("a11b.comparator_identity_audit", payload,
                           max_tokens=C.LLM.comparator_identity_audit_max_tokens,
                           default=None)
    if not parsed:
        return []

    merges: List[Tuple[str, str, str]] = []
    for group in parsed.get("duplicate_groups", []) or []:
        canonical_idx = group.get("canonical_index")
        duplicate_idxs = group.get("duplicate_indices") or []
        reason = (group.get("reason") or "").strip()
        if not isinstance(canonical_idx, int) or not (1 <= canonical_idx <= len(keys)):
            continue
        canonical_key = keys[canonical_idx - 1]
        for idx in duplicate_idxs:
            if isinstance(idx, int) and 1 <= idx <= len(keys) and idx != canonical_idx:
                merges.append((keys[idx - 1], canonical_key, reason))
    return merges


def apply_identity_merges(records: Sequence[EvidenceRecord],
                          identities: Dict[str, Comparator],
                          merges: Sequence[Tuple[str, str, str]]) -> None:
    """Apply audit_resolved_identities()'s merge decisions onto every record
    of the "loser" identity, rewriting it to the canonical identity's fields.
    Runs AFTER apply_identities() and BEFORE group_comparators() (A14), so a
    merged identity lands in the same group from the start -- no downstream
    stage needs to know an audit merge happened."""
    if not merges:
        return
    by_key: Dict[str, Comparator] = {}
    for comp in identities.values():
        by_key.setdefault(identity_key(comp), comp)
    canonical_by_loser = {loser: by_key[canonical] for loser, canonical, _reason in merges
                          if canonical in by_key}
    if not canonical_by_loser:
        return
    for rec in records:
        if not rec.comparator:
            continue
        canonical = canonical_by_loser.get(identity_key(rec.comparator))
        if canonical is not None:
            _copy_identity_fields(rec.comparator, canonical)


# ===========================================================================
# A12 — Scope adjudication
# ===========================================================================

def adjudicate_scope(candidate_key: str, comp: Comparator,
                     records: Sequence[EvidenceRecord], boundary: ScopeBoundary,
                     intervention: Intervention, llm: Optional[LLMClient]
                     ) -> ScopeAdjudication:
    """Is this candidate genuinely in scope for THIS population?

    Retrieval is deliberately broad. This is the step that makes it precise —
    and separating the two is what lets recall and precision be tuned
    independently. Both supporting and opposing evidence are retained, so an
    exclusion is reviewable rather than a silent drop.
    """
    version = "a12.scope_adjudication/v1"

    # Deterministic rejection: nothing here is an active comparator.
    roles = {r.comparator.role for r in records if r.comparator}
    if roles and roles.isdisjoint({C.ROLE_ACTIVE_COMPARATOR, C.ROLE_UNCLEAR}):
        role = sorted(roles)[0]
        return ScopeAdjudication(
            comparator_inn=comp.inn or comp.as_stated, verdict=C.SCOPE_OUT,
            decisive_facet="role", adjudicator_version=version,
            reason=(f"every source describes this as {role.replace('_', ' ')}, "
                    f"not as a comparator for the requested population"),
            evidence_against=[EvidenceCitation(
                source_id=r.source_id, source_url=r.source_url,
                quote=r.evidence_quote[:400], member_state=r.member_state, tier=r.tier,
                facet_conflict="role",
                detail=f"role={r.comparator.role}") for r in records if r.comparator][:5])

    if llm is None:
        # Without an adjudicator, be honest rather than permissive.
        return ScopeAdjudication(
            comparator_inn=comp.inn or comp.as_stated, verdict=C.SCOPE_UNCERTAIN,
            reason="no adjudicator available; surfaced for reviewer judgment",
            adjudicator_version=version,
            evidence_for=[_cite(r) for r in records[:5]])

    lines = []
    for r in records[:20]:
        lines.append(
            f"- source_id: {r.source_id} | tier {r.tier} | {r.source_class} | "
            f"{r.member_state}\n"
            f"  population stated by this source: "
            f"{r.population_context.summary() or '(none stated)'}\n"
            f"  role: {r.comparator.role if r.comparator else 'n/a'}\n"
            f"  quote: {r.evidence_quote[:300]}")

    payload = (
        f"REQUESTED INTERVENTION: {intervention.product_name}\n\n"
        f"REQUESTED POPULATION FACETS:\n{boundary.describe()}\n\n"
        f"LICENSED / CLAIMED INDICATION WORDING: "
        f"{boundary.licensed_indication_wording or '(not established)'}\n\n"
        f"CANDIDATE COMPARATOR: {comp.as_stated}"
        f"{f' (INN {comp.inn})' if comp.inn else ''}\n\n"
        f"EVIDENCE RETRIEVED FOR THIS CANDIDATE:\n" + "\n".join(lines))

    parsed = llm.call_json("a12.scope_adjudication", payload,
                           max_tokens=C.LLM.scope_adjudication_max_tokens,
                           default=None)
    if not parsed:
        return ScopeAdjudication(
            comparator_inn=comp.inn or comp.as_stated, verdict=C.SCOPE_UNCERTAIN,
            reason="adjudicator response could not be parsed; surfaced rather than dropped",
            adjudicator_version=version,
            evidence_for=[_cite(r) for r in records[:5]])

    verdict = parsed.get("verdict", C.SCOPE_UNCERTAIN)
    if verdict not in (C.SCOPE_IN, C.SCOPE_OUT, C.SCOPE_UNCERTAIN):
        verdict = C.SCOPE_UNCERTAIN
    by_id = {r.source_id: r for r in records}

    def _to_citations(items, conflict: bool) -> List[EvidenceCitation]:
        out = []
        for it in items or []:
            rec = by_id.get(it.get("source_id", ""))
            out.append(EvidenceCitation(
                source_id=it.get("source_id", ""),
                source_url=rec.source_url if rec else "",
                quote=(rec.evidence_quote[:400] if rec else ""),
                member_state=rec.member_state if rec else "",
                tier=rec.tier if rec else 3,
                facets_matched=list(it.get("facets_matched") or []),
                facet_conflict=it.get("facet_conflict", "") if conflict else "",
                detail=it.get("detail", "")))
        return out

    return ScopeAdjudication(
        comparator_inn=comp.inn or comp.as_stated,
        verdict=verdict,
        decisive_facet=parsed.get("decisive_facet", ""),
        reason=parsed.get("reason", ""),
        evidence_for=_to_citations(parsed.get("evidence_for"), False),
        evidence_against=_to_citations(parsed.get("evidence_against"), True),
        adjudicator_version=version)


def _cite(r: EvidenceRecord) -> EvidenceCitation:
    return EvidenceCitation(source_id=r.source_id, source_url=r.source_url,
                            quote=r.evidence_quote[:400], member_state=r.member_state,
                            tier=r.tier)


# ===========================================================================
# A13 — Per-state assignment
# ===========================================================================

def assign_member_states(records: Sequence[EvidenceRecord],
                         searched_states: Iterable[str]) -> List[MemberStateVerdict]:
    """Produce a verdict for ALL 27 states, not only those with findings.

    The UI renders every state with a two-value legend, so "absent" must never
    be ambiguous. We assert `standard_of_care` only where a state-attributed
    source says so; everywhere else we say `not_established` and why. We do NOT
    assert `not_used`, because nothing in the evidence supports that claim —
    saying "this is not standard of care in Malta" requires a source that says
    so, and the pipeline does not have one.
    """
    searched = set(searched_states)
    by_state: Dict[str, List[EvidenceRecord]] = {}
    for rec in records:
        if rec.member_state in C.EU_27_MEMBER_STATES:
            by_state.setdefault(rec.member_state, []).append(rec)

    verdicts: List[MemberStateVerdict] = []
    for state in C.EU_27_MEMBER_STATES:
        hits = by_state.get(state, [])
        if hits:
            # Terminated-trial evidence is never excluded, only de-prioritised:
            # at the same tier, a non-terminated record wins the "best"/
            # representative slot for this state. [CONFIRMED FROM SME]
            best = sorted(hits, key=lambda r: (r.tier, r.is_terminated_trial,
                                               -len(r.evidence_quote)))[0]
            verdicts.append(MemberStateVerdict(
                member_state=state, verdict=C.STATE_STANDARD_OF_CARE,
                tier=best.tier, source_id=best.source_id, source_url=best.source_url,
                evidence_quote=best.evidence_quote[:400],
                reason="named by a source attributed to this Member State"))
        elif state in searched:
            verdicts.append(MemberStateVerdict(
                member_state=state, verdict=C.STATE_NOT_ESTABLISHED,
                reason=("this state's sources were searched and returned no "
                        "state-specific evidence naming this comparator")))
        else:
            verdicts.append(MemberStateVerdict(
                member_state=state, verdict=C.STATE_NOT_ESTABLISHED,
                reason="no curated source for this state and no fallback result"))
    return verdicts
