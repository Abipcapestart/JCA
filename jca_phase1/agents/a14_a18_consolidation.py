"""
A14  Harmonization        — SME Agent 11 merge rules, vocabulary-anchored
A15  Outcome Catalog      — the catalog is SUPPLIED to the model, never assumed
A16  Consolidation        — SME Agent 12, kept; rationale batched
A17  Completeness Audit   — deterministic assertions, not an LLM checklist
A18  User Additions       — taken exactly as entered

A15 is where the largest outcome defect is fixed. The legacy prompt asked a
model whether "a standard HTA outcome catalog term applies" and gave it a
six-item `e.g.` list — two of whose items the Ground Truth marks *unlisted*.
Here the catalog is loaded from a versioned data file and passed in full, and a
coverage entry is emitted for every catalog item whether or not it was found.
"""

from __future__ import annotations

import json
import re
import os
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .. import config as C
from ..providers.llm import LLMClient
from ..schema import (ByMemberStateEntry, CatalogCoverageEntry, Comparator,
                      CompletenessReport, ConsolidatedComparator, ConsolidatedOutcome,
                      EvidenceRecord, EvidenceRef, Assertion, Intervention,
                      MemberStateSummary, MemberStateVerdict, ORIGIN_AGENT, ORIGIN_USER,
                      OutcomesView, Phase1Output, POP_LICENSED, ScopeAdjudication,
                      SourceClassAttempt, SourceRef)
from .a09_a13_validation import _normalise, identity_key


# ===========================================================================
# Outcome catalog
# ===========================================================================

def load_outcome_catalog() -> Dict[str, Any]:
    with open(os.path.join(C.DATA_DIR, "outcome_catalog.json"), encoding="utf-8") as f:
        return json.load(f)


CATALOG = load_outcome_catalog()
CATALOG_ITEMS: List[Dict[str, Any]] = CATALOG["items"]
CATALOG_BY_ID = {i["catalog_id"]: i for i in CATALOG_ITEMS}


def catalog_prompt_block() -> str:
    """The catalog, rendered for the harmonisation prompt.

    Passing the real list is the entire fix: a model cannot map to a catalog it
    has never seen, and asking it to guess produced mappings that contradict the
    Ground Truth.
    """
    lines = ["THE STANDARD OUTCOME CATALOG (complete — these are the only valid catalog_id values):"]
    for item in CATALOG_ITEMS:
        lines.append(f'- {item["catalog_id"]}: "{item["display_name"]}" '
                     f'[{item["category"]}] — also called: '
                     f'{", ".join(item["synonyms"][:6])}')
    lines.append('Any outcome that matches NONE of these takes catalog_id "" and is '
                 'reported as an additional, unlisted outcome.')
    return "\n".join(lines)


_PAREN = re.compile(r"\([^)]*\)")
_PUNCT = re.compile(r"[^a-z0-9>=]+")


def _catalog_key(text: str) -> frozenset:
    """Canonical token set for catalog matching.

    Deliberately EXACT on the canonical form, with no substring fallback. A
    loose "is this synonym contained in that concept" pass maps 'Complete
    response rate' onto 'Objective response rate' via the shared words
    'response rate' — a false `listed: true`, which is precisely the class of
    error the legacy six-item prompt produced. Under-matching is recoverable:
    an unmatched concept goes to the LLM harmoniser with the real catalog in
    hand, and failing that is surfaced honestly as unlisted.
    """
    t = _normalise(text)
    t = t.replace("≥", ">=").replace("≤", "<=")
    t = _PAREN.sub(" ", t)                    # drop "(OS)", "(PFS)" etc.
    tokens = [w for w in _PUNCT.split(t) if w]
    # Strip a trailing plural so "adverse event" == "adverse events".
    tokens = [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in tokens]
    return frozenset(tokens)


_CATALOG_KEYS: List[Tuple[frozenset, Dict[str, Any]]] = []
for _item in CATALOG_ITEMS:
    for _name in [_item["display_name"]] + _item["synonyms"]:
        _CATALOG_KEYS.append((_catalog_key(_name), _item))


def match_catalog(concept: str) -> Optional[Dict[str, Any]]:
    """Deterministic canonical match. Tried before any LLM call."""
    key = _catalog_key(concept)
    if not key:
        return None
    for candidate, item in _CATALOG_KEYS:
        if candidate == key:
            return item
    return None


# ===========================================================================
# A14 — Harmonization
# ===========================================================================

def group_comparators(records: Sequence[EvidenceRecord]
                      ) -> Dict[str, List[EvidenceRecord]]:
    """Group on substance identity. SME Agent 11's rules, applied to typed
    fields rather than to bare strings: class similarity never merges, a
    combination is never merged with a component, and two combinations sharing
    an ingredient stay separate because the key is the COMPLETE component set."""
    groups: Dict[str, List[EvidenceRecord]] = {}
    for rec in records:
        if rec.comparator is None:
            continue
        groups.setdefault(identity_key(rec.comparator), []).append(rec)
    return groups


def harmonize_outcomes(records: Sequence[EvidenceRecord], llm: Optional[LLMClient],
                       batch_size: int = 50) -> List[Tuple[str, str, List[EvidenceRecord]]]:
    """Returns [(concept, catalog_id, records)].

    Deterministic catalog matching first; the model handles only what the
    synonym table cannot place, and it is given the real catalog.
    """
    outcome_records = [r for r in records if r.outcome and r.outcome.measure]
    if not outcome_records:
        return []

    groups: Dict[str, Tuple[str, str, List[EvidenceRecord]]] = {}
    residue: List[EvidenceRecord] = []

    for rec in outcome_records:
        item = match_catalog(rec.outcome.measure)
        if item:
            key = item["catalog_id"]
            if key not in groups:
                groups[key] = (item["display_name"], item["catalog_id"], [])
            groups[key][2].append(rec)
        else:
            residue.append(rec)

    if residue and llm is not None:
        for start in range(0, len(residue), batch_size):
            batch = residue[start:start + batch_size]
            payload = (catalog_prompt_block() + "\n\nRAW OUTCOME STRINGS:\n"
                       + "\n".join(f"{i + 1}. {r.outcome.measure}"
                                   for i, r in enumerate(batch)))
            parsed = llm.call_json("a14.outcome_harmonization", payload,
                                   max_tokens=C.LLM.outcome_harmonization_max_tokens,
                                   default=None)
            if not parsed:
                for rec in batch:      # fail-safe: keep its own wording
                    _add_group(groups, rec.outcome.measure, "", rec)
                continue
            if isinstance(parsed, list):
                # Some responses omit the {"groups": ..., "excluded": ...}
                # wrapper and return the groups array directly - tolerate it
                # rather than dropping the whole batch to the fail-safe path.
                parsed = {"groups": parsed, "excluded": []}
            claimed = set()
            for g in parsed.get("groups", []) or []:
                concept = (g.get("concept") or "").strip()
                cid = (g.get("catalog_id") or "").strip()
                if cid and cid not in CATALOG_BY_ID:
                    cid = ""      # never accept an id outside the supplied catalog
                if cid:
                    concept = CATALOG_BY_ID[cid]["display_name"]
                for idx in g.get("member_indices", []) or []:
                    if isinstance(idx, int) and 1 <= idx <= len(batch):
                        claimed.add(idx)
                        _add_group(groups, concept or batch[idx - 1].outcome.measure,
                                   cid, batch[idx - 1])
            for item in parsed.get("excluded", []) or []:
                idx = item.get("index")
                if isinstance(idx, int) and 1 <= idx <= len(batch):
                    claimed.add(idx)
            for i, rec in enumerate(batch, start=1):
                if i not in claimed:
                    _add_group(groups, rec.outcome.measure, "", rec)
    elif residue:
        for rec in residue:
            _add_group(groups, rec.outcome.measure, "", rec)

    return list(groups.values())


def _add_group(groups: Dict[str, Tuple[str, str, List[EvidenceRecord]]],
               concept: str, catalog_id: str, rec: EvidenceRecord) -> None:
    key = catalog_id or _normalise(concept)
    if key not in groups:
        groups[key] = (concept, catalog_id, [])
    groups[key][2].append(rec)


# ===========================================================================
# A16 — Consolidation
# ===========================================================================

def build_comparator(key: str, comp: Comparator, records: Sequence[EvidenceRecord],
                     adjudication: ScopeAdjudication, searched_states: Iterable[str],
                     llm: Optional[LLMClient],
                     or_map: Optional[Dict[str, List[str]]] = None,
                     population_ids: Optional[List[str]] = None,
                     adjudications_by_population: Optional[Dict[str, ScopeAdjudication]] = None
                     ) -> ConsolidatedComparator:
    from .a09_a13_validation import assign_member_states

    # Same de-prioritisation rule as A13's per-state selection, applied here
    # too: terminated-trial evidence is never dropped, only ranked behind
    # non-terminated evidence at the same tier — this governs which record's
    # value wins the per-comparator `next(...)` picks below (line of therapy,
    # scenario, retain status, recommendation strength) and the order
    # sources/evidence are listed in. Stable sort: relative order is otherwise
    # unchanged. [CONFIRMED FROM SME]
    records = sorted(records, key=lambda r: (r.tier, r.is_terminated_trial))

    per_state = assign_member_states(records, searched_states)
    states = [v.member_state for v in per_state if v.verdict == C.STATE_STANDARD_OF_CARE]
    tiers = sorted({r.tier for r in records})

    sources, seen_src = [], set()
    evidence, seen_ev = [], set()
    for r in records:
        if r.source_id not in seen_src:
            seen_src.add(r.source_id)
            sources.append(SourceRef(
                source_id=r.source_id, display_name=_display_name(r),
                url=r.source_url, tier=r.tier, source_class=r.source_class,
                organization=r.organization, document_title=r.document_title,
                document_date=r.document_date, language=r.language))
        ev_key = (r.source_id, r.evidence_quote[:120])
        if r.evidence_quote and ev_key not in seen_ev:
            seen_ev.add(ev_key)
            evidence.append(EvidenceRef(
                source_id=r.source_id, quote=r.evidence_quote[:1200],
                locator=r.evidence_locator, member_state=r.member_state,
                grounded=r.grounded, validation_verdict=r.validation.verdict,
                refetched=r.validation.refetched,
                evidence_status=r.evidence_status, data_maturity=r.data_maturity,
                trial_status=r.trial_status, peer_review_status=r.peer_review_status))

    line_of_therapy = next(
        (r.population_context.line_of_therapy for r in records
         if r.population_context.line_of_therapy), C.NOT_STATED_BY_SOURCE)
    scenario = next((r.comparator.comparator_scenario for r in records
                     if r.comparator and r.comparator.comparator_scenario), "")
    retain = next((r.comparator.retain_all_status for r in records
                   if r.comparator and r.comparator.retain_all_status
                   and r.comparator.retain_all_status != C.RETAIN_UNCONFIRMED),
                  C.RETAIN_UNCONFIRMED)
    strength = next((r.recommendation_strength for r in records
                     if r.recommendation_strength
                     and r.recommendation_strength != C.REC_NOT_STATED), C.REC_NOT_STATED)

    indication, scope_note = _synthesise_indication(records, llm)
    display = comp.as_stated if not comp.inn else comp.inn.capitalize()

    consolidated = ConsolidatedComparator(
        generic_name=display,
        inn=comp.inn,
        atc_code=comp.atc_code,
        class_or_mechanism=comp.class_mechanism,
        class_source=comp.class_source or "unresolved",
        is_combination=comp.is_combination,
        components=list(comp.components),
        role=C.ROLE_ACTIVE_COMPARATOR,
        indication=indication,
        indication_scope_note=scope_note,
        line_of_therapy=line_of_therapy,
        comparator_scenario=scenario,
        retain_all_status=retain,
        recommendation_strength=strength,
        member_states=states,
        per_member_state=per_state,
        tiers=tiers,
        cross_tier_confirmed=len(tiers) > 1,
        or_alternative=bool(or_map and any(
            len(or_map.get(s, [])) > 1 for s in states)),
        general_evidence_flag=all(r.general_evidence_flag for r in records) and bool(records),
        aliases_merged=list(dict.fromkeys(
            (r.comparator.raw_as_stated or r.comparator.as_stated) for r in records
            if r.comparator
            and _normalise(r.comparator.raw_as_stated or r.comparator.as_stated)
               != _normalise(display))),
        scope_adjudication=adjudication,
        adjudications_by_population=dict(adjudications_by_population or {}),
        sources=sources,
        evidence=evidence,
        population_ids=list(population_ids or [POP_LICENSED]))

    consolidated.rationale = _compose_comparator_rationale(consolidated, llm)
    return consolidated


def _display_name(rec: EvidenceRecord) -> str:
    """The UI shows source NAMES, not raw URLs."""
    if rec.organization and rec.document_title:
        return f"{rec.organization} — {rec.document_title}"[:120]
    if rec.document_title:
        return rec.document_title[:120]
    if rec.organization:
        return rec.organization[:120]
    label = {
        C.SRC_HTA_REGULATORY: "National HTA report",
        C.SRC_DRUG_LABEL: "EMA product label (EPAR)",
        C.SRC_CLINICAL_GUIDELINE: "Clinical guideline",
        C.SRC_TRIAL_REGISTRY: "Clinical trial registry",
        C.SRC_PUBMED: "Peer-reviewed publication",
        C.SRC_CONFERENCE: "Conference abstract",
        C.SRC_GENERAL_WEB: "Web source",
    }.get(rec.source_class, "Source")
    host = (rec.source_url or "").split("//")[-1].split("/")[0]
    return f"{label} ({host})" if host else label


def _synthesise_indication(records: Sequence[EvidenceRecord],
                           llm: Optional[LLMClient]) -> Tuple[str, str]:
    contexts = [r.population_context.summary() for r in records
                if r.population_context.summary()]
    contexts = list(dict.fromkeys(contexts))
    if not contexts:
        return "", ""
    if len(contexts) == 1 or llm is None:
        return contexts[0], ("" if len(contexts) == 1 else
                             "Sources state more than one population context; the "
                             "narrowest is shown. Review the full list.")
    payload = "INDICATION CONTEXT AS EACH SOURCE STATED IT:\n" + "\n".join(
        f"- {c}" for c in contexts)
    parsed = llm.call_json("a16.indication_synthesis", payload,
                           max_tokens=C.LLM.indication_synthesis_max_tokens,
                           default=None)
    if not parsed:
        return contexts[0], ""
    return parsed.get("indication", contexts[0]), parsed.get("scope_note", "")


def _compose_comparator_rationale(c: ConsolidatedComparator,
                                  llm: Optional[LLMClient]) -> str:
    if llm is None:
        return (f"{c.generic_name} is included at {c.line_of_therapy.lower()} "
                f"on the evidence of {len(c.sources)} source(s) across "
                f"{len(c.member_states)} Member State(s).")
    payload = (f"COMPARATOR: {c.generic_name}"
               f"{f' (INN {c.inn})' if c.inn else ''}\n"
               f"CLASS / MECHANISM: {c.class_or_mechanism or '(not resolved)'}\n"
               f"LINE OF THERAPY AS SOURCES STATE IT: {c.line_of_therapy}\n"
               f"INDICATION CONTEXT: {c.indication or '(not stated)'}\n"
               f"MEMBER STATES WHERE IT IS STANDARD OF CARE: "
               f"{len(c.member_states)} of 27\n"
               f"TIERS REPRESENTED: {c.tiers}\n"
               f"RECOMMENDATION STRENGTH: {c.recommendation_strength}\n")
    try:
        return llm.call("a16.comparator_rationale", payload,
                        max_tokens=C.LLM.comparator_rationale_max_tokens).strip()
    except Exception:  # noqa: BLE001
        return (f"{c.generic_name} is included at {c.line_of_therapy.lower()}; "
                f"rationale could not be composed automatically.")


def build_outcomes(groups: Sequence[Tuple[str, str, List[EvidenceRecord]]],
                   llm: Optional[LLMClient]) -> OutcomesView:
    view = OutcomesView()
    by_id: Dict[str, ConsolidatedOutcome] = {}

    for concept, catalog_id, records in groups:
        # Same de-prioritisation as build_comparator(): terminated-trial
        # evidence is never dropped, only ranked behind non-terminated
        # evidence at the same tier for the unit/instrument/requirement-type
        # `next(...)` picks below. Stable sort. [CONFIRMED FROM SME]
        records = sorted(records, key=lambda r: (r.tier, r.is_terminated_trial))
        item = CATALOG_BY_ID.get(catalog_id)
        category = item["category"] if item else _infer_category(concept, records)
        tiers = sorted({r.tier for r in records})

        units = [r.outcome.unit for r in records if r.outcome and r.outcome.unit]
        unit = units[0] if units else C.NOT_STATED_BY_SOURCE
        unit_note = ""
        if len({u.lower() for u in units}) > 1:
            unit_note = (f"Sources disagree on unit: {sorted(set(units))}. "
                         f"Noted rather than silently resolved.")

        instrument = next((r.outcome.instrument for r in records
                           if r.outcome and r.outcome.instrument), "")
        if not instrument and item:
            instrument = item.get("instrument") or ""
        req_type = next((r.outcome.requirement_type for r in records
                         if r.outcome and r.outcome.requirement_type), "")
        if not req_type and any(r.outcome and r.outcome.is_requirement for r in records):
            req_type = "relative_effect_required"

        sources, seen = [], set()
        evidence = []
        for r in records:
            if r.source_id not in seen:
                seen.add(r.source_id)
                sources.append(SourceRef(
                    source_id=r.source_id, display_name=_display_name(r),
                    url=r.source_url, tier=r.tier, source_class=r.source_class,
                    organization=r.organization, document_title=r.document_title))
            if r.evidence_quote:
                evidence.append(EvidenceRef(
                    source_id=r.source_id, quote=r.evidence_quote[:1200],
                    locator=r.evidence_locator, grounded=r.grounded,
                    validation_verdict=r.validation.verdict,
                    refetched=r.validation.refetched,
                    evidence_status=r.evidence_status, data_maturity=r.data_maturity,
                    trial_status=r.trial_status, peer_review_status=r.peer_review_status))

        outcome = ConsolidatedOutcome(
            concept=concept, category=category, catalog_id=catalog_id,
            listed=bool(catalog_id), instrument=instrument,
            unit_of_measurement=unit, unit_disagreement_note=unit_note,
            requirement_type=req_type, coverage_status=C.EV_FOUND,
            tiers=tiers, cross_tier_confirmed=len(tiers) > 1,
            sources=sources, evidence=evidence,
            aliases_merged=list(dict.fromkeys(
                r.outcome.measure for r in records
                if r.outcome and _normalise(r.outcome.measure) != _normalise(concept))))
        outcome.rationale = _compose_outcome_rationale(outcome, llm)
        by_id[outcome.catalog_id or outcome.concept] = outcome
        _bucket(view, outcome)

    # Coverage over EVERY catalog item, found or not. This is what makes a
    # missing Quality-of-Life outcome a visible result instead of an absence.
    for item in CATALOG_ITEMS:
        matched = next((o for o in view.all_outcomes()
                        if o.catalog_id == item["catalog_id"]), None)
        view.catalog_coverage.append(CatalogCoverageEntry(
            catalog_id=item["catalog_id"], display_name=item["display_name"],
            category=item["category"],
            status=C.EV_FOUND if matched else C.EV_NONE,
            matched_outcome_id=matched.outcome_id if matched else "",
            detail="" if matched else
                   "no source retrieved in this run stated this outcome"))
    return view


def _bucket(view: OutcomesView, outcome: ConsolidatedOutcome) -> None:
    if outcome.category == C.CAT_SAFETY:
        view.safety.append(outcome)
    elif outcome.category == C.CAT_QOL:
        view.quality_of_life.append(outcome)
    elif outcome.category == C.CAT_COA:
        view.clinician_patient_reported.append(outcome)
    else:
        view.clinical_effectiveness.append(outcome)


_CATEGORY_HINTS = {
    C.CAT_SAFETY: ["adverse", "safety", "toxicit", "discontinuation", "mortality due"],
    C.CAT_QOL: ["quality of life", "qol", "health status", "well-being", "utility",
                "eq-5d", "sf-36"],
    C.CAT_COA: ["patient-reported", "clinician-reported", "symptom", "fatigue",
                "functioning", "functional status"],
}


def _infer_category(concept: str, records: Sequence[EvidenceRecord]) -> str:
    n = _normalise(concept)
    for cat, hints in _CATEGORY_HINTS.items():
        if any(h in n for h in hints):
            return cat
    return C.CAT_CLINICAL


def _compose_outcome_rationale(o: ConsolidatedOutcome, llm: Optional[LLMClient]) -> str:
    if llm is None:
        return (f"{o.concept} is included as a {o.category.lower()} outcome, "
                f"supported by {len(o.sources)} source(s).")
    payload = (f"OUTCOME: {o.concept}\nCATEGORY: {o.category}\n"
               f"UNIT: {o.unit_of_measurement}\n"
               f"INSTRUMENT: {o.instrument or '(none named)'}\n"
               f"STATED AS A SCOPE REQUIREMENT: "
               f"{'yes' if o.requirement_type else 'no — reported result only'}\n")
    try:
        return llm.call("a16.outcome_rationale", payload,
                        max_tokens=C.LLM.outcome_rationale_max_tokens).strip()
    except Exception:  # noqa: BLE001
        return f"{o.concept} is included as a {o.category.lower()} outcome."


def build_by_member_state(comparators: Sequence[ConsolidatedComparator],
                          attempts: Sequence[SourceClassAttempt]
                          ) -> Tuple[List[ByMemberStateEntry], List[str], MemberStateSummary]:
    """Exactly 27 entries. Comparators referenced BY NAME ONLY — SME Agent 12 is
    explicit that no comparator detail is duplicated into this view."""
    failures: Dict[str, List[Dict[str, str]]] = {}
    searched: Dict[str, bool] = {}
    for a in attempts:
        if a.member_state not in C.EU_27_MEMBER_STATES:
            continue
        searched[a.member_state] = searched.get(a.member_state, False) or a.attempted
        if a.status in (C.EV_RETRIEVAL_FAILED, C.EV_SOURCE_INACCESSIBLE,
                        C.EV_NO_CURATED_SOURCE):
            failures.setdefault(a.member_state, []).append(
                {"source_class": a.source_class, "status": a.status, "detail": a.detail})

    entries, not_identified = [], []
    for state in C.EU_27_MEMBER_STATES:
        names = sorted({c.generic_name for c in comparators if state in c.member_states})
        status_flag = ("complete" if searched.get(state) and state not in failures
                       else ("partial" if searched.get(state) else "failed"))
        if names:
            entries.append(ByMemberStateEntry(
                member_state=state, status="identified", comparators=names,
                state_search_status=status_flag,
                search_failure_detail=failures.get(state, [])))
        else:
            not_identified.append(state)
            entries.append(ByMemberStateEntry(
                member_state=state, status="not_identified", comparators=[],
                state_search_status=status_flag,
                finding=C.NOT_IDENTIFIED_COMPARATOR_TEXT,   # exact SME wording
                search_failure_detail=failures.get(state, [])))

    summary = MemberStateSummary(identified_count=len(entries) - len(not_identified),
                                 not_identified_count=len(not_identified))
    return entries, not_identified, summary


def detect_or_alternatives(comparators: Sequence[ConsolidatedComparator]
                           ) -> Dict[str, List[str]]:
    per_state: Dict[str, List[str]] = {s: [] for s in C.EU_27_MEMBER_STATES}
    for c in comparators:
        for s in c.member_states:
            per_state[s].append(c.generic_name)
    return {s: names for s, names in per_state.items() if len(names) > 1}


# ===========================================================================
# A17 — Completeness audit
# ===========================================================================

def audit_completeness(output: Phase1Output,
                       attempts: Sequence[SourceClassAttempt],
                       populations_processed: Sequence[str]) -> CompletenessReport:
    """Deterministic assertions over states x source classes x outcome categories.

    The SME expresses completeness as self-validation checklist items inside each
    agent. A checklist an LLM ticks is not a guarantee; an assertion that fails
    the run is.
    """
    report = CompletenessReport(
        source_class_matrix=list(attempts),
        populations_processed=list(populations_processed))

    searched = {a.member_state for a in attempts if a.attempted}
    failed = {a.member_state for a in attempts if a.status == C.EV_RETRIEVAL_FAILED}
    with_findings = {e.member_state for e in output.by_member_state
                     if e.status == "identified"}
    report.member_states = {
        "total": 27,
        "searched": len(searched & set(C.EU_27_MEMBER_STATES)),
        "with_findings": len(with_findings),
        "no_findings": 27 - len(with_findings),
        "search_failed": len(failed & set(C.EU_27_MEMBER_STATES)),
    }

    for category in C.OUTCOME_CATEGORIES:
        items = [i for i in CATALOG_ITEMS if i["category"] == category]
        covered = [c for c in output.outcomes.catalog_coverage
                   if c.category == category and c.status == C.EV_FOUND]
        found_any = any(o.category == category for o in output.outcomes.all_outcomes())
        report.outcome_categories.append({
            "category": category,
            "status": C.EV_FOUND if found_any else C.EV_NONE,
            "catalog_items_covered": len(covered),
            "catalog_items_total": len(items),
        })

    report.assertions = [
        Assertion("by_member_state_has_27_entries",
                  len(output.by_member_state) == 27,
                  f"found {len(output.by_member_state)}"),
        Assertion("member_state_summary_sums_to_27",
                  bool(output.member_state_summary)
                  and (output.member_state_summary.identified_count
                       + output.member_state_summary.not_identified_count) == 27),
        Assertion("every_comparator_has_grounded_evidence",
                  all(c.evidence for c in output.comparators
                      if c.origin == ORIGIN_AGENT)),
        Assertion("no_comparator_carries_the_intervention_class",
                  all(c.class_source in ("atc_vocabulary", "source_stated",
                                         "user_added", "unresolved", "")
                      for c in output.comparators)),
        Assertion("every_comparator_has_27_state_verdicts",
                  all(len(c.per_member_state) == 27 for c in output.comparators)),
        Assertion("all_catalog_outcomes_have_a_status",
                  len(output.outcomes.catalog_coverage) == len(CATALOG_ITEMS)),
        Assertion("no_pico_sets_in_phase1_output", True,
                  "PICO-set generation is Phase 2 and is absent by design"),
    ]
    return report


# ===========================================================================
# A18 — User additions
# ===========================================================================

def add_user_comparator(output: Phase1Output, generic_name: str,
                        member_states: Sequence[str],
                        class_or_mechanism: str = "",
                        line_of_therapy: str = "",
                        rationale: str = "") -> ConsolidatedComparator:
    """A user-added comparator is taken EXACTLY as entered (SME step 14).

    Drug name and Member State are both mandatory. No validation, no
    enrichment, no adjudication — the user's judgment replaces the pipeline's.
    """
    if not generic_name.strip():
        raise ValueError("A manually added comparator requires a drug name.")
    states = [s for s in (C.canonicalize_member_state(m) for m in member_states) if s]
    if not states:
        raise ValueError("A manually added comparator requires at least one "
                         "valid EU-27 Member State.")

    per_state = [
        MemberStateVerdict(
            member_state=s,
            verdict=(C.STATE_STANDARD_OF_CARE if s in states else C.STATE_NOT_ESTABLISHED),
            reason="entered by the reviewer" if s in states else "not selected by the reviewer")
        for s in C.EU_27_MEMBER_STATES]

    comparator = ConsolidatedComparator(
        generic_name=generic_name.strip(),
        class_or_mechanism=class_or_mechanism.strip(),
        class_source="user_added" if class_or_mechanism.strip() else "",
        line_of_therapy=line_of_therapy.strip() or C.NOT_STATED_BY_SOURCE,
        member_states=states,
        per_member_state=per_state,
        tiers=[],
        rationale=rationale.strip() or "Added by the reviewer.",
        origin=ORIGIN_USER)
    output.comparators.append(comparator)
    _rebuild_state_views(output)
    return comparator


def add_user_outcome(output: Phase1Output, concept: str, category: str,
                     unit_of_measurement: str = "", instrument: str = "",
                     rationale: str = "") -> ConsolidatedOutcome:
    if not concept.strip():
        raise ValueError("A manually added outcome requires a name.")
    if category not in C.OUTCOME_CATEGORIES:
        raise ValueError(f"category must be one of {C.OUTCOME_CATEGORIES}")
    item = match_catalog(concept)
    outcome = ConsolidatedOutcome(
        concept=concept.strip(), category=category,
        catalog_id=item["catalog_id"] if item else "",
        listed=bool(item),
        instrument=instrument.strip(),
        unit_of_measurement=unit_of_measurement.strip() or C.NOT_STATED_BY_SOURCE,
        coverage_status=C.EV_FOUND,
        rationale=rationale.strip() or "Added by the reviewer.",
        origin=ORIGIN_USER)
    _bucket(output.outcomes, outcome)
    if item:
        for cov in output.outcomes.catalog_coverage:
            if cov.catalog_id == item["catalog_id"]:
                cov.status = C.EV_FOUND
                cov.matched_outcome_id = outcome.outcome_id
                cov.detail = "added by the reviewer"
    return outcome


def _rebuild_state_views(output: Phase1Output) -> None:
    entries, not_identified, summary = build_by_member_state(
        output.comparators, output.completeness.source_class_matrix)
    output.by_member_state = entries
    output.not_identified_states = not_identified
    output.member_state_summary = summary
