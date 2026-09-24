"""
Phase 1 orchestrator — the single entry point.

There is exactly ONE run path. The legacy codebase had a main orchestrator plus
two "resume and patch" scripts that rewrote part of a saved run; one of them
rewrote the comparator list without re-deriving the downstream view, which is
how a deliverable shipped with sections that disagreed with each other. Resume
here re-runs from a checkpoint through the SAME code, or it does not resume.

Stage order (see ARCHITECTURE.md for the diagram):
  A1 validate -> A2 structure -> [user confirms] -> A4 indication lock ->
  A3 scope boundary -> A5 areas -> A6 query plan -> A7 retrieve ->
  A8 extract -> A9 ground -> A10 validate claims -> A11 identity ->
  A12 adjudicate -> A14 harmonise -> A13/A15 assign+catalog -> A16 consolidate ->
  A17 completeness
"""

from __future__ import annotations

import contextlib
import copy
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import config as C
from .agents import a01_a05_input_context as inputs
from .agents import a06_a08_retrieval as retrieval
from .agents import a09_a13_validation as validation
from .agents import a14_a18_consolidation as consolidation
from .providers.llm import LLMClient
from .providers.registries import (LiteratureClient, MedicineRegistryClient,
                                   TrialRegistryClient)
from .providers.search import SearchProvider
from .prompts import registry as prompt_registry
from .schema import (EvidenceRecord, FINDING_COMPARATOR, Intervention, ORIGIN_AGENT,
                     Phase1Output, Population, SourceClassAttempt, ValidationSummary)
from .sources.workbook import SourceInventory, load_source_inventory


@dataclass
class Providers:
    """Everything the pipeline talks to. All optional: a provider that is None
    degrades that capability explicitly rather than failing the run."""
    llm: Optional[LLMClient] = None
    search: Optional[SearchProvider] = None
    trials: Optional[TrialRegistryClient] = None
    literature: Optional[LiteratureClient] = None
    medicines: Optional[MedicineRegistryClient] = None


@dataclass
class RunOptions:
    request_id: str = ""
    # Default flipped to True per explicit SME decision (2026-09-23): the
    # drug's own JCA report is now a deliberate, labeled Tier 1 source
    # (SRC_EU_JCA_REPORT), not evaluation-only. Still recorded in the
    # manifest either way, and still a one-line-reversible flag.
    allow_jca_reports: bool = True
    source_workbook: Optional[str] = None
    strict_sources: bool = True
    max_workers: int = C.RETRIEVAL.max_workers
    skip_confirmation: bool = True       # API drives confirmation; CLI auto-confirms
    progress: Optional[Callable[[str, str], None]] = None

    def emit(self, stage: str, message: str) -> None:
        if self.progress:
            try:
                self.progress(stage, message)
            except Exception:  # noqa: BLE001 — progress must never break a run
                pass


class BlockedError(RuntimeError):
    """Raised when P&I validation returns a FIX. The SME is explicit that the
    user cannot proceed past this stage while an unresolved FIX remains."""

    def __init__(self, result):
        super().__init__("P&I validation returned blocking FIX items")
        self.validation = result


def _snapshot(debug_capture: Optional[Dict[str, Any]], key: str, value: Any) -> None:
    if debug_capture is not None:
        debug_capture[key] = value


def _doc_snapshot(doc, max_chars: int = 3000) -> Dict[str, Any]:
    """RetrievedDocument has no to_dict(); asdict() plus a text cap so a debug
    capture with dozens of full documents doesn't balloon the run file."""
    d = asdict(doc)
    if len(d.get("text", "")) > max_chars:
        d["text"] = d["text"][:max_chars] + f"... [truncated, {len(doc.text)} chars total]"
    return d


@dataclass
class _RetrievalPass:
    """A6-A10 for exactly ONE population. Kept as one function so a run with a
    single population (the overwhelming common case) still does exactly what
    it always did -- called once, on populations[0]."""
    records: List[EvidenceRecord] = field(default_factory=list)
    attempts: List[SourceClassAttempt] = field(default_factory=list)
    documents: List[Any] = field(default_factory=list)
    trials: List[Any] = field(default_factory=list)
    publications: List[Any] = field(default_factory=list)
    blocked_count: int = 0
    vocab: Any = None
    plan: List[Any] = field(default_factory=list)
    followup_attempts: List[Dict[str, Any]] = field(default_factory=list)


def _run_retrieval_pass(pop: Population, intervention: Intervention, areas,
                        inventory: SourceInventory, indication_record, guard,
                        providers: Providers, opts: "RunOptions",
                        extraction_cache: Optional[Dict[str, List[EvidenceRecord]]] = None,
                        cache_lock: Optional[threading.Lock] = None
                        ) -> _RetrievalPass:
    result = _RetrievalPass()

    result.vocab = retrieval.build_vocabulary(pop, areas.areas, providers.llm)
    result.plan = retrieval.plan_queries(pop, intervention, areas.areas,
                                         result.vocab, inventory, guard)

    tag = f" ({pop.population_id})"

    if providers.search is not None:
        opts.emit("A7", f"Retrieving documents{tag}")
        web = retrieval.execute_plan(result.plan, providers.search, guard, inventory,
                                     areas.areas, opts.max_workers)
        result.documents = web.documents
        result.attempts.extend(web.attempts)
        result.blocked_count = len(web.blocked_by_guard)

        # Refinement round: plan_refinement() was fully built and enabled by
        # default (config.py's enable_refinement_round) but never actually
        # called anywhere -- every genuine coverage gap the first pass hit
        # (a state/source-class that came back empty or inaccessible) simply
        # stayed a gap for the rest of the run. Retry, with broadened terms,
        # only what was genuinely ATTEMPTED and came back empty/inaccessible
        # -- never something never attempted (EV_NOT_APPLICABLE / no curated
        # source at all), where a broader search wouldn't help anyway.
        # plan_refinement() itself already no-ops when the config flag is off.
        gaps = sorted({(a.member_state, a.source_class) for a in web.attempts
                      if a.attempted and a.status in (C.EV_NONE, C.EV_SOURCE_INACCESSIBLE,
                                                      C.EV_RETRIEVAL_FAILED)})
        if gaps:
            refinement_plan = retrieval.plan_refinement(
                gaps, pop, intervention, result.vocab, inventory, areas.areas)
            if refinement_plan:
                opts.emit("A7", f"Refinement round for {len(refinement_plan)} coverage "
                                f"gap(s){tag}")
                refined = retrieval.execute_plan(refinement_plan, providers.search, guard,
                                                 inventory, areas.areas, opts.max_workers)
                result.documents = result.documents + refined.documents
                result.attempts.extend(refined.attempts)
                result.blocked_count += len(refined.blocked_by_guard)

    opts.emit("A7", f"Querying structured registries{tag}")
    structured = retrieval.retrieve_structured(
        pop, intervention, indication_record, providers.trials, providers.literature,
        result.vocab)
    result.attempts.extend(structured.attempts)
    result.trials = list(structured.trials)
    result.publications = list(getattr(structured, "publications", []))
    # PubMed publications have no typed comparator/outcome fields to bypass
    # extraction with (unlike trials) -- route their abstracts through the
    # same shared extraction contract as every other document.
    result.documents = result.documents + retrieval.documents_from_publications(
        result.publications)

    # Shared by both the main A8 block below and Version B's follow-up round
    # further down -- always defined so the follow-up round can reuse the
    # exact same cache regardless of whether the main block's own condition
    # was true this call.
    cache = extraction_cache if extraction_cache is not None else {}
    lock = cache_lock or contextlib.nullcontext()

    if providers.llm is not None and result.documents:
        # A8 extraction is population-agnostic by design: it extracts every
        # distinct claim a document states, with population RELEVANCE decided
        # later at A12 adjudication, not here (see a08.extraction's own
        # MULTIPLE POPULATIONS note). So the same document fetched for two
        # populations gets re-extracted for identical content -- paying for,
        # and independently re-risking truncation on, an a08.extraction call
        # whose answer cannot legitimately differ. Confirmed real waste in a
        # production run: the richest guideline PDF was independently
        # extracted (and independently truncated) once per population, and
        # the resulting split verdicts on the same comparator were mistaken
        # for an A11-grouping artifact before this was traced to its root
        # cause. Cache by document identity (source_id, stable per URL) so
        # only genuinely new documents pay for a call; deep-copy each reused
        # record with a fresh finding_id, since downstream grounding/claim-
        # validation mutate a record's fields in place and each population's
        # pass must never see another population's mutations.
        with lock:
            cached_docs = [d for d in result.documents if d.source_id in cache]
            new_docs = [d for d in result.documents if d.source_id not in cache]

        for d in cached_docs:
            for rec in cache[d.source_id]:
                clone = copy.deepcopy(rec)
                clone.finding_id = retrieval._next_id(
                    "cmp" if clone.finding_type == FINDING_COMPARATOR else "out")
                result.records.append(clone)

        if new_docs:
            opts.emit("A8", f"Extracting evidence from {len(new_docs)} document(s){tag}")
            new_records = retrieval.extract_from_documents(
                new_docs, pop, intervention, providers.llm, opts.max_workers)
            result.records.extend(new_records)
            by_source: Dict[str, List[EvidenceRecord]] = {}
            for rec in new_records:
                by_source.setdefault(rec.source_id, []).append(rec)
            with lock:
                for d in new_docs:
                    cache[d.source_id] = by_source.get(d.source_id, [])
    result.records.extend(retrieval.records_from_trials(result.trials, intervention))

    # ---- Version B: comparator follow-up round -----------------------------
    # A comparator name that organically surfaced from real round-1 evidence
    # with only a thin mention (comparator.thin_mention) gets ONE additional,
    # name-targeted retrieval round -- closes the gap standard_of_care_
    # candidates can't reach (a comparator the LLM's prior knowledge doesn't
    # already associate with this disease, but real evidence just named).
    # Additive only: new documents/records fall through to the SAME A9/A10
    # calls below, exactly like plan_refinement()'s web documents already do
    # -- no separate grounding/validation pipeline.
    thin_candidates = retrieval._dedup([
        rec.comparator.as_stated for rec in result.records
        if rec.comparator and rec.comparator.thin_mention])
    if C.RETRIEVAL.enable_comparator_followup_round and thin_candidates:
        bounded_candidates = thin_candidates[:C.RETRIEVAL.max_comparator_candidates]
        opts.emit("A7b", f"Follow-up search for {len(bounded_candidates)} thinly-evidenced "
                         f"comparator name(s){tag}")
        followup_docs: List[Any] = []
        # Looped per-candidate (not one execute_plan() call for all of them)
        # so each candidate's own document count/query can be tracked
        # individually for Excel visibility -- execute_plan()'s own
        # concurrency doesn't preserve a plan-item-to-attempt correlation.
        if providers.search is not None:
            for candidate in bounded_candidates:
                cand_plan = retrieval.plan_comparator_followup(
                    [candidate], pop, intervention, areas.areas, inventory)
                if not cand_plan:
                    continue
                cand_result = retrieval.execute_plan(
                    cand_plan, providers.search, guard, inventory, areas.areas,
                    opts.max_workers)
                followup_docs += cand_result.documents
                result.blocked_count += len(cand_result.blocked_by_guard)
                docs_found = len(cand_result.documents)
                result.followup_attempts.append({
                    "candidate_name": candidate,
                    "tier": C.TIER_OF_SOURCE_CLASS.get(C.SRC_CLINICAL_GUIDELINE, 1),
                    "query_sent": cand_plan[0].query, "documents_retrieved": docs_found,
                    "status": C.EV_FOUND if docs_found else C.EV_NONE})
        if providers.literature is not None:
            for candidate in bounded_candidates:
                pubs = retrieval.pubmed_comparator_followup(
                    [candidate], providers.literature, result.vocab,
                    pop.value("indication_disease"), intervention.product_name)
                followup_docs += retrieval.documents_from_publications(pubs)
                result.followup_attempts.append({
                    "candidate_name": candidate,
                    "tier": C.TIER_OF_SOURCE_CLASS.get(C.SRC_PUBMED, 2),
                    "query_sent": f"PubMed: {candidate}", "documents_retrieved": len(pubs),
                    "status": C.EV_FOUND if pubs else C.EV_NONE})

        if followup_docs and providers.llm is not None:
            # Same cached/new_docs split as the main A8 block above, keyed by
            # source_id, so a URL round 1 already fetched/extracted is never
            # re-extracted here.
            with lock:
                cached_followup = [d for d in followup_docs if d.source_id in cache]
                new_followup = [d for d in followup_docs if d.source_id not in cache]
            for d in cached_followup:
                for rec in cache[d.source_id]:
                    clone = copy.deepcopy(rec)
                    clone.finding_id = retrieval._next_id(
                        "cmp" if clone.finding_type == FINDING_COMPARATOR else "out")
                    result.records.append(clone)
            if new_followup:
                opts.emit("A7b", f"Extracting evidence from {len(new_followup)} follow-up "
                                 f"document(s){tag}")
                new_followup_records = retrieval.extract_from_documents(
                    new_followup, pop, intervention, providers.llm, opts.max_workers)
                result.records.extend(new_followup_records)
                by_followup_source: Dict[str, List[EvidenceRecord]] = {}
                for rec in new_followup_records:
                    by_followup_source.setdefault(rec.source_id, []).append(rec)
                with lock:
                    for d in new_followup:
                        cache[d.source_id] = by_followup_source.get(d.source_id, [])
        if followup_docs:
            result.documents = result.documents + followup_docs

    opts.emit("A9", f"Grounding extracted values in their source text{tag}")
    result.records = validation.ground_records(result.records, result.documents)

    if providers.llm is not None and providers.search is not None:
        opts.emit("A10", f"Re-opening cited sources and validating claims{tag}")
        result.records = validation.validate_claims(
            result.records, pop.value("indication_disease"), intervention,
            providers.search, providers.llm, opts.max_workers,
            documents=result.documents)

    return result


def run_phase1(population_input: Dict[str, Any],
               intervention_input: Dict[str, Any],
               providers: Providers,
               free_text: str = "",
               added_fields: Optional[List[str]] = None,
               options: Optional[RunOptions] = None,
               debug_capture: Optional[Dict[str, Any]] = None) -> Phase1Output:
    """`debug_capture`, when passed an empty dict, is filled in-place with a
    raw snapshot of every stage's intermediate state (every extracted record
    before/after grounding and validation, every retrieval attempt, every
    candidate comparator's scope adjudication including the ones that did NOT
    make the final cut) -- for debugging why something is or isn't in the
    final output. None by default: existing callers see no change at all."""
    opts = options or RunOptions()
    started = time.time()
    out = Phase1Output(request_id=opts.request_id or f"jca-{uuid.uuid4().hex[:10]}")
    out.raw_population_input = dict(population_input)
    out.raw_intervention_input = dict(intervention_input)

    # ---- A1 ---------------------------------------------------------------
    opts.emit("A1", "Validating population and intervention fields")
    pi_validation = inputs.validate_pi(population_input, intervention_input,
                                       added_fields, providers.llm)
    out.input_validation = pi_validation.to_dict()
    if not pi_validation.passed:
        raise BlockedError(pi_validation)

    # ---- A2 ---------------------------------------------------------------
    opts.emit("A2", "Structuring input and tagging provenance")
    populations, intervention = inputs.structure_pi(
        population_input, intervention_input, free_text, providers.llm)
    out.populations = populations
    out.intervention = intervention

    # ---- A4 ---------------------------------------------------------------
    opts.emit("A4", "Establishing the licensed indication and leakage guard")
    indication_record, guard = inputs.lock_indication(
        intervention, providers.medicines, providers.search, providers.llm,
        allow_jca_reports=opts.allow_jca_reports)
    out.licensed_indication_record = indication_record
    intervention.inn_resolved = indication_record.inn or intervention.product_name
    intervention.atc_code = indication_record.atc_code

    # ---- A3 ---------------------------------------------------------------
    opts.emit("A3", "Building the population scope boundary")
    boundaries = [inputs.build_scope_boundary(p, intervention, indication_record,
                                              providers.llm)
                  for p in populations]
    out.scope_boundaries = boundaries

    # ---- A5 ---------------------------------------------------------------
    opts.emit("A5", "Resolving therapeutic area(s)")
    areas = inputs.resolve_therapeutic_areas(populations[0], providers.llm)
    out.therapeutic_areas = areas

    # ---- source inventory -------------------------------------------------
    opts.emit("sources", "Loading the curated source list")
    inventory = load_source_inventory(opts.source_workbook, strict=opts.strict_sources)
    for p in inventory.problems:
        out.notes.append(f"[source-list {p.severity}] {p.where}: {p.message}")

    # ---- A6-A10, once per population ---------------------------------------
    # A single population (the overwhelming common case) runs this loop
    # exactly once, with exactly the calls this used to make directly against
    # populations[0] -- zero behaviour change. A second (intended-to-treat)
    # population gets its OWN query plan, retrieval, extraction and claim
    # validation, run against ITS OWN indication wording -- the SME requires
    # the two populations be kept as separate structured objects, and a
    # boundary that is only ever computed and never used to search or
    # validate anything is not actually separate, just decorative.
    opts.emit("A6", "Planning retrieval queries")
    records: List[EvidenceRecord] = []
    attempts: List[SourceClassAttempt] = []
    documents: List[Any] = []
    all_trials: List[Any] = []
    all_publications: List[Any] = []
    plans_by_population: List[tuple] = []
    followup_attempts: List[Dict[str, Any]] = []
    blocked_total = 0
    extraction_cache: Dict[str, List[EvidenceRecord]] = {}
    cache_lock = threading.Lock()

    # Each population's A6-A10 pass is fully independent (its own query plan,
    # retrieval, extraction, claim validation, against its OWN indication
    # wording -- the SME requires the populations be kept as separate
    # structured objects). No correctness reason they can't run
    # CONCURRENTLY: confirmed a real 2-population run pays the full
    # single-population wall-clock time twice, back to back, sequentially --
    # the single largest contributor to a 40-minute run. The only shared
    # mutable state between them is extraction_cache, guarded by cache_lock.
    with ThreadPoolExecutor(max_workers=max(1, len(populations))) as pop_pool:
        pass_results = list(pop_pool.map(
            lambda pop: _run_retrieval_pass(pop, intervention, areas, inventory,
                                            indication_record, guard, providers, opts,
                                            extraction_cache, cache_lock),
            populations))

    for pop, pass_result in zip(populations, pass_results):
        plans_by_population.append((pop.population_id, pass_result.vocab, pass_result.plan))
        records.extend(pass_result.records)
        attempts.extend(pass_result.attempts)
        documents.extend(pass_result.documents)
        all_trials.extend(pass_result.trials)
        all_publications.extend(pass_result.publications)
        followup_attempts.extend(pass_result.followup_attempts)
        blocked_total += pass_result.blocked_count

    if providers.search is None:
        out.notes.append("[A7] No search provider configured; web retrieval skipped.")
    if blocked_total:
        out.notes.append(
            f"[leakage guard] {blocked_total} document(s) blocked as the "
            f"target drug's own JCA report. This is the intended behaviour: the "
            f"product must anticipate a scope, not read the answer key.")
    if providers.llm is None or providers.search is None:
        out.notes.append("[A10] Claim validation skipped: it requires both an LLM and a "
                         "search provider, because it re-fetches the cited source.")

    total_plan_items = sum(len(p) for _, _, p in plans_by_population)
    opts.emit("A6", f"{total_plan_items} query units planned across "
                    f"{len({i.member_state for _, _, p in plans_by_population for i in p})} "
                    f"scopes, across {len(plans_by_population)} population(s)")
    _snapshot(debug_capture, "A6_query_plan", {
        "vocabulary": plans_by_population[0][1].to_dict() if plans_by_population else {},
        "plan": [asdict(item) for _, _, plan in plans_by_population for item in plan]})
    _snapshot(debug_capture, "A7_retrieval", {
        "attempts": [a.to_dict() for a in attempts],
        "documents": [_doc_snapshot(d) for d in documents],
        "trials_found": [asdict(t) for t in all_trials],
        "publications_found": [asdict(p) for p in all_publications]})
    _snapshot(debug_capture, "A7b_comparator_followup", {"attempts": followup_attempts})

    out.validation.records_extracted = len(records)
    _snapshot(debug_capture, "A8_extraction", [r.to_dict() for r in records])

    out.validation.grounded = sum(1 for r in records if r.grounded)
    out.validation.grounding_failed = len(records) - out.validation.grounded
    _snapshot(debug_capture, "A9_grounding", [r.to_dict() for r in records])

    _tally_validation(records, out.validation)
    _snapshot(debug_capture, "A10_claim_validation", [r.to_dict() for r in records])

    usable = [r for r in records if r.usable()]
    _snapshot(debug_capture, "usable_after_A10", [r.to_dict() for r in usable])

    # ---- A11 --------------------------------------------------------------
    opts.emit("A11", "Resolving comparator substance identity")
    identities, a11_excluded = validation.resolve_identities(usable, providers.llm)
    validation.apply_identities(usable, identities)

    # ---- A11b (identity audit) ---------------------------------------------
    # A second, narrower LLM call over A11's own already-resolved output --
    # mirrors A12's "generate broadly, then a separate precise pass" split.
    # Catches cross-language/cross-source duplicates precluster_candidates()
    # can never catch (purely literal-token-based) and A11's single
    # generate+dedupe call sometimes misses.
    opts.emit("A11b", "Auditing resolved identities for cross-language duplicates")
    merges = validation.audit_resolved_identities(identities, providers.llm)
    validation.apply_identity_merges(usable, identities, merges)

    distinct_by_key: Dict[str, Any] = {}
    for comp in identities.values():
        distinct_by_key.setdefault(validation.identity_key(comp), comp)
    canonical_by_loser_key = {loser: canonical for loser, canonical, _reason in merges}
    identity_snapshot = {}
    for k, v in identities.items():
        entry = asdict(v)
        canonical_key = canonical_by_loser_key.get(validation.identity_key(v))
        canonical = distinct_by_key.get(canonical_key) if canonical_key else None
        entry["audit_merged_into"] = canonical.as_stated if canonical else ""
        identity_snapshot[k] = entry
    _snapshot(debug_capture, "A11_identity", identity_snapshot)
    _snapshot(debug_capture, "A11b_identity_audit", {
        "merges": [{"loser_display_name": distinct_by_key[loser].as_stated
                                          if loser in distinct_by_key else loser,
                   "canonical_display_name": distinct_by_key[canonical].as_stated
                                            if canonical in distinct_by_key else canonical,
                   "reason": reason}
                  for loser, canonical, reason in merges]})
    _snapshot(debug_capture, "A11_excluded", a11_excluded)
    for e in a11_excluded:
        out.validation.excluded.append({
            "value": e["value"], "stage": "comparator_identity",
            "reason": e["reason"], "decisive_facet": None})
    usable = validation.filter_a11_excluded(usable, a11_excluded)

    # ---- A14 (grouping) ---------------------------------------------------
    opts.emit("A14", "Harmonising comparator identities")
    groups = consolidation.group_comparators(
        [r for r in usable if r.finding_type == "comparator"])
    _snapshot(debug_capture, "A14_grouping", {
        key: [r.to_dict() for r in recs] for key, recs in groups.items()})

    # ---- A12 --------------------------------------------------------------
    # Adjudicated against EVERY population's boundary, not just boundaries[0]
    # -- a comparator that is only relevant to the broader intended-to-treat
    # population must still surface, tagged with which population(s)
    # confirmed it, rather than being silently judged only against the
    # licensed population's (possibly narrower) facets.
    opts.emit("A12", f"Adjudicating scope for {len(groups)} candidate comparator(s) "
                    f"across {len(boundaries)} population(s)")
    searched_states = {a.member_state for a in attempts
                       if a.attempted and a.member_state in C.EU_27_MEMBER_STATES}

    # One LLM call per (candidate group, population) -- previously a plain
    # nested loop with zero concurrency, calling adjudicate_scope() once at a
    # time. Confirmed the biggest hidden serial block in the whole pipeline:
    # a real run showed 14+ of these calls, each paying full network +
    # generation latency back to back. No shared mutable state between
    # adjudications, so this is a pure fan-out/fan-in, same pattern already
    # used for A7/A8/A10.
    adjudication_work = [(key, recs, b) for key, recs in groups.items() for b in boundaries]

    def _adjudicate_one(item):
        key, recs, b = item
        comp = recs[0].comparator
        return key, b.population_id, validation.adjudicate_scope(
            key, comp, recs, b, intervention, providers.llm)

    adjs_by_key: Dict[str, Dict[str, Any]] = {key: {} for key in groups}
    if adjudication_work:
        with ThreadPoolExecutor(max_workers=opts.max_workers) as pool:
            futures = [pool.submit(_adjudicate_one, item) for item in adjudication_work]
            for fut in as_completed(futures):
                key, pid, adj = fut.result()
                adjs_by_key[key][pid] = adj

    in_scope: Dict[str, Any] = {}
    all_adjudications: Dict[str, Any] = {}
    for key, recs in groups.items():
        comp = recs[0].comparator
        adjs = adjs_by_key[key]
        all_adjudications[key] = {
            "comparator": comp.to_dict(), "num_records": len(recs),
            "adjudication_by_population": {pid: a.to_dict() for pid, a in adjs.items()}}
        pop_ids_in = [pid for pid, a in adjs.items()
                     if a.verdict in (C.SCOPE_IN, C.SCOPE_UNCERTAIN)]
        # Representative adjudication for the comparator's single
        # scope_adjudication/rationale field: prefer an IN verdict over an
        # UNCERTAIN one, in population order (licensed first).
        best = (next((adjs[b.population_id] for b in boundaries
                     if adjs[b.population_id].verdict == C.SCOPE_IN), None)
               or next((adjs[b.population_id] for b in boundaries
                        if adjs[b.population_id].verdict == C.SCOPE_UNCERTAIN), None))
        if pop_ids_in:
            if best.verdict == C.SCOPE_IN:
                out.validation.scope_in += 1
            else:
                out.validation.scope_uncertain += 1
            in_scope[key] = (comp, recs, best, pop_ids_in, adjs)
        else:
            out.validation.scope_out += 1
            out.validation.excluded.append({
                "value": comp.as_stated, "stage": "scope_adjudication",
                "reason": "; ".join(f"{pid}: {a.reason}" for pid, a in adjs.items()),
                "decisive_facet": next(iter(adjs.values())).decisive_facet})
    # Every candidate, in scope or not -- this is the one place to see why a
    # real comparator mention never reached the final output.
    _snapshot(debug_capture, "A12_scope_adjudication", all_adjudications)

    # ---- A16 / A13 --------------------------------------------------------
    opts.emit("A16", "Consolidating comparators")
    comparators = []
    dropped_at_consolidation = []
    for key, (comp, recs, adj, pop_ids, adjs) in in_scope.items():
        try:
            comparators.append(consolidation.build_comparator(
                key, comp, recs, adj, searched_states, providers.llm,
                population_ids=pop_ids, adjudications_by_population=adjs))
        except ValueError as exc:
            out.notes.append(f"[A16] {comp.as_stated!r} dropped: {exc}")
            dropped_at_consolidation.append({"value": comp.as_stated, "reason": str(exc)})
    _snapshot(debug_capture, "A16_dropped_at_consolidation", dropped_at_consolidation)
    or_map = consolidation.detect_or_alternatives(comparators)
    for c in comparators:
        c.or_alternative = any(len(or_map.get(s, [])) > 1 for s in c.member_states)
    comparators.sort(key=lambda c: (-len(c.member_states), c.generic_name.lower()))
    out.comparators = comparators

    # ---- A15 --------------------------------------------------------------
    opts.emit("A15", "Harmonising outcomes and checking catalog coverage")
    outcome_groups = consolidation.harmonize_outcomes(
        [r for r in usable if r.finding_type == "outcome"], providers.llm)
    _snapshot(debug_capture, "A15_outcome_groups", [
        {"concept": concept, "catalog_id": catalog_id, "records": [r.to_dict() for r in recs]}
        for concept, catalog_id, recs in outcome_groups])
    out.outcomes = consolidation.build_outcomes(outcome_groups, providers.llm)

    # ---- by-state view ----------------------------------------------------
    entries, not_identified, summary = consolidation.build_by_member_state(
        comparators, attempts)
    out.by_member_state = entries
    out.not_identified_states = not_identified
    out.member_state_summary = summary

    # ---- source registry --------------------------------------------------
    seen = set()
    for c in comparators:
        for s in c.sources:
            if s.source_id not in seen:
                seen.add(s.source_id)
                out.source_registry.append(s)
    for o in out.outcomes.all_outcomes():
        for s in o.sources:
            if s.source_id not in seen:
                seen.add(s.source_id)
                out.source_registry.append(s)

    # ---- A17 --------------------------------------------------------------
    opts.emit("A17", "Auditing completeness")
    out.completeness = consolidation.audit_completeness(
        out, attempts, [p.population_id for p in populations])

    out.run_manifest = _manifest(out, providers, inventory, guard, started, opts)
    opts.emit("done", f"{len(out.comparators)} comparator(s), "
                      f"{len(out.outcomes.all_outcomes())} outcome(s)")
    return out


def _tally_validation(records: Sequence[EvidenceRecord], summary: ValidationSummary) -> None:
    for r in records:
        v = r.validation
        if v.refetched:
            summary.refetch_attempted += 1
        if v.verdict == C.V_WRONG_SUBJECT_DRUG:
            summary.subject_drug_rejections += 1
        if v.verdict == C.V_WRONG_INTERVENTION:
            summary.role_rejections += 1
        if v.passed:
            summary.claim_validated += 1
        elif v.verdict:
            summary.claim_rejected += 1
            value = (r.comparator.as_stated if r.comparator
                     else (r.outcome.measure if r.outcome else ""))
            summary.excluded.append({"value": value, "stage": "claim_validation",
                                     "reason": v.reason, "source_url": r.source_url})


def _manifest(out: Phase1Output, providers: Providers, inventory: SourceInventory,
              guard, started: float, opts: RunOptions) -> Dict[str, Any]:
    """Everything needed to reproduce and explain this run.

    MAI-34392 requires the model and the source list in the audit trail. Prompt
    versions belong there too: a run that cannot be traced to the prompt text
    that produced it cannot be explained when it is wrong.
    """
    import os
    manifest: Dict[str, Any] = {
        "request_id": out.request_id,
        "phase": "phase1_comparator_outcome_scoping",
        "phase2_included": False,
        "code_version": os.getenv("JCA_CODE_VERSION", "dev"),
        "model": C.LLM.model_id if providers.llm else None,
        "prompt_versions": prompt_registry.versions(),
        "outcome_catalog_version": consolidation.CATALOG["catalog_version"],
        "source_workbook": {
            "path": inventory.path,
            "sheets_matched": inventory.sheets_matched,
            "entries": len(inventory.entries),
            "unique_domains": len(inventory.unique_domains),
            "problems": [p.to_dict() for p in inventory.problems],
            "mtime": (datetime.fromtimestamp(os.path.getmtime(inventory.path),
                                             timezone.utc).isoformat()
                      if inventory.path and os.path.exists(inventory.path) else None),
        },
        "therapeutic_areas_applied": out.therapeutic_areas.areas,
        "leakage_guard": guard.to_dict(),
        "tier3_enabled": C.ENABLE_TIER_3_GENERAL_WEB,
        "providers": {
            "llm": type(providers.llm).__name__ if providers.llm else None,
            "search": type(providers.search).__name__ if providers.search else None,
            "trials": type(providers.trials).__name__ if providers.trials else None,
            "literature": type(providers.literature).__name__ if providers.literature else None,
            "medicines": type(providers.medicines).__name__ if providers.medicines else None,
        },
        "started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_s": round(time.time() - started, 2),
    }
    if providers.llm:
        manifest["llm_usage"] = providers.llm.usage_summary()
    if providers.search:
        manifest["search_usage"] = providers.search.usage_summary()
    return manifest
