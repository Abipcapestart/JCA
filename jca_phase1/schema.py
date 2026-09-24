"""
Typed data model for JCA Phase 1.

The whole point of this module is that concepts which fail differently are
different types. The legacy implementation kept comparator, intervention,
population and outcome in one loosely-typed dict, which is how the
intervention's mechanism ended up written onto every comparator.

Three invariants are enforced structurally here, not by convention:

  1. `Comparator` has its own `class_mechanism`. There is NO code path from
     `Intervention.therapeutic_class_mechanism` to it; `ComparatorIdentity`
     fills it from the comparator's own INN.
  2. Every `EvidenceRecord` carries `subject_drug`. A record whose subject drug
     is not the requested intervention cannot become a comparator.
  3. A `ConsolidatedComparator` cannot be constructed without at least one
     validated, grounded evidence record (`__post_init__` raises).

PICO sets are absent by design. They are Phase 2.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from . import config as C


# ===========================================================================
# Provenance primitives
# ===========================================================================

CONFIRMED = "confirmed"       # the user explicitly stated this
INFERRED = "inferred"         # the system filled it in, from a permitted basis
NOT_PROVIDED = "not_provided"  # no statement and no permitted inference
DERIVED_FROM_LABEL = "derived_from_label"  # taken from the regulatory record


@dataclass
class Field:
    """One population/intervention field with its honesty tag.

    SME Agent 2: 'Every field you populate is tagged exactly one of CONFIRMED,
    INFERRED, or NOT PROVIDED; inferred fields are never presented as if the
    user stated them.'
    """
    value: Optional[str] = None
    provenance: str = NOT_PROVIDED
    inference_basis: str = ""   # required whenever provenance == INFERRED
    label: str = ""             # display label, for custom fields

    def is_stated(self) -> bool:
        return bool(self.value) and self.provenance in (CONFIRMED, DERIVED_FROM_LABEL)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ===========================================================================
# A1 — P&I validation
# ===========================================================================

SEVERITY_FIX = "FIX"      # blocking
SEVERITY_CHECK = "CHECK"  # advisory


@dataclass
class ValidationItem:
    severity: str
    fields: List[str]
    value_flagged: str = ""
    explanation: str = ""


@dataclass
class PIValidationResult:
    overall_status: str = "PASS"      # PASS | BLOCKED
    fix_items: List[ValidationItem] = field(default_factory=list)
    check_items: List[ValidationItem] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.fix_items

    def finalise(self) -> "PIValidationResult":
        # SME Agent 1 step 7: BLOCKED whenever fix_items is non-empty,
        # regardless of check_items.
        self.overall_status = "BLOCKED" if self.fix_items else "PASS"
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status,
            "fix_items": [asdict(i) for i in self.fix_items],
            "check_items": [asdict(i) for i in self.check_items],
        }


# ===========================================================================
# A2 — structured Population / Intervention
# ===========================================================================

POP_LICENSED = "licensed"
POP_ITT = "intended_to_treat"


@dataclass
class Population:
    population_id: str = POP_LICENSED
    fields: Dict[str, Field] = field(default_factory=dict)
    custom_fields: List[Field] = field(default_factory=list)
    user_confirmed: bool = False
    confirmed_at: str = ""
    confirmed_by: str = ""

    def get(self, name: str) -> Field:
        return self.fields.get(name, Field())

    def value(self, name: str) -> str:
        return self.get(name).value or ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "population_id": self.population_id,
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
            "custom_fields": [f.to_dict() for f in self.custom_fields],
            "user_confirmed": self.user_confirmed,
            "confirmed_at": self.confirmed_at,
            "confirmed_by": self.confirmed_by,
        }


@dataclass
class Intervention:
    fields: Dict[str, Field] = field(default_factory=dict)
    custom_fields: List[Field] = field(default_factory=list)
    inn_resolved: str = ""
    atc_code: str = ""
    user_confirmed: bool = False

    @property
    def product_name(self) -> str:
        return (self.fields.get("product_name_inn") or Field()).value or ""

    def value(self, name: str) -> str:
        return (self.fields.get(name) or Field()).value or ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
            "custom_fields": [f.to_dict() for f in self.custom_fields],
            "inn_resolved": self.inn_resolved,
            "atc_code": self.atc_code,
            "user_confirmed": self.user_confirmed,
        }


# ===========================================================================
# A4 — indication lock
# ===========================================================================

@dataclass
class LicensedIndicationRecord:
    """What the product is actually licensed (or claimed) for.

    Without this there is nothing to reject an out-of-indication population
    against, which is how first-line populations survived into a
    relapsed/refractory request.
    """
    source: str = "none"          # ema_epar | ema_smpc | claimed_wording_only | none
    indication_text: str = ""
    source_url: str = ""
    retrieved_at: str = ""
    pivotal_trials: List[str] = field(default_factory=list)
    atc_code: str = ""
    inn: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LeakageGuard:
    """Blocks the target drug's own published JCA report from the runtime path.

    The product's purpose is to anticipate a scope BEFORE the JCA exists.
    Reading the JCA report is reading the answer key. `allow_jca_reports` is
    for evaluation runs only and is recorded in the run manifest either way,
    so the GT-free claim is auditable rather than asserted.
    """
    enabled: bool = True
    allow_jca_reports: bool = False
    drug_tokens: List[str] = field(default_factory=list)
    blocked_domains: List[str] = field(default_factory=lambda: ["health.ec.europa.eu"])
    blocked_urls: List[str] = field(default_factory=list)   # audit log

    def is_blocked(self, url: str) -> bool:
        if not self.enabled or self.allow_jca_reports:
            return False
        low = (url or "").lower()
        if not any(d in low for d in self.blocked_domains):
            return False
        # Only block documents about THIS drug; generic EU HTA methodology
        # guidance on the same domain is legitimate and must stay reachable.
        return any(t and t in low for t in self.drug_tokens)

    def record(self, url: str) -> None:
        # Always append rather than check-then-append: execute_plan() calls
        # this from multiple worker threads, and "not in / append" is a race
        # that can double-log a URL. list.append is itself GIL-atomic, so
        # appending unconditionally is race-free; de-duplication happens at
        # read time in to_dict() instead, where it costs nothing.
        self.blocked_urls.append(url)

    def to_dict(self) -> Dict[str, Any]:
        return {"enabled": self.enabled, "allow_jca_reports": self.allow_jca_reports,
                "blocked_domains": list(self.blocked_domains),
                "blocked_urls": sorted(set(self.blocked_urls))}


# ===========================================================================
# A3 — population scope boundary
# ===========================================================================

@dataclass
class ScopeFacet:
    name: str
    value: str = ""
    provenance: str = NOT_PROVIDED
    must_match: bool = False
    verbatim: str = ""

    @property
    def is_bound(self) -> bool:
        """A facet only constrains evidence when the user actually stated it.
        A facet nobody specified must never silently become a filter."""
        return bool(self.value) and self.provenance in (CONFIRMED, DERIVED_FROM_LABEL)


@dataclass
class ScopeBoundary:
    population_id: str = POP_LICENSED
    facets: Dict[str, ScopeFacet] = field(default_factory=dict)
    licensed_indication_wording: str = ""
    out_of_bounds_rule: str = "reject"   # reject | flag_as_itt
    unbounded_facets: List[str] = field(default_factory=list)

    def must_match_facets(self) -> List[ScopeFacet]:
        return [f for f in self.facets.values() if f.must_match and f.is_bound]

    def describe(self) -> str:
        """Human-readable boundary, handed to the adjudicator prompt."""
        lines = []
        for f in self.facets.values():
            if f.is_bound:
                flag = " [DISCRIMINATING]" if f.must_match else ""
                lines.append(f"- {f.name}: {f.value}{flag}")
        if self.unbounded_facets:
            lines.append(f"- NOT SPECIFIED BY THE USER (do not filter on these): "
                         f"{', '.join(self.unbounded_facets)}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "population_id": self.population_id,
            "facets": {k: asdict(v) for k, v in self.facets.items()},
            "licensed_indication_wording": self.licensed_indication_wording,
            "out_of_bounds_rule": self.out_of_bounds_rule,
            "unbounded_facets": list(self.unbounded_facets),
        }


# ===========================================================================
# A5 — therapeutic area
# ===========================================================================

@dataclass
class AreaResolution:
    areas: List[str] = field(default_factory=list)
    rationales: Dict[str, str] = field(default_factory=dict)
    method: str = "keyword"      # keyword | llm | user_stated
    needed_adjudication: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ===========================================================================
# A6 — query plan
# ===========================================================================

@dataclass
class QueryVocabulary:
    """Produced by ONE LLM call per request. Deliberately has no comparator
    field, with ONE documented exception: standard_of_care_candidates below.
    Every other field must never be able to name the answer."""
    indication_synonyms: List[str] = field(default_factory=list)
    indication_abbreviations: List[str] = field(default_factory=list)
    disease_class_terms: List[str] = field(default_factory=list)
    localised_assessment_terms: Dict[str, List[str]] = field(default_factory=dict)
    outcome_requirement_terms: List[str] = field(default_factory=list)
    # The ONE exception to "no comparator field": DISEASE-GENERAL established
    # treatment names (not an assertion about what compares against the
    # requested drug) so retrieval can search for a real comparator BY NAME
    # instead of only finding it by accident inside a disease-only search.
    # Every document these queries retrieve still goes through the exact
    # same independent a08-a12 extraction/validation/adjudication pipeline as
    # anything else -- this field only widens what gets searched for, it
    # never decides the answer. Confirmed real gap: two GT comparators
    # (Everolimus, Bevacizumab+chemo) had zero hits across multiple runs
    # because no query path ever named a candidate comparator at all.
    standard_of_care_candidates: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


PASS_DRUG_ANCHORED = "drug_anchored"
PASS_LANDSCAPE = "landscape"          # NOT drug-anchored — finds the complete
                                       # treatment-line landscape a guideline states
PASS_OUTCOME_REQUIREMENT = "outcome_requirement"
PASS_REFINEMENT = "refinement"
PASS_COMPARATOR_ANCHORED = "comparator_anchored"  # anchored on a CANDIDATE
                                       # comparator's own name, not the
                                       # requested drug or disease alone
PASS_COMPARATOR_FOLLOWUP = "comparator_followup"  # Version B: a SECOND-round
                                       # query, anchored on a comparator name
                                       # that organically surfaced from real
                                       # round-1 evidence with only a thin
                                       # mention -- not a guessed candidate


@dataclass
class QueryPlanItem:
    query: str
    source_class: str
    member_state: str = C.EU_WIDE
    pass_type: str = PASS_DRUG_ANCHORED
    domains: List[str] = field(default_factory=list)
    max_urls: int = C.RETRIEVAL.max_urls_per_state
    language: str = "en"
    note: str = ""

    def key(self) -> str:
        return f"{self.member_state}|{self.source_class}|{self.pass_type}"


# ===========================================================================
# A7/A8 — retrieval + the typed evidence record
# ===========================================================================

@dataclass
class RetrievedDocument:
    url: str
    resolved_url: str = ""
    text: str = ""
    ok: bool = False
    status: str = C.EV_RETRIEVAL_FAILED
    method: str = "none"
    title: str = ""
    published_date: str = ""
    language: str = "en"
    source_class: str = ""
    member_state: str = C.EU_WIDE
    organization: str = ""
    error: str = ""
    publication_types: List[str] = field(default_factory=list)  # PubMed only,
        # from PublicationRecord.publication_types — the metadata that lets A8
        # tell a preprint from a journal article without asking the LLM to guess.

    @property
    def source_id(self) -> str:
        return "src-" + hashlib.sha1((self.resolved_url or self.url).encode()).hexdigest()[:12]


@dataclass
class Comparator:
    """The comparator as an object in its own right.

    `class_mechanism` is filled by ComparatorIdentity from `inn`. Extraction
    must leave it empty: a model asked for a comparator's class while reading a
    document about the intervention will hand back the intervention's class.
    """
    as_stated: str = ""
    inn: str = ""
    atc_code: str = ""
    class_mechanism: str = ""
    class_source: str = ""          # atc_vocabulary | source_stated | unresolved
    is_combination: bool = False
    components: List[str] = field(default_factory=list)
    role: str = C.ROLE_UNCLEAR
    comparator_scenario: str = ""
    retain_all_status: str = C.RETAIN_UNCONFIRMED
    # The wording exactly as extraction stated it, BEFORE apply_identities()
    # overwrites as_stated with the cluster's unified display wording. Kept
    # so build_comparator() can still report which raw wordings (different
    # language, brand name, dose-annotated citation, etc.) were merged into
    # one comparator -- without this, that history is gone the moment
    # as_stated is unified, and aliases_merged can never be populated.
    raw_as_stated: str = ""
    # True when the source passage only NAMES this comparator with no
    # dosing/regimen/population/outcome detail (a passing background
    # mention). Seeds Version B's name-targeted follow-up retrieval round --
    # a comparator whose own dedicated evidence never ranks in a disease-only
    # search still gets a second, targeted chance once its name has
    # organically surfaced. Independent of `role`: a thinly-mentioned name
    # can still be role=active_comparator.
    thin_mention: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PopulationContext:
    """The population the SOURCE itself states for this finding — in the
    source's own words, distinct from the request's confirmed indication."""
    disease: str = ""
    subtype_histology: str = ""
    stage: str = ""
    biomarker: str = ""
    line_of_therapy: str = ""
    prior_therapy: str = ""
    treatment_setting_intent: str = ""
    age_band: str = ""
    other: str = ""
    verbatim: str = ""

    def summary(self) -> str:
        parts = [self.disease, self.subtype_histology, self.stage, self.biomarker,
                 self.line_of_therapy, self.prior_therapy, self.treatment_setting_intent]
        return ", ".join(dict.fromkeys(p for p in parts if p))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OutcomeMention:
    measure: str = ""
    result: str = ""
    unit: str = ""
    instrument: str = ""
    requirement_type: str = ""   # relative_effect_required | descriptive_only | ""
    is_requirement: bool = False  # True = the source states this as a REQUIRED
                                   # scope outcome, not merely a reported result

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationOutcome:
    verdict: str = ""
    reason: str = ""
    refetched: bool = False
    attempts: int = 0

    @property
    def passed(self) -> bool:
        return self.verdict in C.PASSING_VERDICTS

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


FINDING_COMPARATOR = "comparator"
FINDING_OUTCOME = "outcome"


@dataclass
class EvidenceRecord:
    """One claim, from one source, about one subject drug, in one population.

    This is the unit of work for the whole pipeline. `subject_drug` is
    mandatory: a document about extensive-stage SCLC discusses several drugs,
    and without this field a comparator from one drug's trial silently becomes
    a comparator for another's.
    """
    finding_id: str
    finding_type: str
    subject_drug: str = ""
    source_id: str = ""
    source_url: str = ""
    source_class: str = ""
    tier: int = 3
    member_state: str = C.EU_WIDE
    organization: str = ""
    document_title: str = ""
    document_date: str = ""
    language: str = "en"
    retrieval_method: str = ""

    comparator: Optional[Comparator] = None
    outcome: Optional[OutcomeMention] = None
    population_context: PopulationContext = field(default_factory=PopulationContext)
    recommendation_strength: str = C.REC_NOT_STATED

    evidence_quote: str = ""
    evidence_locator: str = ""
    grounded: bool = False
    grounding_note: str = ""
    validation: ValidationOutcome = field(default_factory=ValidationOutcome)
    general_evidence_flag: bool = False
    evidence_status: str = ""     # e.g. "conference_abstract"
    data_maturity: str = ""       # "interim" | "final" — conference sources only,
                                   # source-stated, wording owned by the SME's own
                                   # tuning of a08.extraction
    trial_status: str = ""        # verbatim TrialRecord.status (e.g. "TERMINATED") —
                                   # trial-registry records only, never excluded on
                                   # this value, only de-prioritised and flagged
    peer_review_status: str = ""  # "peer_reviewed" | "preprint" | "unconfirmed" | ""
                                   # (not applicable) — deterministic, PubMed +
                                   # general-web records only; never used to exclude

    @property
    def is_terminated_trial(self) -> bool:
        return self.trial_status.upper() == C.TRIAL_STATUS_TERMINATED

    def usable(self) -> bool:
        """A record may only reach harmonization when it is grounded in its
        source AND the claim validated against a re-fetch of that source."""
        return self.grounded and self.validation.passed

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# ===========================================================================
# A12 — scope adjudication
# ===========================================================================

@dataclass
class EvidenceCitation:
    source_id: str = ""
    source_url: str = ""
    quote: str = ""
    member_state: str = ""
    tier: int = 3
    facets_matched: List[str] = field(default_factory=list)
    facet_conflict: str = ""
    detail: str = ""


@dataclass
class ScopeAdjudication:
    comparator_inn: str = ""
    verdict: str = C.SCOPE_UNCERTAIN
    evidence_for: List[EvidenceCitation] = field(default_factory=list)
    evidence_against: List[EvidenceCitation] = field(default_factory=list)
    decisive_facet: str = ""
    reason: str = ""
    adjudicator_version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ===========================================================================
# A13/A16 — consolidated output
# ===========================================================================

@dataclass
class MemberStateVerdict:
    """One entry per Member State per comparator. The UI renders all 27 with a
    two-value legend, so 'absent' must never be ambiguous."""
    member_state: str
    verdict: str = C.STATE_NOT_ESTABLISHED
    tier: Optional[int] = None
    source_id: str = ""
    source_url: str = ""
    evidence_quote: str = ""
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SourceRef:
    source_id: str
    display_name: str
    url: str
    tier: int
    source_class: str
    organization: str = ""
    document_title: str = ""
    document_date: str = ""
    language: str = "en"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceRef:
    source_id: str
    quote: str
    locator: str = ""
    member_state: str = ""
    grounded: bool = True
    validation_verdict: str = ""
    refetched: bool = False
    evidence_status: str = ""     # e.g. "conference_abstract" — carried from
                                   # EvidenceRecord.evidence_status, previously
                                   # dropped at this exact conversion boundary
    data_maturity: str = ""       # "interim" | "final" — conference sources only
    trial_status: str = ""        # verbatim trial-registry status, e.g. "TERMINATED"
    peer_review_status: str = ""  # "peer_reviewed" | "preprint" | "unconfirmed" | ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


ORIGIN_AGENT = "agent_found"
ORIGIN_USER = "user_added"


@dataclass
class ConsolidatedComparator:
    generic_name: str
    comparator_id: str = ""
    brand_names: List[str] = field(default_factory=list)
    inn: str = ""
    atc_code: str = ""
    class_or_mechanism: str = ""
    class_source: str = ""
    is_combination: bool = False
    components: List[str] = field(default_factory=list)
    role: str = C.ROLE_ACTIVE_COMPARATOR
    indication: str = ""
    indication_scope_note: str = ""
    line_of_therapy: str = C.NOT_STATED_BY_SOURCE
    comparator_scenario: str = ""
    retain_all_status: str = C.RETAIN_UNCONFIRMED
    recommendation_strength: str = C.REC_NOT_STATED
    member_states: List[str] = field(default_factory=list)
    member_state_count: int = 0
    per_member_state: List[MemberStateVerdict] = field(default_factory=list)
    tiers: List[int] = field(default_factory=list)
    cross_tier_confirmed: bool = False
    or_alternative: bool = False
    general_evidence_flag: bool = False
    rationale: str = ""
    scope_adjudication: Optional[ScopeAdjudication] = None
    # Every population's OWN adjudication, keyed by ScopeBoundary.population_id
    # -- scope_adjudication above is only the single representative verdict
    # (an IN verdict preferred over an UNCERTAIN one). A comparator merged
    # across multiple wording variants can genuinely get different verdicts
    # per population (e.g. in_scope for "licensed" but uncertain for
    # "intended_to_treat", for a different reason) -- that discarded verdict
    # was previously invisible anywhere in the export. Empty when only one
    # population was adjudicated (nothing was actually discarded).
    adjudications_by_population: Dict[str, ScopeAdjudication] = field(default_factory=dict)
    origin: str = ORIGIN_AGENT
    sources: List[SourceRef] = field(default_factory=list)
    evidence: List[EvidenceRef] = field(default_factory=list)
    # Which population(s) (ScopeBoundary.population_id) this comparator was
    # judged in-scope or uncertain for -- e.g. ["licensed"], or
    # ["licensed", "intended_to_treat"], or just ["intended_to_treat"] for a
    # comparator that is only relevant to the broader ITT population. Never
    # empty for an agent-found comparator: it was adjudicated in for at least
    # one population, or it would not exist.
    population_ids: List[str] = field(default_factory=list)
    # Distinct raw comparator.as_stated wordings (e.g. "monoterapia con
    # topotecan", "Topotecan (1997)") that A11/precluster merged into this
    # one comparator, excluding the final generic_name itself -- mirrors
    # ConsolidatedOutcome.aliases_merged. Without this, dedup across
    # wording/language/brand-name variants has no visible trace in the
    # export; a reviewer has no way to see what was folded into this row.
    aliases_merged: List[str] = field(default_factory=list)

    def __post_init__(self):
        # INVARIANT 3. A comparator with no evidence cannot exist. User-added
        # comparators are exempt: the SME says they are taken exactly as entered.
        if self.origin == ORIGIN_AGENT and not self.evidence:
            raise ValueError(
                f"ConsolidatedComparator({self.generic_name!r}) constructed with no "
                f"evidence. Every agent-found comparator must carry at least one "
                f"grounded, validated evidence reference.")
        # INVARIANT 1. Never let an intervention's class land on a comparator.
        if self.class_or_mechanism and self.class_source not in (
                "atc_vocabulary", "source_stated", "user_added", ""):
            raise ValueError(
                f"ConsolidatedComparator({self.generic_name!r}) has class_source "
                f"{self.class_source!r}; a comparator's class may only come from its "
                f"own INN lookup or from the source's own words.")
        self.member_state_count = len(self.member_states)
        if not self.comparator_id:
            self.comparator_id = "cmp-" + hashlib.sha1(
                (self.inn or self.generic_name).lower().encode()).hexdigest()[:10]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["member_state_count"] = len(self.member_states)
        return d


@dataclass
class ConsolidatedOutcome:
    concept: str
    outcome_id: str = ""
    category: str = C.CAT_CLINICAL
    catalog_id: str = ""
    listed: bool = False
    instrument: str = ""
    unit_of_measurement: str = C.NOT_STATED_BY_SOURCE
    unit_disagreement_note: str = ""
    requirement_type: str = ""
    coverage_status: str = C.EV_FOUND
    tiers: List[int] = field(default_factory=list)
    cross_tier_confirmed: bool = False
    rationale: str = ""
    origin: str = ORIGIN_AGENT
    sources: List[SourceRef] = field(default_factory=list)
    evidence: List[EvidenceRef] = field(default_factory=list)
    aliases_merged: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.outcome_id:
            self.outcome_id = "out-" + hashlib.sha1(
                (self.catalog_id or self.concept).lower().encode()).hexdigest()[:10]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CatalogCoverageEntry:
    catalog_id: str
    display_name: str
    category: str
    status: str = C.EV_NONE
    matched_outcome_id: str = ""
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ByMemberStateEntry:
    member_state: str
    status: str = "not_identified"       # identified | not_identified
    comparators: List[str] = field(default_factory=list)   # generic_name ONLY
    state_search_status: str = "complete"
    finding: str = ""
    search_failure_detail: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemberStateSummary:
    identified_count: int
    not_identified_count: int
    total: int = 27

    def __post_init__(self):
        if self.identified_count + self.not_identified_count != self.total:
            raise ValueError(
                "member_state_summary must always sum to exactly 27 — this is the "
                "completeness proof the whole product rests on.")


@dataclass
class OutcomesView:
    clinical_effectiveness: List[ConsolidatedOutcome] = field(default_factory=list)
    safety: List[ConsolidatedOutcome] = field(default_factory=list)
    quality_of_life: List[ConsolidatedOutcome] = field(default_factory=list)
    clinician_patient_reported: List[ConsolidatedOutcome] = field(default_factory=list)
    catalog_coverage: List[CatalogCoverageEntry] = field(default_factory=list)

    def all_outcomes(self) -> List[ConsolidatedOutcome]:
        return (self.clinical_effectiveness + self.safety + self.quality_of_life
                + self.clinician_patient_reported)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clinical_effectiveness": [o.to_dict() for o in self.clinical_effectiveness],
            "safety": [o.to_dict() for o in self.safety],
            "quality_of_life": [o.to_dict() for o in self.quality_of_life],
            "clinician_patient_reported": [o.to_dict() for o in self.clinician_patient_reported],
            "catalog_coverage": [c.to_dict() for c in self.catalog_coverage],
        }


# ===========================================================================
# A17 — completeness
# ===========================================================================

@dataclass
class SourceClassAttempt:
    member_state: str
    source_class: str
    attempted: bool = False
    documents_retrieved: int = 0
    status: str = C.EV_NONE
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Assertion:
    name: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CompletenessReport:
    member_states: Dict[str, int] = field(default_factory=dict)
    source_class_matrix: List[SourceClassAttempt] = field(default_factory=list)
    outcome_categories: List[Dict[str, Any]] = field(default_factory=list)
    populations_processed: List[str] = field(default_factory=list)
    assertions: List[Assertion] = field(default_factory=list)

    @property
    def all_assertions_passed(self) -> bool:
        return all(a.passed for a in self.assertions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "member_states": self.member_states,
            "source_class_matrix": [s.to_dict() for s in self.source_class_matrix],
            "outcome_categories": self.outcome_categories,
            "populations_processed": self.populations_processed,
            "assertions": [a.to_dict() for a in self.assertions],
        }


@dataclass
class ValidationSummary:
    records_extracted: int = 0
    grounded: int = 0
    grounding_failed: int = 0
    claim_validated: int = 0
    claim_rejected: int = 0
    refetch_attempted: int = 0
    subject_drug_rejections: int = 0
    role_rejections: int = 0
    scope_in: int = 0
    scope_out: int = 0
    scope_uncertain: int = 0
    excluded: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ===========================================================================
# The Phase 1 output
# ===========================================================================

@dataclass
class Phase1Output:
    """The complete Phase 1 deliverable. Consumed by UI screen 2 ("PICO Scope")
    and handed to Phase 2 PICO-set consolidation.

    Deliberately contains NO pico_sets field.
    """
    request_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    input_validation: Dict[str, Any] = field(default_factory=dict)
    raw_population_input: Dict[str, Any] = field(default_factory=dict)
    raw_intervention_input: Dict[str, Any] = field(default_factory=dict)

    populations: List[Population] = field(default_factory=list)
    scope_boundaries: List[ScopeBoundary] = field(default_factory=list)
    intervention: Intervention = field(default_factory=Intervention)
    licensed_indication_record: LicensedIndicationRecord = field(
        default_factory=LicensedIndicationRecord)
    therapeutic_areas: AreaResolution = field(default_factory=AreaResolution)

    comparators: List[ConsolidatedComparator] = field(default_factory=list)
    by_member_state: List[ByMemberStateEntry] = field(default_factory=list)
    member_state_summary: Optional[MemberStateSummary] = None
    not_identified_states: List[str] = field(default_factory=list)

    outcomes: OutcomesView = field(default_factory=OutcomesView)

    source_registry: List[SourceRef] = field(default_factory=list)
    validation: ValidationSummary = field(default_factory=ValidationSummary)
    completeness: CompletenessReport = field(default_factory=CompletenessReport)
    run_manifest: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "created_at": self.created_at,
            "input_validation": self.input_validation,
            "raw_population_input": self.raw_population_input,
            "raw_intervention_input": self.raw_intervention_input,
            "populations": [p.to_dict() for p in self.populations],
            "scope_boundaries": [s.to_dict() for s in self.scope_boundaries],
            "intervention": self.intervention.to_dict(),
            "licensed_indication_record": self.licensed_indication_record.to_dict(),
            "therapeutic_areas": self.therapeutic_areas.to_dict(),
            "comparators": [c.to_dict() for c in self.comparators],
            "by_member_state": [b.to_dict() for b in self.by_member_state],
            "member_state_summary": (asdict(self.member_state_summary)
                                     if self.member_state_summary else None),
            "not_identified_states": list(self.not_identified_states),
            "outcomes": self.outcomes.to_dict(),
            "source_registry": [s.to_dict() for s in self.source_registry],
            "validation": self.validation.to_dict(),
            "completeness": self.completeness.to_dict(),
            "run_manifest": self.run_manifest,
            "notes": list(self.notes),
            # Explicit, so a consumer never wonders whether it was forgotten.
            "pico_sets": None,
            "pico_sets_note": "PICO-set generation is Phase 2 and is out of scope for this module.",
        }
