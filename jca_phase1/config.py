"""
Configuration constants for JCA Phase 1 — Comparator & Outcome Scoping.

Scope note: this package implements PHASE 1 ONLY (Population + Intervention ->
confirmed Comparator and Outcome scope). PICO-set generation is Phase 2 and is
deliberately absent. See ARCHITECTURE.md §"Scope boundary".

Every constant here is traceable to a requirement source; the label in the
comment says which. Nothing in this file is inferred.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# The 27 EU Member States. [CONFIRMED FROM SME — Agents 3/5/12 input lists]
# Spelled exactly as the SME base prompt spells them.
# ---------------------------------------------------------------------------

EU_27_MEMBER_STATES: List[str] = [
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic",
    "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary",
    "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta",
    "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia",
    "Spain", "Sweden",
]
assert len(EU_27_MEMBER_STATES) == 27, "EU-27 list must contain exactly 27 states"

# ISO-3166 alpha-2 codes, for the UI's member-state chips.
MEMBER_STATE_CODE: Dict[str, str] = {
    "Austria": "AT", "Belgium": "BE", "Bulgaria": "BG", "Croatia": "HR",
    "Cyprus": "CY", "Czech Republic": "CZ", "Denmark": "DK", "Estonia": "EE",
    "Finland": "FI", "France": "FR", "Germany": "DE", "Greece": "GR",
    "Hungary": "HU", "Ireland": "IE", "Italy": "IT", "Latvia": "LV",
    "Lithuania": "LT", "Luxembourg": "LU", "Malta": "MT", "Netherlands": "NL",
    "Poland": "PL", "Portugal": "PT", "Romania": "RO", "Slovakia": "SK",
    "Slovenia": "SI", "Spain": "ES", "Sweden": "SE",
}

# Spelling variants real source workbooks use. Unnormalised names fragment one
# state's findings across several entries.
_STATE_ALIASES: Dict[str, str] = {
    "czechia": "Czech Republic", "czech rep.": "Czech Republic",
    "czech republic ": "Czech Republic",
    "slovak republic": "Slovakia", "the netherlands": "Netherlands",
    "holland": "Netherlands", "hellas": "Greece", "eire": "Ireland",
    "deutschland": "Germany", "espana": "Spain", "españa": "Spain",
    "italia": "Italy", "sverige": "Sweden", "suomi": "Finland",
    "danmark": "Denmark", "belgie": "Belgium", "belgique": "Belgium",
    "osterreich": "Austria", "österreich": "Austria",
}
_CANONICAL_STATE = {s.lower(): s for s in EU_27_MEMBER_STATES}
_CANONICAL_STATE.update(_STATE_ALIASES)
_CANONICAL_STATE.update({c.lower(): s for s, c in MEMBER_STATE_CODE.items()})

# Sentinels. Not Member States; never counted toward the 27.
EU_WIDE = "EU-wide"            # an EMA label / EU-level record
GENERAL_EVIDENCE = "EU/general"  # Tier 2/3 evidence with no country attribution


def canonicalize_member_state(raw: Optional[str]) -> Optional[str]:
    """Resolve any spelling/alias/ISO code to a canonical EU-27 name.

    Returns the sentinels unchanged. Returns None when the input does not
    resolve — callers must treat None as "a person needs to look at this",
    never silently drop or guess.
    """
    if raw in (EU_WIDE, GENERAL_EVIDENCE):
        return raw
    return _CANONICAL_STATE.get((raw or "").strip().lower())


# ---------------------------------------------------------------------------
# The 11 fixed therapeutic areas the Standard-of-Care source material is
# organised around. [CONFIRMED FROM SME — Agent 5]
# ---------------------------------------------------------------------------

THERAPEUTIC_AREAS: List[str] = [
    "Multi Disciplinary", "Oncology", "Cardiovascular", "CNS",
    "Metabolic & Endocrine", "Infectious", "Immunology", "Respiratory",
    "Gastroenterology", "Musculoskeletal", "Rare Diseases",
]
MULTI_DISCIPLINARY = "Multi Disciplinary"
RARE_DISEASES = "Rare Diseases"

# Source workbooks label the same area differently. Mapping is data, not logic.
AREA_SYNONYMS: Dict[str, str] = {
    "oncology": "Oncology",
    "cardiovascular diseases": "Cardiovascular", "cardiovascular": "Cardiovascular",
    "central nervous system": "CNS", "cns": "CNS", "neurology": "CNS",
    "metabolic & endocrine disorders": "Metabolic & Endocrine",
    "metabolic and endocrine disorders": "Metabolic & Endocrine",
    "metabolic & endocrine": "Metabolic & Endocrine",
    "infectious diseases": "Infectious", "infectious": "Infectious",
    "immunology & autoimmune diseases": "Immunology",
    "immunology and autoimmune diseases": "Immunology", "immunology": "Immunology",
    "respiratory diseases": "Respiratory", "respiratory": "Respiratory",
    "gastroenterology": "Gastroenterology",
    "musculoskeletal & pain": "Musculoskeletal", "musculoskeletal and pain": "Musculoskeletal",
    "musculoskeletal": "Musculoskeletal",
    "rare diseases": "Rare Diseases", "rare disease": "Rare Diseases",
    "multi disciplinary": "Multi Disciplinary", "multidisciplinary": "Multi Disciplinary",
    "multi-disciplinary": "Multi Disciplinary",
}


def canonicalize_area(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    key = " ".join(str(raw).split()).strip().lower()
    return AREA_SYNONYMS.get(key) or (raw if raw in THERAPEUTIC_AREAS else None)


# ---------------------------------------------------------------------------
# The 4 fixed outcome categories. [CONFIRMED FROM SME — Agent 12; JIRA; UI]
# ---------------------------------------------------------------------------

CAT_CLINICAL = "Clinical Effectiveness"
CAT_SAFETY = "Safety"
CAT_QOL = "Quality of Life"
CAT_COA = "Clinician / Patient-Reported Outcomes (COA)"
OUTCOME_CATEGORIES: List[str] = [CAT_CLINICAL, CAT_SAFETY, CAT_QOL, CAT_COA]


# ---------------------------------------------------------------------------
# Source classes and tiers. [CONFIRMED FROM SME — Tier definitions]
#
# `eu_jca_report` was previously DELIBERATELY ABSENT from the runtime source
# classes: the target drug's own published JCA report is the answer key, and
# using it at runtime seemed to defeat the product's stated purpose
# (anticipate the scope BEFORE the JCA exists). Per explicit SME decision
# (2026-09-23), that is no longer the policy -- the SME wants comparators
# pulled from the drug's own JCA report too, not just for evaluation. It is
# now a real, labeled Tier 1 source, gated on LeakageGuard.allow_jca_reports
# (default now True -- see RunOptions.allow_jca_reports / lock_indication()
# in agents/a01_a05_input_context.py) rather than being excluded outright.
# ---------------------------------------------------------------------------

SRC_HTA_REGULATORY = "hta_regulatory"
SRC_DRUG_LABEL = "drug_registry_label"
SRC_CLINICAL_GUIDELINE = "clinical_guideline"
SRC_EU_JCA_REPORT = "eu_jca_report"
SRC_TRIAL_REGISTRY = "trial_registry"
SRC_PUBMED = "pubmed"
SRC_CONFERENCE = "conference_evidence"
SRC_GENERAL_WEB = "general_web"

TIER_OF_SOURCE_CLASS: Dict[str, int] = {
    SRC_HTA_REGULATORY: 1,
    SRC_DRUG_LABEL: 1,
    SRC_CLINICAL_GUIDELINE: 1,
    SRC_EU_JCA_REPORT: 1,
    SRC_TRIAL_REGISTRY: 2,
    SRC_PUBMED: 2,
    SRC_CONFERENCE: 2,
    SRC_GENERAL_WEB: 3,
}

TIER_1_SOURCE_CLASSES = [SRC_HTA_REGULATORY, SRC_DRUG_LABEL, SRC_CLINICAL_GUIDELINE,
                         SRC_EU_JCA_REPORT]
TIER_2_SOURCE_CLASSES = [SRC_TRIAL_REGISTRY, SRC_PUBMED, SRC_CONFERENCE]
TIER_3_SOURCE_CLASSES = [SRC_GENERAL_WEB]

# Source classes whose documents must be read WHOLE. A guideline's recommendation
# table and an HTA report's scope table are never in the top relevance chunks.
FULL_DOCUMENT_SOURCE_CLASSES = {
    SRC_DRUG_LABEL, SRC_CLINICAL_GUIDELINE, SRC_HTA_REGULATORY, SRC_EU_JCA_REPORT,
}

# MAI-34392 states "Non-peer-reviewed web content is excluded", which contradicts
# the SME prompt's Tier 3 agent. Enabled by default as of 2026-09-23 per explicit
# product decision -- the MAI-34392 conflict is UNRESOLVED and should still be
# raised with the SME; this is a visible default, not a resolution of that
# conflict. Still overridable via JCA_ENABLE_TIER3=0 without a code change.
# [OPEN — SME CONFIRMATION REQUIRED]
ENABLE_TIER_3_GENERAL_WEB = os.getenv("JCA_ENABLE_TIER3", "1") == "1"

# Debug only: log every raw Tavily response (search + extract) to the console
# before this module trims it down to SearchHit/RetrievedDocument, so an SME
# or developer can see exactly what the API sent back. Off by default -- a
# real run's response bodies are large and this is verbose.
DEBUG_LOG_TAVILY_RESPONSES = os.getenv("JCA_DEBUG_TAVILY", "0") == "1"


# ---------------------------------------------------------------------------
# Comparator roles. Only `active_comparator` is a comparator for JCA purposes.
# Everything else is context that must never become a comparator row.
# ---------------------------------------------------------------------------

ROLE_ACTIVE_COMPARATOR = "active_comparator"
ROLE_PRIOR_THERAPY = "prior_therapy"
ROLE_BACKGROUND = "background_therapy"
ROLE_INTERVENTION_ARM = "intervention_arm"
ROLE_UNCLEAR = "unclear"
COMPARATOR_ROLES = [
    ROLE_ACTIVE_COMPARATOR, ROLE_PRIOR_THERAPY, ROLE_BACKGROUND,
    ROLE_INTERVENTION_ARM, ROLE_UNCLEAR,
]

# comparator_scenario / retain_all_status. [CONFIRMED FROM SME — Agent 3 step 4]
SCENARIO_UNIQUE = "unique"
SCENARIO_EACH_REQUIRED = "each_required"
SCENARIO_AT_LEAST_ONE = "at_least_one"
SCENARIO_INDIVIDUALISED = "individualised"
COMPARATOR_SCENARIOS = [
    SCENARIO_UNIQUE, SCENARIO_EACH_REQUIRED, SCENARIO_AT_LEAST_ONE,
    SCENARIO_INDIVIDUALISED,
]

RETAIN_DROPPABLE = "confirmed_droppable"
RETAIN_MUST_RETAIN = "confirmed_must_retain"
RETAIN_UNCONFIRMED = "unconfirmed"  # the correct default; the SME says it will
                                     # be the value for the great majority of sources

# Guideline recommendation strength. [CONFIRMED FROM SME — Agent 5 step 7]
REC_PREFERRED = "preferred"
REC_CONDITIONAL = "conditional_alternative"
REC_NOT_RECOMMENDED = "not_recommended"
REC_NOT_STATED = "not_stated"

# Trial registry status. Verbatim ClinicalTrials.gov statusModule.overallStatus
# vocabulary — we only need to recognise "terminated", so only that one value
# is a named constant; everything else is stored and displayed as-is.
# [CONFIRMED FROM SME — terminated trial evidence is flagged, weighted lower,
# never excluded]
TRIAL_STATUS_TERMINATED = "TERMINATED"

# Peer-review status. Deterministic, never LLM-judged: derived from PubMed's
# own publication_types metadata for pubmed records, and a flat "unconfirmed"
# default for general-web records with no such metadata available.
# [CONFIRMED FROM SME — preprints and unconfirmed-status publications are never
# excluded, only clearly marked]
PEER_REVIEW_CONFIRMED = "peer_reviewed"
PEER_REVIEW_PREPRINT = "preprint"
PEER_REVIEW_UNCONFIRMED = "unconfirmed"

# Fallback signal for peer_review_status when a pubmed-tagged document has no
# publication_types metadata (only the direct PubMed-API path populates that).
# Not exhaustive -- just the well-known preprint servers likely to be cited.
PREPRINT_SERVER_DOMAINS = [
    "biorxiv.org", "medrxiv.org", "ssrn.com", "researchsquare.com", "preprints.org",
]

# Per-state comparator verdicts, rendered by the UI's two-value legend.
# [CONFIRMED FROM UI — comparator modal "standard of care" / "not used"]
STATE_STANDARD_OF_CARE = "standard_of_care"
STATE_NOT_USED = "not_used"
STATE_NOT_ESTABLISHED = "not_established"  # searched, nothing said either way

# Evidence / coverage statuses. Only `evidence_found` and
# `no_relevant_evidence_found` are directly SME-supported; the rest are
# [PROPOSED] and exist so a gap can never render as a silent blank.
EV_FOUND = "evidence_found"
EV_NONE = "no_relevant_evidence_found"
EV_INSUFFICIENT = "insufficient_evidence"
EV_SOURCE_INACCESSIBLE = "source_inaccessible"
EV_RETRIEVAL_FAILED = "retrieval_failed"
EV_NOT_APPLICABLE = "not_applicable"
EV_NO_CURATED_SOURCE = "no_curated_source"

# Exact wording the SME mandates. Agent 12 is explicit that the by-state view
# must use the narrower phrase, because that view never covers outcomes.
NOT_IDENTIFIED_COMPARATOR_TEXT = "No comparator is identified"
NOT_IDENTIFIED_GENERAL_TEXT = "No comparator or outcome is identified"

# Explicit "the source genuinely did not state this". Never fabricate a unit or
# a line of therapy to avoid a blank — a reviewer trusting an invented 95% CI is
# a worse outcome than an honest gap. [CONFIRMED FROM JIRA success criteria]
NOT_STATED_BY_SOURCE = "Not stated by source"

# Validation verdicts.
V_SUPPORTED = "SUPPORTED"
V_SUPPORTED_SUBPOPULATION = "SUPPORTED_SUBPOPULATION"
V_SUPPORTED_BROADER = "SUPPORTED_BROADER"
V_WRONG_POPULATION = "WRONG_POPULATION"
V_WRONG_INTERVENTION = "WRONG_INTERVENTION"
V_WRONG_SUBJECT_DRUG = "WRONG_SUBJECT_DRUG"
V_SCOPE_ONLY_NO_DATA = "SCOPE_ONLY_NO_DATA"
V_NOT_SUPPORTED = "NOT_SUPPORTED"
V_SOURCE_INACCESSIBLE = "SOURCE_INACCESSIBLE"
# Verdicts that let a record continue to harmonization.
PASSING_VERDICTS = {V_SUPPORTED, V_SUPPORTED_SUBPOPULATION, V_SUPPORTED_BROADER,
                    V_SCOPE_ONLY_NO_DATA}

# Scope-adjudication verdicts.
SCOPE_IN = "in_scope"
SCOPE_OUT = "out_of_scope"
SCOPE_UNCERTAIN = "uncertain"


# ---------------------------------------------------------------------------
# Retrieval tuning.
# ---------------------------------------------------------------------------

class RetrievalSettings:
    """Retrieval budget. Defaults favour recall: MAI-35277 states the run
    "can take longer to run in the background, being right matters more here
    than being fast". Cost is not the binding constraint; accuracy is."""

    # Candidate documents retrieved per (state, source class). The legacy
    # implementation used 2 and that was a measured recall ceiling.
    max_urls_per_state: int = 4
    # Ask the search for more candidates than are retrieved — search is billed
    # flat per call, so extra candidates are free and feed balanced selection.
    search_max_results: int = 10
    # States processed concurrently. I/O-bound.
    max_workers: int = 8
    # Retries for the search provider. The legacy implementation had none, so a
    # transient rate-limit was indistinguishable from "nothing found".
    search_max_attempts: int = 3
    search_backoff_seconds: float = 5.0
    # Minimum spacing between Tavily calls, enforced globally across every
    # worker thread. Our key's dashboard states 100 requests/minute; 0.75s
    # spacing caps us at ~80/min, leaving headroom below the hard limit
    # rather than riding the edge of it. Raise it if still rate-limited,
    # lower it if the key's limit is later raised.
    search_min_interval_seconds: float = 0.75
    # One refinement round, fired only where a coverage assertion fails.
    enable_refinement_round: bool = True
    # Per-request document cache. The same URL is otherwise fetched by
    # extraction, by the retry path and again by re-fetch validation.
    enable_document_cache: bool = True


RETRIEVAL = RetrievalSettings()


class LLMSettings:
    model_id: str = os.getenv("JCA_MODEL_ID", "anthropic.claude-sonnet-5")
    region: str = os.getenv("AWS_REGION", "us-east-1")
    max_attempts: int = 3
    backoff_seconds: float = 3.0
    # Chunking (agents/a06_a08_retrieval.py's _chunk_document_text) already
    # bounds how much any single call can need to generate, by keeping the
    # INPUT to ~30000 chars per call -- so this cap's job is now headroom for
    # one unusually dense chunk, not "hold an entire document's findings."
    # Raised modestly (not to the model's full output ceiling): BedrockLLM
    # calls invoke_model synchronously with a fixed client-side read_timeout
    # (providers/llm.py), so a much larger max_tokens risks trading a fast,
    # cheap truncation for a slow, expensive read-timeout-then-retry failure
    # instead -- raised together with read_timeout, in proportion, below.
    extraction_max_tokens: int = 24000
    # Was flat 4000 -- same truncation shape already fixed at a12/a14/a16: a
    # batch of many claims from one re-fetched source needs headroom per
    # claim, not a single shared ceiling. Confirmed real defect: a source
    # discussing a dozen+ claims in one batch (e.g. a national HTA transcript
    # naming the subject drug plus several comparators) could truncate
    # mid-JSON, failing EVERY claim in that batch closed -- including
    # genuinely on-target ones. Raised in line with validation_max_tokens's
    # siblings (outcome_harmonization_max_tokens: 8000). Paired with
    # _MAX_CLAIMS_PER_VALIDATION_BATCH in a09_a13_validation.py so the cap is
    # never the ONLY thing standing between one dense source and a mass
    # failed-closed loss.
    validation_max_tokens: int = 8000
    default_max_tokens: int = 2000
    # Named per-call caps, centralized here instead of literals scattered
    # across agent files -- that scattering is exactly what let a06 (1500)
    # and a16.indication_synthesis (600) sit at 100%-truncation caps across
    # three real production runs before anyone noticed. Confirmed via
    # Bedrock Calls telemetry from those runs, cross-referenced against each
    # call site's actual output_tokens.
    query_vocabulary_max_tokens: int = 4000       # was inline 1500
    scope_adjudication_max_tokens: int = 4000     # was inline 2000
    outcome_harmonization_max_tokens: int = 8000  # was ALREADY 4000 and still
                                                    # truncating -- needs more
                                                    # headroom than a10, not the same
    indication_synthesis_max_tokens: int = 1500   # was inline 600


LLM = LLMSettings()


# ---------------------------------------------------------------------------
# Source workbook location. Configurable — the single most consequential defect
# in the legacy implementation was a hardcoded path pointing at the wrong file,
# which meant Data Team edits reached nothing.
# ---------------------------------------------------------------------------

SOURCE_WORKBOOK_PATH: str = os.getenv(
    "JCA_SOURCE_WORKBOOK",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "fixtures", "MadeAi_JCA_Source_List.xlsx"),
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
