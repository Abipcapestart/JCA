"""
A self-contained, credential-free demo scenario.

Purpose: run the entire Phase 1 pipeline end to end with deterministic fakes, so
the wiring, the invariants and the completeness assertions can be exercised in
CI and demonstrated locally. The clinical content is realistic but synthetic —
it is a harness, not a data source.

The scenario deliberately includes the failure modes that mattered:
  * a comparator whose class must come from ITS OWN INN, not the intervention's
  * a record about a DIFFERENT subject drug, which must be rejected
  * an induction-phase backbone whose role is prior_therapy, not comparator
  * a guideline landscape document naming a comparator the drug-anchored query
    would never surface
  * outcome categories where nothing is found, which must still be reported
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from . import config as C
from .providers.llm import ScriptedLLM
from .providers.registries import (FixtureLiterature, FixtureMedicineRegistry,
                                   FixtureTrialRegistry, MedicineRecord,
                                   PublicationRecord, TrialArm, TrialRecord)
from .providers.search import FixtureSearchProvider, SearchHit

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

DEMO_POPULATION: Dict[str, Any] = {
    "indication_disease": "Extensive-stage small cell lung cancer",
    "stage_severity": "Extensive-stage disease",
    "prior_therapy_line": "Progressed on or after first-line platinum-based chemotherapy",
    "age_group": "Adults",
    "orphan_rare_disease": "Yes",
    "treatment_setting_intent": "Palliative, second line or later",
    "other_characteristics": "Chemotherapy-free interval of at least 90 days since the last platinum dose",
}

DEMO_INTERVENTION: Dict[str, Any] = {
    "product_name_inn": "Examplimab",
    "line_of_therapy": "Second line or later",
    "therapeutic_class_mechanism": "Bispecific T-cell engager",
    "claimed_indication_wording": (
        "Monotherapy for the treatment of adult patients with extensive-stage small "
        "cell lung cancer who require systemic therapy following progression on or "
        "after platinum-based chemotherapy"),
    "pivotal_trial_identifiers": "NCT09999999",
}

# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

_GERMAN_HTA = """G-BA Nutzenbewertung — Examplimab

Anwendungsgebiet: Erwachsene mit kleinzelligem Lungenkarzinom im
fortgeschrittenen Stadium (extensive-stage SCLC) nach Progression unter oder nach
einer platinbasierten Chemotherapie.

Zweckmaessige Vergleichstherapie: Topotecan.

The appropriate comparator therapy determined by the G-BA for this population is
topotecan, for patients in the second line following progression after
platinum-based chemotherapy. The committee requires relative effect estimates for
overall survival and for health-related quality of life measured with a
disease-specific instrument.
"""

_GUIDELINE_LANDSCAPE = """ESMO Clinical Practice Guideline — Small Cell Lung Cancer

Second-line treatment of relapsed extensive-stage small cell lung cancer.

For patients relapsing after first-line platinum-based chemotherapy, topotecan
remains a recommended option [I, A]. CAV (cyclophosphamide, doxorubicin and
vincristine) is an alternative option [II, B] and may be preferred where
myelosuppression is a concern. For patients with a chemotherapy-free interval of
at least 90 days, re-challenge with a platinum doublet is a valid option [II, B].

Recommended outcomes for assessment in this setting are overall survival,
progression-free survival, objective response rate, and health-related quality of
life assessed with a disease-specific instrument. Patient-reported symptom burden
should also be captured.
"""

_TRIAL_PAGE = """Examplimab in relapsed extensive-stage SCLC — assessment report

The pivotal study randomised patients with extensive-stage SCLC who had
progressed after first-line platinum-based chemotherapy to examplimab or to
topotecan. All patients had previously received carboplatin plus etoposide as
first-line induction therapy before enrolment; that induction regimen is prior
therapy and was not a randomised comparison in this study.

Reported outcomes were overall survival, progression-free survival, objective
response rate, serious adverse events, and grade >=3 adverse events.
"""

_OTHER_DRUG = """Otherdrugimab in extensive-stage SCLC maintenance

Otherdrugimab was compared against atezolizumab monotherapy in the maintenance
setting for patients who had not progressed after first-line induction. Overall
survival was the primary endpoint.
"""

# Also the PubMed abstract's full text -- claim validation (A10) re-fetches
# the source URL, so this must resolve the same way a real Tavily fetch of a
# PubMed page would, for the fixture to exercise that step faithfully.
_PUBMED_ABSTRACT = "Topotecan and CAV remain the principal second-line options."

DEMO_DOCUMENTS: Dict[str, str] = {
    "https://g-ba.de/bewertungsverfahren/examplimab": _GERMAN_HTA,
    "https://esmo.org/guidelines/lung-and-chest-tumours/small-cell-lung-cancer": _GUIDELINE_LANDSCAPE,
    "https://ema.europa.eu/en/medicines/human/EPAR/examplimab": _TRIAL_PAGE,
    "https://esmo.org/guidelines/other-drug-maintenance": _OTHER_DRUG,
    "https://pubmed.ncbi.nlm.nih.gov/00000001/": _PUBMED_ABSTRACT,
}

DEMO_INDEX: Dict[str, List[SearchHit]] = {
    "g-ba.de": [SearchHit(url="https://g-ba.de/bewertungsverfahren/examplimab",
                          title="G-BA Nutzenbewertung Examplimab")],
    "esmo.org": [
        SearchHit(url="https://esmo.org/guidelines/lung-and-chest-tumours/small-cell-lung-cancer",
                  title="ESMO SCLC Clinical Practice Guideline"),
        SearchHit(url="https://esmo.org/guidelines/other-drug-maintenance",
                  title="Otherdrugimab maintenance guideline section"),
    ],
    "ema.europa.eu": [SearchHit(url="https://ema.europa.eu/en/medicines/human/EPAR/examplimab",
                                title="Examplimab EPAR")],
}

DEMO_TRIALS = [TrialRecord(
    identifier="NCT09999999",
    title="Examplimab versus topotecan in relapsed ES-SCLC",
    url="https://clinicaltrials.gov/study/NCT09999999",
    phase="PHASE3", status="COMPLETED",
    conditions=["Small Cell Lung Carcinoma"],
    arms=[
        TrialArm(label="Examplimab", arm_type="EXPERIMENTAL", interventions=["Examplimab"]),
        TrialArm(label="Topotecan", arm_type="ACTIVE_COMPARATOR", interventions=["Topotecan"],
                 description="Topotecan administered per local standard of care"),
    ],
    primary_outcomes=["Overall survival"],
    secondary_outcomes=["Progression-free survival", "Objective response rate"],
)]

DEMO_PUBLICATIONS = [PublicationRecord(
    pmid="00000001", title="Management of relapsed small cell lung cancer",
    abstract=_PUBMED_ABSTRACT,
    journal="Annals of Oncology", year="2025",
    publication_types=["Practice Guideline"],
    url="https://pubmed.ncbi.nlm.nih.gov/00000001/")]

DEMO_MEDICINES = {"examplimab": MedicineRecord(
    name="Examplimab", inn="examplimab", atc_code="L01FX99",
    indication_text=DEMO_INTERVENTION["claimed_indication_wording"],
    product_info_url="https://ema.europa.eu/en/medicines/human/EPAR/examplimab",
    pivotal_trials=["NCT09999999"])}


# ---------------------------------------------------------------------------
# Scripted LLM
# ---------------------------------------------------------------------------

def _extraction(user_prompt: str) -> str:
    """Return records appropriate to whichever document is in the prompt."""
    if "G-BA" in user_prompt or "Nutzenbewertung" in user_prompt:
        return json.dumps([
            {"finding_type": "comparator", "subject_drug": "Examplimab",
             "member_state": "Germany",
             "comparator": {"as_stated": "Topotecan", "role": "active_comparator",
                            "is_combination": False, "components": ["Topotecan"],
                            "comparator_scenario": "unique",
                            "retain_all_status": "unconfirmed"},
             "population_context": {
                 "disease": "extensive-stage SCLC", "stage": "extensive-stage",
                 "line_of_therapy": "second line",
                 "prior_therapy": "platinum-based chemotherapy",
                 "verbatim": "nach Progression unter oder nach einer platinbasierten Chemotherapie"},
             "recommendation_strength": "preferred",
             "evidence_quote": "The appropriate comparator therapy determined by the G-BA for this population is topotecan",
             "evidence_locator": "Zweckmaessige Vergleichstherapie"},
            {"finding_type": "outcome", "subject_drug": "Examplimab",
             "member_state": "Germany",
             "outcome": {"measure": "Overall survival", "unit": "months, median",
                         "is_requirement": True,
                         "requirement_type": "relative_effect_required"},
             "population_context": {"disease": "extensive-stage SCLC"},
             "evidence_quote": "The committee requires relative effect estimates for overall survival"},
            {"finding_type": "outcome", "subject_drug": "Examplimab",
             "member_state": "Germany",
             "outcome": {"measure": "health-related quality of life measured with a disease-specific instrument",
                         "unit": "points, change from baseline", "is_requirement": True,
                         "requirement_type": "relative_effect_required"},
             "population_context": {"disease": "extensive-stage SCLC"},
             "evidence_quote": "health-related quality of life measured with a disease-specific instrument"},
        ])
    if "ESMO Clinical Practice Guideline" in user_prompt:
        return json.dumps([
            {"finding_type": "comparator", "subject_drug": "Examplimab",
             "comparator": {"as_stated": "Topotecan", "role": "active_comparator",
                            "comparator_scenario": "at_least_one",
                            "retain_all_status": "unconfirmed", "components": ["Topotecan"]},
             "population_context": {"disease": "relapsed extensive-stage small cell lung cancer",
                                    "line_of_therapy": "second-line"},
             "recommendation_strength": "preferred",
             "evidence_quote": "topotecan remains a recommended option [I, A]"},
            {"finding_type": "comparator", "subject_drug": "Examplimab",
             "comparator": {"as_stated": "CAV (cyclophosphamide, doxorubicin and vincristine)",
                            "role": "active_comparator", "is_combination": True,
                            "components": ["cyclophosphamide", "doxorubicin", "vincristine"],
                            "comparator_scenario": "at_least_one",
                            "retain_all_status": "unconfirmed"},
             "population_context": {"disease": "relapsed extensive-stage small cell lung cancer",
                                    "line_of_therapy": "second-line"},
             "recommendation_strength": "conditional_alternative",
             "evidence_quote": "CAV (cyclophosphamide, doxorubicin and vincristine) is an alternative option [II, B]"},
            {"finding_type": "outcome", "subject_drug": "Examplimab",
             "outcome": {"measure": "Patient-reported symptom burden",
                         "is_requirement": True, "requirement_type": "relative_effect_required"},
             "population_context": {"disease": "relapsed extensive-stage SCLC"},
             "evidence_quote": "Patient-reported symptom burden should also be captured"},
        ])
    if "principal second-line options" in user_prompt:
        # The PubMed abstract wired via FixtureLiterature/DEMO_PUBLICATIONS.
        # Names the same CAV regimen the ESMO guideline names, but via a
        # different source class/tier -- exercises cross-tier confirmation.
        return json.dumps([
            {"finding_type": "comparator", "subject_drug": "Examplimab",
             # "as_stated" matches the quote verbatim (just "CAV") -- the
             # source text never spells out the regimen, unlike the ESMO
             # guideline branch above, which does.
             "comparator": {"as_stated": "CAV",
                            "role": "active_comparator", "is_combination": True,
                            "components": ["cyclophosphamide", "doxorubicin", "vincristine"],
                            "comparator_scenario": "at_least_one",
                            "retain_all_status": "unconfirmed"},
             "population_context": {"disease": "relapsed extensive-stage small cell lung cancer",
                                    "line_of_therapy": "second-line"},
             "recommendation_strength": "not_stated",
             "evidence_quote": "Topotecan and CAV remain the principal second-line options."},
        ])
    if "Otherdrugimab" in user_prompt:
        # MUST be rejected: different subject drug.
        return json.dumps([
            {"finding_type": "comparator", "subject_drug": "Otherdrugimab",
             "comparator": {"as_stated": "Atezolizumab", "role": "active_comparator",
                            "components": ["Atezolizumab"]},
             "population_context": {"disease": "extensive-stage SCLC",
                                    "treatment_setting_intent": "maintenance"},
             "evidence_quote": "Otherdrugimab was compared against atezolizumab monotherapy in the maintenance setting"},
        ])
    if "assessment report" in user_prompt or "EPAR" in user_prompt:
        return json.dumps([
            {"finding_type": "comparator", "subject_drug": "Examplimab",
             "member_state": "EU-wide",
             "comparator": {"as_stated": "Topotecan", "role": "active_comparator",
                            "components": ["Topotecan"]},
             "population_context": {"disease": "extensive-stage SCLC",
                                    "line_of_therapy": "second line",
                                    "prior_therapy": "first-line platinum-based chemotherapy"},
             "evidence_quote": "randomised patients with extensive-stage SCLC who had progressed after first-line platinum-based chemotherapy to examplimab or to topotecan"},
            # Induction backbone: prior therapy, NOT a comparator.
            {"finding_type": "comparator", "subject_drug": "Examplimab",
             "member_state": "EU-wide",
             "comparator": {"as_stated": "Carboplatin plus etoposide", "role": "prior_therapy",
                            "is_combination": True, "components": ["carboplatin", "etoposide"]},
             "population_context": {"disease": "extensive-stage SCLC",
                                    "line_of_therapy": "first-line induction"},
             "evidence_quote": "All patients had previously received carboplatin plus etoposide as first-line induction therapy before enrolment; that induction regimen is prior therapy"},
            {"finding_type": "outcome", "subject_drug": "Examplimab", "member_state": "EU-wide",
             "outcome": {"measure": "Serious adverse events", "unit": "% of patients"},
             "population_context": {"disease": "extensive-stage SCLC"},
             "evidence_quote": "Reported outcomes were overall survival, progression-free survival, objective response rate, serious adverse events"},
            {"finding_type": "outcome", "subject_drug": "Examplimab", "member_state": "EU-wide",
             "outcome": {"measure": "Grade >=3 adverse events", "unit": "% of patients"},
             "population_context": {"disease": "extensive-stage SCLC"},
             "evidence_quote": "serious adverse events, and grade >=3 adverse events"},
        ])
    return "[]"


def _claim_validation(user_prompt: str) -> str:
    """Approve everything the deterministic pre-filter let through, except the
    clearly out-of-population induction regimen."""
    import re
    n = len(re.findall(r"^\d+\.", user_prompt, re.MULTILINE))
    results = []
    for i in range(1, max(n, 1) + 1):
        block = user_prompt.split(f"\n{i}. ")[-1][:400] if f"\n{i}. " in user_prompt else ""
        if "prior_therapy" in block or "induction" in block.lower():
            results.append({"index": i, "verdict": "WRONG_POPULATION",
                            "reason": "induction-phase regimen is prior therapy for this population"})
        else:
            results.append({"index": i, "verdict": "SUPPORTED",
                            "reason": "the re-fetched source states this for this population"})
    return json.dumps({"results": results})


def _scope_adjudication(user_prompt: str) -> str:
    if "Carboplatin" in user_prompt and "prior_therapy" in user_prompt:
        return json.dumps({"verdict": "out_of_scope", "decisive_facet": "line_of_therapy",
                           "reason": "first-line induction backbone, not a second-line comparator",
                           "evidence_for": [], "evidence_against": []})
    return json.dumps({"verdict": "in_scope", "decisive_facet": "line_of_therapy",
                       "reason": "named as a second-line option for this population",
                       "evidence_for": [], "evidence_against": []})


def _identity(user_prompt: str) -> str:
    import re
    items = []
    for line in user_prompt.splitlines():
        m = re.match(r"(\d+)\.\s+(.*)", line.strip())
        if not m:
            continue
        idx, name = int(m.group(1)), m.group(2)
        low = name.lower()
        if "cav" in low or ("cyclophosphamide" in low and "doxorubicin" in low):
            items.append({"index": idx, "inn": "",
                          "display_name": "Cyclophosphamide + Doxorubicin + Vincristine",
                          "is_combination": True,
                          "components": ["cyclophosphamide", "doxorubicin", "vincristine"],
                          "class_mechanism": "alkylating agent + anthracycline + vinca alkaloid combination",
                          "brand_names": [], "is_category": False, "atc_code": ""})
        else:
            items.append({"index": idx, "inn": low.split()[0], "display_name": name,
                          "is_combination": False, "components": [], "is_category": False,
                          "class_mechanism": "", "brand_names": [], "atc_code": ""})
    return json.dumps({"items": items, "excluded": []})


DEMO_LLM_RESPONSES: Dict[str, Any] = {
    "a01.pi_validation": json.dumps({"fix_items": [], "check_items": []}),
    "a02.input_structuring": json.dumps({"populations": [], "intervention": {"fields": {}}}),
    "a03.scope_facet_normalise": json.dumps({"facets": [
        {"name": "treatment_free_interval", "value": "at least 90 days since the last platinum dose",
         "discriminating": True,
         "verbatim": "Chemotherapy-free interval of at least 90 days since the last platinum dose"}]}),
    "a04.indication_lock": json.dumps({
        "indication_text": DEMO_INTERVENTION["claimed_indication_wording"],
        "pivotal_trials": ["NCT09999999"], "atc_code": "L01FX99", "notes": ""}),
    "a05.area_adjudication": json.dumps({"areas": [
        {"area": "Oncology", "rationale": "Small cell lung cancer is a thoracic malignancy."}]}),
    "a06.query_vocabulary": json.dumps({
        "indication_synonyms": ["extensive-stage small cell lung cancer", "ES-SCLC"],
        "indication_abbreviations": ["ES-SCLC", "SCLC"],
        "disease_class_terms": ["small cell lung cancer", "lung neoplasm"],
        "localised_assessment_terms": {"de": ["Nutzenbewertung"], "fr": ["avis"]},
        "outcome_requirement_terms": ["required outcomes", "outcome measures for assessment"]}),
    "a08.extraction": _extraction,
    "a10.claim_validation": _claim_validation,
    "a11.comparator_identity": _identity,
    "a12.scope_adjudication": _scope_adjudication,
    "a14.outcome_harmonization": json.dumps({"groups": [], "excluded": []}),
    "a16.comparator_rationale":
        "Positioned at second line or later; named as a recommended option for patients "
        "who have progressed after first-line platinum-based chemotherapy.",
    "a16.outcome_rationale":
        "Required by assessing bodies as a relative-effect endpoint in this setting.",
    "a16.indication_synthesis": json.dumps({
        "indication": "Extensive-stage small cell lung cancer, second line or later following "
                      "progression on platinum-based chemotherapy",
        "scope_note": ""}),
}


def build_demo_providers():
    """Providers wired to the fixture scenario. No credentials, no network."""
    from .orchestrator import Providers
    return Providers(
        llm=ScriptedLLM(DEMO_LLM_RESPONSES),
        search=FixtureSearchProvider(index=DEMO_INDEX, documents=DEMO_DOCUMENTS),
        trials=FixtureTrialRegistry(DEMO_TRIALS),
        literature=FixtureLiterature(DEMO_PUBLICATIONS),
        medicines=FixtureMedicineRegistry(DEMO_MEDICINES),
    )


def run_demo(**kwargs):
    from .orchestrator import RunOptions, run_phase1
    return run_phase1(DEMO_POPULATION, DEMO_INTERVENTION, build_demo_providers(),
                      options=RunOptions(request_id="demo-run", **kwargs))
