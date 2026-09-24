"""
Parses `fixtures/jca gt.xlsx` (the GT reference workbook) into Population +
Intervention input dicts per drug sheet, for feeding into `run_phase1`.

Only the LICENSED population block + disease-classification fields are taken;
the broader "intended-to-treat" sub-population rows are ground truth for
Phase 2 PICO scoping, not Phase 1 input, so they are skipped. A value of
"Not stated in JCA report reviewed" (or "N/A ...") means the field is left
unset -- A2 will mark it `not_provided` rather than receiving a fabricated
value.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import openpyxl

POPULATION_LABELS = {
    "Indication / disease": "indication_disease",
    "Age group": "age_group",
    "Sex": "sex",
    "Stage / severity": "stage_severity",
    "Molecular / biomarker status": "molecular_biomarker_status",
    "Prior therapy / line": "prior_therapy_line",
    "Performance status": "performance_status",
    "Other characteristics (free text)": "other_characteristics",
    "Is the intended-to-treat population the same?": "itt_differs",
    "Therapeutic area": "therapeutic_area",
    "Treatment setting / intent": "treatment_setting_intent",
    "Disease subtype / histology": "disease_subtype_histology",
    "Key organ function / comorbidity": "organ_function_comorbidity",
    "ICD disease code (optional)": "icd_disease_code",
    "Orphan / rare-disease": "orphan_rare_disease",
}

INTERVENTION_LABELS = {
    "Product name (INN)": "product_name_inn",
    "Therapeutic class / mechanism": "therapeutic_class_mechanism",
    "Route of administration": "route_of_administration",
    "Dose(s)": "dose",
    "Treatment duration": "treatment_duration",
    "Line of therapy": "line_of_therapy",
    "Dosing schedule": "dosing_schedule",
    "Product modality / type": "product_modality",
    "MA type / regulatory pathway": "regulatory_pathway",
    "Pharmaceutical form / strength": "pharmaceutical_form_strength",
    "Companion diagnostic": "companion_diagnostic",
    "Pivotal trial identifier(s)": "pivotal_trial_identifiers",
    "Claimed / anticipated indication wording": "claimed_indication_wording",
}

_UNSTATED_PREFIXES = ("not stated in jca report reviewed", "n/a", "none required / not stated")


def _clean_label(raw: Optional[str]) -> str:
    if not raw:
        return ""
    # Strip leading circled-number markers (①②③) some sheets use.
    return raw.strip().lstrip("①②③④⑤⑥⑦⑧⑨").strip()


def _usable(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    return not any(text.lower().startswith(p) for p in _UNSTATED_PREFIXES)


def parse_gt_sheet(path: str, sheet_name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[sheet_name]

    population: Dict[str, Any] = {}
    intervention: Dict[str, Any] = {}
    section = None  # "population" | "itt" | "disease" | "intervention" | "output"

    for row in ws.iter_rows(values_only=True):
        label = _clean_label(row[0] if len(row) > 0 else None)
        value = row[1] if len(row) > 1 else None

        if not label:
            continue
        if label.startswith("INPUT — POPULATION") or label.startswith("INPUT - POPULATION"):
            section = "population"
            continue
        if label.startswith("LICENSED POPULATION"):
            continue
        if label.startswith("INTENDED-TO-TREAT POPULATION"):
            section = "itt"
            continue
        if label.startswith("DISEASE CLASSIFICATION"):
            section = "disease"
            continue
        if label.startswith("INPUT — INTERVENTION") or label.startswith("INPUT - INTERVENTION"):
            section = "intervention"
            continue
        if label.startswith("OUTPUT —") or label.startswith("OUTPUT -"):
            section = "output"
            continue
        if section == "output":
            break
        if label in ("Field", "Value", "Source (JCA report)"):
            continue

        if section in ("population", "disease") and label in POPULATION_LABELS and _usable(value):
            population[POPULATION_LABELS[label]] = str(value).strip()
        elif section == "intervention" and label in INTERVENTION_LABELS and _usable(value):
            intervention[INTERVENTION_LABELS[label]] = str(value).strip()

    return population, intervention


GT_DRUGS = ["Tovorafenib", "Lurbinectedin", "Tarlatamab"]


def parse_all(path: str) -> Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]]:
    return {drug: parse_gt_sheet(path, drug) for drug in GT_DRUGS}


if __name__ == "__main__":
    import json
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "fixtures/jca gt.xlsx"
    for drug, (pop, interv) in parse_all(path).items():
        print(f"=== {drug} ===")
        print("population:", json.dumps(pop, indent=2, ensure_ascii=True))
        print("intervention:", json.dumps(interv, indent=2, ensure_ascii=True))
        print()
