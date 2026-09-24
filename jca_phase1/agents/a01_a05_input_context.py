"""
A1  P&I Validation            — SME Agent 1, kept as specified
A2  Input Structuring         — SME Agent 2, kept as specified
A4  Indication Lock           — NEW (nothing in the SME bounds the indication)
A3  Population Scope Boundary — NEW (the SME has no machine-checkable boundary)
A5  Therapeutic Area          — SME Agent 5 rules; the fix is that the result is USED

A3 runs after A4 because the licensed indication wording is part of the boundary.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .. import config as C
from ..providers.llm import LLMClient
from ..providers.registries import MedicineRegistryClient
from ..providers.search import SearchProvider
from ..schema import (
    AreaResolution, CONFIRMED, DERIVED_FROM_LABEL, Field, INFERRED, Intervention,
    LeakageGuard, LicensedIndicationRecord, NOT_PROVIDED, POP_ITT, POP_LICENSED,
    PIValidationResult, Population, ScopeBoundary, ScopeFacet, SEVERITY_CHECK,
    SEVERITY_FIX, ValidationItem,
)

# ---------------------------------------------------------------------------
# Field taxonomy, loaded once.
# ---------------------------------------------------------------------------

def _load_taxonomy() -> Dict[str, Any]:
    with open(os.path.join(C.DATA_DIR, "field_taxonomy.json"), encoding="utf-8") as f:
        return json.load(f)


TAXONOMY = _load_taxonomy()
POPULATION_FIELDS = {f["field"]: f for f in TAXONOMY["population"]}
INTERVENTION_FIELDS = {f["field"]: f for f in TAXONOMY["intervention"]}
MANDATORY_POPULATION = [f["field"] for f in TAXONOMY["population"] if f["mandatory"]]
MANDATORY_INTERVENTION = [f["field"] for f in TAXONOMY["intervention"] if f["mandatory"]]


# ===========================================================================
# A1 — P&I Validation
# ===========================================================================

def validate_pi(population_input: Dict[str, Any],
                intervention_input: Dict[str, Any],
                added_fields: Optional[List[str]] = None,
                llm: Optional[LLMClient] = None) -> PIValidationResult:
    """FIX blocks, CHECK advises. Mandatory-field emptiness is deterministic;
    field-appropriateness is the LLM's judgment.

    `added_fields` lists fields the user explicitly ADDED to the form. A field
    never added is never a CHECK item — the SME is explicit that there is
    nothing to check if it was never added.
    """
    result = PIValidationResult()

    # -- deterministic: mandatory fields ------------------------------------
    for fname in MANDATORY_POPULATION:
        if not str(population_input.get(fname, "") or "").strip():
            label = POPULATION_FIELDS[fname]["label"]
            result.fix_items.append(ValidationItem(
                severity=SEVERITY_FIX, fields=[fname],
                explanation=f"{label} is mandatory and has not been provided."))
    for fname in MANDATORY_INTERVENTION:
        if not str(intervention_input.get(fname, "") or "").strip():
            label = INTERVENTION_FIELDS[fname]["label"]
            result.fix_items.append(ValidationItem(
                severity=SEVERITY_FIX, fields=[fname],
                explanation=f"{label} is mandatory and has not been provided."))

    # -- deterministic: added-but-empty optional fields ---------------------
    added = set(added_fields or [])
    empty_added = sorted(
        f for f in added
        if not str(population_input.get(f, intervention_input.get(f, "")) or "").strip()
        and f not in MANDATORY_POPULATION and f not in MANDATORY_INTERVENTION)
    if empty_added:
        labels = [POPULATION_FIELDS.get(f, INTERVENTION_FIELDS.get(f, {})).get("label", f)
                  for f in empty_added]
        result.check_items.append(ValidationItem(
            severity=SEVERITY_CHECK, fields=empty_added,
            explanation=(f"{len(empty_added)} field(s) you added are empty: "
                         f"{', '.join(labels)}. They narrow nothing as they stand — "
                         f"either fill them in or remove them.")))

    # -- LLM: field-appropriateness ----------------------------------------
    if llm is not None:
        filled = {k: v for k, v in {**population_input, **intervention_input}.items()
                  if str(v or "").strip()}
        if filled:
            lines = []
            for k, v in filled.items():
                spec = POPULATION_FIELDS.get(k) or INTERVENTION_FIELDS.get(k) or {}
                lines.append(f'- {k} ("{spec.get("label", k)}" — '
                             f'{spec.get("definition", "no definition")}): {v}')
            payload = "FIELDS WITH CONTENT:\n" + "\n".join(lines)
            parsed = llm.call_json("a01.pi_validation", payload,
                                   max_tokens=C.LLM.pi_validation_max_tokens, default={})
            for item in (parsed or {}).get("fix_items", []) or []:
                fields = item.get("fields") or []
                if any(f in {i.fields[0] for i in result.fix_items if i.fields} for f in fields):
                    continue  # already flagged deterministically
                result.fix_items.append(ValidationItem(
                    severity=SEVERITY_FIX, fields=fields,
                    value_flagged=item.get("value_flagged", ""),
                    explanation=item.get("explanation", "")))
    return result.finalise()


# ===========================================================================
# A2 — Input Structuring
# ===========================================================================

def structure_pi(population_input: Dict[str, Any],
                 intervention_input: Dict[str, Any],
                 free_text: str = "",
                 llm: Optional[LLMClient] = None) -> Tuple[List[Population], Intervention]:
    """Map whatever arrived onto the taxonomy, tagging provenance honestly.

    Runs even when the form was filled in directly — as a lightweight
    confirmation pass — because the provenance tags themselves are the output.
    """
    populations: List[Population] = []
    licensed = Population(population_id=POP_LICENSED)
    intervention = Intervention()

    for fname in POPULATION_FIELDS:
        raw = str(population_input.get(fname, "") or "").strip()
        licensed.fields[fname] = Field(value=raw or None,
                                       provenance=CONFIRMED if raw else NOT_PROVIDED,
                                       label=POPULATION_FIELDS[fname]["label"])
    for fname in INTERVENTION_FIELDS:
        raw = str(intervention_input.get(fname, "") or "").strip()
        intervention.fields[fname] = Field(value=raw or None,
                                           provenance=CONFIRMED if raw else NOT_PROVIDED,
                                           label=INTERVENTION_FIELDS[fname]["label"])

    # Custom fields the caller supplied explicitly.
    for label, value in (population_input.get("_custom") or {}).items():
        if str(value or "").strip():
            licensed.custom_fields.append(
                Field(value=str(value).strip(), provenance=CONFIRMED, label=str(label)))
    for label, value in (intervention_input.get("_custom") or {}).items():
        if str(value or "").strip():
            intervention.custom_fields.append(
                Field(value=str(value).strip(), provenance=CONFIRMED, label=str(label)))

    # LLM pass: free text, and permitted inferences.
    if llm is not None and (free_text.strip() or any(
            f.value for f in licensed.fields.values())):
        payload = (
            "POPULATION FIELD TAXONOMY:\n"
            + "\n".join(f'- {k}: {v["definition"]}' for k, v in POPULATION_FIELDS.items())
            + "\n\nINTERVENTION FIELD TAXONOMY:\n"
            + "\n".join(f'- {k}: {v["definition"]}' for k, v in INTERVENTION_FIELDS.items())
            + "\n\nSTRUCTURED INPUT AS ENTERED:\n"
            + json.dumps({"population": population_input, "intervention": intervention_input},
                         indent=2, default=str)
            + (f"\n\nFREE TEXT:\n{free_text}" if free_text.strip() else ""))
        parsed = llm.call_json("a02.input_structuring", payload,
                               max_tokens=C.LLM.input_structuring_max_tokens, default={})
        _apply_structuring(parsed, licensed, intervention, populations)

    populations.insert(0, licensed)

    # SME Agent 2 step 8: an ITT population that differs becomes its own object.
    itt_decl = licensed.value("itt_differs")
    already_has_itt = any(p.population_id == POP_ITT for p in populations)
    if itt_decl and not already_has_itt and _declares_difference(itt_decl):
        itt = Population(population_id=POP_ITT)
        for k, f in licensed.fields.items():
            itt.fields[k] = Field(value=f.value, provenance=f.provenance,
                                  inference_basis=f.inference_basis, label=f.label)
        itt.fields["other_characteristics"] = Field(
            value=itt_decl, provenance=CONFIRMED,
            label="Other characteristics (intended-to-treat)")
        populations.append(itt)

    return populations, intervention


def _declares_difference(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    if t.startswith(("no", "differ", "broader", "wider")):
        return True
    return not t.startswith(("yes, same", "same", "yes"))


def _apply_structuring(parsed: Dict[str, Any], licensed: Population,
                       intervention: Intervention, extra: List[Population]) -> None:
    if not parsed:
        return
    for i, pop in enumerate(parsed.get("populations", []) or []):
        target = licensed
        if i > 0:
            target = Population(population_id=pop.get("population_id") or POP_ITT)
            extra.append(target)
        for fname, spec in (pop.get("fields") or {}).items():
            if fname not in POPULATION_FIELDS:
                continue
            value = (spec or {}).get("value") or ""
            prov = (spec or {}).get("provenance") or NOT_PROVIDED
            existing = target.fields.get(fname)
            # A user statement always wins over an inference.
            if existing and existing.provenance == CONFIRMED and existing.value:
                continue
            if not value:
                continue
            # SME Agent 2 rule 5: therapeutic class is NEVER inferred.
            if fname == "therapeutic_class_mechanism" and prov == INFERRED:
                continue
            target.fields[fname] = Field(
                value=value, provenance=prov if prov in (CONFIRMED, INFERRED) else NOT_PROVIDED,
                inference_basis=(spec or {}).get("inference_basis", ""),
                label=POPULATION_FIELDS[fname]["label"])
        for cf in (pop.get("custom_fields") or []):
            if cf.get("value"):
                target.custom_fields.append(Field(
                    value=cf["value"], provenance=CONFIRMED, label=cf.get("label", "custom")))

    inter = parsed.get("intervention") or {}
    for fname, spec in (inter.get("fields") or {}).items():
        if fname not in INTERVENTION_FIELDS:
            continue
        value = (spec or {}).get("value") or ""
        prov = (spec or {}).get("provenance") or NOT_PROVIDED
        existing = intervention.fields.get(fname)
        if existing and existing.provenance == CONFIRMED and existing.value:
            continue
        if not value:
            continue
        if fname == "therapeutic_class_mechanism" and prov == INFERRED:
            continue   # absolute rule
        intervention.fields[fname] = Field(
            value=value, provenance=prov if prov in (CONFIRMED, INFERRED) else NOT_PROVIDED,
            inference_basis=(spec or {}).get("inference_basis", ""),
            label=INTERVENTION_FIELDS[fname]["label"])


# ===========================================================================
# A4 — Indication Lock
# ===========================================================================

def lock_indication(intervention: Intervention,
                    medicine_registry: Optional[MedicineRegistryClient] = None,
                    search: Optional[SearchProvider] = None,
                    llm: Optional[LLMClient] = None,
                    # Default flipped to True per explicit SME decision
                    # (2026-09-23): the drug's own JCA report is now a
                    # deliberate, labeled Tier 1 source, not evaluation-only.
                    allow_jca_reports: bool = True
                    ) -> Tuple[LicensedIndicationRecord, LeakageGuard]:
    """Establish what the product is licensed (or claimed) for, and build the
    leakage guard.

    Falls back to the user's claimed indication wording when no label exists —
    which is the product's PRIMARY case, since the whole point is to scope
    before authorisation. A pipeline that only works post-approval fails its
    own use case.
    """
    product = intervention.product_name
    guard = LeakageGuard(
        enabled=True,
        allow_jca_reports=allow_jca_reports,
        drug_tokens=[t for t in (product or "").lower().split() if len(t) > 3])

    record = LicensedIndicationRecord(inn=product)

    medicine = medicine_registry.lookup(product) if medicine_registry else None
    if medicine:
        record.source = "ema_smpc" if medicine.product_info_url else "ema_epar"
        record.source_url = medicine.product_info_url or medicine.epar_url
        record.atc_code = medicine.atc_code
        record.indication_text = medicine.indication_text
        record.pivotal_trials = list(medicine.pivotal_trials)

    # Read the actual document when we have a URL and no indication text yet.
    if search and llm and record.source_url and not record.indication_text:
        doc = search.fetch(record.source_url, full_document=True)
        if doc.ok and doc.text:
            parsed = llm.call_json(
                "a04.indication_lock",
                f"PRODUCT: {product}\n\nDOCUMENT:\n{doc.text[:60000]}",
                max_tokens=C.LLM.indication_lock_max_tokens, default={}) or {}
            record.indication_text = parsed.get("indication_text", "") or ""
            record.pivotal_trials = list(dict.fromkeys(
                record.pivotal_trials + (parsed.get("pivotal_trials") or [])))
            record.atc_code = record.atc_code or parsed.get("atc_code", "")
            record.notes = parsed.get("notes", "")

    # Fall back to the claimed wording — the pre-approval path.
    if not record.indication_text:
        claimed = intervention.value("claimed_indication_wording")
        if claimed:
            record.source = "claimed_wording_only"
            record.indication_text = claimed
            record.notes = ("No authorised indication found; the boundary is the "
                            "manufacturer's claimed wording. Scope decisions are only "
                            "as reliable as that wording.")
        else:
            record.source = "none"
            record.notes = ("No authorised indication and no claimed indication wording "
                            "supplied. The population boundary rests on the user's "
                            "population fields alone, which is weaker — supplying the "
                            "claimed indication wording would materially improve it.")

    # Pivotal trial identifiers the user gave are authoritative.
    user_trials = intervention.value("pivotal_trial_identifiers")
    if user_trials:
        record.pivotal_trials = list(dict.fromkeys(
            record.pivotal_trials + [t.strip() for t in user_trials.replace(";", ",").split(",")
                                     if t.strip()]))
    return record, guard


# ===========================================================================
# A3 — Population Scope Boundary
# ===========================================================================

# Facets that exist as named slots but have no field in the SME taxonomy.
# They are reachable only via free text — see ARCHITECTURE.md open question 4.
_EXTRA_FACETS = ("treatment_free_interval", "recurrence_free_interval")


def build_scope_boundary(population: Population, intervention: Intervention,
                         indication: LicensedIndicationRecord,
                         llm: Optional[LLMClient] = None) -> ScopeBoundary:
    """Project the SME's own population fields into a machine-checkable predicate.

    Two properties do the real work:
      * a facet the user did NOT state is listed in `unbounded_facets` and never
        becomes a filter — this is the SME's honesty rule applied to retrieval,
        and it is what stops a real comparator being rejected on an axis nobody
        specified;
      * `must_match` marks only the facets that genuinely narrow the population,
        derived from the input rather than hardcoded.
    """
    boundary = ScopeBoundary(population_id=population.population_id,
                             licensed_indication_wording=indication.indication_text)

    for fname, spec in POPULATION_FIELDS.items():
        if not spec.get("scope_facet"):
            continue
        f = population.get(fname)
        facet = ScopeFacet(
            name=fname,
            value=f.value or "",
            provenance=f.provenance,
            must_match=bool(spec.get("discriminating")) and f.is_stated(),
            verbatim=f.value or "")
        boundary.facets[fname] = facet

    # Line of therapy lives on the intervention side but is a population axis.
    lot = intervention.fields.get("line_of_therapy")
    if lot is not None:
        boundary.facets["line_of_therapy"] = ScopeFacet(
            name="line_of_therapy", value=lot.value or "", provenance=lot.provenance,
            must_match=lot.is_stated(), verbatim=lot.value or "")

    # Lift facets out of free text — including axes with no named field.
    if llm is not None:
        free_bits = []
        oc = population.get("other_characteristics")
        if oc.value:
            free_bits.append(f"other_characteristics: {oc.value}")
        for cf in population.custom_fields:
            free_bits.append(f"{cf.label}: {cf.value}")
        if free_bits:
            payload = ("NAMED FIELDS ALREADY SET:\n"
                       + "\n".join(f"- {k}: {v.value}" for k, v in boundary.facets.items()
                                   if v.value)
                       + "\n\nFREE-TEXT FIELDS TO NORMALISE:\n" + "\n".join(free_bits))
            parsed = llm.call_json("a03.scope_facet_normalise", payload,
                                   max_tokens=C.LLM.scope_facet_normalise_max_tokens,
                                   default={}) or {}
            for item in parsed.get("facets", []) or []:
                name = item.get("name", "")
                value = item.get("value", "")
                if not name or not value:
                    continue
                existing = boundary.facets.get(name)
                if existing and existing.is_bound:
                    continue  # never overwrite what the user placed themselves
                boundary.facets[name] = ScopeFacet(
                    name=name, value=value, provenance=CONFIRMED,
                    must_match=bool(item.get("discriminating")),
                    verbatim=item.get("verbatim", value))

    for name in _EXTRA_FACETS:
        boundary.facets.setdefault(name, ScopeFacet(name=name))

    boundary.unbounded_facets = sorted(
        name for name, f in boundary.facets.items() if not f.is_bound)
    boundary.out_of_bounds_rule = (
        "flag_as_itt" if population.population_id == POP_ITT else "reject")
    return boundary


# ===========================================================================
# A5 — Therapeutic Area Resolution
# ===========================================================================

# Deterministic first pass. Keyword lists are cheap and cover the unambiguous
# majority; the LLM is reserved for genuine judgment (multi-area, and the
# Multi-Disciplinary call, which is about how literature is PUBLISHED and
# cannot be keyword-matched).
_AREA_KEYWORDS: Dict[str, List[str]] = {
    "Oncology": ["cancer", "carcinoma", "tumour", "tumor", "oncolog", "sarcoma",
                 "lymphoma", "leukemia", "leukaemia", "myeloma", "melanoma",
                 "glioma", "neoplasm", "metastat", "malignan"],
    "Cardiovascular": ["cardiac", "cardio", "heart", "hypertension", "atrial",
                       "myocardial", "thrombo", "stroke", "vascular", "lipid"],
    "CNS": ["alzheimer", "parkinson", "epilep", "multiple sclerosis", "migraine",
            "neurolog", "dementia", "schizophren", "depress", "seizure"],
    "Metabolic & Endocrine": ["diabet", "thyroid", "obesity", "metabolic",
                              "endocrin", "adrenal", "pituitary", "cushing"],
    "Infectious": ["infection", "viral", "hiv", "hepatitis", "tubercul",
                   "bacterial", "sepsis", "influenza", "covid", "antimicrob"],
    "Immunology": ["autoimmune", "lupus", "psoria", "immune", "allerg",
                   "atopic", "vasculitis", "rheumatoid"],
    "Respiratory": ["asthma", "copd", "pulmonary", "respiratory", "cystic fibrosis",
                    "bronch"],
    "Gastroenterology": ["crohn", "colitis", "hepatic", "liver", "gastro",
                         "bowel", "pancrea", "oesophag", "esophag"],
    "Musculoskeletal": ["arthritis", "osteo", "muscular", "bone", "spondyl",
                        "fibromyalgia", "myopathy"],
    "Rare Diseases": ["rare disease", "orphan"],
}


def resolve_therapeutic_areas(population: Population,
                              llm: Optional[LLMClient] = None) -> AreaResolution:
    """Resolve the area(s) whose guideline literature governs this indication.

    The legacy pipeline computed this correctly and then discarded it. Here the
    result is returned and CONSUMED by the query planner — which is the single
    highest-value change in the whole redesign, because without it ~89% of the
    per-state guideline search surface is the wrong specialty.
    """
    stated = C.canonicalize_area(population.value("therapeutic_area"))
    if stated:
        return AreaResolution(areas=_with_rare(stated, population),
                              rationales={stated: "Stated by the user."},
                              method="user_stated")

    haystack = " ".join([
        population.value("indication_disease"),
        population.value("disease_subtype_histology"),
        population.value("icd_disease_code"),
    ]).lower()

    hits = [area for area, kws in _AREA_KEYWORDS.items()
            if area != C.RARE_DISEASES and any(k in haystack for k in kws)]

    if len(hits) == 1 and llm is None:
        return AreaResolution(areas=_with_rare(hits[0], population),
                              rationales={hits[0]: "Deterministic keyword match on the indication."},
                              method="keyword")

    if len(hits) == 1:
        # One confident deterministic answer: do not spend an LLM call.
        return AreaResolution(areas=_with_rare(hits[0], population),
                              rationales={hits[0]: "Deterministic keyword match on the indication."},
                              method="keyword")

    # 0 hits, or genuinely more than one area: this is a judgment call.
    if llm is not None:
        payload = (
            f"INDICATION: {population.value('indication_disease')}\n"
            f"DISEASE SUBTYPE / HISTOLOGY: {population.value('disease_subtype_histology') or '(not provided)'}\n"
            f"ICD CODE: {population.value('icd_disease_code') or '(not provided)'}\n"
            f"ORPHAN / RARE DISEASE: {population.value('orphan_rare_disease') or '(not provided)'}\n"
            f"DETERMINISTIC KEYWORD CANDIDATES: {hits or '(none)'}\n")
        parsed = llm.call_json("a05.area_adjudication", payload,
                               max_tokens=C.LLM.area_adjudication_max_tokens,
                               default={}) or {}
        areas, rationales = [], {}
        for item in parsed.get("areas", []) or []:
            a = C.canonicalize_area(item.get("area", ""))
            if a and a not in areas:
                areas.append(a)
                rationales[a] = item.get("rationale", "")
        if areas:
            final = areas[:]
            if _is_orphan(population) and C.RARE_DISEASES not in final:
                final.append(C.RARE_DISEASES)
                rationales[C.RARE_DISEASES] = "Orphan/rare-disease flag is set."
            return AreaResolution(areas=final, rationales=rationales, method="llm",
                                  needed_adjudication=True)

    # Fail-safe. Never leave the area unresolved — the SME is explicit that
    # there is no "nothing fits" — but say honestly that this is a fallback.
    fallback = hits or [C.MULTI_DISCIPLINARY]
    return AreaResolution(
        areas=_with_rare_list(fallback, population),
        rationales={a: "Fallback: deterministic match only, no adjudication available."
                    for a in fallback},
        method="keyword", needed_adjudication=True)


def _is_orphan(population: Population) -> bool:
    v = (population.value("orphan_rare_disease") or "").strip().lower()
    return v.startswith(("yes", "true", "y"))


def _with_rare(area: str, population: Population) -> List[str]:
    return _with_rare_list([area], population)


def _with_rare_list(areas: List[str], population: Population) -> List[str]:
    out = list(dict.fromkeys(areas))
    if _is_orphan(population) and C.RARE_DISEASES not in out:
        out.append(C.RARE_DISEASES)   # SME Agent 5: ALWAYS added alongside
    return out
