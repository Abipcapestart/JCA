"""
Excel export for a single full-workflow run record (see run_store.py for the
record shape). Two halves:

  1. The FINAL output's nested detail that's already in Phase1Output.to_dict()
     but wasn't being surfaced before: each comparator's/outcome's sources,
     evidence, and (for comparators) scope-adjudication reasoning -- exactly
     the "why is this included" + "sources" detail the production mock UI
     shows per item.
  2. The full per-stage debug trail (run["debug_capture"], populated by
     jca_phase1.orchestrator.run_phase1's debug_capture hook): every record
     extracted at A8, its grounding state at A9, its validation verdict at
     A10, and every candidate comparator's scope adjudication at A12 --
     including the ones that did NOT make the final comparator list. This is
     the "where did it get lost" trail an SME needs when a real comparator or
     outcome is missing from the final output.
"""

from __future__ import annotations

import io
import json
from typing import Any, Dict, List

from openpyxl import Workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from jca_phase1 import config as C

HEADER_FILL = PatternFill("solid", fgColor="1F2937")
HEADER_FONT = Font(color="FFFFFF", bold=True)
DEBUG_HEADER_FILL = PatternFill("solid", fgColor="7C2D12")

# Every step's classification, verified against the actual code (not the
# module docstrings, which in a couple of places disagree with what the
# orchestrator actually calls -- see the "note" column below):
#   (LLM)  = every call in this step is a model call, no deterministic branch
#   (Det)  = no `llm` parameter reaches this step at all
#   (Mix)  = the step combines a deterministic part with an LLM call
_STEPS_LEGEND = [
    ("A1", "Input (P&I) validation", "Mix", "A1 Input Validation (Mix)",
     "Deterministic required-field checks + a01.pi_validation LLM plausibility check."),
    ("A2", "Input structuring", "Mix", "A2 Population/Intervention Fields (Mix)",
     "Deterministic provenance tagging + a02.input_structuring LLM call."),
    ("A3", "Scope boundary", "Mix", "A3 Scope Boundary (Mix)",
     "a03.scope_facet_normalise LLM call + deterministic rejection rules "
     "(e.g. prior-therapy-only)."),
    ("A4", "Licensed indication lock", "Mix", "A4 Licensed Indication (Mix)",
     "Deterministic registry fetch + a04.indication_lock LLM parse of the record."),
    ("A5", "Therapeutic area(s)", "Det", "A5 Therapeutic Areas (Det)",
     "Deterministic indication->area map; a05.area_adjudication LLM call fires "
     "ONLY when that map is ambiguous (see this run's 'method' field)."),
    ("A6", "Query planning", "Mix", "A6 Query Plan (Mix)",
     "a06.query_vocabulary LLM call for terminology; query templates themselves "
     "are pure code, no model involved."),
    ("A7", "Source-routed retrieval", "Det", "A7 Retrieval Attempts/Documents (Det)",
     "Tavily + registry API calls only; no `llm` parameter exists on this step."),
    ("A8", "Evidence extraction", "LLM", "A8 Extraction (LLM)",
     "One a08.extraction LLM call per retrieved document, no deterministic branch."),
    ("A9", "Grounding", "Det", "A9 Grounding (Det)",
     "Deterministic lexical check: does the value appear in the retrieved text?"),
    ("A10", "Claim validation", "Mix", "A10 Claim Validation/Usable Records (Mix)",
     "Deterministic re-fetch of the cited source + a10.claim_validation LLM verdict."),
    ("A11", "Comparator identity", "Mix", "A11 Comparator Identity (Mix)",
     "a11.comparator_identity LLM call, then deterministic application onto records."),
    ("A12", "Scope adjudication", "LLM", "A12 Scope Adjudication (LLM)",
     "a12.scope_adjudication LLM call per candidate comparator, no deterministic branch."),
    ("A13", "Per-state assignment", "Det", "A13 Per-State Verdicts (Det)",
     "assign_member_states() takes no `llm` parameter at all."),
    ("A14", "Comparator grouping", "Det", "A14 Comparator Grouping (Det)",
     "Pure dict grouping by identity_key; no model call. NOTE: the module's own "
     "docstring labels the OUTCOME-harmonization LLM call 'A14' -- that call "
     "actually fires during this workbook's A15 step, not here. Verified against "
     "orchestrator.py, not the docstring."),
    ("A15", "Outcome harmonization + catalog coverage", "Mix",
     "A15 Outcomes/Outcome Groups/Catalog Coverage (Mix)",
     "Deterministic catalog match first; a14.outcome_harmonization LLM call only "
     "for residue outcomes the catalog can't place; a16.outcome_rationale LLM "
     "call per outcome; catalog coverage bookkeeping itself is deterministic."),
    ("A16", "Comparator consolidation", "Mix", "A16 Comparators/Sources/Evidence (Mix)",
     "Deterministic field assembly (incl. embedded A13) + conditional "
     "a16.indication_synthesis LLM call + a16.comparator_rationale LLM call."),
    ("A17", "Completeness audit", "Det", "A17 Completeness (Det)",
     "audit_completeness() is called with no `llm` argument at all."),
    ("A18", "User additions", "Det", "A16 Comparators/A15 Outcomes (origin=user_added)",
     "Taken exactly as entered by the SME; runs outside run_phase1() entirely, as "
     "a separate API action (add_comparator/add_outcome) after the run finishes. "
     "Filter the 'origin' column in the A16/A15 sheets to see these rows."),
]


def _write_legend(wb: Workbook) -> None:
    ws = wb.create_sheet("Pipeline Steps Legend")
    _write_table(ws, ["step", "what it does", "classification", "sheet(s)", "note"],
                [[s, what, cls, sheet, note] for s, what, cls, sheet, note in _STEPS_LEGEND])
    ws.column_dimensions["E"].width = 100


def _sanitize(value: Any) -> Any:
    """Strip characters illegal in XLSX XML (control chars other than tab/CR/LF).

    Scraped source text (e.g. a mis-decoded HTA/legal PDF) can carry raw control
    bytes that openpyxl's Cell.value setter rejects outright, which otherwise
    crashes the whole export over a single bad cell deep in a debug trail."""
    if isinstance(value, str):
        return ILLEGAL_CHARACTERS_RE.sub("", value)
    return value


def _write_table(ws, headers, rows, header_fill=HEADER_FILL):
    ws.append(headers)
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = HEADER_FONT
        cell.alignment = Alignment(vertical="center")
    for row in rows:
        ws.append([_sanitize(v) for v in row])
    ws.freeze_panes = "A2"
    for i, h in enumerate(headers, start=1):
        col = get_column_letter(i)
        ws.column_dimensions[col].width = max(12, min(60, len(str(h)) + 4))


def _record_row(r: Dict[str, Any]) -> list:
    comp = r.get("comparator") or {}
    outc = r.get("outcome") or {}
    v = r.get("validation") or {}
    pc = r.get("population_context") or {}
    is_outcome = r.get("finding_type") != "comparator"
    # Same summary a10.claim_validation's own payload already shows the LLM
    # (see agents/a09_a13_validation.py's _validate_batch) -- the source's
    # OWN stated population for this specific finding, which is exactly what
    # A12 scope adjudication reasons over per-record. Previously invisible
    # at the record level; only inferable after the fact from the final verdict.
    pc_summary = ", ".join(dict.fromkeys(
        val for val in [pc.get("disease"), pc.get("subtype_histology"), pc.get("stage"),
                       pc.get("biomarker"), pc.get("line_of_therapy"), pc.get("prior_therapy"),
                       pc.get("treatment_setting_intent")] if val))
    return [
        r.get("finding_id"), r.get("finding_type"), r.get("subject_drug"),
        r.get("member_state"), r.get("source_class"), r.get("tier"),
        r.get("source_url"), r.get("organization"),
        comp.get("as_stated") or outc.get("measure"),
        comp.get("role") if r.get("finding_type") == "comparator" else outc.get("result"),
        (r.get("evidence_quote") or "")[:300],
        r.get("grounded"), r.get("grounding_note"),
        v.get("verdict"), v.get("reason"), v.get("refetched"),
        r.get("evidence_status"), r.get("data_maturity"),
        r.get("trial_status"), r.get("peer_review_status"),
        comp.get("thin_mention") if r.get("finding_type") == "comparator" else None,
        pc_summary, r.get("recommendation_strength"), r.get("retrieval_method"),
        r.get("document_title"), r.get("document_date"),
        outc.get("unit") if is_outcome else None,
        outc.get("instrument") if is_outcome else None,
        outc.get("is_requirement") if is_outcome else None,
    ]


_RECORD_HEADERS = ["finding_id", "finding_type", "subject_drug", "member_state",
                   "source_class", "tier", "source_url", "organization",
                   "comparator_or_outcome", "role_or_result", "evidence_quote",
                   "grounded", "grounding_note", "validation_verdict",
                   "validation_reason", "refetched",
                   "evidence_status", "data_maturity", "trial_status",
                   "peer_review_status", "thin_mention",
                   "population_context", "recommendation_strength", "retrieval_method",
                   "document_title", "document_date",
                   "outcome_unit", "outcome_instrument", "outcome_is_requirement"]


def build_excel(run: Dict[str, Any]) -> bytes:
    wb = Workbook()
    out = run.get("output") or {}
    debug = run.get("debug_capture") or {}

    # ---- Run Summary ----------------------------------------------------
    ws = wb.active
    ws.title = "Run Summary"
    v = out.get("validation", {})
    # member_state_summary is the literal "sums to 27" completeness invariant
    # MemberStateSummary.__post_init__ enforces on every run -- previously
    # never surfaced anywhere in the export.
    mss = out.get("member_state_summary") or {}
    _write_table(ws, ["field", "value"], [
        ["run_id", run.get("run_id")], ["status", run.get("status")],
        ["total_latency_s", run.get("total_latency_s")],
        ["total_cost_usd", run.get("total_cost_usd")],
        ["bedrock_cost_usd", run.get("bedrock_summary", {}).get("total_cost_usd")],
        ["tavily_cost_usd", run.get("tavily_summary", {}).get("total_cost_usd")],
        ["num_comparators", len(out.get("comparators", []))],
        ["records_extracted", v.get("records_extracted")],
        ["grounded", v.get("grounded")], ["grounding_failed", v.get("grounding_failed")],
        ["claim_validated", v.get("claim_validated")],
        ["claim_rejected", v.get("claim_rejected")],
        ["subject_drug_rejections", v.get("subject_drug_rejections")],
        ["role_rejections", v.get("role_rejections")],
        ["scope_in", v.get("scope_in")], ["scope_out", v.get("scope_out")],
        ["scope_uncertain", v.get("scope_uncertain")],
        ["member_states_identified", mss.get("identified_count")],
        ["member_states_not_identified", mss.get("not_identified_count")],
        ["member_states_total", mss.get("total")],
    ])

    ws = wb.create_sheet("Prompt Versions Used")
    _write_table(ws, ["prompt_id", "version"],
                [[k, v_] for k, v_ in run.get("prompt_versions_used", {}).items()])

    # Operational warnings (e.g. "claim validation skipped this run", a
    # blocked-JCA-report count) -- previously invisible; a reviewer looking
    # at empty validation columns had no way to learn WHY.
    ws = wb.create_sheet("Run Notes")
    _write_table(ws, ["note"], [[n] for n in out.get("notes", [])])

    _write_legend(wb)

    # ---- A1 input validation ----------------------------------------------
    ws = wb.create_sheet("A1 Input Validation (Mix)")
    iv = out.get("input_validation", {}) or run.get("input_validation", {}) or {}
    rows = [["FIX", "; ".join(i.get("fields", [])), i.get("value_flagged", ""), i.get("explanation", "")]
           for i in iv.get("fix_items", [])]
    rows += [["CHECK", "; ".join(i.get("fields", [])), "", i.get("explanation", "")]
            for i in iv.get("check_items", [])]
    _write_table(ws, ["severity", "fields", "value_flagged", "explanation"], rows)

    # ---- A2 structured population / intervention ---------------------------
    ws = wb.create_sheet("A2 Population Fields (Mix)")
    rows = []
    for p in out.get("populations", []):
        for fname, f in p.get("fields", {}).items():
            rows.append([p.get("population_id"), fname, f.get("value"), f.get("provenance"),
                        f.get("inference_basis")])
    _write_table(ws, ["population_id", "field", "value", "provenance", "inference_basis"], rows)

    ws = wb.create_sheet("A2 Intervention Fields (Mix)")
    interv = out.get("intervention", {}) or {}
    rows = [[fname, f.get("value"), f.get("provenance"), f.get("inference_basis")]
           for fname, f in interv.get("fields", {}).items()]
    # The resolved substance identity of the drug being ASSESSED -- core
    # metadata, previously absent (only the raw stated fields were shown).
    rows += [["inn_resolved", interv.get("inn_resolved"), "", ""],
            ["atc_code", interv.get("atc_code"), "", ""],
            ["user_confirmed", interv.get("user_confirmed"), "", ""]]
    _write_table(ws, ["field", "value", "provenance", "inference_basis"], rows)

    # ---- A3 scope boundary --------------------------------------------------
    ws = wb.create_sheet("A3 Scope Boundary (Mix)")
    rows = []
    for b in out.get("scope_boundaries", []):
        unbounded = set(b.get("unbounded_facets", []))
        for fname, f in b.get("facets", {}).items():
            is_bound = bool(f.get("value")) and f.get("provenance") in ("confirmed", "derived_from_label")
            rows.append([b.get("population_id"), fname, f.get("value"), f.get("provenance"),
                        f.get("must_match"), is_bound, fname in unbounded])
    _write_table(ws, ["population_id", "facet", "value", "provenance", "must_match",
                     "is_bound_filters_evidence", "explicitly_unbounded"], rows)

    # ---- A4 licensed indication ---------------------------------------------
    ws = wb.create_sheet("A4 Licensed Indication (Mix)")
    lic = out.get("licensed_indication_record", {}) or {}
    _write_table(ws, ["field", "value"], [
        ["source", lic.get("source")], ["indication_text", lic.get("indication_text")],
        ["source_url", lic.get("source_url")], ["retrieved_at", lic.get("retrieved_at")],
        ["pivotal_trials", "; ".join(lic.get("pivotal_trials", []))],
        ["atc_code", lic.get("atc_code")], ["inn", lic.get("inn")], ["notes", lic.get("notes")],
    ])

    # ---- A5 therapeutic area(s) ---------------------------------------------
    ws = wb.create_sheet("A5 Therapeutic Areas (Det)")
    areas = out.get("therapeutic_areas", {}) or {}
    _write_table(ws, ["area", "rationale"],
                [[a, areas.get("rationales", {}).get(a, "")] for a in areas.get("areas", [])])
    ws.append([])
    ws.append(["method", _sanitize(areas.get("method"))])
    ws.append(["needed_adjudication", _sanitize(areas.get("needed_adjudication"))])

    # ---- Run manifest --------------------------------------------------------
    ws = wb.create_sheet("Run Manifest")
    m = out.get("run_manifest", {}) or {}
    workbook_info = m.get("source_workbook", {}) or {}
    _write_table(ws, ["field", "value"], [
        ["model", m.get("model")], ["code_version", m.get("code_version")],
        ["outcome_catalog_version", m.get("outcome_catalog_version")],
        ["therapeutic_areas_applied", "; ".join(m.get("therapeutic_areas_applied", []))],
        ["tier3_enabled", m.get("tier3_enabled")],
        ["source_workbook_path", workbook_info.get("path")],
        ["source_workbook_entries", workbook_info.get("entries")],
        ["source_workbook_unique_domains", workbook_info.get("unique_domains")],
        ["source_workbook_problems", json.dumps(workbook_info.get("problems", []))],
        ["leakage_guard", json.dumps(m.get("leakage_guard", {}))],
        ["providers", json.dumps(m.get("providers", {}))],
        ["started_at", m.get("started_at")], ["finished_at", m.get("finished_at")],
        ["duration_s", m.get("duration_s")],
        ["llm_usage", json.dumps(m.get("llm_usage", {}))],
        ["search_usage", json.dumps(m.get("search_usage", {}))],
    ])

    # ---- Validation: every excluded item, any stage --------------------------
    ws = wb.create_sheet("Validation Excluded Items")
    rows = [[e.get("value"), e.get("stage"), e.get("reason"), e.get("decisive_facet", ""),
            e.get("source_url", "")] for e in out.get("validation", {}).get("excluded", [])]
    _write_table(ws, ["value", "stage", "reason", "decisive_facet", "source_url"], rows)

    # ---- Final comparators + their full sources/evidence/adjudication ----
    # NOTE (A18): user-added comparators land in this same sheet with
    # origin="user_added" -- filter that column to isolate them; see the
    # Pipeline Steps Legend sheet.
    ws = wb.create_sheet("A16 Comparators (Mix)")
    rows = []
    for c in out.get("comparators", []):
        rows.append([
            c.get("generic_name"), c.get("comparator_id"), "; ".join(c.get("brand_names", [])),
            c.get("inn"), c.get("atc_code"), c.get("class_or_mechanism"), c.get("class_source"),
            c.get("is_combination"), "; ".join(c.get("components", [])), c.get("role"),
            c.get("indication"), c.get("indication_scope_note"), c.get("line_of_therapy"),
            c.get("comparator_scenario"), c.get("retain_all_status"), c.get("recommendation_strength"),
            c.get("member_state_count"), "; ".join(c.get("member_states", [])),
            "; ".join(str(t) for t in c.get("tiers", [])), c.get("cross_tier_confirmed"),
            c.get("or_alternative"), c.get("general_evidence_flag"),
            "; ".join(c.get("population_ids", [])),
            c.get("rationale"), c.get("origin"), len(c.get("sources", [])), len(c.get("evidence", [])),
            "; ".join(c.get("aliases_merged", [])),
        ])
    _write_table(ws, ["generic_name", "comparator_id", "brand_names", "inn", "atc_code",
                     "class_or_mechanism", "class_source", "is_combination", "components", "role",
                     "indication", "indication_scope_note", "line_of_therapy", "comparator_scenario",
                     "retain_all_status", "recommendation_strength", "member_state_count",
                     "member_states", "tiers", "cross_tier_confirmed", "or_alternative",
                     "general_evidence_flag", "population_ids", "rationale", "origin",
                     "num_sources", "num_evidence", "aliases_merged"], rows)

    ws = wb.create_sheet("A16 Comparator Sources (Mix)")
    rows = []
    for c in out.get("comparators", []):
        for s in c.get("sources", []):
            rows.append([c.get("generic_name"), s.get("source_id"), s.get("tier"),
                        s.get("source_class"), s.get("organization"), s.get("display_name"),
                        s.get("url"), s.get("document_title"), s.get("document_date")])
    _write_table(ws, ["generic_name", "source_id", "tier", "source_class", "organization",
                     "display_name", "url", "document_title", "document_date"], rows)

    ws = wb.create_sheet("A16 Comparator Evidence (Mix)")
    rows = []
    for c in out.get("comparators", []):
        for e in c.get("evidence", []):
            rows.append([c.get("generic_name"), e.get("source_id"), (e.get("quote") or "")[:400],
                        e.get("locator"), e.get("member_state"), e.get("grounded"),
                        e.get("validation_verdict"), e.get("refetched"),
                        e.get("evidence_status"), e.get("data_maturity"),
                        e.get("trial_status"), e.get("peer_review_status")])
    _write_table(ws, ["generic_name", "source_id", "quote", "locator", "member_state",
                     "grounded", "validation_verdict", "refetched",
                     "evidence_status", "data_maturity", "trial_status",
                     "peer_review_status"], rows)

    ws = wb.create_sheet("A12 Final Adjudication (LLM)")
    rows = []
    for c in out.get("comparators", []):
        adj = c.get("scope_adjudication") or {}
        rows.append([
            c.get("generic_name"), adj.get("verdict"), adj.get("decisive_facet"),
            adj.get("reason"), adj.get("adjudicator_version"),
            len(adj.get("evidence_for", [])), len(adj.get("evidence_against", [])),
        ])
    _write_table(ws, ["generic_name", "verdict", "decisive_facet", "reason",
                     "adjudicator_version", "num_evidence_for", "num_evidence_against"], rows)

    # A comparator merged from several wording variants can get genuinely
    # different verdicts per population boundary (e.g. in_scope for
    # "licensed" but uncertain for "intended_to_treat", for a different
    # reason) -- only the single "best" verdict above reaches the main
    # sheet. This sheet is where the discarded verdict(s) become visible,
    # rather than only living in debug_capture.
    ws = wb.create_sheet("A12 All Population Verdicts (LLM)")
    rows = []
    for c in out.get("comparators", []):
        by_pop = c.get("adjudications_by_population") or {}
        for pop_id, adj in by_pop.items():
            rows.append([
                c.get("generic_name"), pop_id, adj.get("verdict"),
                adj.get("decisive_facet"), adj.get("reason"),
            ])
    _write_table(ws, ["generic_name", "population_id", "verdict", "decisive_facet", "reason"],
                rows)

    ws = wb.create_sheet("A13 Per-State Verdicts (Det)")
    rows = []
    for c in out.get("comparators", []):
        for pv in c.get("per_member_state", []):
            rows.append([c.get("generic_name"), pv.get("member_state"), pv.get("verdict"),
                        pv.get("tier"), pv.get("source_url", ""),
                        (pv.get("evidence_quote") or "")[:300], pv.get("reason", "")])
    _write_table(ws, ["generic_name", "member_state", "verdict", "tier", "source_url",
                     "evidence_quote", "reason"], rows)

    # ---- Final outcomes + their full sources/evidence ---------------------
    outcomes = out.get("outcomes", {}) or {}
    # NOTE (A18): user-added outcomes land in this same sheet with
    # origin="user_added" -- filter that column to isolate them.
    ws = wb.create_sheet("A15 Outcomes (Mix)")
    rows = []
    for cat in ("clinical_effectiveness", "safety", "quality_of_life", "clinician_patient_reported"):
        for o in outcomes.get(cat, []):
            rows.append([cat, o.get("concept"), o.get("outcome_id"), o.get("category"),
                        o.get("catalog_id"), o.get("listed"), o.get("instrument"),
                        o.get("unit_of_measurement"), o.get("unit_disagreement_note"),
                        o.get("requirement_type"), o.get("coverage_status"),
                        "; ".join(str(t) for t in o.get("tiers", [])), o.get("cross_tier_confirmed"),
                        o.get("rationale"), o.get("origin"), "; ".join(o.get("aliases_merged", [])),
                        len(o.get("sources", [])), len(o.get("evidence", []))])
    _write_table(ws, ["bucket", "concept", "outcome_id", "category", "catalog_id", "listed",
                     "instrument", "unit_of_measurement", "unit_disagreement_note",
                     "requirement_type", "coverage_status", "tiers", "cross_tier_confirmed",
                     "rationale", "origin", "aliases_merged", "num_sources", "num_evidence"], rows)

    ws = wb.create_sheet("A15 Outcome Sources (Mix)")
    rows = []
    for cat in ("clinical_effectiveness", "safety", "quality_of_life", "clinician_patient_reported"):
        for o in outcomes.get(cat, []):
            for s in o.get("sources", []):
                rows.append([o.get("concept"), s.get("source_id"), s.get("tier"),
                            s.get("source_class"), s.get("organization"), s.get("display_name"),
                            s.get("url")])
    _write_table(ws, ["concept", "source_id", "tier", "source_class", "organization",
                     "display_name", "url"], rows)

    ws = wb.create_sheet("A15 Outcome Evidence (Mix)")
    rows = []
    for cat in ("clinical_effectiveness", "safety", "quality_of_life", "clinician_patient_reported"):
        for o in outcomes.get(cat, []):
            for e in o.get("evidence", []):
                rows.append([o.get("concept"), e.get("source_id"), (e.get("quote") or "")[:400],
                            e.get("member_state"), e.get("grounded"), e.get("validation_verdict"),
                            e.get("evidence_status"), e.get("data_maturity"),
                            e.get("trial_status"), e.get("peer_review_status")])
    _write_table(ws, ["concept", "source_id", "quote", "member_state", "grounded",
                     "validation_verdict", "evidence_status", "data_maturity",
                     "trial_status", "peer_review_status"], rows)

    ws = wb.create_sheet("A15 Catalog Coverage (Det)")
    _write_table(ws, ["catalog_id", "display_name", "category", "status", "matched_outcome_id"],
                [[cc.get("catalog_id"), cc.get("display_name"), cc.get("category"),
                  cc.get("status"), cc.get("matched_outcome_id", "")]
                 for cc in outcomes.get("catalog_coverage", [])])

    ws = wb.create_sheet("By Member State (Det)")
    _write_table(ws, ["member_state", "status", "comparators", "state_search_status"],
                [[e.get("member_state"), e.get("status"), "; ".join(e.get("comparators", [])),
                  e.get("state_search_status")] for e in out.get("by_member_state", [])])

    # ---- Cost / latency telemetry -----------------------------------------
    ws = wb.create_sheet("Bedrock Calls")
    _write_table(ws, ["stage", "prompt_id", "prompt_version", "input_tokens", "output_tokens",
                     "cache_creation_input_tokens", "cache_read_input_tokens",
                     "cost_usd", "latency_s", "attempts", "error"],
                [[c.get("stage"), c["prompt_id"], c["prompt_version"], c["input_tokens"],
                  c["output_tokens"], c.get("cache_creation_input_tokens", 0),
                  c.get("cache_read_input_tokens", 0), c["cost_usd"], c["latency_s"],
                  c["attempts"], c["error"]]
                 for c in run.get("bedrock_calls", [])])

    ws = wb.create_sheet("Tavily Calls")
    _write_table(ws, ["op", "detail", "credits", "cost_usd", "latency_s", "attempts", "results", "error"],
                [[c["op"], c["detail"], c["credits"], c["cost_usd"], c["latency_s"],
                  c["attempts"], c["results"], c["error"]] for c in run.get("tavily_calls", [])])

    ws = wb.create_sheet("Stage Latency")
    _write_table(ws, ["stage", "latency_s"],
                [[stage, secs] for stage, secs in run.get("stage_latency_s", {}).items()])

    ws = wb.create_sheet("Errors and Retries")
    _write_table(ws, ["t_s", "level", "message"],
                [[e["t"], e["level"], e["message"]] for e in run.get("error_log", [])])

    # =========================================================================
    # DEBUG TRAIL -- every stage's raw intermediate state, including
    # candidates that did NOT survive to the final output. Only present if
    # this run was executed after the debug_capture hook was added; older
    # cached runs simply produce empty sheets here.
    # =========================================================================

    ws = wb.create_sheet("A6 Query Plan (Mix)")
    plan = (debug.get("A6_query_plan") or {}).get("plan", [])
    _write_table(ws, ["query", "source_class", "tier", "member_state", "pass_type", "domains",
                     "max_urls", "language", "note"],
                [[p.get("query"), p.get("source_class"),
                  C.TIER_OF_SOURCE_CLASS.get(p.get("source_class"), 3), p.get("member_state"),
                  p.get("pass_type"), "; ".join(p.get("domains", [])), p.get("max_urls"),
                  p.get("language"), p.get("note")] for p in plan], DEBUG_HEADER_FILL)

    ws = wb.create_sheet("A7 Retrieval Attempts (Det)")
    attempts = (debug.get("A7_retrieval") or {}).get("attempts", [])
    _write_table(ws, ["member_state", "source_class", "tier", "attempted",
                     "documents_retrieved", "status", "query_sent"],
                [[a.get("member_state"), a.get("source_class"),
                  C.TIER_OF_SOURCE_CLASS.get(a.get("source_class"), 3), a.get("attempted"),
                  a.get("documents_retrieved"), a.get("status"), a.get("detail")]
                 for a in attempts], DEBUG_HEADER_FILL)

    ws = wb.create_sheet("A7 Retrieved Documents (Det)")
    docs = (debug.get("A7_retrieval") or {}).get("documents", [])
    _write_table(ws, ["url", "member_state", "source_class", "organization", "ok", "status",
                     "title", "text_preview"],
                [[d.get("url"), d.get("member_state"), d.get("source_class"),
                  d.get("organization"), d.get("ok"), d.get("status"), d.get("title"),
                  (d.get("text") or "")[:300]] for d in docs], DEBUG_HEADER_FILL)

    ws = wb.create_sheet("A7b Comparator Follow-up (Det)")
    followup_attempts = (debug.get("A7b_comparator_followup") or {}).get("attempts", [])
    _write_table(ws, ["candidate_name", "tier", "query_sent", "documents_retrieved", "status"],
                [[a.get("candidate_name"), a.get("tier"), a.get("query_sent"),
                  a.get("documents_retrieved"), a.get("status")] for a in followup_attempts],
                DEBUG_HEADER_FILL)

    for key, title in [("A8_extraction", "A8 Extraction (LLM)"),
                       ("A9_grounding", "A9 Grounding (Det)"),
                       ("A10_claim_validation", "A10 Claim Validation (Mix)"),
                       ("usable_after_A10", "A10 Usable Records (Mix)")]:
        ws = wb.create_sheet(title)
        recs = debug.get(key) or []
        _write_table(ws, _RECORD_HEADERS, [_record_row(r) for r in recs], DEBUG_HEADER_FILL)

    ws = wb.create_sheet("A11 Comparator Identity (Mix)")
    identities = debug.get("A11_identity") or {}
    # A component can be a plain string or (older/pre-fix runs, and any model
    # response the ambiguous a11 schema still nudges this way) a rich object
    # -- render either as its display name rather than crashing str.join().
    _component_display = lambda c: c if isinstance(c, str) else str(
        (c or {}).get("display_name") or (c or {}).get("inn") or c)
    _write_table(ws, ["as_stated_key", "inn", "atc_code", "class_mechanism", "class_source",
                     "is_combination", "components", "audit_merged_into"],
                [[k, v_.get("inn"), v_.get("atc_code"), v_.get("class_mechanism"),
                  v_.get("class_source"), v_.get("is_combination"),
                  "; ".join(_component_display(c) for c in (v_.get("components") or [])),
                  v_.get("audit_merged_into")]
                 for k, v_ in identities.items()],
                DEBUG_HEADER_FILL)

    ws = wb.create_sheet("A11 Identity Audit (LLM)")
    audit_merges = (debug.get("A11b_identity_audit") or {}).get("merges") or []
    _write_table(ws, ["canonical_display_name", "duplicate_display_name", "reason"],
                [[m.get("canonical_display_name"), m.get("loser_display_name"), m.get("reason")]
                 for m in audit_merges],
                DEBUG_HEADER_FILL)

    ws = wb.create_sheet("A12 All Candidates (LLM)")
    all_adj = debug.get("A12_scope_adjudication") or {}
    rows = []
    for key, d in all_adj.items():
        comp = d.get("comparator", {})
        adj = d.get("adjudication", {})
        rows.append([
            key, comp.get("as_stated"), comp.get("role"), d.get("num_records"),
            adj.get("verdict"), adj.get("decisive_facet"), adj.get("reason"),
            len(adj.get("evidence_for", [])), len(adj.get("evidence_against", [])),
        ])
    _write_table(ws, ["candidate_key", "as_stated", "role", "num_records", "verdict",
                     "decisive_facet", "reason", "num_evidence_for", "num_evidence_against"],
                rows, DEBUG_HEADER_FILL)
    ws2 = wb.create_sheet("A12 Evidence For-Against (LLM)")
    rows2 = []
    for key, d in all_adj.items():
        comp = d.get("comparator", {})
        adj = d.get("adjudication", {})
        for side, items in (("for", adj.get("evidence_for", [])), ("against", adj.get("evidence_against", []))):
            for ec in items:
                rows2.append([key, comp.get("as_stated"), side, ec.get("source_id"),
                            ec.get("member_state"), ec.get("tier"), (ec.get("quote") or "")[:300],
                            "; ".join(ec.get("facets_matched", [])), ec.get("facet_conflict"),
                            ec.get("detail")])
    _write_table(ws2, ["candidate_key", "as_stated", "side", "source_id", "member_state", "tier",
                      "quote", "facets_matched", "facet_conflict", "detail"], rows2, DEBUG_HEADER_FILL)

    # A14_grouping was already captured into debug_capture by the orchestrator
    # but had no sheet of its own until now -- these are the identity-key
    # groups BEFORE scope adjudication (A12) decides which ones survive.
    ws = wb.create_sheet("A14 Comparator Grouping (Det)")
    rows = []
    for key, recs in (debug.get("A14_grouping") or {}).items():
        for r in recs:
            rows.append([key] + _record_row(r))
    _write_table(ws, ["group_key"] + _RECORD_HEADERS, rows, DEBUG_HEADER_FILL)

    ws = wb.create_sheet("A16 Dropped Comparators (Mix)")
    _write_table(ws, ["value", "reason"],
                [[d.get("value"), d.get("reason")] for d in (debug.get("A16_dropped_at_consolidation") or [])],
                DEBUG_HEADER_FILL)

    ws = wb.create_sheet("A15 Outcome Groups (Mix)")
    rows = []
    for g in (debug.get("A15_outcome_groups") or []):
        for r in g.get("records", []):
            rows.append([g.get("concept"), g.get("catalog_id")] + _record_row(r))
    _write_table(ws, ["group_concept", "group_catalog_id"] + _RECORD_HEADERS, rows, DEBUG_HEADER_FILL)

    # ---- A17 completeness audit --------------------------------------------
    ws = wb.create_sheet("A17 Completeness (Det)")
    comp_report = out.get("completeness", {}) or {}
    rows = [["assertion", a.get("name"), a.get("passed"), a.get("detail")]
           for a in comp_report.get("assertions", [])]
    rows += [["member_state_count", ms, n, ""]
            for ms, n in (comp_report.get("member_states") or {}).items()]
    rows += [["outcome_category", oc.get("category"), oc.get("count", oc.get("status", "")),
              json.dumps(oc)] for oc in comp_report.get("outcome_categories", [])]
    rows += [["population_processed", p, "", ""]
            for p in comp_report.get("populations_processed", [])]
    _write_table(ws, ["kind", "name", "value", "detail"], rows)

    # The AUDITED final coverage matrix -- distinct from "A7 Retrieval
    # Attempts (Det)" above, which is the pre-audit debug snapshot. This is
    # the actual CompletenessReport.source_class_matrix field on the final
    # Phase1Output, previously never read by this export at all.
    ws = wb.create_sheet("A17 Source Class Matrix (Det)")
    matrix = comp_report.get("source_class_matrix") or []
    _write_table(ws, ["member_state", "source_class", "attempted", "documents_retrieved",
                     "status", "detail"],
                [[m.get("member_state"), m.get("source_class"), m.get("attempted"),
                  m.get("documents_retrieved"), m.get("status"), m.get("detail")]
                 for m in matrix])

    _write_ui_sheets(wb, out)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# =============================================================================
# UI OUTPUT -- remaps the internal field names above into the field set the
# production PICO Scope screen actually asks for/displays (verified against
# the demo's HTML, not guessed): Comparator / EU Member States / Class /
# mechanism / Indication / Source tier / Standard of care / Outcome / Unit of
# measurement / Rationale. This is a presentation transformation only -- the
# underlying granular data (with every field the UI doesn't show) stays in
# the A16/A15/A13 sheets above for traceability.
# =============================================================================

def _tier_label(tiers: List[int]) -> str:
    if not tiers:
        return ""
    if len(tiers) == 1:
        return f"Tier {tiers[0]}"
    return " + ".join(f"Tier {t}" for t in sorted(tiers))


def _write_ui_sheets(wb: Workbook, out: Dict[str, Any]) -> None:
    ws = wb.create_sheet("UI - Comparators")
    rows = []
    for c in out.get("comparators", []):
        rows.append([
            c.get("generic_name"), "; ".join(c.get("member_states", [])),
            c.get("class_or_mechanism"), c.get("indication"),
            _tier_label(c.get("tiers", [])), c.get("line_of_therapy"),
            c.get("rationale"),
            "; ".join(s.get("display_name", "") for s in c.get("sources", [])),
        ])
    _write_table(ws, ["Comparator", "EU Member States", "Class / mechanism", "Indication",
                     "Source tier", "Line of therapy", "Rationale", "Sources"], rows)

    ws = wb.create_sheet("UI - By EU Member State")
    rows = []
    for c in out.get("comparators", []):
        for pv in c.get("per_member_state", []):
            rows.append([
                pv.get("member_state"), c.get("generic_name"),
                "Yes" if pv.get("verdict") == "standard_of_care" else "Not established",
                pv.get("reason", ""),
            ])
    _write_table(ws, ["EU Member State", "Comparator", "Standard of care", "reason"], rows)

    ws = wb.create_sheet("UI - Outcomes")
    rows = []
    outcomes = out.get("outcomes", {}) or {}
    for cat in ("clinical_effectiveness", "safety", "quality_of_life", "clinician_patient_reported"):
        for o in outcomes.get(cat, []):
            rows.append([
                o.get("concept"), cat, o.get("unit_of_measurement"),
                _tier_label(o.get("tiers", [])), o.get("rationale"),
                "; ".join(s.get("display_name", "") for s in o.get("sources", [])),
            ])
    _write_table(ws, ["Outcome", "Category", "Unit of measurement", "Source tier",
                     "Rationale", "Sources"], rows)
