"""
JCA Phase 1 -- SME Prompt Tuning UI.

Lets an SME edit any of the 14 Phase 1 prompts (jca_phase1/prompts/registry.py),
run the REAL Phase 1 workflow (jca_phase1.orchestrator.run_phase1) with those
edits actually in effect, inspect/download the result, and iterate -- with
every run's prompt versions, input, output and cost/latency telemetry kept in
run_history/ for traceability.

Run with:  streamlit run streamlit_app/app.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime, timezone

import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(APP_DIR)
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
for p in (REPO_ROOT, APP_DIR, SCRIPTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import prompt_store  # noqa: E402
import run_store  # noqa: E402
import telemetry  # noqa: E402
import excel_export  # noqa: E402
import workflow_runner  # noqa: E402
import gt_workbook  # noqa: E402

from jca_phase1.agents import a01_a05_input_context as inputs  # noqa: E402
from jca_phase1.providers.llm import BedrockLLM, ScriptedLLM  # noqa: E402

st.set_page_config(page_title="JCA Phase 1 -- SME Prompt Tuning", layout="wide")

POP_FIELDS = inputs.TAXONOMY["population"]
INT_FIELDS = inputs.TAXONOMY["intervention"]

# Default inputs, parsed live from the GT reference workbook (fixtures/jca
# gt.xlsx) via the same parser used for the GT evaluation earlier this
# session -- single source of truth, not retyped here. GT_DRUGS (from
# gt_workbook, not re-listed here) is every drug sheet that workbook
# actually has, so a fourth GT scenario added there is picked up
# automatically rather than needing a matching hardcoded list in this file.
GT_WORKBOOK_PATH = os.path.join(REPO_ROOT, "fixtures", "jca gt.xlsx")
DEFAULT_DRUG = "Tovorafenib"


def _load_gt_defaults(drug: str):
    try:
        population, intervention = gt_workbook.parse_gt_sheet(GT_WORKBOOK_PATH, drug)
    except Exception:  # noqa: BLE001 -- workbook missing/moved: fall back to empty fields
        return {}, {}
    if drug == "Tovorafenib":
        # Both fields duplicate content that belongs elsewhere in the taxonomy
        # (other_characteristics restates the Companion diagnostic requirement;
        # treatment_setting_intent restates Stage/severity + Prior therapy/line)
        # and real A1 validation correctly blocks on them -- observed earlier
        # this session, non-deterministically, for THIS drug's sheet. Applied
        # only here, not to every GT sheet, since the other two haven't been
        # checked for the same duplication and dropping a field GT actually
        # states distinct content for would be its own (different) defect.
        population.pop("other_characteristics", None)
        population.pop("treatment_setting_intent", None)
    return population, intervention


DEFAULT_POPULATION, DEFAULT_INTERVENTION = _load_gt_defaults(DEFAULT_DRUG)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("provider_mode", "production")
    ss.setdefault("running", False)
    ss.setdefault("progress_log", [])
    ss.setdefault("error_log", [])
    ss.setdefault("run_result_holder", {})
    ss.setdefault("last_run_id", None)


_init_state()


# ---------------------------------------------------------------------------
# Input collection / validation
# ---------------------------------------------------------------------------

def _collect_inputs():
    population = {f["field"]: st.session_state.get(f"field__population__{f['field']}", "")
                 for f in POP_FIELDS}
    population = {k: v for k, v in population.items() if v}
    intervention = {f["field"]: st.session_state.get(f"field__intervention__{f['field']}", "")
                    for f in INT_FIELDS}
    intervention = {k: v for k, v in intervention.items() if v}
    return population, intervention


def _validate_inputs(population, intervention):
    errors = []
    for f in POP_FIELDS:
        if f.get("mandatory") and not population.get(f["field"]):
            errors.append(f"Population field '{f['label']}' is mandatory but empty.")
    for f in INT_FIELDS:
        if f.get("mandatory") and not intervention.get(f["field"]):
            errors.append(f"Intervention field '{f['label']}' is mandatory but empty.")
    return errors


# ---------------------------------------------------------------------------
# Full-workflow run (background thread)
# ---------------------------------------------------------------------------

def _start_full_run(population_input, intervention_input) -> None:
    provider_mode = st.session_state["provider_mode"]
    allow_jca_reports = not st.session_state.get("benchmarking_mode", False)
    run_id = run_store.new_run_id()
    overrides = prompt_store.active_overrides()
    prompt_versions_used = {pid: prompt_store.get_active_version(pid)
                            for pid in prompt_store.all_prompt_ids()}
    started_at = datetime.now(timezone.utc).isoformat()

    progress_log: list = []
    error_log: list = []
    result_holder: dict = {}
    st.session_state["progress_log"] = progress_log
    st.session_state["error_log"] = error_log
    st.session_state["run_result_holder"] = result_holder
    st.session_state["running"] = True

    thread = threading.Thread(
        target=workflow_runner.run_full_workflow_async,
        args=(run_id, population_input, intervention_input, provider_mode, overrides,
             prompt_versions_used, progress_log, result_holder, started_at, error_log,
             allow_jca_reports),
        daemon=True)
    st.session_state["run_thread"] = thread
    thread.start()


def _render_running_status() -> None:
    st.info("Running Phase 1 with the current prompt set...")
    log = st.session_state.get("progress_log", [])
    if log:
        with st.container(height=220):
            for evt in log[-50:]:
                st.text(f"[{evt['t']:>7.1f}s] {evt['stage']:<10s} {evt['message']}")
    errs = st.session_state.get("error_log", [])
    if errs:
        with st.expander(f"Retries / errors ({len(errs)})", expanded=True):
            for e in errs[-50:]:
                line = f"[{e['t']:>7.1f}s] {e['message']}"
                st.warning(line) if e["level"] == "retry" else st.error(line)
    thread = st.session_state.get("run_thread")
    if thread is not None and not thread.is_alive():
        st.session_state["running"] = False
        record = st.session_state["run_result_holder"].get("record")
        if record:
            run_store.save_run(record)
            st.session_state["last_run_id"] = record["run_id"]
        st.rerun()
    else:
        time.sleep(1.5)
        st.rerun()


# ---------------------------------------------------------------------------
# Prompt-only test mode
# ---------------------------------------------------------------------------

def _render_prompt_only_mode(prompt_id: str, edited_text: str) -> None:
    examples = run_store.find_example_payloads(prompt_id)
    if not examples:
        st.info("No captured example payload yet for this prompt. Run the full workflow once "
               "(any input) to capture a real example -- then come back here to iterate fast "
               "without re-running retrieval.")
        return

    labels = [f"{e['run_id']}  --  {e['user_prompt'][:90].replace(chr(10), ' ')}..." for e in examples]
    idx = st.selectbox("Example payload (captured from a real full run)", range(len(examples)),
                       format_func=lambda i: labels[i], key=f"example_idx__{prompt_id}")
    example = examples[idx]

    with st.expander("Original captured user prompt (the data sent to the model)"):
        st.text(example["user_prompt"])
    with st.expander("Original captured response"):
        st.text(example["response_text"])

    provider_mode = st.session_state["provider_mode"]
    if provider_mode == "demo":
        st.caption("Demo mode uses the scripted fake LLM -- it will not give a meaningful "
                  "comparison. Switch to production mode for a real prompt-only test.")

    if st.button("Test edited prompt against this example", key=f"test_btn__{prompt_id}"):
        if not edited_text.strip():
            st.error("Prompt is empty.")
            return
        try:
            llm = BedrockLLM() if provider_mode == "production" else ScriptedLLM()
            started = time.time()
            new_response = llm.call(prompt_id, example["user_prompt"], max_tokens=4000,
                                    system_override=edited_text)
            latency = time.time() - started
            call = llm.call_log[-1]
            cost = (call.input_tokens / 1_000_000) * telemetry.BEDROCK_INPUT_PER_MTOK + \
                  (call.output_tokens / 1_000_000) * telemetry.BEDROCK_OUTPUT_PER_MTOK

            st.success(f"Done in {latency:.1f}s -- ${cost:.4f} "
                      f"({call.input_tokens} in / {call.output_tokens} out tokens)")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original response**")
                st.text(example["response_text"])
            with col2:
                st.markdown("**New response (edited prompt)**")
                st.text(new_response)

            run_id = run_store.new_run_id()
            record = {
                "run_id": run_id, "mode": "prompt_only", "component": prompt_id,
                "status": "complete", "provider_mode": provider_mode,
                "prompt_versions_used": {prompt_id: prompt_store.get_active_version(prompt_id)},
                "source_example_run_id": example["run_id"],
                "system_prompt_used": edited_text, "user_prompt": example["user_prompt"],
                "original_response": example["response_text"], "new_response": new_response,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "total_latency_s": round(latency, 3), "total_cost_usd": round(cost, 6),
                "bedrock_calls": [{
                    "prompt_id": prompt_id, "prompt_version": "edited-in-ui",
                    "stage": telemetry.stage_for_prompt(prompt_id),
                    "input_tokens": call.input_tokens, "output_tokens": call.output_tokens,
                    "cost_usd": round(cost, 6), "latency_s": call.latency_s,
                    "attempts": call.attempts, "error": call.error,
                }],
                "error": None,
            }
            run_store.save_run(record)
            st.session_state["last_run_id"] = run_id
            st.caption(f"Saved as {run_id}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"LLM call failed: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------

def _render_result(run_id: str) -> None:
    record = run_store.load_run(run_id)
    if record is None:
        st.warning("Selected run not found on disk.")
        return

    status_icon = {"complete": "✅", "blocked": "⛔", "failed": "❌"}.get(record["status"], "")
    st.write(f"**Run ID:** {record['run_id']}  |  **Mode:** {record['mode']}  |  "
            f"**Status:** {status_icon} {record['status']}")

    if record["status"] == "blocked":
        st.error("Blocked at A1 (P&I validation)")
        st.json(record.get("input_validation"))
        return
    if record["status"] == "failed":
        st.error(record.get("error", "Run failed"))
        if record.get("traceback"):
            with st.expander("Traceback"):
                st.code(record["traceback"])
        errs = record.get("error_log") or []
        if errs:
            st.markdown(f"##### Retries / errors leading up to the failure ({len(errs)})")
            st.dataframe([{"t_s": e["t"], "level": e["level"], "message": e["message"]}
                         for e in errs], use_container_width=True, hide_index=True)
        return

    if record["mode"] == "prompt_only":
        st.write(f"**Component:** {record['component']}  |  "
                f"**Cost:** ${record['total_cost_usd']:.4f}  |  "
                f"**Latency:** {record['total_latency_s']:.1f}s  |  "
                f"**Example source:** {record['source_example_run_id']}")
        with st.expander("Prompt used", expanded=False):
            st.text(record["system_prompt_used"])
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original response**")
            st.text(record["original_response"])
        with col2:
            st.markdown("**New response**")
            st.text(record["new_response"])
        st.download_button("Download JSON", data=json.dumps(record, indent=2, default=str),
                           file_name=f"JCA_Phase1_{run_id}.json", mime="application/json")
        return

    # full_workflow, complete
    out = record["output"]
    st.write(f"**Cost:** ${record['total_cost_usd']:.4f}  |  "
            f"**Latency:** {record['total_latency_s']:.1f}s  |  "
            f"**Comparators found:** {record['num_comparators']}")

    tab_formatted, tab_raw, tab_debug = st.tabs(["Formatted", "Raw JSON", "Debug Trail (per-step)"])
    with tab_formatted:
        st.markdown("##### Prompt versions used this run")
        st.dataframe([{"prompt_id": k, "version": v} for k, v in record["prompt_versions_used"].items()],
                    use_container_width=True, hide_index=True)

        st.markdown("##### Comparators -- why included, sources, evidence")
        if out["comparators"]:
            for c in out["comparators"]:
                with st.expander(f"{c['generic_name']}  --  {c['member_state_count']} state(s), "
                                 f"{len(c.get('sources', []))} source(s), {len(c.get('evidence', []))} "
                                 f"evidence item(s)"):
                    st.write(f"**Class/mechanism:** {c['class_or_mechanism']} "
                            f"({c['class_source']})  |  **Line of therapy:** {c['line_of_therapy']}")
                    st.write(f"**Why included:** {c['rationale']}")
                    adj = c.get("scope_adjudication") or {}
                    if adj:
                        st.caption(f"Scope adjudication: {adj.get('verdict')} -- {adj.get('reason')}")
                    st.markdown("**Sources**")
                    st.dataframe([{"tier": s.get("tier"), "organization": s.get("organization"),
                                  "display_name": s.get("display_name"), "url": s.get("url")}
                                 for s in c.get("sources", [])], use_container_width=True, hide_index=True)
                    st.markdown("**Evidence**")
                    st.dataframe([{"member_state": e.get("member_state"),
                                  "quote": (e.get("quote") or "")[:200],
                                  "validation_verdict": e.get("validation_verdict"),
                                  "grounded": e.get("grounded")}
                                 for e in c.get("evidence", [])], use_container_width=True, hide_index=True)
        else:
            st.caption("No comparators in this run.")

        st.markdown("##### Outcome catalog coverage")
        st.dataframe(out["outcomes"].get("catalog_coverage", []), use_container_width=True, hide_index=True)

        st.markdown("##### Completeness assertions")
        st.dataframe(out["completeness"]["assertions"], use_container_width=True, hide_index=True)

        if out.get("notes"):
            st.markdown("##### Notes")
            for n in out["notes"]:
                st.caption(n)

        st.markdown("##### Bedrock / Tavily cost by stage")
        st.json({"bedrock": record["bedrock_summary"], "tavily": record["tavily_summary"],
                "stage_latency_s": record["stage_latency_s"]})

        errs = record.get("error_log") or []
        st.markdown(f"##### Retries / errors during this run ({len(errs)})")
        if errs:
            st.dataframe([{"t_s": e["t"], "level": e["level"], "message": e["message"]}
                         for e in errs], use_container_width=True, hide_index=True)
        else:
            st.caption("None -- every Bedrock/Tavily call succeeded on the first attempt.")
    with tab_raw:
        st.json(out)
    with tab_debug:
        debug = record.get("debug_capture") or {}
        if not debug:
            st.caption("No debug trail captured for this run (older run, or debug_capture was empty).")
        else:
            st.caption("Every stage's raw intermediate state -- including candidate comparators "
                      "that did NOT make the final list. Also in the downloadable Excel, one sheet "
                      "per stage.")
            all_adj = debug.get("A12_scope_adjudication") or {}
            st.markdown("##### A12 -- every candidate comparator (in scope or not)")
            st.dataframe([{
                "as_stated": d.get("comparator", {}).get("as_stated"),
                "num_records": d.get("num_records"),
                "verdict": d.get("adjudication", {}).get("verdict"),
                "decisive_facet": d.get("adjudication", {}).get("decisive_facet"),
                "reason": d.get("adjudication", {}).get("reason"),
            } for d in all_adj.values()], use_container_width=True, hide_index=True)
            if debug.get("A16_dropped_at_consolidation"):
                st.markdown("##### A16 -- dropped during consolidation")
                st.dataframe(debug["A16_dropped_at_consolidation"], use_container_width=True,
                            hide_index=True)
            st.markdown("##### Full stage-by-stage JSON")
            for key in ["A6_query_plan", "A7_retrieval", "A8_extraction", "A9_grounding",
                       "A10_claim_validation", "usable_after_A10", "A11_identity",
                       "A14_grouping", "A12_scope_adjudication", "A16_dropped_at_consolidation",
                       "A15_outcome_groups"]:
                if key in debug:
                    with st.expander(key):
                        st.json(debug[key])

    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Download JSON", data=json.dumps(record, indent=2, default=str),
                           file_name=f"JCA_Phase1_{run_id}.json", mime="application/json")
    with col2:
        st.download_button("Download Excel", data=excel_export.build_excel(record),
                           file_name=f"JCA_Phase1_{run_id}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.title("JCA Phase 1 -- SME Prompt Tuning")

with st.sidebar:
    st.header("Phase 1 Component")
    prompt_ids = prompt_store.all_prompt_ids()

    selected_prompt_id = st.selectbox("Component", prompt_ids, key="selected_prompt_id")
    meta = prompt_store.prompt_meta(selected_prompt_id)
    st.caption(meta["purpose"])

    versions = prompt_store.list_versions(selected_prompt_id)
    active_version = prompt_store.get_active_version(selected_prompt_id)
    default_idx = versions.index(active_version) if active_version in versions else 0
    selected_version = st.selectbox("Prompt Version", versions, index=default_idx,
                                    key=f"version_select__{selected_prompt_id}")

    st.divider()
    st.header("Provider Mode")
    provider_mode = st.radio("Providers", ["production", "demo"],
                             index=0 if st.session_state["provider_mode"] == "production" else 1,
                             help="production = real Bedrock + Tavily (real cost, minutes). "
                                  "demo = free fixture data, seconds, for wiring/UI checks only.")
    st.session_state["provider_mode"] = provider_mode
    if provider_mode == "production":
        st.warning("PRODUCTION mode: real Bedrock + Tavily calls -- real cost and several "
                  "minutes per full run.")

    st.divider()
    st.header("GT Test Drug")
    gt_drug = st.selectbox(
        "Drug", gt_workbook.GT_DRUGS,
        index=gt_workbook.GT_DRUGS.index(DEFAULT_DRUG) if DEFAULT_DRUG in gt_workbook.GT_DRUGS else 0,
        key="gt_test_drug",
        help="Which fixtures/jca gt.xlsx sheet 'Reset to GT defaults' (below) loads. "
            "Picking a drug here does not itself refill the input fields -- click "
            "Reset to GT defaults after picking to load that sheet's values.")

    default_benchmarking = gt_drug in gt_workbook.GT_DRUGS
    benchmarking_mode = st.checkbox(
        "Block the target drug's own JCA report (benchmarking mode)",
        value=st.session_state.get("benchmarking_mode", default_benchmarking),
        help="Enable for GT-validation runs so the drug's own published JCA report "
            "can never be read as evidence. Leave off for genuine production runs.")
    st.session_state["benchmarking_mode"] = benchmarking_mode

    st.divider()
    st.header("Execution Mode")
    exec_mode = st.radio("Mode", ["Full Phase 1 Workflow", "Test Selected Prompt Only"],
                         key="exec_mode")

    st.divider()
    st.caption("Active version per component")
    st.dataframe(
        [{"prompt_id": pid, "active_version": prompt_store.get_active_version(pid)}
         for pid in prompt_ids],
        use_container_width=True, hide_index=True, height=250)


# ---------------------------------------------------------------------------
# 1. JCA Input
# ---------------------------------------------------------------------------

st.subheader("1. JCA Input")
st.caption(f"'Reset to {gt_drug} GT defaults' below loads that drug's sheet from the GT "
          f"reference workbook (fixtures/jca gt.xlsx, GT test drug picked in the sidebar) "
          f"-- fields shown now are whatever was last loaded (or edited); edit freely "
          f"before running.")

col_reset1, col_reset2 = st.columns(2)
with col_reset1:
    if st.button(f"Reset to {gt_drug} GT defaults"):
        gt_population, gt_intervention = _load_gt_defaults(gt_drug)
        for k, v in gt_population.items():
            st.session_state[f"field__population__{k}"] = v
        for k, v in gt_intervention.items():
            st.session_state[f"field__intervention__{k}"] = v
        st.rerun()
with col_reset2:
    if st.button("Load demo example (jca_phase1.demo -- credential-free scenario)"):
        from jca_phase1.demo import DEMO_INTERVENTION, DEMO_POPULATION
        for k, v in DEMO_POPULATION.items():
            st.session_state[f"field__population__{k}"] = v
        for k, v in DEMO_INTERVENTION.items():
            st.session_state[f"field__intervention__{k}"] = v
        st.rerun()

col_pop, col_int = st.columns(2)
with col_pop:
    st.markdown("**Population**")
    by_group: dict = {}
    for f in POP_FIELDS:
        by_group.setdefault(f.get("group", "other"), []).append(f)
    for group, fields in by_group.items():
        st.caption(group.replace("_", " ").title())
        for f in fields:
            label = f["label"] + (" *" if f.get("mandatory") else "")
            st.text_area(label, key=f"field__population__{f['field']}", height=68,
                        value=DEFAULT_POPULATION.get(f["field"], ""), help=f.get("definition", ""))
with col_int:
    st.markdown("**Intervention**")
    for f in INT_FIELDS:
        label = f["label"] + (" *" if f.get("mandatory") else "")
        st.text_area(label, key=f"field__intervention__{f['field']}", height=68,
                    value=DEFAULT_INTERVENTION.get(f["field"], ""), help=f.get("definition", ""))

st.caption("* mandatory field")

# ---------------------------------------------------------------------------
# 2. SME Prompt
# ---------------------------------------------------------------------------

st.subheader("2. SME Prompt")

current_text = prompt_store.get_text(selected_prompt_id, selected_version)
loaded_key = f"editor_loaded_version__{selected_prompt_id}"
editor_key = f"editor_text__{selected_prompt_id}"
if st.session_state.get(loaded_key) != selected_version:
    st.session_state[editor_key] = current_text
    st.session_state[loaded_key] = selected_version

edited_text = st.text_area(f"{selected_prompt_id} -- prompt text", key=editor_key, height=400)
note = st.text_input("Version note (optional)", key=f"note__{selected_prompt_id}")

col1, col2 = st.columns(2)
with col1:
    if st.button("Apply as new version"):
        if edited_text.strip() == "":
            st.error("Prompt is empty.")
        elif edited_text == current_text and selected_version != prompt_store.BASE_VERSION:
            st.info("No changes from the selected version -- nothing new saved.")
        else:
            new_version = prompt_store.save_version(selected_prompt_id, edited_text, note)
            # The version-select widget below only honours `index=` the FIRST
            # time it's created in a session -- without this, it would keep
            # showing the version chosen before Apply, even though the new
            # version is now active, which risks the SME then clicking
            # "Make '<stale version>' active" and silently reverting the edit.
            st.session_state[f"version_select__{selected_prompt_id}"] = new_version
            st.success(f"Saved {selected_prompt_id} {new_version} and made it active.")
            st.rerun()
with col2:
    if st.button(f"Make '{selected_version}' the active version"):
        prompt_store.set_active_version(selected_prompt_id, selected_version)
        st.success(f"{selected_prompt_id} active version set to {selected_version}.")
        st.rerun()

# ---------------------------------------------------------------------------
# 3. Execution
# ---------------------------------------------------------------------------

st.subheader("3. Execution")

if exec_mode == "Full Phase 1 Workflow":
    st.caption("Prompt versions that will be used this run:")
    st.json({pid: prompt_store.get_active_version(pid) for pid in prompt_ids})

    run_disabled = st.session_state["running"]
    if st.button("Run Phase 1 with Current Prompts", disabled=run_disabled, type="primary"):
        population_input, intervention_input = _collect_inputs()
        errors = _validate_inputs(population_input, intervention_input)
        if errors:
            for e in errors:
                st.error(e)
        else:
            _start_full_run(population_input, intervention_input)
            st.rerun()

    if st.session_state["running"]:
        _render_running_status()
else:
    _render_prompt_only_mode(selected_prompt_id, edited_text)

# ---------------------------------------------------------------------------
# 4. Result
# ---------------------------------------------------------------------------

st.subheader("4. Result")
if st.session_state.get("last_run_id"):
    _render_result(st.session_state["last_run_id"])
else:
    st.caption("No run selected yet. Run the workflow above, or pick a past run from history below.")

# ---------------------------------------------------------------------------
# 5. Run History
# ---------------------------------------------------------------------------

st.subheader("5. Run History")
rows = run_store.list_runs()
if not rows:
    st.caption("No runs yet.")
else:
    st.dataframe(rows, use_container_width=True, hide_index=True)
    run_ids = [r["run_id"] for r in rows]
    picked = st.selectbox("View a past run", run_ids, key="history_select")
    if st.button("Load selected run into Result view"):
        st.session_state["last_run_id"] = picked
        st.rerun()
