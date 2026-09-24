"""Throwaway validation of the streamlit_app plumbing, no browser needed.
Uses demo providers (free) to prove: prompt override actually reaches the run
manifest, run persistence round-trips, Excel export builds, and prompt-only
mode can find a captured example payload afterward.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.join(REPO_ROOT, "streamlit_app")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, APP_DIR)

import prompt_store  # noqa: E402
import run_store  # noqa: E402
import workflow_runner  # noqa: E402
import excel_export  # noqa: E402

PROMPT_ID = "a01.pi_validation"

base = prompt_store.base_text(PROMPT_ID)
edited = base + "\n\n[VALIDATION-SENTINEL: this line proves an SME edit reached the run]"
new_version = prompt_store.save_version(PROMPT_ID, edited, note="validation smoke test")
print("saved version:", new_version, "| active now:", prompt_store.get_active_version(PROMPT_ID))
assert prompt_store.get_active_version(PROMPT_ID) == new_version

overrides = prompt_store.active_overrides()
assert PROMPT_ID in overrides, "override not picked up by active_overrides()"
prompt_versions_used = {pid: prompt_store.get_active_version(pid) for pid in prompt_store.all_prompt_ids()}

run_id = run_store.new_run_id()
progress_log = []
from datetime import datetime, timezone
record = workflow_runner.run_full_workflow(
    run_id=run_id,
    population_input={"indication_disease": "Extensive-stage small cell lung cancer"},
    intervention_input={"product_name_inn": "Examplimab"},
    provider_mode="demo",
    overrides=overrides,
    prompt_versions_used=prompt_versions_used,
    progress_log=progress_log,
    started_at=datetime.now(timezone.utc).isoformat(),
)
print("status:", record["status"])
assert record["status"] == "complete", record.get("error")

manifest_versions = record["output"]["run_manifest"]["prompt_versions"]
print("manifest prompt_versions for", PROMPT_ID, "=", manifest_versions.get(PROMPT_ID))
assert manifest_versions.get(PROMPT_ID) == f"ui-{new_version}", (
    "override version did not reach the run manifest -- injection mechanism broken")

run_store.save_run(record)
loaded = run_store.load_run(run_id)
assert loaded is not None and loaded["run_id"] == run_id
print("run persisted and reloaded OK:", run_id)

rows = run_store.list_runs()
assert any(r["run_id"] == run_id for r in rows)
print("appears in list_runs():", any(r["run_id"] == run_id for r in rows))

xlsx_bytes = excel_export.build_excel(record)
assert len(xlsx_bytes) > 1000
print("excel built OK, bytes:", len(xlsx_bytes))

examples = run_store.find_example_payloads(PROMPT_ID)
print("captured example payloads for", PROMPT_ID, ":", len(examples))
assert examples, "expected at least one captured payload after a full run"
assert "VALIDATION-SENTINEL" not in examples[0]["user_prompt"], (
    "sentinel should be in the SYSTEM prompt, not leak into the captured user prompt")

print()
print("ALL CHECKS PASSED")
