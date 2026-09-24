"""
Simple on-disk run history: one JSON file per run under run_history/. No
database -- a flat directory is enough for a single-user internal tool, and
keeps every run's full prompt-version provenance and output inspectable by
just opening the file.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_history")

# Module-level (not session-level) so it's a real process-wide mutex: Streamlit
# re-executes app.py on every rerun, but modules it imports via `import` stay
# cached in sys.modules, so this lock instance persists across those reruns.
# Enforces "single run at a time" per the chosen concurrency model -- a second
# Run attempt fails fast with a clear message instead of corrupting the first
# run's registered prompt overrides.
RUN_LOCK = threading.Lock()


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"RUN-{stamp}-{uuid.uuid4().hex[:6]}"


def save_run(record: Dict[str, Any]) -> str:
    os.makedirs(RUNS_DIR, exist_ok=True)
    path = os.path.join(RUNS_DIR, f"{record['run_id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)
    return path


def list_runs() -> List[Dict[str, Any]]:
    """Newest first. Reads full files (fine at this tool's scale -- an
    internal SME tool doing tens to low hundreds of runs, not thousands)."""
    if not os.path.isdir(RUNS_DIR):
        return []
    rows = []
    for fname in sorted(os.listdir(RUNS_DIR), reverse=True):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(RUNS_DIR, fname), "r", encoding="utf-8") as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        rows.append({
            "run_id": d.get("run_id"), "mode": d.get("mode"), "status": d.get("status"),
            "component": d.get("component"),
            "prompt_versions_used": d.get("prompt_versions_used", {}),
            "started_at": d.get("started_at"),
            "total_latency_s": d.get("total_latency_s"),
            "total_cost_usd": d.get("total_cost_usd"),
            "num_comparators": d.get("num_comparators"),
        })
    return rows


def load_run(run_id: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(RUNS_DIR, f"{run_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_example_payloads(prompt_id: str, max_runs: int = 8, max_per_run: int = 5) -> List[Dict[str, Any]]:
    """Real captured (system_prompt, user_prompt, response_text) triples for
    this prompt_id, pulled from the most recent full-workflow runs -- the
    payload source for "test selected prompt only" mode. Returns [] if no
    full run has ever exercised this prompt yet."""
    out: List[Dict[str, Any]] = []
    for row in list_runs():
        if row.get("mode") != "full_workflow":
            continue
        run = load_run(row["run_id"])
        if not run:
            continue
        calls = [c for c in run.get("bedrock_calls", []) if c["prompt_id"] == prompt_id and not c["error"]]
        for c in calls[:max_per_run]:
            out.append({
                "run_id": run["run_id"], "started_at": run.get("started_at"),
                "system_prompt": c["system_prompt"], "user_prompt": c["user_prompt"],
                "response_text": c["response_text"],
            })
        if len(out) >= max_runs * max_per_run:
            break
    return out
