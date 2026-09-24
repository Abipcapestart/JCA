"""
Full-workflow execution logic, kept free of any Streamlit dependency so it can
be exercised directly (scripts, tests) without a running Streamlit session.
app.py wraps run_full_workflow() in a background thread and relays its
progress_log/result via st.session_state.
"""

from __future__ import annotations

import time
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import run_store
import telemetry

from jca_phase1.api.handlers import _demo_providers, production_providers
from jca_phase1.orchestrator import BlockedError, RunOptions, run_phase1
from jca_phase1.prompts import registry as prompt_registry


def base_record(run_id: str, mode: str, component: Optional[str], provider_mode: str,
               prompt_versions_used: Dict[str, str], population_input: Dict[str, Any],
               intervention_input: Dict[str, Any], started_at: str) -> Dict[str, Any]:
    return {
        "run_id": run_id, "mode": mode, "component": component, "status": "running",
        "provider_mode": provider_mode, "prompt_versions_used": prompt_versions_used,
        "population_input": population_input, "intervention_input": intervention_input,
        "started_at": started_at, "total_latency_s": None, "total_cost_usd": None,
        "num_comparators": None, "output": None, "bedrock_calls": [], "bedrock_summary": {},
        "tavily_calls": [], "tavily_summary": {}, "stage_latency_s": {}, "progress_log": [],
        "debug_capture": {}, "error_log": [], "error": None,
    }


def run_full_workflow(run_id: str, population_input: Dict[str, Any],
                      intervention_input: Dict[str, Any], provider_mode: str,
                      overrides: Dict[str, Dict[str, str]], prompt_versions_used: Dict[str, str],
                      progress_log: List[Dict[str, Any]], started_at: str,
                      error_log: Optional[List[Dict[str, Any]]] = None,
                      allow_jca_reports: bool = True) -> Dict[str, Any]:
    """Blocking. Returns the full run record (see base_record for shape).
    Acquires run_store.RUN_LOCK for the duration -- refuses to start a second
    concurrent run rather than letting two runs clobber each other's prompt
    overrides (this tool is single-run-at-a-time by design).

    `error_log`, when passed a list, is appended in-place with every LLM/search
    retry and final failure (also always logged server-side via `logging`,
    regardless of whether error_log is passed) -- for live UI display."""
    error_log = error_log if error_log is not None else []
    acquired = run_store.RUN_LOCK.acquire(blocking=False)
    if not acquired:
        rec = base_record(run_id, "full_workflow", None, provider_mode, prompt_versions_used,
                          population_input, intervention_input, started_at)
        rec["status"] = "failed"
        rec["error"] = "Another run is already in progress on this server. Wait for it to finish."
        return rec

    started = time.time()
    try:
        prompt_registry.clear_overrides()
        for pid, ov in overrides.items():
            # "ui-" prefix keeps this distinguishable in the run manifest from
            # the registry's own built-in version label (both start at "v1").
            prompt_registry.register_override(pid, ov["text"], f"ui-{ov['version']}")

        try:
            providers = production_providers() if provider_mode == "production" else _demo_providers()
        except Exception as exc:  # noqa: BLE001
            rec = base_record(run_id, "full_workflow", None, provider_mode, prompt_versions_used,
                              population_input, intervention_input, started_at)
            rec["status"] = "failed"
            rec["error"] = f"Provider setup failed: {type(exc).__name__}: {exc}"
            return rec

        def on_provider_event(level: str, message: str) -> None:
            error_log.append({"t": round(time.time() - started, 3), "level": level, "message": message})

        if providers.llm is not None:
            providers.llm.on_event = on_provider_event
        if providers.search is not None:
            providers.search.on_event = on_provider_event

        def on_progress(stage: str, message: str) -> None:
            progress_log.append({"t": round(time.time() - started, 3), "stage": stage, "message": message})

        options = RunOptions(request_id=run_id, progress=on_progress,
                            allow_jca_reports=allow_jca_reports)
        rec = base_record(run_id, "full_workflow", None, provider_mode, prompt_versions_used,
                          population_input, intervention_input, started_at)
        debug_capture: Dict[str, Any] = {}
        try:
            output = run_phase1(population_input, intervention_input, providers, options=options,
                               debug_capture=debug_capture)
            total_latency = time.time() - started
            bedrock_calls, bedrock_summary = telemetry.summarize_bedrock(providers.llm)
            tavily_calls, tavily_summary = ([], {"calls": 0, "failed_calls": 0,
                                                 "total_credits": 0, "total_cost_usd": 0.0})
            if providers.search is not None:
                tavily_calls, tavily_summary = telemetry.summarize_tavily(providers.search)
            rec.update({
                "status": "complete", "total_latency_s": round(total_latency, 3),
                "total_cost_usd": round(bedrock_summary["total_cost_usd"] +
                                        tavily_summary["total_cost_usd"], 4),
                "num_comparators": len(output.comparators), "output": output.to_dict(),
                "bedrock_calls": bedrock_calls, "bedrock_summary": bedrock_summary,
                "tavily_calls": tavily_calls, "tavily_summary": tavily_summary,
                "stage_latency_s": telemetry.build_stage_timeline(progress_log, total_latency),
                "progress_log": progress_log, "debug_capture": debug_capture,
                "error_log": error_log,
            })
        except BlockedError as exc:
            rec.update({"status": "blocked", "total_latency_s": round(time.time() - started, 3),
                       "error": "P&I validation returned blocking FIX items",
                       "input_validation": exc.validation.to_dict(), "progress_log": progress_log,
                       "debug_capture": debug_capture, "error_log": error_log})
        except Exception as exc:  # noqa: BLE001
            # A crash partway through must not throw away the real cost/call
            # telemetry already accumulated on providers.llm/providers.search
            # -- those objects hold it in-memory regardless of where run_phase1
            # failed, and it's real money already spent. Guarded separately so
            # a problem summarizing telemetry never masks the original error.
            bedrock_calls, bedrock_summary = [], {}
            tavily_calls, tavily_summary = [], {}
            try:
                if providers.llm is not None:
                    bedrock_calls, bedrock_summary = telemetry.summarize_bedrock(providers.llm)
                if providers.search is not None:
                    tavily_calls, tavily_summary = telemetry.summarize_tavily(providers.search)
            except Exception:  # noqa: BLE001
                pass
            rec.update({"status": "failed", "total_latency_s": round(time.time() - started, 3),
                       "total_cost_usd": round(bedrock_summary.get("total_cost_usd", 0.0)
                                               + tavily_summary.get("total_cost_usd", 0.0), 4),
                       "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                       "progress_log": progress_log, "debug_capture": debug_capture,
                       "error_log": error_log, "bedrock_calls": bedrock_calls,
                       "bedrock_summary": bedrock_summary, "tavily_calls": tavily_calls,
                       "tavily_summary": tavily_summary})
        return rec
    finally:
        prompt_registry.clear_overrides()
        run_store.RUN_LOCK.release()


def run_full_workflow_async(run_id: str, population_input: Dict[str, Any],
                            intervention_input: Dict[str, Any], provider_mode: str,
                            overrides: Dict[str, Dict[str, str]], prompt_versions_used: Dict[str, str],
                            progress_log: List[Dict[str, Any]], result_holder: Dict[str, Any],
                            started_at: str, error_log: Optional[List[Dict[str, Any]]] = None,
                            allow_jca_reports: bool = True) -> None:
    """Thread target: runs run_full_workflow() and drops the record into
    result_holder['record'] for the caller (app.py) to pick up and persist."""
    result_holder["record"] = run_full_workflow(
        run_id, population_input, intervention_input, provider_mode, overrides,
        prompt_versions_used, progress_log, started_at, error_log=error_log,
        allow_jca_reports=allow_jca_reports)
