"""
Runs Phase 1 end to end, with real providers (Bedrock + Tavily + registries),
against one drug from the GT reference workbook, and records full telemetry:
per-stage latency, every LLM/search call with its cost, and failures.

Usage:
    python scripts/run_gt_eval.py <DrugName> [--allow-jca-reports]

Output: scripts/gt_eval_output/<DrugName>.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gt_workbook import parse_gt_sheet  # noqa: E402

from jca_phase1.orchestrator import BlockedError, Providers, RunOptions, run_phase1  # noqa: E402
from jca_phase1.providers.llm import BedrockLLM  # noqa: E402
from jca_phase1.providers.registries import (ClinicalTrialsGovClient,  # noqa: E402
                                             EMAMedicineClient, PubMedClient)
from jca_phase1.providers.search import TavilySearchProvider  # noqa: E402

WORKBOOK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "fixtures", "jca gt.xlsx")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gt_eval_output")

# Bedrock pricing: Anthropic first-party list price for Claude Sonnet 5
# ($2.00 / $10.00 per 1M input/output tokens). Bedrock is partner-operated
# and may bill at a different rate -- treat this as an estimate and reconcile
# against AWS Cost Explorer / the Bedrock console for the authoritative figure.
BEDROCK_INPUT_PER_MTOK = 2.00
BEDROCK_OUTPUT_PER_MTOK = 10.00

# Tavily pay-as-you-go: $0.008 / credit (docs.tavily.com/documentation/api-credits).
# Actual credits per call are read back from Tavily's own `usage.credits` field
# (include_usage=True), not estimated -- this is the real billed amount at
# whatever per-credit rate the account is on. If the account is on a plan with
# included monthly credits rather than pay-as-you-go, the true out-of-pocket
# cost is lower than shown here.
TAVILY_USD_PER_CREDIT = 0.008


def build_stage_timeline(progress: List[Dict[str, Any]], total_latency: float) -> Dict[str, float]:
    """Attribute wall-clock time to each stage from progress-emit timestamps.

    The orchestrator emits at the START of a stage (sometimes more than once,
    e.g. A6/A7). Duration between consecutive emits is charged to the stage of
    the earlier one; the final stage runs until the process's measured total.
    """
    by_stage: Dict[str, float] = {}
    for i, entry in enumerate(progress):
        start = entry["t"]
        end = progress[i + 1]["t"] if i + 1 < len(progress) else (progress[0]["t"] + total_latency if progress else start)
        by_stage[entry["stage"]] = by_stage.get(entry["stage"], 0.0) + max(0.0, end - start)
    return {k: round(v, 3) for k, v in by_stage.items()}


def stage_of_prompt(prompt_id: str) -> str:
    digits = "".join(ch for ch in prompt_id.split(".")[0] if ch.isdigit())
    return f"A{int(digits)}" if digits else "?"


def summarize_bedrock(llm: BedrockLLM) -> Dict[str, Any]:
    calls = []
    total_cost = 0.0
    by_stage: Dict[str, Dict[str, Any]] = {}
    for c in llm.call_log:
        cost = (c.input_tokens / 1_000_000) * BEDROCK_INPUT_PER_MTOK + \
               (c.output_tokens / 1_000_000) * BEDROCK_OUTPUT_PER_MTOK
        total_cost += cost
        stage = stage_of_prompt(c.prompt_id)
        bucket = by_stage.setdefault(stage, {"calls": 0, "failed": 0, "input_tokens": 0,
                                             "output_tokens": 0, "cost_usd": 0.0,
                                             "latency_s": 0.0})
        bucket["calls"] += 1
        bucket["failed"] += 1 if c.error else 0
        bucket["input_tokens"] += c.input_tokens
        bucket["output_tokens"] += c.output_tokens
        bucket["cost_usd"] += cost
        bucket["latency_s"] += c.latency_s
        calls.append({
            "prompt_id": c.prompt_id, "prompt_version": c.prompt_version, "stage": stage,
            "input_tokens": c.input_tokens, "output_tokens": c.output_tokens,
            "cost_usd": round(cost, 6), "latency_s": c.latency_s, "attempts": c.attempts,
            "error": c.error,
        })
    for bucket in by_stage.values():
        bucket["cost_usd"] = round(bucket["cost_usd"], 6)
        bucket["latency_s"] = round(bucket["latency_s"], 3)
    return {
        "model_id": llm.model_id, "region": llm.region,
        "calls": len(llm.call_log), "failed_calls": sum(1 for c in llm.call_log if c.error),
        "total_input_tokens": sum(c.input_tokens for c in llm.call_log),
        "total_output_tokens": sum(c.output_tokens for c in llm.call_log),
        "total_cost_usd": round(total_cost, 4),
        "pricing_usd_per_mtok": {"input": BEDROCK_INPUT_PER_MTOK, "output": BEDROCK_OUTPUT_PER_MTOK},
        "by_stage": by_stage,
        "calls_detail": calls,
    }


def summarize_tavily(search: TavilySearchProvider) -> Dict[str, Any]:
    calls = []
    total_cost = 0.0
    by_op: Dict[str, Dict[str, Any]] = {}
    for c in search.call_log:
        cost = c.credits * TAVILY_USD_PER_CREDIT
        total_cost += cost
        bucket = by_op.setdefault(c.op, {"calls": 0, "failed": 0, "credits": 0.0, "cost_usd": 0.0,
                                         "latency_s": 0.0})
        bucket["calls"] += 1
        bucket["failed"] += 1 if c.error else 0
        bucket["credits"] += c.credits
        bucket["cost_usd"] += cost
        bucket["latency_s"] += c.latency_s
        calls.append({
            "op": c.op, "detail": c.detail, "credits": c.credits, "cost_usd": round(cost, 6),
            "latency_s": c.latency_s, "attempts": c.attempts, "results": c.results,
            "error": c.error,
        })
    for bucket in by_op.values():
        bucket["cost_usd"] = round(bucket["cost_usd"], 6)
        bucket["credits"] = round(bucket["credits"], 3)
        bucket["latency_s"] = round(bucket["latency_s"], 3)
    return {
        "calls": len(search.call_log), "failed_calls": sum(1 for c in search.call_log if c.error),
        "total_credits": round(sum(c.credits for c in search.call_log), 3),
        "total_cost_usd": round(total_cost, 4),
        "pricing_usd_per_credit": TAVILY_USD_PER_CREDIT,
        "by_op": by_op,
        "calls_detail": calls,
    }


def run_one(drug: str, allow_jca_reports: bool = False,
           drop_population_fields: List[str] = None) -> Dict[str, Any]:
    population_input, intervention_input = parse_gt_sheet(WORKBOOK, drug)
    for field in (drop_population_fields or []):
        population_input.pop(field, None)

    search = TavilySearchProvider()
    llm = BedrockLLM()
    providers = Providers(llm=llm, search=search,
                         trials=ClinicalTrialsGovClient(),
                         literature=PubMedClient(),
                         medicines=EMAMedicineClient(search_provider=search))

    progress: List[Dict[str, Any]] = []

    def on_progress(stage: str, message: str) -> None:
        progress.append({"t": time.time(), "stage": stage, "message": message})

    options = RunOptions(request_id=f"gt-{drug.lower()}", allow_jca_reports=allow_jca_reports,
                        progress=on_progress)

    started = time.time()
    result: Dict[str, Any] = {
        "drug": drug, "started_at": started, "population_input": population_input,
        "intervention_input": intervention_input, "allow_jca_reports": allow_jca_reports,
    }
    try:
        output = run_phase1(population_input, intervention_input, providers, options=options)
        total_latency = time.time() - started
        result["status"] = "complete"
        result["output"] = output.to_dict()
        result["num_comparators"] = len(output.comparators)
        result["num_outcomes"] = len(output.outcomes.all_outcomes())
    except BlockedError as exc:
        total_latency = time.time() - started
        result["status"] = "blocked"
        result["input_validation"] = exc.validation.to_dict()
    except Exception as exc:  # noqa: BLE001
        total_latency = time.time() - started
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()

    result["total_latency_s"] = round(total_latency, 3)
    result["stage_latency_s"] = build_stage_timeline(progress, total_latency)
    result["progress_log"] = [{"t": round(p["t"] - started, 3), "stage": p["stage"],
                               "message": p["message"]} for p in progress]
    result["bedrock"] = summarize_bedrock(llm)
    result["tavily"] = summarize_tavily(search)
    result["total_cost_usd"] = round(result["bedrock"]["total_cost_usd"] +
                                     result["tavily"]["total_cost_usd"], 4)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("drug")
    ap.add_argument("--allow-jca-reports", action="store_true")
    ap.add_argument("--drop-population-field", action="append", default=[])
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    result = run_one(args.drug, allow_jca_reports=args.allow_jca_reports,
                    drop_population_fields=args.drop_population_field)
    out_path = os.path.join(OUT_DIR, f"{args.drug}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"[{args.drug}] status={result['status']} "
         f"comparators={result.get('num_comparators')} "
         f"total_latency_s={result['total_latency_s']} "
         f"bedrock_cost_usd={result['bedrock']['total_cost_usd']} "
         f"tavily_cost_usd={result['tavily']['total_cost_usd']} "
         f"-> {out_path}")


if __name__ == "__main__":
    main()
