"""
Cost/latency telemetry helpers, shared by app.py and excel_export.py.

Pricing notes (same caveats as scripts/run_gt_eval.py, built earlier this
session for the GT-drug evaluation):
  - Bedrock cost uses Anthropic's first-party list price for Claude Sonnet 5
    ($2.00 / $10.00 per 1M input/output tokens) as an ESTIMATE. Bedrock is
    partner-billed separately -- reconcile against AWS Cost Explorer for the
    authoritative figure.
  - Tavily cost uses the real `usage.credits` value Tavily returns per call
    (captured in jca_phase1/providers/search.py) at the documented
    pay-as-you-go rate of $0.008/credit.
"""

from __future__ import annotations

from typing import Any, Dict, List

BEDROCK_INPUT_PER_MTOK = 2.00
BEDROCK_OUTPUT_PER_MTOK = 10.00
# Anthropic's standard prompt-caching multipliers on the base input rate:
# a cache WRITE costs 1.25x (creating the cached prefix), a cache READ costs
# 0.1x (reusing it) -- both against BEDROCK_INPUT_PER_MTOK, not separate list
# prices. Same "estimate, reconcile against AWS Cost Explorer" caveat as the
# base rates above.
BEDROCK_CACHE_WRITE_MULTIPLIER = 1.25
BEDROCK_CACHE_READ_MULTIPLIER = 0.1
TAVILY_USD_PER_CREDIT = 0.008

# The prompt_id's own numeric prefix does NOT always match the orchestrator
# stage it actually runs in -- verified against jca_phase1/agents/
# a14_a18_consolidation.py and orchestrator.py directly. harmonize_outcomes()
# and build_outcomes()'s prompts execute inside the orchestrator's "A15"
# block; build_comparator()'s two prompts execute inside "A16".
PROMPT_TO_STAGE = {
    "a01.pi_validation": "A1",
    "a02.input_structuring": "A2",
    "a03.scope_facet_normalise": "A3",
    "a04.indication_lock": "A4",
    "a05.area_adjudication": "A5",
    "a06.query_vocabulary": "A6",
    "a08.extraction": "A8",
    "a10.claim_validation": "A10",
    "a11.comparator_identity": "A11",
    "a12.scope_adjudication": "A12",
    "a14.outcome_harmonization": "A15",
    "a16.outcome_rationale": "A15",
    "a16.comparator_rationale": "A16",
    "a16.indication_synthesis": "A16",
}


def stage_for_prompt(prompt_id: str) -> str:
    return PROMPT_TO_STAGE.get(prompt_id, "?")


def summarize_bedrock(llm) -> Dict[str, Any]:
    """llm: a jca_phase1.providers.llm.LLMClient (BedrockLLM/ScriptedLLM)."""
    calls: List[Dict[str, Any]] = []
    total_cost = 0.0
    for c in llm.call_log:
        cache_write = getattr(c, "cache_creation_input_tokens", 0)
        cache_read = getattr(c, "cache_read_input_tokens", 0)
        cost = ((c.input_tokens / 1_000_000) * BEDROCK_INPUT_PER_MTOK
               + (c.output_tokens / 1_000_000) * BEDROCK_OUTPUT_PER_MTOK
               + (cache_write / 1_000_000) * BEDROCK_INPUT_PER_MTOK
                 * BEDROCK_CACHE_WRITE_MULTIPLIER
               + (cache_read / 1_000_000) * BEDROCK_INPUT_PER_MTOK
                 * BEDROCK_CACHE_READ_MULTIPLIER)
        total_cost += cost
        calls.append({
            "prompt_id": c.prompt_id, "prompt_version": c.prompt_version,
            "stage": stage_for_prompt(c.prompt_id),
            "input_tokens": c.input_tokens, "output_tokens": c.output_tokens,
            "cache_creation_input_tokens": cache_write, "cache_read_input_tokens": cache_read,
            "cost_usd": round(cost, 6), "latency_s": c.latency_s, "attempts": c.attempts,
            "error": c.error, "system_prompt": c.system_prompt, "user_prompt": c.user_prompt,
            "response_text": c.response_text,
        })
    summary = {
        "calls": len(llm.call_log), "failed_calls": sum(1 for c in llm.call_log if c.error),
        "total_input_tokens": sum(c.input_tokens for c in llm.call_log),
        "total_output_tokens": sum(c.output_tokens for c in llm.call_log),
        "total_cache_creation_input_tokens": sum(
            getattr(c, "cache_creation_input_tokens", 0) for c in llm.call_log),
        "total_cache_read_input_tokens": sum(
            getattr(c, "cache_read_input_tokens", 0) for c in llm.call_log),
        "total_cost_usd": round(total_cost, 4),
    }
    return calls, summary


def summarize_tavily(search) -> Dict[str, Any]:
    """search: a jca_phase1.providers.search.SearchProvider (Tavily/Fixture)."""
    calls: List[Dict[str, Any]] = []
    total_cost = 0.0
    for c in search.call_log:
        cost = c.credits * TAVILY_USD_PER_CREDIT
        total_cost += cost
        calls.append({
            "op": c.op, "detail": c.detail, "credits": c.credits, "cost_usd": round(cost, 6),
            "latency_s": c.latency_s, "attempts": c.attempts, "results": c.results, "error": c.error,
        })
    summary = {
        "calls": len(search.call_log), "failed_calls": sum(1 for c in search.call_log if c.error),
        "total_credits": round(sum(c.credits for c in search.call_log), 3),
        "total_cost_usd": round(total_cost, 4),
    }
    return calls, summary


def build_stage_timeline(progress_log: List[Dict[str, Any]], total_latency: float) -> Dict[str, float]:
    """Same attribution method as scripts/run_gt_eval.py: duration between
    consecutive progress-emit timestamps is charged to the earlier stage."""
    by_stage: Dict[str, float] = {}
    for i, entry in enumerate(progress_log):
        start = entry["t"]
        end = progress_log[i + 1]["t"] if i + 1 < len(progress_log) else total_latency
        by_stage[entry["stage"]] = by_stage.get(entry["stage"], 0.0) + max(0.0, end - start)
    return {k: round(v, 3) for k, v in by_stage.items()}
