"""
Framework-free API handlers.

Kept independent of any web framework so the same logic serves the FastAPI app
(production) and the stdlib dev server (runs anywhere, no install). Each handler
takes a plain dict and returns (status_code, dict).

Runs are held in an in-process store. That is correct for a single-node demo and
explicitly NOT correct for production — swap RunStore for your persistence
layer; the interface is four methods.
"""

from __future__ import annotations

import json
import threading
import traceback
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from .. import config as C
from ..agents import a01_a05_input_context as inputs
from ..agents import a14_a18_consolidation as consolidation
from ..orchestrator import BlockedError, Providers, RunOptions, run_phase1
from ..prompts import registry as prompt_registry
from ..schema import Phase1Output
from ..sources.workbook import load_source_inventory

JSON = Dict[str, Any]


class RunStore:
    """In-process run store. Replace with your persistence layer."""

    def __init__(self) -> None:
        self._runs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(self, request_id: str) -> None:
        with self._lock:
            self._runs[request_id] = {"request_id": request_id, "status": "running",
                                      "progress": [], "output": None, "error": None}

    def progress(self, request_id: str, stage: str, message: str) -> None:
        with self._lock:
            run = self._runs.get(request_id)
            if run is not None:
                run["progress"].append({"stage": stage, "message": message})

    def finish(self, request_id: str, output: Phase1Output) -> None:
        with self._lock:
            run = self._runs.setdefault(request_id, {"request_id": request_id,
                                                     "progress": []})
            run["status"] = "complete"
            run["output"] = output
            run["error"] = None

    def fail(self, request_id: str, error: str, detail: Any = None) -> None:
        with self._lock:
            run = self._runs.setdefault(request_id, {"request_id": request_id,
                                                     "progress": []})
            run["status"] = "failed"
            run["error"] = {"message": error, "detail": detail}

    def get(self, request_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._runs.get(request_id)

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [{"request_id": r["request_id"], "status": r["status"],
                     "comparators": (len(r["output"].comparators)
                                     if r.get("output") else 0)}
                    for r in self._runs.values()]


STORE = RunStore()

# Provider factory. Default is the credential-free demo scenario so the UI is
# clickable out of the box; wire the real one in your deployment.
_provider_factory: Callable[[], Providers] = lambda: _demo_providers()


def _demo_providers() -> Providers:
    from ..demo import build_demo_providers
    return build_demo_providers()


def set_provider_factory(factory: Callable[[], Providers]) -> None:
    global _provider_factory
    _provider_factory = factory


def production_providers() -> Providers:
    """Real providers. Requires TAVILY_API_KEY and AWS credentials."""
    from ..providers.llm import BedrockLLM
    from ..providers.registries import (ClinicalTrialsGovClient, EMAMedicineClient,
                                        PubMedClient)
    from ..providers.search import TavilySearchProvider
    search = TavilySearchProvider()
    return Providers(llm=BedrockLLM(), search=search,
                     trials=ClinicalTrialsGovClient(),
                     literature=PubMedClient(),
                     medicines=EMAMedicineClient(search_provider=search))


# ===========================================================================
# Handlers
# ===========================================================================

def get_schema(_: JSON) -> Tuple[int, JSON]:
    """Everything the UI needs to render screen 1 without hardcoding anything."""
    return 200, {
        "population_fields": inputs.TAXONOMY["population"],
        "intervention_fields": inputs.TAXONOMY["intervention"],
        "member_states": [{"name": s, "code": C.MEMBER_STATE_CODE[s]}
                          for s in C.EU_27_MEMBER_STATES],
        "outcome_categories": C.OUTCOME_CATEGORIES,
        "therapeutic_areas": C.THERAPEUTIC_AREAS,
        "outcome_catalog": [
            {"catalog_id": i["catalog_id"], "display_name": i["display_name"],
             "category": i["category"]}
            for i in consolidation.CATALOG_ITEMS],
        "outcome_catalog_provenance": consolidation.CATALOG["provenance"],
        "phase": "phase1_comparator_outcome_scoping",
        "phase2_note": ("PICO-set generation is Phase 2 and is not produced by "
                        "this service."),
    }


def validate(payload: JSON) -> Tuple[int, JSON]:
    """Screen 1's FIX/CHECK check, callable before a full run."""
    providers = _provider_factory()
    result = inputs.validate_pi(payload.get("population") or {},
                               payload.get("intervention") or {},
                               payload.get("added_fields") or [],
                               providers.llm)
    return 200, result.to_dict()


def create_run(payload: JSON) -> Tuple[int, JSON]:
    """Start a Phase 1 run. Synchronous by default; set `async_run` for a
    background thread and poll GET /runs/{id}."""
    request_id = payload.get("request_id") or f"jca-{uuid.uuid4().hex[:10]}"
    STORE.create(request_id)

    options = RunOptions(
        request_id=request_id,
        # A missing key must inherit the new default (True), not silently
        # revert to the old evaluation-only behaviour via bool(None).
        allow_jca_reports=bool(payload.get("allow_jca_reports", True)),
        source_workbook=payload.get("source_workbook"),
        strict_sources=payload.get("strict_sources", True),
        progress=lambda stage, msg: STORE.progress(request_id, stage, msg))

    def _run() -> Tuple[int, JSON]:
        try:
            output = run_phase1(
                payload.get("population") or {},
                payload.get("intervention") or {},
                _provider_factory(),
                free_text=payload.get("free_text", ""),
                added_fields=payload.get("added_fields") or [],
                options=options)
            STORE.finish(request_id, output)
            return 200, {"request_id": request_id, "status": "complete",
                         "output": output.to_dict()}
        except BlockedError as exc:
            STORE.fail(request_id, "P&I validation returned blocking FIX items",
                       exc.validation.to_dict())
            return 422, {"request_id": request_id, "status": "blocked",
                         "input_validation": exc.validation.to_dict()}
        except Exception as exc:  # noqa: BLE001
            STORE.fail(request_id, str(exc), traceback.format_exc())
            return 500, {"request_id": request_id, "status": "failed",
                         "error": str(exc)}

    if payload.get("async_run"):
        threading.Thread(target=_run, daemon=True).start()
        return 202, {"request_id": request_id, "status": "running"}
    return _run()


def get_run(payload: JSON) -> Tuple[int, JSON]:
    run = STORE.get(payload.get("request_id", ""))
    if run is None:
        return 404, {"error": "unknown request_id"}
    body: JSON = {"request_id": run["request_id"], "status": run["status"],
                  "progress": run["progress"]}
    if run.get("output") is not None:
        body["output"] = run["output"].to_dict()
    if run.get("error"):
        body["error"] = run["error"]
    return 200, body


def list_runs(_: JSON) -> Tuple[int, JSON]:
    return 200, {"runs": STORE.list()}


def add_comparator(payload: JSON) -> Tuple[int, JSON]:
    """SME workflow step 14: the reviewer adds a comparator the pipeline missed.
    Drug name and Member State are both mandatory; the entry is taken exactly as
    given."""
    run = STORE.get(payload.get("request_id", ""))
    if run is None or run.get("output") is None:
        return 404, {"error": "unknown or incomplete run"}
    try:
        comparator = consolidation.add_user_comparator(
            run["output"],
            generic_name=payload.get("generic_name", ""),
            member_states=payload.get("member_states") or [],
            class_or_mechanism=payload.get("class_or_mechanism", ""),
            line_of_therapy=payload.get("line_of_therapy", ""),
            rationale=payload.get("rationale", ""))
    except ValueError as exc:
        return 400, {"error": str(exc)}
    return 200, {"comparator": comparator.to_dict(),
                 "output": run["output"].to_dict()}


def add_outcome(payload: JSON) -> Tuple[int, JSON]:
    run = STORE.get(payload.get("request_id", ""))
    if run is None or run.get("output") is None:
        return 404, {"error": "unknown or incomplete run"}
    try:
        outcome = consolidation.add_user_outcome(
            run["output"],
            concept=payload.get("concept", ""),
            category=payload.get("category", ""),
            unit_of_measurement=payload.get("unit_of_measurement", ""),
            instrument=payload.get("instrument", ""),
            rationale=payload.get("rationale", ""))
    except ValueError as exc:
        return 400, {"error": str(exc)}
    return 200, {"outcome": outcome.to_dict(), "output": run["output"].to_dict()}


def source_preflight(payload: JSON) -> Tuple[int, JSON]:
    """Source-list coverage BEFORE any retrieval runs.

    This is the report that would have surfaced, on day one, that the workbook
    the Data Team maintains was not the workbook the code read.
    """
    path = payload.get("source_workbook") or C.SOURCE_WORKBOOK_PATH
    # A query string gives `areas` as a scalar. Passing a bare string into a
    # membership check iterates it character by character and silently matches
    # nothing — the filter appears to work and returns an empty set.
    raw_areas = payload.get("areas") or ["Oncology"]
    if isinstance(raw_areas, str):
        raw_areas = [a.strip() for a in raw_areas.split(",") if a.strip()]
    areas = [C.canonicalize_area(a) or a for a in raw_areas]
    try:
        inv = load_source_inventory(path, strict=False)
    except Exception as exc:  # noqa: BLE001
        return 400, {"error": str(exc)}

    per_state = []
    for state in C.EU_27_MEMBER_STATES:
        hta = inv.domains(C.SRC_HTA_REGULATORY, member_state=state, include_shared=False)
        guide_all = inv.domains(C.SRC_CLINICAL_GUIDELINE, member_state=state)
        guide_area = inv.domains(C.SRC_CLINICAL_GUIDELINE, member_state=state, areas=areas)
        per_state.append({
            "member_state": state, "code": C.MEMBER_STATE_CODE[state],
            "hta_domains": len(hta), "guideline_domains_all_areas": len(guide_all),
            "guideline_domains_in_area": len(guide_area),
            "area_relevant_share": (round(len(guide_area) / len(guide_all), 3)
                                     if guide_all else None),
            "has_hta_source": bool(hta),
        })

    by_class: Dict[str, int] = {}
    for e in inv.entries:
        by_class[e.source_class] = by_class.get(e.source_class, 0) + 1

    return 200, {
        "workbook": inv.path,
        "sheets_seen": inv.sheets_seen,
        "sheets_matched": inv.sheets_matched,
        "entries": len(inv.entries),
        "unique_domains": len(inv.unique_domains),
        "entries_by_source_class": by_class,
        "document_urls": sum(1 for e in inv.entries if e.is_document),
        "hub_urls": sum(1 for e in inv.entries if not e.is_document),
        "areas_applied": areas,
        "states_without_hta_source": [s["member_state"] for s in per_state
                                      if not s["has_hta_source"]],
        "per_member_state": per_state,
        "problems": [p.to_dict() for p in inv.problems],
    }


def list_prompts_handler(_: JSON) -> Tuple[int, JSON]:
    return 200, {"prompts": prompt_registry.list_prompts(),
                 "note": ("SME-editable prompts should be promoted through a "
                          "regression gate on a frozen eval set with retrieval "
                          "cached, or retrieval variance swamps the prompt effect.")}


ROUTES: Dict[Tuple[str, str], Callable[[JSON], Tuple[int, JSON]]] = {
    ("GET", "/api/schema"): get_schema,
    ("POST", "/api/validate"): validate,
    ("POST", "/api/runs"): create_run,
    ("GET", "/api/runs"): list_runs,
    ("GET", "/api/run"): get_run,               # ?request_id=...
    ("POST", "/api/comparators"): add_comparator,
    ("POST", "/api/outcomes"): add_outcome,
    ("GET", "/api/sources/preflight"): source_preflight,
    ("GET", "/api/prompts"): list_prompts_handler,
}
