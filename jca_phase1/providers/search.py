"""
Search / document-retrieval providers.

Three things the legacy engine did not do, all of which are implemented here:

  * RETRY. The legacy Tavily wrapper made one attempt and swallowed any
    exception into `{}`, so a transient rate-limit was indistinguishable from
    "nothing found". That is a plausible driver of the measured run-to-run
    variance. Here a failure is retried, and a final failure is reported as
    RETRIEVAL_FAILED, distinct from "searched, nothing found".
  * A DOCUMENT CACHE. The same URL was previously fetched by extraction, again
    by the retry path, and again by re-verification.
  * FULL-DOCUMENT READS for guidelines and HTA assessments. A guideline's
    recommendation table and an HTA report's scope table are never in the top
    relevance chunks.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .. import config as C
from ..schema import RetrievedDocument
from ..sources.workbook import normalise_domain

LOGGER = logging.getLogger(__name__)


def _debug_log_tavily(op: str, params: Dict[str, Any], resp: Any) -> None:
    """Print the raw Tavily request/response when JCA_DEBUG_TAVILY=1, so an
    SME or developer can see exactly what the API sent back before this
    module trims it down to a SearchHit/RetrievedDocument."""
    if not C.DEBUG_LOG_TAVILY_RESPONSES:
        return
    print(f"\n=== [Tavily {op}] request ===")
    print(json.dumps(params, indent=2, ensure_ascii=False))
    print(f"=== [Tavily {op}] raw response ===")
    print(json.dumps(resp, indent=2, ensure_ascii=False))
    print(f"=== [Tavily {op}] end ===\n")


@dataclass
class SearchHit:
    url: str
    title: str = ""
    snippet: str = ""
    score: float = 0.0


@dataclass
class ProviderCall:
    op: str
    detail: str
    latency_s: float = 0.0
    attempts: int = 1
    error: str = ""
    results: int = 0
    credits: float = 0.0


class SearchProvider:
    """Base class. `search` discovers, `fetch` turns a URL into text."""

    def __init__(self) -> None:
        self.call_log: List[ProviderCall] = []
        self._doc_cache: Dict[str, RetrievedDocument] = {}
        self._credits_local = threading.local()
        # Optional (level, message) sink for retries/failures, in addition to
        # the LOGGER calls below which always fire regardless -- set by a UI
        # layer that wants to show these live, not just in the server log.
        self.on_event: Optional[Callable[[str, str], None]] = None
        # Global pacing across the whole thread pool. A provider's rate limit
        # (e.g. Tavily free tier) is enforced on the API key, not per
        # connection -- reducing max_workers alone still lets N threads each
        # fire immediately and burst past it. This makes every call, from
        # every thread, wait its turn before hitting the network.
        self._rate_lock = threading.Lock()
        self._next_call_at = 0.0

    def _throttle(self) -> None:
        min_interval = C.RETRIEVAL.search_min_interval_seconds
        if min_interval <= 0:
            return
        with self._rate_lock:
            wait = self._next_call_at - time.time()
            if wait > 0:
                time.sleep(wait)
            self._next_call_at = time.time() + min_interval

    def _emit(self, level: str, message: str) -> None:
        (LOGGER.warning if level == "retry" else LOGGER.error)(message)
        if self.on_event is not None:
            try:
                self.on_event(level, message)
            except Exception:  # noqa: BLE001 - a broken UI hook must never break a run
                pass

    def _record_credits(self, credits: float) -> None:
        """Retrieval runs on a thread pool; thread-local avoids one call's
        credit figure leaking onto another's log entry."""
        self._credits_local.value = credits

    def _pop_credits(self) -> float:
        credits = getattr(self._credits_local, "value", 0.0)
        self._credits_local.value = 0.0
        return credits

    # -- public -------------------------------------------------------------

    def search(self, query: str, domains: List[str], max_results: int) -> List[SearchHit]:
        return self._retry("search", f"{query} @{len(domains)} domains",
                           lambda: self._search(query, domains, max_results))

    def fetch(self, url: str, query: str = "", full_document: bool = False,
              use_cache: bool = True) -> RetrievedDocument:
        key = f"{url}|{'full' if full_document else 'scoped'}"
        if use_cache and C.RETRIEVAL.enable_document_cache and key in self._doc_cache:
            return self._doc_cache[key]
        try:
            doc = self._retry("fetch", url, lambda: self._fetch(url, query, full_document))
        except Exception as exc:  # noqa: BLE001
            doc = RetrievedDocument(url=url, ok=False, status=C.EV_RETRIEVAL_FAILED,
                                    error=f"{type(exc).__name__}: {exc}")
        if C.RETRIEVAL.enable_document_cache:
            self._doc_cache[key] = doc
        return doc

    def usage_summary(self) -> Dict[str, Any]:
        return {
            "calls": len(self.call_log),
            "failed_calls": sum(1 for c in self.call_log if c.error),
            "searches": sum(1 for c in self.call_log if c.op == "search"),
            "fetches": sum(1 for c in self.call_log if c.op == "fetch"),
            "cached_documents": len(self._doc_cache),
            "latency_s": round(sum(c.latency_s for c in self.call_log), 2),
            "credits": round(sum(c.credits for c in self.call_log), 3),
        }

    # -- internals ----------------------------------------------------------

    def _retry(self, op: str, detail: str, fn: Callable[[], Any]) -> Any:
        started = time.time()
        last = ""
        for attempt in range(1, C.RETRIEVAL.search_max_attempts + 1):
            try:
                self._throttle()
                result = fn()
                self.call_log.append(ProviderCall(
                    op=op, detail=detail[:160], latency_s=round(time.time() - started, 3),
                    attempts=attempt,
                    results=len(result) if isinstance(result, list) else 1,
                    credits=self._pop_credits()))
                return result
            except Exception as exc:  # noqa: BLE001
                last = f"{type(exc).__name__}: {exc}"
                if attempt == C.RETRIEVAL.search_max_attempts:
                    break
                self._emit("retry", f"[Tavily {op}] {detail[:120]}: attempt {attempt}/"
                                    f"{C.RETRIEVAL.search_max_attempts} failed - {last} - retrying")
                time.sleep(C.RETRIEVAL.search_backoff_seconds * (2 ** (attempt - 1)))
        self._emit("error", f"[Tavily {op}] {detail[:120]}: failed after "
                            f"{C.RETRIEVAL.search_max_attempts} attempts - {last}")
        self.call_log.append(ProviderCall(
            op=op, detail=detail[:160], latency_s=round(time.time() - started, 3),
            attempts=C.RETRIEVAL.search_max_attempts, error=last))
        if op == "search":
            # A failed search is NOT an empty search. Callers inspect the log.
            return []
        raise RuntimeError(last)

    def _search(self, query: str, domains: List[str], max_results: int) -> List[SearchHit]:
        raise NotImplementedError

    def _fetch(self, url: str, query: str, full_document: bool) -> RetrievedDocument:
        raise NotImplementedError


def select_balanced(hits: List[SearchHit], max_urls: int,
                    domain_groups: Dict[str, List[str]]) -> List[str]:
    """Pick up to `max_urls` URLs, guaranteeing at least one slot per domain
    group that actually returned something.

    Merging several domain categories into one ranked pool halves search cost,
    but a blind top-k slice lets one category crowd another out entirely. This
    keeps the saving without the starvation.
    """
    if not hits:
        return []
    if len(domain_groups) <= 1:
        return [h.url for h in hits[:max_urls]]

    by_group: Dict[str, List[str]] = {g: [] for g in domain_groups}
    ungrouped: List[str] = []
    for h in hits:
        d = normalise_domain(h.url)
        placed = False
        for group, domains in domain_groups.items():
            if any(d == nd or d.endswith("." + nd) for nd in domains):
                by_group[group].append(h.url)
                placed = True
                break
        if not placed:
            ungrouped.append(h.url)

    active = [g for g, urls in by_group.items() if urls]
    selected: List[str] = []
    if active:
        per_group = max(1, max_urls // len(active))
        for g in active:
            selected.extend(by_group[g][:per_group])
    # Backfill in original rank order.
    for h in hits:
        if len(selected) >= max_urls:
            break
        if h.url not in selected:
            selected.append(h.url)
    return selected[:max_urls]


# ---------------------------------------------------------------------------
# Production: Tavily
# ---------------------------------------------------------------------------

class TavilySearchProvider(SearchProvider):
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        import os
        self.api_key = api_key or os.getenv("TAVILY_API_KEY", "")
        self._client = None

    def _lazy_client(self):
        if self._client is None:
            from tavily import TavilyClient
            if not self.api_key:
                raise RuntimeError("TAVILY_API_KEY is not set")
            self._client = TavilyClient(api_key=self.api_key)
        return self._client

    def _search(self, query: str, domains: List[str], max_results: int) -> List[SearchHit]:
        params: Dict[str, Any] = {
            "query": query,
            "search_depth": "basic",
            "chunks_per_source": 3,
            "max_results": max_results,
            "include_usage": True,
        }
        if domains:
            params["include_domains"] = domains[:300]  # documented API ceiling
        resp = self._lazy_client().search(**params)
        _debug_log_tavily("search", params, resp)
        self._record_credits((resp or {}).get("usage", {}).get("credits", 0) or 0)
        hits = []
        for r in (resp or {}).get("results", []) or []:
            hits.append(SearchHit(url=r.get("url", ""), title=r.get("title", ""),
                                  snippet=r.get("content", ""), score=r.get("score", 0.0)))
        return [h for h in hits if h.url]

    def _fetch(self, url: str, query: str, full_document: bool) -> RetrievedDocument:
        # "basic" extract_depth is documented to handle PDFs and complex
        # table layouts poorly -- confirmed in production: a real guideline
        # PDF fetched with full_document=True (the exact signal this
        # codebase uses for "this source's real content is a table never in
        # the top relevance chunks, read it whole") came back as ~3KB of
        # generic template boilerplate, never reaching the actual
        # recommendations. "advanced" costs more Tavily credits but is the
        # documented fix for PDF/table extraction, and full_document is
        # already a deliberately small subset of all fetches.
        params: Dict[str, Any] = {"urls": [url],
                                  "extract_depth": "advanced" if full_document else "basic",
                                  "format": "markdown", "include_usage": True}
        if query and not full_document:
            params["query"] = query
            params["chunks_per_source"] = 5
        resp = self._lazy_client().extract(**params)
        _debug_log_tavily("extract", params, resp)
        self._record_credits((resp or {}).get("usage", {}).get("credits", 0) or 0)
        results = (resp or {}).get("results", []) or []
        if not results:
            return RetrievedDocument(url=url, ok=False, status=C.EV_SOURCE_INACCESSIBLE,
                                     error="extract returned no results")
        first = results[0]
        text = first.get("raw_content") or first.get("content") or ""
        resolved = first.get("url", url)
        # Low enough to still catch a genuinely empty/error response, but not
        # so high it marks a legitimately short, single-fact page (a brief
        # national HTA decision notice, a "no assessment on file"
        # confirmation -- itself informative) as inaccessible just because
        # it's short. Was 500, which is easily longer than a real short page.
        if len(text) < 100:
            return RetrievedDocument(url=url, resolved_url=resolved, text=text, ok=False,
                                     status=C.EV_SOURCE_INACCESSIBLE,
                                     method="tavily_extract",
                                     error=f"only {len(text)} chars retrieved")
        return RetrievedDocument(
            url=url, resolved_url=resolved, text=text, ok=True,
            status=C.EV_FOUND,
            method="tavily_extract_full" if full_document else "tavily_extract",
            title=first.get("title", ""))


# ---------------------------------------------------------------------------
# Deterministic fake
# ---------------------------------------------------------------------------

class FixtureSearchProvider(SearchProvider):
    """Serves search hits and documents from an in-memory fixture.

    `index`  : {domain: [SearchHit, ...]}  — what a domain-scoped search returns
    `documents`: {url: text}               — what fetching that URL returns
    """

    def __init__(self, index: Optional[Dict[str, List[SearchHit]]] = None,
                 documents: Optional[Dict[str, str]] = None,
                 fail_domains: Optional[List[str]] = None):
        super().__init__()
        self.index = index or {}
        self.documents = documents or {}
        self.fail_domains = set(fail_domains or [])

    def _search(self, query: str, domains: List[str], max_results: int) -> List[SearchHit]:
        wanted = [normalise_domain(d) for d in domains] if domains else list(self.index)
        for d in wanted:
            if d in self.fail_domains:
                raise RuntimeError(f"simulated search failure for {d}")
        hits: List[SearchHit] = []
        for d in wanted:
            hits.extend(self.index.get(d, []))
        return hits[:max_results]

    def _fetch(self, url: str, query: str, full_document: bool) -> RetrievedDocument:
        if normalise_domain(url) in self.fail_domains:
            raise RuntimeError(f"simulated fetch failure for {url}")
        text = self.documents.get(url)
        if text is None:
            return RetrievedDocument(url=url, ok=False, status=C.EV_SOURCE_INACCESSIBLE,
                                     error="not in fixture")
        return RetrievedDocument(url=url, resolved_url=url, text=text, ok=True,
                                 status=C.EV_FOUND,
                                 method="fixture_full" if full_document else "fixture")
