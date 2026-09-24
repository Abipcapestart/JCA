"""
LLM provider interface.

Two implementations:
  * BedrockLLM  — production, with retry/backoff and per-call accounting.
  * ScriptedLLM — a deterministic fake for tests and offline runs. It answers
    from a fixture map keyed by prompt id, so the full pipeline can be exercised
    end to end with no credentials and no network.

Every call goes through `LLMClient.call()`, which records the prompt id and the
prompt VERSION into the call log. That log becomes part of the run manifest, so
a run can always be traced to the exact prompt text that produced it.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .. import config as C
from ..prompts import registry as prompt_registry

LOGGER = logging.getLogger(__name__)


@dataclass
class LLMCall:
    prompt_id: str
    prompt_version: str
    input_tokens: int = 0
    output_tokens: int = 0
    # Anthropic prompt-caching accounting -- 0 for every prompt not marked
    # cacheable, and for a cacheable one until enough calls have happened to
    # actually populate/hit the cache. Present here so caching's real effect
    # is measured, not assumed.
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    latency_s: float = 0.0
    attempts: int = 1
    error: str = ""
    # Captured so a UI can replay/inspect a real call later (e.g. re-testing an
    # edited prompt against a genuine past payload without re-running retrieval).
    system_prompt: str = ""
    user_prompt: str = ""
    response_text: str = ""


class LLMError(RuntimeError):
    pass


class LLMClient:
    """Base class. Subclasses implement `_invoke`."""

    def __init__(self) -> None:
        self.call_log: List[LLMCall] = []
        # Optional (level, message) sink for retries/failures, in addition to
        # the LOGGER calls below which always fire regardless -- set by a UI
        # layer that wants to show these live, not just in the server log.
        self.on_event: Optional[Callable[[str, str], None]] = None

    def _emit(self, level: str, message: str) -> None:
        (LOGGER.warning if level == "retry" else LOGGER.error)(message)
        if self.on_event is not None:
            try:
                self.on_event(level, message)
            except Exception:  # noqa: BLE001 - a broken UI hook must never break a run
                pass

    # -- public -------------------------------------------------------------

    def call(self, prompt_id: str, user_prompt: str,
             max_tokens: int = C.LLM.default_max_tokens,
             system_override: Optional[str] = None) -> str:
        system, version = prompt_registry.get(prompt_id)
        cacheable = prompt_registry.is_cacheable(prompt_id)
        if system_override is not None:
            system, version = system_override, "override"
        started = time.time()
        last_error = ""
        for attempt in range(1, C.LLM.max_attempts + 1):
            try:
                text, usage = self._invoke(system, user_prompt, max_tokens, prompt_id,
                                           cacheable=cacheable)
                self.call_log.append(LLMCall(
                    prompt_id=prompt_id, prompt_version=version,
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
                    cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
                    latency_s=round(time.time() - started, 3), attempts=attempt,
                    system_prompt=system, user_prompt=user_prompt, response_text=text))
                return text
            except Exception as exc:  # noqa: BLE001 - retried below
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt == C.LLM.max_attempts:
                    break
                self._emit("retry", f"[Bedrock] {prompt_id}: attempt {attempt}/"
                                    f"{C.LLM.max_attempts} failed - {last_error} - retrying")
                time.sleep(C.LLM.backoff_seconds * (2 ** (attempt - 1)))
        self._emit("error", f"[Bedrock] {prompt_id}: failed after "
                            f"{C.LLM.max_attempts} attempts - {last_error}")
        self.call_log.append(LLMCall(prompt_id=prompt_id, prompt_version=version,
                                     latency_s=round(time.time() - started, 3),
                                     attempts=C.LLM.max_attempts, error=last_error,
                                     system_prompt=system, user_prompt=user_prompt))
        raise LLMError(f"{prompt_id} failed after {C.LLM.max_attempts} attempts: {last_error}")

    def call_json(self, prompt_id: str, user_prompt: str,
                  max_tokens: int = C.LLM.default_max_tokens,
                  default: Any = None) -> Any:
        """Call and parse JSON. Returns `default` rather than raising when the
        response cannot be parsed — every caller has a fail-safe path, because
        a malformed batch response must never silently drop findings."""
        try:
            raw = self.call(prompt_id, user_prompt, max_tokens=max_tokens)
        except LLMError:
            return default
        return parse_json(raw, default)

    def usage_summary(self) -> Dict[str, Any]:
        return {
            "calls": len(self.call_log),
            "failed_calls": sum(1 for c in self.call_log if c.error),
            "input_tokens": sum(c.input_tokens for c in self.call_log),
            "output_tokens": sum(c.output_tokens for c in self.call_log),
            "cache_creation_input_tokens": sum(
                c.cache_creation_input_tokens for c in self.call_log),
            "cache_read_input_tokens": sum(c.cache_read_input_tokens for c in self.call_log),
            "latency_s": round(sum(c.latency_s for c in self.call_log), 2),
            "by_prompt": _count_by(self.call_log, lambda c: c.prompt_id),
            "prompt_versions": {c.prompt_id: c.prompt_version for c in self.call_log},
        }

    # -- to implement -------------------------------------------------------

    def _invoke(self, system: str, user: str, max_tokens: int,
                prompt_id: str, cacheable: bool = False) -> "tuple[str, Dict[str, int]]":
        raise NotImplementedError


def _count_by(items, key) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for i in items:
        k = key(i)
        out[k] = out.get(k, 0) + 1
    return out


def parse_json(raw: str, default: Any = None) -> Any:
    """Tolerant JSON extraction: strips code fences and finds the outermost
    object or array.

    The order of the two patterns matters. Searching for `{...}` first on a
    response that is a JSON ARRAY of objects matches the first inner object and
    returns a dict — so an extraction call that correctly returned a list of
    findings silently yields one malformed dict, and every caller that checks
    `isinstance(parsed, list)` discards the whole document. The delimiter is
    chosen from the first non-whitespace character instead of guessed.
    """
    cleaned = re.sub(r"```(?:json)?", "", raw or "").strip()
    if not cleaned:
        return default

    # Fast path: the response is already clean JSON.
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        pass

    first = next((ch for ch in cleaned if ch in "[{"), "")
    patterns = (r"(\[.*\])", r"(\{.*\})") if first == "[" else (r"(\{.*\})", r"(\[.*\])")
    for pattern in patterns:
        m = re.search(pattern, cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except (json.JSONDecodeError, TypeError):
                continue
    return default


# ---------------------------------------------------------------------------
# Production
# ---------------------------------------------------------------------------

class BedrockLLM(LLMClient):
    def __init__(self, model_id: Optional[str] = None, region: Optional[str] = None):
        super().__init__()
        self.model_id = model_id or C.LLM.model_id
        self.region = region or C.LLM.region
        self._client = None

    def _lazy_client(self):
        if self._client is None:
            import boto3
            from botocore.config import Config as BotoConfig
            self._client = boto3.client(
                "bedrock-runtime", region_name=self.region,
                # invoke_model is synchronous -- the client blocks for the
                # full generation time, so read_timeout must comfortably
                # exceed how long the largest configured max_tokens (see
                # config.py's LLMSettings) can plausibly take to generate.
                # Raised alongside extraction_max_tokens's 16000->24000 bump
                # so a genuinely long generation is a slow success, not a
                # read-timeout failure that then burns a full retry cycle.
                config=BotoConfig(read_timeout=450, connect_timeout=20,
                                  retries={"max_attempts": 0}))
        return self._client

    def _invoke(self, system: str, user: str, max_tokens: int, prompt_id: str,
               cacheable: bool = False):
        import json as _json
        client = self._lazy_client()
        # Anthropic prompt caching (native cache_control, not Bedrock's
        # Converse-only cachePoint -- this uses invoke_model with the raw
        # Anthropic Messages body). Only worth it for a prompt whose static
        # system text is read many times per run; see prompts/registry.py's
        # "cacheable" flag for which ones qualify. Below Anthropic's per-model
        # minimum cacheable length, the block is silently not cached -- no
        # error, just no benefit -- so this is always safe to attempt.
        system_field = ([{"type": "text", "text": system,
                          "cache_control": {"type": "ephemeral"}}]
                        if cacheable else system)
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system_field,
            "messages": [{"role": "user", "content": [{"type": "text", "text": user}]}],
        }
        resp = client.invoke_model(modelId=self.model_id, body=_json.dumps(body))
        payload = _json.loads(resp["body"].read())
        text = "".join(b.get("text", "") for b in payload.get("content", []))
        usage = payload.get("usage", {}) or {}
        return text, {"input_tokens": usage.get("input_tokens", 0),
                      "output_tokens": usage.get("output_tokens", 0),
                      "cache_creation_input_tokens": usage.get(
                          "cache_creation_input_tokens", 0),
                      "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0)}


# ---------------------------------------------------------------------------
# Deterministic fake
# ---------------------------------------------------------------------------

class ScriptedLLM(LLMClient):
    """Answers from a `{prompt_id: callable-or-literal}` map.

    A callable receives the user prompt and returns the response string, so a
    fixture can react to its input — enough to exercise batching, index
    accounting and fail-safe paths without a network.
    """

    def __init__(self, responses: Optional[Dict[str, Any]] = None,
                 default: str = "{}"):
        super().__init__()
        self.responses: Dict[str, Any] = responses or {}
        self.default = default
        self.seen: List[tuple] = []

    def _invoke(self, system: str, user: str, max_tokens: int, prompt_id: str,
               cacheable: bool = False):
        self.seen.append((prompt_id, user))
        handler = self.responses.get(prompt_id, self.default)
        text = handler(user) if callable(handler) else handler
        return text, {"input_tokens": len(system) // 4 + len(user) // 4,
                      "output_tokens": len(str(text)) // 4}
