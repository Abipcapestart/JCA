"""
On-disk version history for SME-edited prompts.

This sits ON TOP of jca_phase1.prompts.registry's existing runtime-override
mechanism (register_override / clear_overrides / get) -- it does not replace
it and does not touch jca_phase1/prompts/registry.py. The registry's built-in
PROMPTS dict is read directly (never mutated) as the "base" version every
prompt starts from; every SME edit becomes a new numbered version stored here.

Layout (created on first use):
    streamlit_app/prompt_versions/<prompt_id>/v1.json, v2.json, ...
    streamlit_app/prompt_versions/<prompt_id>/active.json   -- {"active_version": "v2"}

Each version file: {"text": "...", "created_at": "ISO8601", "note": ""}
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from jca_phase1.prompts import registry as prompt_registry

STORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_versions")

BASE_VERSION = "base"


def all_prompt_ids() -> List[str]:
    return sorted(prompt_registry.PROMPTS.keys())


def prompt_meta(prompt_id: str) -> Dict[str, Any]:
    """purpose / sme_editable, straight from the registry -- never duplicated."""
    return prompt_registry.list_prompts()[prompt_id]


def base_text(prompt_id: str) -> str:
    """The shipped, unedited prompt text. Reads PROMPTS directly so this never
    depends on (or clears) any runtime override currently in effect."""
    return prompt_registry.PROMPTS[prompt_id]["text"]


def _prompt_dir(prompt_id: str) -> str:
    d = os.path.join(STORE_DIR, prompt_id)
    os.makedirs(d, exist_ok=True)
    return d


def _version_sort_key(v: str):
    m = re.match(r"v(\d+)$", v)
    return int(m.group(1)) if m else -1


def list_versions(prompt_id: str) -> List[str]:
    """[BASE_VERSION, 'v1', 'v2', ...] oldest first."""
    d = _prompt_dir(prompt_id)
    saved = sorted(
        (f[:-5] for f in os.listdir(d) if f.endswith(".json") and f != "active.json"),
        key=_version_sort_key)
    return [BASE_VERSION] + saved


def get_text(prompt_id: str, version: str) -> str:
    if version == BASE_VERSION:
        return base_text(prompt_id)
    path = os.path.join(_prompt_dir(prompt_id), f"{version}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["text"]


def get_version_detail(prompt_id: str, version: str) -> Dict[str, Any]:
    if version == BASE_VERSION:
        return {"text": base_text(prompt_id), "created_at": None, "note": "shipped default"}
    path = os.path.join(_prompt_dir(prompt_id), f"{version}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_version(prompt_id: str, text: str, note: str = "") -> str:
    """Persist a new SME-edited version. Returns the new version string."""
    existing = [v for v in list_versions(prompt_id) if v != BASE_VERSION]
    next_n = max((_version_sort_key(v) for v in existing), default=0) + 1
    version = f"v{next_n}"
    path = os.path.join(_prompt_dir(prompt_id), f"{version}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"text": text, "created_at": datetime.now(timezone.utc).isoformat(),
                  "note": note}, f, indent=2)
    set_active_version(prompt_id, version)
    return version


def get_active_version(prompt_id: str) -> str:
    path = os.path.join(_prompt_dir(prompt_id), "active.json")
    if not os.path.exists(path):
        return BASE_VERSION
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("active_version", BASE_VERSION)


def set_active_version(prompt_id: str, version: str) -> None:
    path = os.path.join(_prompt_dir(prompt_id), "active.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"active_version": version}, f, indent=2)


def active_overrides() -> Dict[str, Dict[str, str]]:
    """{prompt_id: {"version": ..., "text": ...}} for every prompt whose active
    version differs from the shipped base -- exactly what needs registering as
    a runtime override before a run."""
    out = {}
    for prompt_id in all_prompt_ids():
        version = get_active_version(prompt_id)
        if version != BASE_VERSION:
            out[prompt_id] = {"version": version, "text": get_text(prompt_id, version)}
    return out
