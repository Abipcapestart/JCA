"""
Source-list loader — schema-tolerant, hyperlink-aware, therapeutic-area-aware.

This module exists because of three proven defects in the legacy loaders:

  1. The workbook path was hardcoded to the wrong file, so Data Team edits
     reached nothing.
  2. Sheet names were matched exactly ("SoC" vs "Standard of Care  Guidelines",
     singular vs plural), and a non-match hit a silent `continue`.
  3. The Standard-of-Care sheet stores every URL as a CELL HYPERLINK. The
     legacy loaders iterate with `values_only=True`, which discards hyperlink
     objects, so all 271 URLs were structurally invisible — not mis-addressed,
     unreadable.

So this loader:
  * matches sheets fuzzily (case, whitespace, singular/plural, punctuation);
  * finds the URL column by HEADER NAME, not a fixed index;
  * supports BOTH the flat layout (Country/Department/Link Label/URL) and the
    matrix layout (countries x therapeutic-area columns);
  * reads `cell.hyperlink.target` as well as cell text;
  * FAILS LOUDLY — a sheet that cannot be matched, or that yields zero URLs,
    raises or records a prominent problem. Silence is what hid this for months.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse

import openpyxl

from .. import config as C


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://[^\s;,)\"'<>\]]+", re.IGNORECASE)


def normalise_domain(value: str) -> str:
    """Registrable host, lowercased, `www.` stripped.

    Accepts EITHER a full URL or a bare domain. Both forms occur: the workbook
    holds URLs, while include_domains lists and the balanced-selection helper
    pass bare domains. Assuming a URL made this return "" for every bare domain,
    which silently emptied every domain-scoped search — the failure is invisible
    because an empty include_domains list looks exactly like "nothing found".
    """
    text = str(value or "").strip()
    if not text:
        return ""
    if "//" not in text:
        text = "//" + text          # let urlparse treat it as a netloc
    try:
        netloc = urlparse(text).netloc.lower()
    except Exception:
        return ""
    netloc = netloc.split("@")[-1].split(":")[0]
    return netloc[4:] if netloc.startswith("www.") else netloc


def _norm_label(text: str) -> str:
    """Collapse whitespace, strip punctuation and a trailing plural 's'.
    'Standard of Care  Guidelines' and 'standard of care guideline' match."""
    t = re.sub(r"[^a-z0-9 ]+", " ", str(text or "").lower())
    t = " ".join(t.split())
    return re.sub(r"s\b", "", t)


def is_document_url(url: str) -> bool:
    """A deep/document URL can be extracted directly. A shallow/hub URL needs a
    site-scoped search first — the distinction that decides retrieval routing."""
    try:
        path = urlparse(str(url)).path or "/"
    except Exception:
        return False
    if re.search(r"\.(pdf|docx?|pptx?|xlsx?)$", path, re.IGNORECASE):
        return True
    segments = [s for s in path.split("/") if s]
    return len(segments) >= 3


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class SourceEntry:
    url: str
    domain: str
    sheet: str
    source_class: str
    member_state: Optional[str] = None      # None => applies to all states
    therapeutic_area: Optional[str] = None  # None => applies to all areas
    label: str = ""
    organization: str = ""
    is_document: bool = False
    row: int = 0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class LoadProblem:
    severity: str        # ERROR | WARNING
    where: str
    message: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass
class SourceInventory:
    path: str = ""
    entries: List[SourceEntry] = field(default_factory=list)
    problems: List[LoadProblem] = field(default_factory=list)
    sheets_seen: List[str] = field(default_factory=list)
    sheets_matched: Dict[str, str] = field(default_factory=dict)  # role -> actual name

    # -- queries the retrieval planner uses ---------------------------------

    def domains(self, source_class: Optional[str] = None,
                member_state: Optional[str] = None,
                areas: Optional[Iterable[str]] = None,
                include_shared: bool = True) -> List[str]:
        """Domains for a (source class, member state, therapeutic area) slice.

        `areas` is the fix for the single largest measured defect: without it,
        a per-state guideline search is handed every specialty's society sites,
        and only ~11% of the domains offered are relevant to the request.
        """
        area_set = {a for a in (areas or []) if a}
        out: List[str] = []
        for e in self.entries:
            if source_class and e.source_class != source_class:
                continue
            if member_state is not None:
                if e.member_state is not None and e.member_state != member_state:
                    continue
                if e.member_state is None and not include_shared:
                    continue
            if area_set and e.therapeutic_area is not None:
                if e.therapeutic_area not in area_set:
                    continue
            if e.domain and e.domain not in out:
                out.append(e.domain)
        return out

    def document_urls(self, source_class: Optional[str] = None,
                      member_state: Optional[str] = None,
                      areas: Optional[Iterable[str]] = None) -> List[SourceEntry]:
        """Entries that are genuine document URLs — extract these directly
        rather than searching for them."""
        area_set = {a for a in (areas or []) if a}
        out = []
        for e in self.entries:
            if not e.is_document:
                continue
            if source_class and e.source_class != source_class:
                continue
            if member_state is not None and e.member_state not in (None, member_state):
                continue
            if area_set and e.therapeutic_area is not None and e.therapeutic_area not in area_set:
                continue
            out.append(e)
        return out

    def states_with_source(self, source_class: str) -> Set[str]:
        return {e.member_state for e in self.entries
                if e.source_class == source_class and e.member_state}

    @property
    def unique_domains(self) -> List[str]:
        return sorted({e.domain for e in self.entries if e.domain})

    @property
    def has_errors(self) -> bool:
        return any(p.severity == "ERROR" for p in self.problems)

    def to_dict(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "entry_count": len(self.entries),
            "unique_domains": len(self.unique_domains),
            "sheets_seen": self.sheets_seen,
            "sheets_matched": self.sheets_matched,
            "problems": [p.to_dict() for p in self.problems],
        }


# ---------------------------------------------------------------------------
# Sheet-role matching
# ---------------------------------------------------------------------------

# role -> candidate labels. Matched on the normalised form, so plural/singular,
# double spaces and punctuation differences all resolve.
_SHEET_ROLES: Dict[str, Tuple[str, ...]] = {
    "soc": ("standard of care guidelines", "standard of care guideline",
            "standard of care", "soc", "clinical guidelines", "guidelines"),
    "hta": ("hta licensing sources", "hta licensing source", "hta sources",
            "hta bodies", "hta"),
    "emea": ("emea licensing sources", "emea licensing source", "ema licensing sources",
             "emea sources", "ema sources", "emea", "ema"),
    "trials": ("clinical trials watch list", "clinical trial watch list",
               "clinical trials", "trial registries", "registries"),
    "conference": ("conference sources", "conference evidence sources",
                   "conference proceedings", "conference watch list"),
}

_ROLE_TO_SOURCE_CLASS = {
    "soc": C.SRC_CLINICAL_GUIDELINE,
    "hta": C.SRC_HTA_REGULATORY,
    "emea": C.SRC_DRUG_LABEL,
    "trials": C.SRC_TRIAL_REGISTRY,
    "conference": C.SRC_CONFERENCE,
}

# A curator can legitimately cite a PubMed-hosted paper as a guideline/HTA
# source (e.g. a national guideline published as a journal article). But the
# retrieval side treats PubMed as its own Tier 2 evidence class with its own
# retrieval path (retrieve_structured -> documents_from_publications) and its
# own peer-review handling -- a pubmed.ncbi.nlm.nih.gov domain must never
# surface as Tier 1, whichever sheet happened to cite it. Applied as a
# post-load correction so it holds regardless of sheet/role.
_FORCED_SOURCE_CLASS_BY_DOMAIN = {
    "pubmed.ncbi.nlm.nih.gov": C.SRC_PUBMED,
}

# Header names to look for, per role. Header-based, never a fixed index.
_URL_HEADERS = ("url", "website", "link", "web site", "source url", "homepage")
_COUNTRY_HEADERS = ("country", "countries", "member state", "eu jca member state")
_AREA_HEADERS = ("department", "therapeutic area", "area", "specialty")
_LABEL_HEADERS = ("link label", "label", "sub resource", "sub-resource", "purpose",
                  "official registry name", "details of the source")
_ORG_HEADERS = ("hta body full name", "parent organization", "parent organisation",
                "managing organization", "managing organisation", "acronym",
                "organisation", "organization")


def _match_role(sheet_name: str) -> Optional[str]:
    n = _norm_label(sheet_name)
    for role, candidates in _SHEET_ROLES.items():
        for cand in candidates:
            if n == _norm_label(cand):
                return role
    # Fall back to containment, so an unexpected suffix still resolves.
    for role, candidates in _SHEET_ROLES.items():
        for cand in candidates:
            c = _norm_label(cand)
            if c and (c in n or n in c):
                return role
    return None


def _header_index(headers: List[str], candidates: Iterable[str]) -> Optional[int]:
    norm = [_norm_label(h) for h in headers]
    for cand in candidates:
        c = _norm_label(cand)
        for i, h in enumerate(norm):
            if h == c:
                return i
    for cand in candidates:
        c = _norm_label(cand)
        for i, h in enumerate(norm):
            if c and c in h:
                return i
    return None


def _cell_urls(cell) -> List[str]:
    """Every URL a cell carries — hyperlink target AND text.

    This is the fix for defect 3. `cell.hyperlink.target` is the only place the
    Standard-of-Care URLs live; iterating with values_only=True cannot see them.
    """
    urls: List[str] = []
    link = getattr(cell, "hyperlink", None)
    if link is not None and getattr(link, "target", None):
        urls.append(str(link.target))
    if isinstance(cell.value, str):
        urls.extend(_URL_RE.findall(cell.value))
    # Some cells hold several URLs joined by 'or' / ';'
    cleaned, seen = [], set()
    for u in urls:
        u = u.strip().strip(",;")
        if not u.lower().startswith("http"):
            continue
        if u not in seen:
            seen.add(u)
            cleaned.append(u)
    return cleaned


# ---------------------------------------------------------------------------
# The loader
# ---------------------------------------------------------------------------

def load_source_inventory(path: Optional[str] = None,
                          strict: bool = True) -> SourceInventory:
    """Load every URL from every recognisable sheet of the source workbook.

    `strict=True` raises when a required sheet role cannot be matched or a
    matched sheet yields zero URLs. That is deliberate: the legacy behaviour —
    skip silently — is precisely how a wrong workbook produced an empty result
    indistinguishable from "no sources exist".
    """
    path = path or C.SOURCE_WORKBOOK_PATH
    inv = SourceInventory(path=path)

    if not os.path.exists(path):
        msg = f"Source workbook not found at {path!r}. Set JCA_SOURCE_WORKBOOK."
        inv.problems.append(LoadProblem("ERROR", "workbook", msg))
        if strict:
            raise FileNotFoundError(msg)
        return inv

    # NOT data_only, NOT values_only — hyperlinks must survive.
    wb = openpyxl.load_workbook(path)
    inv.sheets_seen = list(wb.sheetnames)

    for sheet_name in wb.sheetnames:
        role = _match_role(sheet_name)
        if role is None:
            inv.problems.append(LoadProblem(
                "WARNING", sheet_name,
                f"Sheet {sheet_name!r} does not match any known source role "
                f"({', '.join(_SHEET_ROLES)}); it was not loaded."))
            continue
        if role in inv.sheets_matched:
            inv.problems.append(LoadProblem(
                "WARNING", sheet_name,
                f"Second sheet matching role {role!r}; "
                f"{inv.sheets_matched[role]!r} already claimed it. Loading both."))
        else:
            inv.sheets_matched[role] = sheet_name

        before = len(inv.entries)
        _load_sheet(wb[sheet_name], role, inv)
        gained = len(inv.entries) - before
        if gained == 0:
            inv.problems.append(LoadProblem(
                "ERROR", sheet_name,
                f"Sheet {sheet_name!r} matched role {role!r} but yielded ZERO URLs. "
                f"Check the header row and whether URLs are stored as hyperlinks."))

    for entry in inv.entries:
        forced = _FORCED_SOURCE_CLASS_BY_DOMAIN.get(entry.domain)
        if forced and entry.source_class != forced:
            entry.source_class = forced

    for required in ("soc", "hta", "emea"):
        if required not in inv.sheets_matched:
            inv.problems.append(LoadProblem(
                "ERROR", "workbook",
                f"No sheet matched the required role {required!r}. "
                f"Sheets present: {inv.sheets_seen}"))

    if strict and inv.has_errors:
        detail = "; ".join(p.message for p in inv.problems if p.severity == "ERROR")
        raise ValueError(f"Source workbook failed to load cleanly: {detail}")
    return inv


def _load_sheet(ws, role: str, inv: SourceInventory) -> None:
    headers = [c.value for c in ws[1]]
    url_col = _header_index(headers, _URL_HEADERS)
    if url_col is not None:
        _load_flat_sheet(ws, role, inv, headers, url_col)
    else:
        # No URL column => the matrix layout (countries x therapeutic areas),
        # which is how the Data Team maintains the Standard-of-Care sheet.
        _load_matrix_sheet(ws, role, inv, headers)


def _load_flat_sheet(ws, role: str, inv: SourceInventory,
                     headers: List[str], url_col: int) -> None:
    country_col = _header_index(headers, _COUNTRY_HEADERS)
    area_col = _header_index(headers, _AREA_HEADERS)
    label_col = _header_index(headers, _LABEL_HEADERS)
    org_col = _header_index(headers, _ORG_HEADERS)
    source_class = _ROLE_TO_SOURCE_CLASS[role]

    for r_idx, row in enumerate(ws.iter_rows(min_row=2), start=2):
        if url_col >= len(row):
            continue
        urls = _cell_urls(row[url_col])
        if not urls:
            continue

        def _txt(i: Optional[int]) -> str:
            if i is None or i >= len(row) or row[i].value is None:
                return ""
            return str(row[i].value).strip()

        state = C.canonicalize_member_state(_txt(country_col)) if country_col is not None else None
        area = C.canonicalize_area(_txt(area_col)) if area_col is not None else None
        label = _txt(label_col)
        org = _txt(org_col)

        for u in urls:
            d = normalise_domain(u)
            if not d or "." not in d:
                continue
            inv.entries.append(SourceEntry(
                url=u, domain=d, sheet=ws.title, source_class=source_class,
                member_state=state, therapeutic_area=area, label=label,
                organization=org, is_document=is_document_url(u), row=r_idx))


def _load_matrix_sheet(ws, role: str, inv: SourceInventory,
                       headers: List[str]) -> None:
    """Countries down the first column, therapeutic areas across the header row.

    Every URL here is a cell hyperlink. The therapeutic area is the COLUMN, and
    capturing it is what makes the SME's area filter possible at all.
    """
    source_class = _ROLE_TO_SOURCE_CLASS[role]
    area_by_col: Dict[int, Optional[str]] = {}
    for i, h in enumerate(headers):
        if i == 0:
            continue
        area_by_col[i] = C.canonicalize_area(h)

    for r_idx, row in enumerate(ws.iter_rows(min_row=2), start=2):
        if not row or row[0].value is None:
            continue
        raw_country = str(row[0].value).strip()
        state = C.canonicalize_member_state(raw_country)
        # A row labelled e.g. "All 27 member states" is shared, not a state.
        shared_row = state is None

        for col_idx, cell in enumerate(row):
            if col_idx == 0:
                continue
            urls = _cell_urls(cell)
            if not urls:
                continue
            area = area_by_col.get(col_idx)
            if area is None:
                inv.problems.append(LoadProblem(
                    "WARNING", ws.title,
                    f"Column {col_idx + 1} header "
                    f"{str(headers[col_idx] if col_idx < len(headers) else '')!r} "
                    f"is not a recognised therapeutic area; its links are loaded "
                    f"without an area filter and will be searched for every request."))
            label = str(cell.value).strip().splitlines()[0] if isinstance(cell.value, str) else ""
            for u in urls:
                d = normalise_domain(u)
                if not d or "." not in d:
                    continue
                inv.entries.append(SourceEntry(
                    url=u, domain=d, sheet=ws.title, source_class=source_class,
                    member_state=None if shared_row else state,
                    therapeutic_area=area, label=label[:120],
                    organization="", is_document=is_document_url(u), row=r_idx))
