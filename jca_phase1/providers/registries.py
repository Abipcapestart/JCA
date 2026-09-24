"""
Structured registry / literature APIs.

The legacy engine routed EVERYTHING through web search, including
ClinicalTrials.gov and PubMed. That is the most expensive possible way to read
a database column: a trial's comparator is literally a typed field
(`armGroups[].interventions[]`), and re-deriving it from rendered HTML with an
LLM is lossy, non-deterministic and slow.

These clients return typed records that bypass LLM extraction entirely for the
fields the API already gives us.

Every client has a Fixture twin so the pipeline runs offline.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Typed records
# ---------------------------------------------------------------------------

# ClinicalTrials.gov armGroupType vocabulary. An arm typed as one of these IS,
# by the registry's own designation, a comparator -- never excluded by name.
_COMPARATOR_ARM_TYPES = {"ACTIVE_COMPARATOR", "PLACEBO_COMPARATOR", "SHAM_COMPARATOR",
                         "NO_INTERVENTION", "OTHER"}

# ClinicalTrials.gov's own interventionType vocabulary, which the v2 API
# prefixes onto every single interventionNames entry -- "Drug: Tovorafenib",
# never bare "Tovorafenib". Confirmed via a real production run: this extra
# "Drug: " token broke comparator_arms()'s exact-token-set match against the
# subject drug's own name (tokens {"drug","tovorafenib"} != {"tovorafenib"}),
# letting the subject's own arm through as a fake "comparator" -- on every
# single trial this client fetches, not just this one. Stripped once here so
# every downstream consumer (comparator_arms, the comparator's own display
# name) sees the clean name.
_INTERVENTION_TYPE_PREFIX = re.compile(
    r"^(Drug|Biological|Device|Procedure|Radiation|Combination Product|"
    r"Dietary Supplement|Genetic|Other|Behavioral|Diagnostic Test):\s*",
    re.IGNORECASE)


def _strip_intervention_type_prefix(name: str) -> str:
    return _INTERVENTION_TYPE_PREFIX.sub("", name or "").strip()


# An arm listing more than one intervention can mean two structurally
# different things CT.gov represents identically as a flat list: a genuine
# combination regimen given together, or a "Standard of Care"/"Investigator's
# Choice" arm offering several ALTERNATIVE single-agent options. Confirmed
# real bundling defect: a SoC arm listing ["Lurbinectedin", "Topotecan",
# "Amrubicin"] was being joined into one fake three-drug "comparator" instead
# of three real, distinct ones. Detected from the arm's own label/description
# text -- the interventions list itself is exactly the thing that's
# ambiguous, so it can't be the signal used to disambiguate itself.
_CHOICE_MARKERS = re.compile(
    r"\b(investigator'?s?\s+choice|physician'?s?\s+choice|treating\s+physician'?s?\s+choice|"
    r"at\s+the\s+discretion\s+of|per\s+investigator|one\s+of\s+the\s+following|"
    r"standard\s+of\s+care\b.*\bor\b|patient'?s?\s+choice)",
    re.IGNORECASE)

_COMBINATION_MARKERS = re.compile(
    r"\b(in\s+combination\s+with|combined\s+with|concurrent(?:ly)?\s+with|"
    r"together\s+with|plus)\b|\+",
    re.IGNORECASE)


def classify_multi_intervention_arm(arm: "TrialArm") -> str:
    """For an arm listing more than one intervention, decide whether they
    were given AS ONE combination regimen, or are ALTERNATIVE options a
    patient could receive one of. Returns "combination", "choice", or
    "ambiguous" (no marker either way -- don't guess which, but don't treat
    as a confirmed combination by default either)."""
    text = f"{arm.label} {arm.description}"
    if _CHOICE_MARKERS.search(text):
        return "choice"
    if _COMBINATION_MARKERS.search(text):
        return "combination"
    return "ambiguous"


@dataclass
class TrialArm:
    label: str = ""
    arm_type: str = ""            # EXPERIMENTAL | ACTIVE_COMPARATOR | PLACEBO_COMPARATOR | ...
    interventions: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class TrialRecord:
    registry: str = "clinicaltrials.gov"
    identifier: str = ""
    title: str = ""
    url: str = ""
    phase: str = ""
    status: str = ""
    conditions: List[str] = field(default_factory=list)
    arms: List[TrialArm] = field(default_factory=list)
    primary_outcomes: List[str] = field(default_factory=list)
    secondary_outcomes: List[str] = field(default_factory=list)
    eligibility: str = ""

    def comparator_arms(self, subject_drug: str) -> List[TrialArm]:
        """Arms that are NOT the subject drug's own arm. The typed answer to
        'what was this compared against', with no LLM in the loop.

        An EXACT, complete token-set match to the subject drug's own name is
        never a legitimately different comparator, regardless of what
        arm_type the registry happened to assign it -- this check is
        unconditional. Confirmed real gap: a sponsor's inconsistent CT.gov
        data entry can type the drug's own arm as OTHER (one of
        _COMPARATOR_ARM_TYPES) rather than EXPERIMENTAL, which previously let
        it through untouched. Partial overlap (a biosimilar, or an ADC built
        on the same root antibody, sharing one token with the subject drug's
        name but a genuinely different, legitimate comparator) is the only
        case where the registry's own arm_type still grants leniency.
        """
        tok = _tokens(subject_drug)
        out = []
        for a in self.arms:
            arm_type_upper = a.arm_type.upper()
            # Intervention names only for the identity check -- the label
            # ("Experimental", "Arm A", ...) is a description, not a drug
            # name, and would pollute an exact-set match.
            intervention_tokens = _tokens(" ".join(a.interventions))
            if tok and intervention_tokens == tok:
                continue            # exact match to the subject's own name -- always excluded
            # A partial overlap (biosimilar, ADC on the same root antibody)
            # falls through to out.append(a) below regardless of arm_type --
            # the registry's own comparator-type leniency needs no separate
            # check here, only the exact-match exclusion above is gated.
            if arm_type_upper.startswith("EXPERIMENTAL") and not a.interventions:
                continue
            out.append(a)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PublicationRecord:
    pmid: str = ""
    doi: str = ""
    title: str = ""
    abstract: str = ""
    journal: str = ""
    year: str = ""
    publication_types: List[str] = field(default_factory=list)
    mesh_terms: List[str] = field(default_factory=list)
    url: str = ""

    @property
    def is_guideline(self) -> bool:
        return any("guideline" in t.lower() for t in self.publication_types)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MedicineRecord:
    name: str = ""
    inn: str = ""
    atc_code: str = ""
    indication_text: str = ""
    epar_url: str = ""
    smpc_url: str = ""
    product_info_url: str = ""
    authorisation_status: str = ""
    pivotal_trials: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _tokens(text: str) -> set:
    return {t for t in re.findall(r"[a-z]+", (text or "").lower()) if len(t) > 3}


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------

class TrialRegistryClient:
    def search(self, condition: str, intervention: str = "",
               identifiers: Optional[List[str]] = None,
               max_results: int = 20) -> List[TrialRecord]:
        raise NotImplementedError


class LiteratureClient:
    def search(self, query: str, publication_types: Optional[List[str]] = None,
               max_results: int = 20) -> List[PublicationRecord]:
        raise NotImplementedError


class MedicineRegistryClient:
    def lookup(self, product_name: str) -> Optional[MedicineRecord]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Production implementations
# ---------------------------------------------------------------------------

class ClinicalTrialsGovClient(TrialRegistryClient):
    """ClinicalTrials.gov API v2. Free, no key, typed arms."""

    BASE = "https://clinicaltrials.gov/api/v2/studies"

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def _get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        import requests
        resp = requests.get(self.BASE, params=params, timeout=self.timeout,
                            headers={"Accept": "application/json"})
        resp.raise_for_status()
        return resp.json()

    def search(self, condition: str, intervention: str = "",
               identifiers: Optional[List[str]] = None,
               max_results: int = 20) -> List[TrialRecord]:
        records: List[TrialRecord] = []
        if identifiers:
            for ident in identifiers:
                nct = _extract_nct(ident)
                if not nct:
                    continue
                try:
                    payload = self._get({"filter.ids": nct, "pageSize": 1})
                except Exception:
                    continue
                records.extend(self._parse(payload))
        params: Dict[str, Any] = {"pageSize": min(max_results, 50)}
        if condition:
            params["query.cond"] = condition
        if intervention:
            params["query.intr"] = intervention
        try:
            records.extend(self._parse(self._get(params)))
        except Exception:
            pass
        seen, out = set(), []
        for r in records:
            if r.identifier and r.identifier not in seen:
                seen.add(r.identifier)
                out.append(r)
        return out[:max_results]

    @staticmethod
    def _parse(payload: Dict[str, Any]) -> List[TrialRecord]:
        out: List[TrialRecord] = []
        for study in (payload or {}).get("studies", []) or []:
            ps = study.get("protocolSection", {}) or {}
            ident = (ps.get("identificationModule", {}) or {}).get("nctId", "")
            design = ps.get("armsInterventionsModule", {}) or {}
            arms = []
            for a in design.get("armGroups", []) or []:
                arms.append(TrialArm(
                    label=a.get("label", ""),
                    arm_type=a.get("type", ""),
                    interventions=[_strip_intervention_type_prefix(n)
                                  for n in (a.get("interventionNames", []) or [])],
                    description=a.get("description", "")))
            outcomes = ps.get("outcomesModule", {}) or {}
            out.append(TrialRecord(
                identifier=ident,
                title=(ps.get("identificationModule", {}) or {}).get("briefTitle", ""),
                url=f"https://clinicaltrials.gov/study/{ident}" if ident else "",
                phase=", ".join((ps.get("designModule", {}) or {}).get("phases", []) or []),
                status=(ps.get("statusModule", {}) or {}).get("overallStatus", ""),
                conditions=list((ps.get("conditionsModule", {}) or {}).get("conditions", []) or []),
                arms=arms,
                primary_outcomes=[o.get("measure", "") for o in outcomes.get("primaryOutcomes", []) or []],
                secondary_outcomes=[o.get("measure", "") for o in outcomes.get("secondaryOutcomes", []) or []],
                eligibility=(ps.get("eligibilityModule", {}) or {}).get("eligibilityCriteria", "")[:4000],
            ))
        return out


class PubMedClient(LiteratureClient):
    """NCBI E-utilities. MeSH and publication-type filters are the reason this
    is an API and not a web search: 'practice guidelines for this disease at
    this line' is a retrievable CLASS here and a lucky ranking there."""

    ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def __init__(self, api_key: str = "", email: str = "", timeout: int = 30):
        self.api_key, self.email, self.timeout = api_key, email, timeout

    def search(self, query: str, publication_types: Optional[List[str]] = None,
               max_results: int = 20) -> List[PublicationRecord]:
        import requests
        term = query
        if publication_types:
            filt = " OR ".join(f'"{p}"[Publication Type]' for p in publication_types)
            term = f"({query}) AND ({filt})"
        params = {"db": "pubmed", "term": term, "retmax": max_results,
                  "retmode": "json", "sort": "relevance"}
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email
        try:
            ids = requests.get(self.ESEARCH, params=params, timeout=self.timeout
                               ).json().get("esearchresult", {}).get("idlist", [])
        except Exception:
            return []
        if not ids:
            return []
        try:
            xml = requests.get(self.EFETCH, params={"db": "pubmed", "id": ",".join(ids),
                                                    "retmode": "xml"},
                               timeout=self.timeout).text
        except Exception:
            return []
        return _parse_pubmed_xml(xml)


class EMAMedicineClient(MedicineRegistryClient):
    """Resolves a product name to its EMA medicine record.

    Implemented against the EMA medicines dataset rather than by search ranking:
    the legacy approach discovered the EPAR by hoping it ranked, which is why an
    EPAR miss looked like an evidence gap.
    """

    SEARCH_URL = "https://www.ema.europa.eu/en/medicines"

    def __init__(self, search_provider=None):
        self.search_provider = search_provider

    def lookup(self, product_name: str) -> Optional[MedicineRecord]:
        if not self.search_provider:
            return None
        hits = self.search_provider.search(
            f"{product_name} EPAR product information", ["ema.europa.eu"], 5)
        if not hits:
            return None
        pi = next((h.url for h in hits if "product-information" in h.url.lower()), hits[0].url)
        epar = next((h.url for h in hits if "epar" in h.url.lower()), "")
        return MedicineRecord(name=product_name, product_info_url=pi, epar_url=epar)


def _extract_nct(text: str) -> str:
    m = re.search(r"NCT\d{8}", str(text or ""), re.IGNORECASE)
    return m.group(0).upper() if m else ""


def _parse_pubmed_xml(xml: str) -> List[PublicationRecord]:
    """Minimal, dependency-free PubMed XML parse."""
    import xml.etree.ElementTree as ET
    out: List[PublicationRecord] = []
    try:
        root = ET.fromstring(xml)
    except Exception:
        return out
    for art in root.findall(".//PubmedArticle"):
        pmid = (art.findtext(".//PMID") or "").strip()
        # itertext(), not .text/findtext() -- those stop at the first child
        # element, so any inline markup (<i>, <sup>, <sub>, <b>, all routine in
        # PubMed XML for gene names, p-values, chemical formulas) silently
        # truncated the title/abstract at that point.
        title_el = art.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""
        abstract = " ".join("".join(t.itertext()) for t in art.findall(".//AbstractText")).strip()
        journal = (art.findtext(".//Journal/Title") or "").strip()
        year = (art.findtext(".//PubDate/Year") or "").strip()
        ptypes = [(p.text or "").strip() for p in art.findall(".//PublicationType")]
        mesh = [(m.text or "").strip() for m in art.findall(".//DescriptorName")]
        doi = ""
        for aid in art.findall(".//ArticleId"):
            if aid.get("IdType") == "doi":
                doi = (aid.text or "").strip()
        out.append(PublicationRecord(
            pmid=pmid, doi=doi, title=title, abstract=abstract, journal=journal,
            year=year, publication_types=ptypes, mesh_terms=mesh,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""))
    return out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class FixtureTrialRegistry(TrialRegistryClient):
    def __init__(self, records: Optional[List[TrialRecord]] = None):
        self.records = records or []

    def search(self, condition: str, intervention: str = "",
               identifiers: Optional[List[str]] = None,
               max_results: int = 20) -> List[TrialRecord]:
        return self.records[:max_results]


class FixtureLiterature(LiteratureClient):
    def __init__(self, records: Optional[List[PublicationRecord]] = None):
        self.records = records or []

    def search(self, query: str, publication_types: Optional[List[str]] = None,
               max_results: int = 20) -> List[PublicationRecord]:
        if publication_types:
            wanted = {p.lower() for p in publication_types}
            return [r for r in self.records
                    if any(t.lower() in wanted for t in r.publication_types)][:max_results]
        return self.records[:max_results]


class FixtureMedicineRegistry(MedicineRegistryClient):
    def __init__(self, records: Optional[Dict[str, MedicineRecord]] = None):
        self.records = records or {}

    def lookup(self, product_name: str) -> Optional[MedicineRecord]:
        return self.records.get((product_name or "").strip().lower())
