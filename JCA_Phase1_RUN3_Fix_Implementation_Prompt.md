# Implementation prompt — JCA Phase 1 pipeline fixes (Tovorafenib Run 3 gap analysis)

Paste this to whoever (or whichever coding assistant) is implementing the fixes. It is self-contained: problem, evidence, exact files to change, and — critically — a **targeted test for each fix that does not require a full pipeline run**. A full run costs ~23 minutes and ~$15 and only tells you the aggregate outcome; these tests isolate the one function/prompt/document that's broken so you get a pass/fail in seconds and can see exactly what the model or code returned.

Background: this codebase is `jca_phase1/`, an 18-stage (A1-A18) JCA comparator-and-outcome scoping pipeline. Three production runs against Tovorafenib have been validated against the actual EU JCA ground-truth report (GT has 7 comparators: Vinblastine, Carboplatin+Vincristine, TPCV, Dabrafenib+Trametinib, Everolimus, Bevacizumab+chemotherapy, Trametinib monotherapy). The most recent run produced only 2, one of which was itself wrong. Every defect below was confirmed either by reading the actual code or by executing it directly against real data from that run — nothing here is inferred from output alone.

Do the fixes in this order: **2 → 3 → 7 (diagnose first) → 4 → 5 → 6 → 1 → 8.** Reasoning: 2 and 3 are small, low-risk, and immediately testable in isolation. 7 needs a diagnostic step before any code is written, so start that diagnostic early since it may take longest to get an answer from. 1 and 8 aren't code changes — do them last, right before the next full validation run.

---

## Fix 1 — Confirm the Tovorafenib-self-comparator bug is actually gone

**Problem:** In the Run 3 export, "Tovorafenib" appeared as its own comparator, backed by evidence quotes reading `"Arm 'Arm 1: Low-Grade Glioma' (EXPERIMENTAL): Drug: Tovorafenib..."` — the unstripped `"Drug: "` prefix from the ClinicalTrials.gov API broke `TrialRecord.comparator_arms()`'s exact-token-set exclusion in `providers/registries.py` (`_tokens("Drug: Tovorafenib")` = `{"drug","tovorafenib"}` ≠ `_tokens("Tovorafenib")` = `{"tovorafenib"}`, so the drug's own arm was never excluded).

**Finding:** The fix (`_strip_intervention_type_prefix`, applied in `TrialRecord._parse()`) already exists in the current codebase and was confirmed correct by direct execution. **No code change needed** — the Run 3 export predates this fix being live.

**Targeted test (no full run):**
```python
from jca_phase1.providers.registries import TrialRecord, TrialArm

t = TrialRecord(identifier="NCT04775485", arms=[
    TrialArm(label="Arm 1: Low-Grade Glioma", arm_type="EXPERIMENTAL",
             interventions=["Drug: Tovorafenib"]),
])
result = t.comparator_arms("Tovorafenib")
assert result == [], f"Expected the drug's own arm to be excluded, got {result}"
print("PASS: own-arm exclusion works with the unstripped API prefix")
```
Run this alone — it takes under a second and definitively answers whether this defect is live, without needing a full pipeline run or any network/LLM calls.

---

## Fix 2 — Diacritic/accent normalization gap

**Problem:** `agents/a09_a13_validation.py`'s `_normalise()` uses only NFKC, which does not strip accents. `_tokens("Tovorafénib")` → `['tovoraf', 'nib']` (the "é" splits the word, and "nib" is filtered by the `len(t) > 3` rule) vs. `_tokens("Tovorafenib")` → `['tovorafenib']` — zero token overlap. `_is_other_drug()` therefore concludes the French-spelled drug name is a *different* drug, and `_same_substance()` never matches it either. This is silently discarding legitimate French-market evidence (HAS-sourced documents routinely use the accented spelling) on every drug this pipeline evaluates, not just Tovorafenib.

**Fix — file: `agents/a09_a13_validation.py`**, in `_normalise()` (~line 30):
```python
def _normalise(text: str) -> str:
    """NFKC then strip combining diacritics, so 'Tovorafénib' and 'Tovorafenib'
    tokenize identically."""
    t = unicodedata.normalize("NFKC", text or "").lower()
    t = unicodedata.normalize("NFKD", t)
    return "".join(ch for ch in t if not unicodedata.combining(ch))
```
This single change propagates to every caller (`_tokens`, `_substance_tokens`, `_same_substance`, `_is_other_drug`) since they all route through `_normalise()`.

**Targeted test (no full run, no LLM calls):**
```python
from jca_phase1.agents.a09_a13_validation import _substance_tokens, _is_other_drug, _same_substance

assert _substance_tokens("Tovorafénib") == _substance_tokens("Tovorafenib"), \
    f"{_substance_tokens('Tovorafénib')} != {_substance_tokens('Tovorafenib')}"
assert not _is_other_drug("Tovorafénib", "Tovorafenib"), \
    "French spelling wrongly flagged as a different drug"
assert _same_substance("Tovorafénib", "Tovorafenib")

# Regression guard: make sure this doesn't start conflating genuinely different drugs
assert _is_other_drug("Dabrafenib plus trametinib", "Tovorafenib")

print("PASS: accented and unaccented spellings resolve to the same substance")
```
Add this as a permanent unit test (e.g. `tests/test_a09_a13_validation.py`) — it should also be run against a couple of other known EU-language variants (e.g. German/Italian INN spelling differences) if any exist in your source data, to make sure the fix generalizes rather than special-casing "é."

---

## Fix 3 — Token-cap truncation (`a06`, `a12`, `a14`, `a16.indication_synthesis`)

**Problem:** Comparing `output_tokens` in the `Bedrock Calls` export sheet against each prompt's configured `max_tokens`, across 3 runs: `a06.query_vocabulary` hits its cap 100% of the time (3/3 runs), `a14.outcome_harmonization` hits its cap 100% of the time (3/3 runs), `a12.scope_adjudication` hit its cap this run and produced the exact truncated, unparseable response that degraded the Dabrafenib+Trametinib verdict from `in_scope` to `uncertain`, and `a16.indication_synthesis` hit its cap 100% this run (newly observed). Your team already fixed this pattern once — raising `a10.claim_validation`'s cap from 2000 to 4000 measurably cut its truncation rate — so this is the same fix applied to the remaining offenders.

**Fix — files and exact caps to raise:**
- `agents/a06_a08_retrieval.py` line 75 — `a06.query_vocabulary`: `max_tokens=1500` → raise to at least 3000-4000 (100% truncation means 1500 isn't close; don't just nudge it)
- `agents/a09_a13_validation.py` line 506 — `a12.scope_adjudication`: `max_tokens=2000` → raise to match `a10`'s 4000
- `agents/a14_a18_consolidation.py` line 154 — `a14.outcome_harmonization`: check what cap it's actually using (it calls `llm.call_json("a14.outcome_harmonization", payload, ...)` — confirm whether it passes an explicit `max_tokens` or falls through to `C.LLM.default_max_tokens = 2000` in `config.py` line 343); either way, raise it
- `agents/a14_a18_consolidation.py` line 322 — `a16.indication_synthesis`: `max_tokens=600` → raise substantially; 600 tokens is very tight for a synthesis prompt and this is a newly observed 100%-truncation site

Consider centralizing all of these as named fields on the `C.LLM` config dataclass in `config.py` (next to `extraction_max_tokens`/`validation_max_tokens`) instead of leaving them as literals scattered across agent files — that's what let this drift unnoticed for 3 runs.

**Targeted test (no full run — isolate each prompt with a single LLM call):**
For each of the four prompts, construct the smallest realistic payload that provokes a long response, call it directly through your `providers/llm.py` client, and check `output_tokens` against the new cap with headroom to spare:
```python
from jca_phase1.providers.llm import LLMClient  # adjust import to actual client class
llm = LLMClient(...)

# Example for a12 — reuse the exact payload shape a12 builds, or a known-large real one
# (e.g. reconstruct the dabrafenib+trametinib candidate from Run 3's A12 All Candidates sheet)
resp = llm.call_json("a12.scope_adjudication", known_large_payload, max_tokens=<new_cap>, default=None)
call = llm.call_log[-1]
assert call.output_tokens < call.max_tokens * 0.9, \
    f"Still near cap: {call.output_tokens}/{call.max_tokens}"
assert resp is not None, "Response failed to parse even with the raised cap"
print(f"PASS: a12 completed in {call.output_tokens}/{call.max_tokens} tokens")
```
Do this once per prompt with one real, previously-truncated payload from the Run 3 export (you have the exact inputs already, since these are the calls that hit their old caps) rather than waiting for a full run to re-exercise them. This gives you a direct before/after token count for the exact case that broke.

---

## Fix 4 — `a11.comparator_identity` prompt/schema mismatch for combinations

**Problem:** The `_COMPARATOR_IDENTITY` prompt in `prompts/registry.py` (~lines 787-902) instructs the model, in its own "Self-Validation Checklist," to resolve each combination component to its own INN — but the JSON return schema is single-valued per candidate (`"inn": "", "class_mechanism": "", "atc_code": ""` — one value for the whole candidate, not one per component). The prose asks for something the schema cannot represent. Result, confirmed independent of truncation: every combination comparator across all 3 runs reviewed shows `class_source: unresolved`.

**Fix — file: `prompts/registry.py`**, the `_COMPARATOR_IDENTITY` prompt's return schema (~line 898):
```python
# Before:
{"index": 1, "inn": "", "display_name": "", "brand_names": [],
 "is_combination": false, "components": [], "is_category": false,
 "class_mechanism": "", "atc_code": ""}

# After:
{"index": 1, "display_name": "", "brand_names": [], "is_combination": false,
 "components": [{"name": "", "inn": "", "atc_code": "", "class_mechanism": ""}],
 "is_category": false}
```
(Drop the flat `inn`/`atc_code`/`class_mechanism` fields entirely, or keep them only meaningful for single-agent, non-combination items — decide one convention and document it in the prompt.)

- **File: `agents/a09_a13_validation.py`**, wherever the `a11.comparator_identity` response is consumed (~line 382-383) — update the parsing to read `components[].inn`/`.atc_code`/`.class_mechanism` for combinations.
- **File: `agents/a14_a18_consolidation.py`** — wherever the `A16 Comparators` export row's `class_source`/`inn` columns are populated, update the rollup logic to handle a list of per-component identities for combination rows (e.g. join component INNs, or pick a representative class if multiple components share one).

**Targeted test (no full run):**
```python
from jca_phase1.providers.llm import LLMClient
llm = LLMClient(...)

payload = build_a11_payload_for(["Dabrafenib plus trametinib"])  # reuse Run 3's actual candidate string
resp = llm.call_json("a11.comparator_identity", payload, max_tokens=4000, default=None)

item = resp["items"][0]
assert item["is_combination"] is True
assert len(item["components"]) == 2, item["components"]
for comp in item["components"]:
    assert comp["inn"], f"Component {comp['name']} has no INN resolved"
print("PASS: both dabrafenib and trametinib resolved their own INN")
```
Feed it the exact "Dabrafenib plus trametinib" string from Run 3's `A11 Comparator Identity` sheet — you already know this is the case that fails, so there's no need to run the whole pipeline to re-trigger it.

---

## Fix 5 — Role ambiguity cascading into wrong `WRONG_SUBJECT_DRUG` rejections

**Problem:** In `Validation Excluded Items`, several genuinely relevant sentences pulled from the tovorafenib EMA dossier itself (e.g. *"Patients with BRAF V600 mutant low-grade gliomas who have disease progression on or after dabrafenib plus trametinib will also need further options"*) were rejected as `WRONG_SUBJECT_DRUG`. Pattern: when A08 extraction tags `role_or_result: unclear` (rather than a definite role like `active_comparator`), A10's `_is_other_drug()` check treats the passage as unrelated background noise and rejects it outright — even when the text explicitly relates the comparator to the subject drug. Contrast with near-identical source material where the role was tagged `active_comparator` and correctly validated `SUPPORTED_SUBPOPULATION`.

**Fix:**
- **File: `agents/a09_a13_validation.py`**, the claim-validation logic around the `a10.claim_validation` call (~line 236) and wherever its verdict is decided — when `role_or_result == "unclear"`, don't let `_is_other_drug()` alone force a hard `WRONG_SUBJECT_DRUG` rejection. Either (a) route it to `uncertain` instead of a hard rejection (matching the fail-open pattern already used for a12's parse failures), or (b) add a secondary check — does the quote text itself explicitly name or reference the requested intervention — before rejecting.
- **File: `prompts/registry.py`**, `_EXTRACTION` prompt (a08.extraction, `comparator.role` field definition ~lines 576-586) — tighten the guidance so the model defaults to a definite role (`active_comparator`, `prior_therapy`, `background_therapy`) rather than `unclear` whenever the passage clearly relates the named drug to the subject drug, reserving `unclear` for genuinely ambiguous cases.

**Targeted test (no full run):**
```python
# Step 1: re-run just the extraction prompt on the exact source passage that mis-tagged role=unclear
extraction_payload = build_a08_payload(
    text="Patients with BRAF V600 mutant low-grade gliomas who have disease progression "
         "on or after dabrafenib plus trametinib will also need further options.",
    subject_drug="Tovorafenib")
ext_resp = llm.call_json("a08.extraction", extraction_payload, max_tokens=16000, default=[])
role = ext_resp[0]["comparator"]["role"]
print(f"role now tagged as: {role}")  # confirm it's no longer 'unclear', or proceed to step 2 regardless

# Step 2: even if role stays 'unclear', confirm validation no longer hard-rejects it
validation_result = validate_claim(ext_resp[0], subject_drug="Tovorafenib")  # call the actual a10 function
assert validation_result.verdict != "WRONG_SUBJECT_DRUG", validation_result.reason
print(f"PASS: verdict is now {validation_result.verdict}")
```
Use the exact quote text from `Validation Excluded Items` — you have it verbatim in the Run 3 export, so this reproduces the failure with zero retrieval/network calls.

---

## Fix 6 — Line-of-therapy retrieval coverage for CV and Vinblastine

**Problem:** Carboplatin+Vincristine and Vinblastine were retrieved, extracted, and correctly reasoned about at adjudication — but excluded because the *only* evidence retrieved for them was first-line context, when GT confirms both are valid at the requested 2L+ setting (CV: "1L / 2L (chemo backbone)"; Vinblastine: "2L+ (individualised)"). A12 reasoned correctly given what it was shown; the gap is upstream, at retrieval.

**Fix — file: `agents/a06_a08_retrieval.py`**, the query-plan generation logic (~lines 195-271, where per-state `clinical_guideline`/`landscape` queries are built) — add or verify a query variant that explicitly requests relapsed/refractory or "after ≥1 prior systemic therapy" framing for chemotherapy-backbone comparator classes, not only the generic drug-anchored query.

**Targeted test (no full run):**
```python
# Inspect the actual query plan generated for this drug/population without executing any retrieval
from jca_phase1.agents.a06_a08_retrieval import build_query_plan  # adjust to actual function name

plan = build_query_plan(population=<Tovorafenib LGG population fixture>, intervention=<Tovorafenib fixture>)
line_of_therapy_queries = [q for q in plan if "2L" in q.query or "relapsed" in q.query.lower()
                           or "prior systemic therapy" in q.query.lower()]
assert line_of_therapy_queries, "No query targets the 2L+/relapsed-refractory setting explicitly"
print(f"PASS: {len(line_of_therapy_queries)} queries target the 2L+ setting")
```
This checks the query *plan* — a deterministic, non-LLM, non-network step — so you can validate the fix without spending on retrieval or extraction calls at all. Only after this passes is it worth spending a real retrieval call to confirm 2L+ evidence actually comes back.

---

## Fix 7 — `clinical_guideline` source-class extraction failure (do this first among the code fixes — highest leverage)

**Problem:** Across Run 3, every one of 356 successfully-retrieved documents was compared by `source_class` against whether it produced ≥1 A08 extraction record:

| source_class | retrieved | extracted from |
|---|---|---|
| hta_regulatory | 205 | 12 |
| pubmed | 60 | 4 |
| **clinical_guideline** | **59** | **0** |
| drug_registry_label | 22 | 2 |
| conference_evidence | 6 | 3 |
| general_web | 4 | 3 |

Every `clinical_guideline` document — including `https://siope.eu/media/documents/escp-low-grade-gliomas-lgg.pdf`, which is GT's own cited source for TPCV, Carboplatin+Vincristine, and Vinblastine — was successfully retrieved (`ok: True`) but produced **zero** extraction records. This is why TPCV, everolimus, and bevacizumab+chemotherapy never appear anywhere in the run: they're guideline-sourced per GT, and the entire guideline source class fails at extraction. Confirmed **not** a code-level filter: `extract_from_documents()` in `agents/a06_a08_retrieval.py` submits every retrieved document uniformly regardless of `source_class`, and `SRC_CLINICAL_GUIDELINE` is correctly included in `FULL_DOCUMENT_SOURCE_CLASSES` in `config.py`, so these documents get full text, not a truncated preview.

**Do the diagnostic before writing any fix code:**
```python
# Re-fetch (or reuse a cached copy of) the exact document and run extraction on it in isolation
from jca_phase1.providers.search import SearchProvider  # or however fetch is exposed
from jca_phase1.agents.a06_a08_retrieval import _extract_one

doc = search.fetch("https://siope.eu/media/documents/escp-low-grade-gliomas-lgg.pdf",
                    query="Tovorafenib paediatric low-grade glioma", full_document=True)
print(f"fetched ok={doc.ok}, text length={len(doc.text)}")
print(doc.text[:2000])  # SANITY CHECK: does the fetched text actually contain readable
                        # drug names and recommendations, or is table structure garbled?

records = _extract_one(doc, population=<Tovorafenib LGG population fixture>,
                        intervention=<Tovorafenib fixture>, llm=llm)
print(f"extracted {len(records)} records")
for r in records[:5]:
    print(r.subject_drug, r.comparator_or_outcome, r.evidence_quote[:120])
```
This tells you definitively which of the two live hypotheses is correct:
- **If `doc.text` looks garbled/unstructured** (guideline PDFs often present recommendations in multi-column tables that flatten badly to plain text) → the fix is in whatever performs PDF-to-text conversion before `_extract_one()` receives it (locate the actual `search.fetch()` implementation — not yet inspected in this pass — likely `providers/search.py` or similar); consider a table-aware extraction step or passing the document through a layout-preserving converter for this source class specifically.
- **If `doc.text` looks fine but `_extract_one()` / the `a08.extraction` LLM call returns an empty list** → the fix is in `prompts/registry.py`'s `_EXTRACTION` prompt (~line 525 onward): add an explicit worked example for the "landscape guideline names multiple treatment options for a population, none framed as directly about the requested drug" case, since the prompt's current framing (extract claims "relevant to comparator or outcome scoping," `subject_drug` = "the drug this specific claim is ABOUT") may be causing the model to treat a multi-option guideline table as having no single extractable subject and return nothing, rather than emitting one record per named option.

Either way, also add logging in `_extract_one()` (~line 536-547 in `agents/a06_a08_retrieval.py`) for the case where `parsed` is an empty list or fails `isinstance(parsed, list)` — right now this fails completely silently with no entry in `Errors and Retries`, which is why this defect took manual cross-tabulation across 356 rows to find instead of being visible from the export directly.

**Once you know which hypothesis is correct, the fix + retest is a single document, single LLM call, no full run.** Re-run the diagnostic script above after the fix and confirm `len(records) > 0` and that TPCV/everolimus/vinblastine/carboplatin+vincristine-style entries actually appear in the output.

---

## Fix 8 — `allow_jca_reports` policy for GT-benchmarking runs

**Problem:** Run 3 had `leakage_guard.allow_jca_reports: true`, `blocked_urls: []` (Run 2 had `false` with populated `blocked_urls`). The actual Tovorafenib JCA answer-key report was fetched into the retrieval pool this run — confirmed by URL match in `A7 Retrieved Documents` — but I traced it forward through `A8 Extraction`, `A16 Comparator Sources`, `A15 Outcome Sources`, and `A10 Usable Records` and found **zero** downstream references, so it did not contaminate this run's answers. But nothing currently *guarantees* that; it depends on which documents extraction happens to prioritize.

**Fix:** No code change — a config/runbook decision. Locate the `leakage_guard`/`allow_jca_reports` setting (surfaced in `Run Manifest`; likely a field on a config object in `config.py`) and set `allow_jca_reports: false` with `blocked_urls` populated (the JCA report's own URL, e.g. the `health.ec.europa.eu` PDF link) for any run whose purpose is GT validation. Leave it `true` only for genuine production runs where there's no answer key to protect. Add a one-line comment at that config site documenting this as a benchmarking-vs-production toggle so it isn't flipped by accident again.

**Targeted test:** none needed beyond confirming, for the next GT run, that `Run Manifest.leakage_guard.blocked_urls` contains the JCA report URL and `A7 Retrieved Documents` shows it either absent or explicitly `blocked` rather than `evidence_found`.

---

## After all fixes: what the next full run should confirm

Once 1-7 are done, run the full pipeline once against Tovorafenib and check, directly against `jca gt.xlsx`:
1. Tovorafenib no longer appears as its own comparator (Fix 1).
2. Dabrafenib+Trametinib gets a clean `in_scope` verdict, resolves both component INNs, and its French-sourced HAS evidence is no longer excluded (Fixes 2, 3, 4).
3. Carboplatin+Vincristine and Vinblastine either get admitted (if 2L+ evidence is now found) or, if still excluded, the exclusion reason changes to something other than "no 2L+ evidence retrieved" (Fix 6).
4. TPCV, Everolimus, and Bevacizumab+chemotherapy appear at least as candidates in `A12 All Candidates`, even if they don't all survive adjudication (Fix 7).
5. `Bedrock Calls` shows all four previously-100%-truncated prompts now comfortably under their caps (Fix 3).
