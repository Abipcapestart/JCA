# JCA Phase 1 — retrieval-layer (A6/A7) deep dive: 5 issues not yet flagged

You asked for a deeper pass specifically on retrieval, beyond what's already been covered (query-vocabulary truncation, extraction truncation/chunking, CT.gov bundling, self-referential comparator, pre-A11 dedup). I read `providers/search.py`, `sources/workbook.py`, the rest of `agents/a06_a08_retrieval.py`, and `orchestrator.py` in full this time. Found five things worth fixing, ranked by how much they're likely costing you. Evidence labels: `[CODE]` = read directly, `[RUN4]` = cross-checked against your last run's actual data.

---

## 1. Biggest one: every document gets fetched and extracted once PER POPULATION, with zero reuse — and this is exactly why your richest document hit the truncation cap twice

**`[CODE]`** — `orchestrator.py`, lines 224-250. The comment is explicit about this being intentional design:
> *"A single population... runs this loop exactly once... A second (intended-to-treat) population gets its OWN query plan, retrieval, extraction and claim validation, run against ITS OWN indication wording — the SME requires the two populations be kept as separate structured objects."*

So `for pop in populations: _run_retrieval_pass(pop, ...)` runs the **entire A6→A10 pipeline** — query planning, retrieval, extraction, grounding, claim validation — once per population, completely independently, with no sharing of fetched documents or extracted findings across populations.

That isolation is a reasonable design goal for the *validation* steps (a claim really should only be checked against the specific population it's being validated for). But it's being paid for by re-fetching and re-extracting *content that doesn't change based on which population is asking* — a guideline PDF's text is the same document regardless of which BRAF-subtype population you're currently scoping. Right now, that population-agnostic content gets pulled through a brand-new, independent `a08.extraction` LLM call for every population, each with its own 16000-token budget and its own independent chance of truncating.

**`[RUN4]`** — this is not theoretical; it's exactly what happened. The SIOPE LGG guideline PDF appears in exactly **2** `a08.extraction` calls in your last run, both hitting the 16000-token cap. Your run had exactly **2 populations**. Every one of the 4 distinct comparators that reached `a12.scope_adjudication` was adjudicated in exactly **2** separate calls each (chemotherapy ×2, dabrafenib+trametinib ×2 — with two *different* verdicts, `in_scope` and `uncertain` — "another medicine" ×2, "surgery and chemotherapy" ×2). That's not coincidence or an A11-grouping artifact as I initially guessed last round — it's the direct, mechanical consequence of running the whole pipeline twice, once per population, with no document-level cache between the two runs. The Dabrafenib+Trametinib split-verdict mystery from the last report is now fully explained: two independent adjudications, on two independently-extracted evidence pools, and something downstream silently kept the more favorable one.

**Cost of this, concretely:** you're paying for `a08.extraction` on the same document N times (N = population count) for content that's identical every time, and each of those N independent attempts is an independent roll of the dice on hitting the truncation cap. Fixing the chunking (already prescribed) reduces the *chance* of truncation per attempt; this fix reduces the *number of attempts* on the same content from N to 1, which is strictly better and also cuts LLM spend roughly in half for every population-agnostic document.

**The fix:** separate "fetch and extract a document's raw content" (population-independent) from "does this specific extracted finding apply to Population N" (population-dependent, and this part should stay exactly as isolated as it is today). Concretely:
- Add a cache in `orchestrator.py`'s population loop, keyed by document URL, that stores each document's already-extracted `EvidenceRecord`s the first time any population encounters that URL.
- Before calling `extract_from_documents()` inside `_run_retrieval_pass()` (`agents/a06_a08_retrieval.py` is where the function lives; the call site is `orchestrator.py` line 146), split `result.documents` into "already extracted this run" (reuse cached records, tagging them with this population's own `population_context` matching logic) and "genuinely new" (extract normally). Only the genuinely new documents pay for an LLM call.
- Claim validation, grounding, and adjudication all continue running per-population exactly as they do now, on the (now population-tagged, but not re-extracted) record — this preserves the SME's stated isolation requirement completely; only the wasteful, truncation-risk-doubling re-extraction of identical text goes away.

**Test without a full run:** count `a08.extraction` calls per unique `source_url` in your last run's `bedrock_calls` — you already have the data to verify this diagnosis directly:
```python
from collections import Counter
import json, re

with open("JCA_Phase1_RUN-20260923T184232-dbbdf4.json") as f:
    data = json.load(f)
calls = [c for c in data["bedrock_calls"] if c["prompt_id"] == "a08.extraction"]
urls = []
for c in calls:
    m = re.search(r"SOURCE URL:\s*(\S+)", c["user_prompt"])
    if m:
        urls.append(m.group(1))
dupes = {u: n for u, n in Counter(urls).items() if n > 1}
print(f"{len(dupes)} documents extracted more than once in a single run")
for u, n in sorted(dupes.items(), key=lambda x: -x[1])[:10]:
    print(n, u)
```
Every URL that shows up more than once here is a document being paid for and re-risked N times for content that's identical every time — this will show you the full scope of the waste, not just the one SIOPE example.

---

## 2. `select_balanced()`'s anti-starvation logic never actually runs — it's always called with at most one domain group

**`[CODE]`** — `providers/search.py`, `select_balanced()` (lines 183-222) is explicitly designed to prevent one domain from crowding out another within a multi-domain search: *"a blind top-k slice lets one category crowd another out entirely. This keeps the saving without the starvation."* It does this by bucketing hits into `domain_groups` and guaranteeing each active group at least one slot.

But `[CODE]` — its only call site, `agents/a06_a08_retrieval.py` line 398:
```python
urls = select_balanced(hits, item.max_urls, {"primary": item.domains} if item.domains else {})
```
always passes either zero groups or exactly one group (`"primary"`). Look at the function's own early-exit: `if len(domain_groups) <= 1: return [h.url for h in hits[:max_urls]]` — a plain top-k-by-relevance-score slice, no domain awareness at all. **The anti-starvation logic this function exists for has never once executed in production**, because it's never handed more than one group to balance across.

This matters most for your `clinical_guideline` queries, which routinely search across several domains at once in one query (e.g., `"esmo.org; eurordis.org; siope.eu"` for a single per-state landscape pass, per your own A6 query plan). If Tavily's relevance ranking happens to favor pages from one of those domains (more content, better SEO, whatever the reason) for a given query, the current top-k slice can fill all of `max_urls` (often only 3-4) with pages from one domain and never touch the others — even though the function specifically built to prevent exactly this sits right there, unused.

**The fix:** at the call site (line 398), bucket by the query's own curated domain list instead of collapsing it into one `"primary"` group:
```python
domain_groups = {d: [d] for d in item.domains} if item.domains else {}
urls = select_balanced(hits, item.max_urls, domain_groups)
```
This activates the existing balancing logic with zero changes to `select_balanced()` itself — the function was already correct, it just never got the input shape it needed to do anything.

**Test without a full run:**
```python
from jca_phase1.providers.search import select_balanced, SearchHit

hits = [SearchHit(url=f"https://eurordis.org/page{i}") for i in range(8)] + \
       [SearchHit(url="https://siope.eu/media/documents/escp-low-grade-gliomas-lgg.pdf")]
domain_groups = {"eurordis.org": ["eurordis.org"], "siope.eu": ["siope.eu"]}
selected = select_balanced(hits, max_urls=4, domain_groups=domain_groups)
assert any("siope.eu" in u for u in selected), \
    f"siope.eu was crowded out entirely: {selected}"
print("PASS: siope.eu guaranteed a slot even though eurordis.org dominates the ranking")
```

---

## 3. `plan_refinement()` — a fully built, config-enabled coverage-gap retry — is never called anywhere

**`[CODE]`** — `agents/a06_a08_retrieval.py`, `plan_refinement()` (lines 251-274) exists, is documented ("One bounded refinement round, fired only where coverage actually failed... Broadens terms"), and is gated by `config.py`'s `enable_refinement_round: bool = True` — enabled by default. **I grepped the entire codebase for callers of this function: there are none.** It's dead code. Nothing computes the `gaps` list it needs as input, and nothing ever invokes it from `orchestrator.py` or anywhere else.

This means every retrieval gap your `A7 Retrieval Attempts`/completeness sheet already shows you (states and source classes that came back `source_inaccessible` or with zero documents on the first pass) simply stays a gap for the rest of that run. The mechanism specifically designed to retry those with broader terms was built and configured on, but never wired in.

**The fix:** in `orchestrator.py`, after the population loop (`run_phase1`, after line 250), compute the actual coverage gaps from `attempts` (any `SourceClassAttempt` with `status` other than `EV_FOUND`, i.e. zero or inaccessible documents) and call `plan_refinement()` with them, then run those refinement-plan items through `execute_plan()`/`extract_from_documents()` the same way the initial plan does — reusing the exact same pattern already in `_run_retrieval_pass()`, just for the gap list instead of the full plan.

**Test without a full run:** this one's simplest to verify by just calling the function directly with a synthetic gap list and confirming it produces a sane query plan (it already has no bugs I could find in its own logic — the only defect is that nothing calls it):
```python
from jca_phase1.agents.a06_a08_retrieval import plan_refinement

gaps = [("Austria", "clinical_guideline"), ("Malta", "hta_regulatory")]
items = plan_refinement(gaps, population=<fixture>, intervention=<fixture>,
                        vocab=<fixture>, inventory=<fixture>, areas=["Oncology"])
assert len(items) == len(gaps)
print(f"PASS: refinement plan built for {len(items)} gaps — now wire this into orchestrator.py")
```

---

## 4. A hard 500-character floor silently marks short (but possibly genuine) pages as "source inaccessible"

**`[CODE]`** — `providers/search.py`, `TavilySearchProvider._fetch()`, lines 279-283:
```python
if len(text) < 500:
    return RetrievedDocument(url=url, resolved_url=resolved, text=text, ok=False,
                             status=C.EV_SOURCE_INACCESSIBLE, ...
                             error=f"only {len(text)} chars retrieved")
```
Any extracted page under 500 characters is treated identically to a genuinely broken/blocked fetch. This is a reasonable heuristic for filtering out error pages and empty stubs, but it will also discard a legitimately short, single-fact page — e.g., a brief national HTA decision notice, or a short "no assessment on file" confirmation that's itself informative (it tells you the state genuinely has no finding, vs. the search failing). Right now both cases produce the exact same `source_inaccessible` status, so a reviewer looking at your completeness sheet can't tell "we tried and this state has nothing" apart from "we tried and the page just happened to be short."

**The fix (lower priority than 1-3, but cheap):** don't hard-fail purely on length. Lower the threshold significantly (e.g. 100 characters, to still catch truly empty responses) and/or check for actual error-page markers in the text (e.g., "404", "not found", "access denied") rather than length alone. At minimum, distinguish the status — e.g. `EV_SHORT_CONTENT` vs `EV_SOURCE_INACCESSIBLE` — so downstream completeness reporting doesn't conflate "nothing there" with "briefly, something's there."

---

## 5. Worth testing, not yet confirmed: `extract_depth="basic"` may be part of why table-heavy guideline PDFs extract poorly

**`[CODE]`** — `providers/search.py`, `TavilySearchProvider._fetch()`, line 264: every fetch, including full-document reads of guideline/HTA PDFs (`extract_depth="basic"`), uses Tavily's basic extraction depth. This is a live hypothesis from the last round of analysis (whether the SIOPE guideline's zero/failed extraction was a token-cap problem, a text-quality problem, or both) that I couldn't fully resolve without inspecting Tavily's actual extracted text. I still can't confirm this from code alone — it needs a side-by-side comparison. But it's cheap to test and directly relevant to the same document that's already been the center of this investigation:
```python
resp_basic = tavily_client.extract(urls=[siope_url], extract_depth="basic", format="markdown")
resp_advanced = tavily_client.extract(urls=[siope_url], extract_depth="advanced", format="markdown")
print("basic length:", len(resp_basic["results"][0].get("raw_content", "")))
print("advanced length:", len(resp_advanced["results"][0].get("raw_content", "")))
# Eyeball both, specifically around the comparator recommendation table —
# does "advanced" preserve which drug goes with which recommendation, where
# "basic" flattens the table into unstructured, hard-to-parse text?
```
If `advanced` meaningfully improves the table's readability, switch `extract_depth` to `"advanced"` for the `FULL_DOCUMENT_SOURCE_CLASSES` fetch path specifically (`config.py` line ~174-176) — leave it `"basic"` for the scoped, query-relevant-chunk fetches where full-document fidelity matters less. This costs more per Tavily call (advanced extraction is typically priced higher), so it's worth confirming it actually helps on a real document before rolling it out broadly, rather than assuming.

---

## Priority order

1. **Fix 1 (cross-population dedup)** — highest leverage: cuts extraction cost roughly in half for every population-agnostic document and directly reduces truncation risk on your richest sources, on top of the chunking fix already planned.
2. **Fix 2 (`select_balanced` domain groups)** — one-line change at the call site, zero risk, activates logic that already exists and was already tested in its own right.
3. **Fix 3 (wire in `plan_refinement`)** — recovers genuine coverage gaps (states/source-classes with zero documents) that currently never get a second attempt, using code that's already written and configured on.
4. **Fix 4 (500-char threshold)** — cheap, low-risk, improves completeness-reporting accuracy more than raw recall.
5. **Fix 5 (extract_depth)** — test first, don't ship blind; only adopt if the side-by-side actually shows a difference on a real table-heavy document.
