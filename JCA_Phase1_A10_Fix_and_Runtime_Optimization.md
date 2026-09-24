# Two fixes: a10.claim_validation cap + runtime (40 min → target ~15-20 min)

Both grounded in the current codebase (`orchestrator.py`, `config.py`, `agents/a06_a08_retrieval.py`, `agents/a09_a13_validation.py`) — no guessing, line numbers below are from the files as sent.

---

## Fix 1 — a10.claim_validation cap (the D+T evidence-loss bug from the Run 5 correction)

`[CODE]` — `config.py` line 351: `validation_max_tokens: int = 4000`. Every other stage that batches multiple items into one JSON response and kept truncating at 4000 has already been raised in this file (`outcome_harmonization_max_tokens: 8000`, `scope_adjudication_max_tokens: 4000` was also raised once from 2000). `a10.claim_validation` never got the same treatment, and it's structurally the same shape of problem: `_validate_batch()` (`agents/a09_a13_validation.py` lines 222-278) batches **every claim from one re-fetched source URL** into a single call and asks for a `verdict` + free-text `reason` per claim. A source like the HAS France Ojemda transcript — which discusses tovorafenib itself plus several comparators — can put a dozen-plus claims in one batch, and the model has to write a reasoned verdict for every one of them inside a 4000-token ceiling. When it runs out, the response is truncated mid-JSON, `parsed.get("results")` comes back empty, and **every claim in that batch** — including any genuinely on-target ones — is failed closed:

```python
if not results:
    for rec in batch:
        rec.validation = ValidationOutcome(
            verdict=C.V_NOT_SUPPORTED, refetched=True, attempts=1,
            reason="validator response could not be parsed; failed closed")
```

This is exactly what happened to the HAS France transcript's dabrafenib+trametinib `active_comparator` claim in Run 5.

**Fix A — raise the cap** (`config.py` line 351):
```python
# Was flat 4000 -- same truncation shape already fixed at a12/a14/a16: a
# batch of many claims from one re-fetched source needs headroom per claim,
# not a single shared ceiling. Raised in line with validation_max_tokens's
# siblings (outcome_harmonization_max_tokens: 8000) rather than guessed.
validation_max_tokens: int = 8000
```

**Fix B — bound batch size directly, so the cap is never the only thing standing between one dense source and a mass failed-closed loss** (`agents/a09_a13_validation.py`, `validate_claims()`, lines 206-213). Split each URL's claims into sub-batches instead of one unbounded batch per URL, same philosophy as the `_CHUNK_CHARS` fix already applied to `a08.extraction`:

```python
_MAX_CLAIMS_PER_VALIDATION_BATCH = 10

by_source: Dict[str, List[EvidenceRecord]] = {}
for rec in survivors:
    by_source.setdefault(rec.source_url, []).append(rec)

sub_batches: List[tuple] = []
for url, batch in by_source.items():
    for start in range(0, len(batch), _MAX_CLAIMS_PER_VALIDATION_BATCH):
        sub_batches.append((url, batch[start:start + _MAX_CLAIMS_PER_VALIDATION_BATCH]))

with ThreadPoolExecutor(max_workers=max_workers) as pool:
    futures = [pool.submit(_validate_batch, url, sub, population_indication, drug, search, llm)
               for url, sub in sub_batches]
    for fut in as_completed(futures):
        try:
            fut.result()
        except Exception:
            continue
return list(records)
```

`_validate_batch()` itself needs no change — it already just takes whatever `batch` it's given. One caveat to flag to engineering: this issues one re-fetch of the same URL per sub-batch instead of one re-fetch per URL. `search.fetch(..., use_cache=False)` deliberately bypasses the cache for a "genuine re-fetch," so as written this would re-fetch the same document N times for a source with N sub-batches. Cheapest correct fix: split the re-fetch out of `_validate_batch` so it happens once per URL and the parsed document is passed into each sub-batch's validation call — worth doing together with the batching change, not after.

---

## Fix 2 — runtime: 40 min is coming from three specific serial sections, not from the model being slow

I checked what's already parallelized (`ThreadPoolExecutor`, confirmed via grep) vs. what silently runs one call at a time. Retrieval (A7 `execute_plan`), extraction (A8 `extract_from_documents`), and claim validation (A10 `validate_claims`) are **already** parallel with `max_workers=8`. Three other places are not, and they're the ones adding the extra ~20 minutes:

### 2a. The two populations run fully sequentially

`[CODE]` — `orchestrator.py` lines 303-313:
```python
for pop in populations:
    pass_result = _run_retrieval_pass(pop, intervention, areas, inventory,
                                      indication_record, guard, providers, opts,
                                      extraction_cache)
    ...
```
`_run_retrieval_pass()` runs the *entire* A6→A10 chain for one population — vocabulary build, query planning, web retrieval, PubMed/CT.gov, extraction, grounding, claim validation. The code comment above this loop is explicit that the two populations are intentionally isolated with no shared state (see the correction I sent earlier on the extraction-cache addition) — which also means there's no correctness reason they can't run **concurrently**. Right now a 2-population run pays the full single-population wall-clock time twice, back to back. This is very likely the single largest contributor to the 20-minute increase.

Fix — run the passes in parallel, guarding the shared `extraction_cache` dict with a lock since two threads can now race on the same document:
```python
import threading
from concurrent.futures import ThreadPoolExecutor as _TPE

extraction_cache: Dict[str, List[EvidenceRecord]] = {}
cache_lock = threading.Lock()

def _run_one(pop):
    return _run_retrieval_pass(pop, intervention, areas, inventory, indication_record,
                               guard, providers, opts, extraction_cache, cache_lock)

with _TPE(max_workers=len(populations)) as pool:
    pass_results = list(pool.map(_run_one, populations))

for pop, pass_result in zip(populations, pass_results):
    plans_by_population.append((pop.population_id, pass_result.vocab, pass_result.plan))
    records.extend(pass_result.records)
    attempts.extend(pass_result.attempts)
    documents.extend(pass_result.documents)
    all_trials.extend(pass_result.trials)
    all_publications.extend(pass_result.publications)
    blocked_total += pass_result.blocked_count
```
`_run_retrieval_pass()` needs the lock threaded through to the cache read/write block (lines 188-208 of the current file) — wrap the `cache[d.source_id] = ...` write and the `d.source_id in cache` reads in `with cache_lock:`. `opts.emit(...)` calls from two threads will interleave in the progress log, which is cosmetic only.

### 2b. a06.query_vocabulary's language batches run sequentially

`[CODE]` — `agents/a06_a08_retrieval.py`, `build_vocabulary()`, lines 106-125: the very split that fixed the truncation bug (23 languages → ~4 batches of 6) turned 1 slow, truncating call into 4 small, sequential, non-truncating calls — correct for accuracy, but each is a full LLM round trip and they run one after another, once per population. Fix — same `ThreadPoolExecutor` pattern already used elsewhere in this file:
```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def _call_batch(batch_languages):
    payload = (f"INDICATION: {indication}\nDISEASE SUBTYPE: {subtype}\n"
               f"THERAPEUTIC AREA(S): {area_text}\n"
               f"LANGUAGES: {', '.join(batch_languages)}\n")
    return llm.call_json("a06.query_vocabulary", payload,
                         max_tokens=C.LLM.query_vocabulary_max_tokens, default={}) or {}

with ThreadPoolExecutor(max_workers=min(4, len(batches))) as pool:
    parsed_batches = list(pool.map(_call_batch, batches))

for parsed in parsed_batches:
    if not isinstance(parsed, dict):
        continue
    synonyms.extend(parsed.get("indication_synonyms") or [])
    abbreviations.extend(parsed.get("indication_abbreviations") or [])
    disease_class_terms.extend(parsed.get("disease_class_terms") or [])
    outcome_requirement_terms.extend(parsed.get("outcome_requirement_terms") or [])
    batch_localised = parsed.get("localised_assessment_terms") or {}
    for k, v in batch_localised.items():
        if isinstance(v, list):
            localised.setdefault(k, []).extend(v)
```
Combined with fix 2a (populations running concurrently too), this turns roughly `2 populations × 4 sequential calls` into effectively one round trip's worth of wall time.

### 2c. A12 scope adjudication is the biggest hidden serial loop in the whole pipeline

`[CODE]` — `orchestrator.py` lines 383-390, inside `for key, recs in groups.items(): ...`:
```python
adjs = {b.population_id: validation.adjudicate_scope(
            key, comp, recs, b, intervention, providers.llm)
        for b in boundaries}
```
This dict comprehension calls the LLM **once per candidate comparator group, per population**, one at a time, with zero concurrency — this is a plain `for`, not a `ThreadPoolExecutor`. Your Run 5 `Bedrock Calls` sheet already showed 14 of these calls in one run; with more candidate groups or more populations this scales linearly and serially. Every one of these 14+ calls pays full network + generation latency back to back, and this is very likely the second-largest contributor to the slowdown, right behind 2a.

Fix — flatten to `(key, population)` work items and run them through a shared pool, same pattern as A7/A8/A10 already use:
```python
from concurrent.futures import ThreadPoolExecutor, as_completed

work = [(key, recs, b) for key, recs in groups.items() for b in boundaries]

def _adjudicate_one(item):
    key, recs, b = item
    comp = recs[0].comparator
    return key, b.population_id, validation.adjudicate_scope(
        key, comp, recs, b, intervention, providers.llm)

adjs_by_key: Dict[str, Dict[str, Any]] = {key: {} for key in groups}
with ThreadPoolExecutor(max_workers=opts.max_workers) as pool:
    futures = [pool.submit(_adjudicate_one, item) for item in work]
    for fut in as_completed(futures):
        key, pid, adj = fut.result()
        adjs_by_key[key][pid] = adj

for key, recs in groups.items():
    comp = recs[0].comparator
    adjs = adjs_by_key[key]
    all_adjudications[key] = {...}   # unchanged from here down
    ...
```
The rest of the loop body (building `all_adjudications`, choosing `best`, the `in_scope`/`out.validation.excluded` branch) is untouched — only the adjudication calls themselves move off the serial path.

### What I'd expect this to buy you

2a and 2c are the two changes worth doing first — they're the largest serial blocks and the lowest-risk to parallelize (no shared mutable state in 2c at all; 2a needs one lock). 2b is smaller but free to do at the same time since it's the same pattern. None of these change what gets sent to the model or how a verdict is decided — they only change how many of those calls happen at once, so accuracy should be unaffected; Fix 1 (the cap raise + batch-size bound) is the only change here that affects what a10 actually decides, and it should only ever let MORE genuinely-supported claims through, not fewer.

I can't promise an exact number without a timed re-run, but structurally: if population count is 2, fix 2a alone should remove close to half of the A6-A10 wall time, and fix 2c should remove most of A12's current serial cost (14+ sequential LLM round trips down to roughly 2 in parallel, capped by `max_workers`). Worth timing a run after 2a+2c alone before also doing 2b, so you can see each fix's real contribution rather than one combined number.
