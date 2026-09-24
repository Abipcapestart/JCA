# PubMed query construction: why `indication_synonyms` isn't reaching the search string

Your SME's diagnosis is exactly right, and I traced it to the precise lines. Answering both parts of their message.

## "A6 Query Plan (Mix) doesn't have PubMed" — expected, not a bug

`[CODE]` — `agents/a06_a08_retrieval.py`, `retrieve_structured()` (lines 418-468). PubMed (and ClinicalTrials.gov) retrieval never goes through `plan_queries()` — the function that populates the "A6 Query Plan" export sheet. It's called separately, directly against the NCBI E-utilities API via `PubMedClient.search()`, with its own two hardcoded query strings built inline. So "not in A6 Query Plan" is correct and by design — same reason trial-registry queries don't show up there either. Nothing to fix on that point.

## The real finding — `indication_synonyms` is almost entirely discarded before it ever reaches PubMed

`[CODE]` — the two PubMed queries, lines 426-452:
```python
condition = population.value("indication_disease")   # ONE raw string
drug = intervention.product_name
...
broad = (vocab.disease_class_terms or vocab.indication_synonyms or [condition])[0]
pubs = literature.search(f"{drug} {condition}", max_results=15)
pubs += literature.search(broad, publication_types=["Practice Guideline", "Guideline"], max_results=15)
```

Walking through exactly what happens to `vocab.indication_synonyms` (the field your query-vocabulary fix was meant to improve):

- **Query 1** (`f"{drug} {condition}"`) — the main drug+disease query, the one most likely to surface a specific therapeutic paper — uses `condition`, the raw `indication_disease` input string. **It never references `vocab.indication_synonyms` at all.** Whatever synonyms the vocabulary call now successfully returns, this query is blind to every one of them.
- **Query 2** (`broad`, restricted to `Practice Guideline`/`Guideline` publication types) — this is the only place `indication_synonyms` can enter at all, and only as a fallback: `disease_class_terms` wins if it has anything, `indication_synonyms` is only tried if `disease_class_terms` is empty. And even then, **`[0]`** — only the first element of whichever list wins is used. If `indication_synonyms` now correctly returns 5 good alternate phrasings, 4 of them are thrown away, and the 1 that survives only ever reaches a guideline-type-restricted query, not a general search.

So your SME's instinct is exactly correct: **"the query vocabulary fix appears to be working structurally, but I can't yet confirm it changed the actual search string enough"** — it structurally *can't*, as currently wired. If the specific paper they're looking for uses a phrasing that isn't the literal `indication_disease` string and isn't the one lucky first synonym that happens to survive into the guideline-only query, no combination of vocabulary-fix improvements will ever surface it, because the code discards the rest before it's ever tried.

## The fix

`PubMedClient.search()` (`providers/registries.py`, lines 260-286) passes its `query` argument straight to NCBI's ESearch as a free-text `term` — confirmed it already supports standard boolean syntax (`" OR "` is literally used one line later, at line 265, to join publication-type filters the same way). So combining multiple synonyms is a query-string change only; no client changes needed.

**File: `agents/a06_a08_retrieval.py`**, `retrieve_structured()`, replace lines 446-452:
```python
# Use every synonym PubMed can search on, not just one. PubMed/Entrez syntax
# supports OR directly in the term string -- no need for N separate calls.
synonym_terms = _dedup([condition] + (vocab.indication_synonyms or []))
disease_query = " OR ".join(f'"{s}"' if " " in s else s for s in synonym_terms)

pubs = literature.search(f"{drug} AND ({disease_query})" if len(synonym_terms) > 1
                         else f"{drug} {condition}", max_results=15)

broad_terms = _dedup((vocab.disease_class_terms or []) + (vocab.indication_synonyms or [condition]))
broad_query = " OR ".join(f'"{s}"' if " " in s else s for s in broad_terms)
pubs += literature.search(broad_query, publication_types=["Practice Guideline", "Guideline"],
                          max_results=15)
```
This uses `_dedup()` (already defined and imported in this file, line 109) to merge and de-duplicate the synonym list, and quotes any multi-word phrase so PubMed treats it as one term rather than an implicit AND of separate words. The de-dup-by-pmid merge that already happens right after (lines 453-457) is unaffected — it's still just combining two result lists.

**Test without a full run** — call the client directly with the two query shapes and confirm the synonym-bearing one changes what comes back:
```python
from jca_phase1.providers.registries import PubMedClient

client = PubMedClient(api_key=<your key>, email=<your email>)

old_style = client.search("Tovorafenib paediatric low-grade glioma", max_results=15)
new_style = client.search(
    'Tovorafenib AND ("paediatric low-grade glioma" OR "pediatric low-grade glioma" OR pLGG)',
    max_results=15)

old_pmids = {p.pmid for p in old_style}
new_pmids = {p.pmid for p in new_style}
print(f"old query: {len(old_pmids)} results")
print(f"new query: {len(new_pmids)} results, {len(new_pmids - old_pmids)} new vs old")
# Check specifically whether the paper your SME was chasing is in new_pmids but not old_pmids
```
This directly answers "did widening the query actually surface anything new" in one call, rather than needing another full run to find out.

One more thing worth flagging to the ML team while they're in this code: `condition = population.value("indication_disease")` is read fresh from the population object independently of `vocab` — worth double-checking that whatever normalization/abbreviation-expansion the vocabulary call does to `indication_disease` (if any) is consistent with what feeds `condition` here, since right now they're two separate reads of what should be the same underlying concept.
