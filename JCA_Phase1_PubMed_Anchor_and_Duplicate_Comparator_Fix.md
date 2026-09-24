# Two code fixes for the ML team

## Fix 1 — anchor the disease term in PubMed queries (query-construction, not a prompt)

**File:** `agents/a06_a08_retrieval.py`, function `retrieve_structured()`.

**Problem:** the guideline-restricted and landscape PubMed queries are built from `vocab.disease_class_terms` / `vocab.indication_synonyms` (A6's output) joined with plain `OR`, with a silent fallback to `condition` (A2's full raw `indication_disease` sentence) when A6's vocab returns nothing for those fields:
```python
broad_terms = _dedup((vocab.disease_class_terms or []) + (vocab.indication_synonyms or [condition]))
broad_query = " OR ".join(f'"{s}"' if " " in s else s for s in broad_terms)
```
Neither case forces the actual disease name to be present in a match — an `OR`'d list lets a document match on one loosely-related term alone, and the fallback submits an entire clause-heavy sentence as one unstructured string. Both produce noise (confirmed: unrelated guideline documents — gout, UTI, Cushing's syndrome — matching on generic words like "systemic," "therapy," "progression").

**Fix — pick one anchor term and AND it against the rest, using PubMed's own field-tag syntax so a match always requires the real disease name:**
```python
def _build_anchored_disease_query(vocab: QueryVocabulary, condition: str) -> str:
    """Force the core disease term as a required match (PubMed field-tag
    syntax), ANDed with the broader synonym/class-term list -- rather than
    OR'ing everything together, where one loosely-related term alone can
    satisfy the whole query, or falling back to an entire unstructured
    sentence as a single search term."""
    anchor_candidates = (vocab.disease_class_terms or []) + (vocab.indication_synonyms or [])
    anchor = min(anchor_candidates, key=len) if anchor_candidates else condition
    anchor_term = f'"{anchor}"[Title/Abstract]' if " " in anchor else f"{anchor}[Title/Abstract]"

    other_terms = _dedup([t for t in anchor_candidates if t != anchor])
    if not other_terms:
        return anchor_term
    other_query = " OR ".join(f'"{s}"' if " " in s else s for s in other_terms)
    return f"{anchor_term} AND ({other_query})"
```
Replace the `broad_query` construction (both the guideline-restricted call and the landscape/unrestricted call, since they currently reuse the same `broad_query` variable) with `broad_query = _build_anchored_disease_query(vocab, condition)`.

**Also worth checking before this ships:** pull the specific run that showed the gout/UTI/Cushing's noise and look at its `A6 Query Plan` export for that request's `disease_class_terms` and `indication_synonyms`. If both were empty, that confirms the fallback-to-`condition` path is what fired — which also means it's worth making `a06.query_vocabulary` more reliably non-empty for `disease_class_terms` (a prompt-side improvement, separate from this fix), so the anchor function above has real short terms to pick from rather than needing `condition` as its own anchor as often.

---

## Fix 2 — propagate the pre-clustered comparator's display wording, not just its identity

**File:** `agents/a09_a13_validation.py`, function `apply_identities()`.

**Problem:** `precluster_candidates()` correctly clusters near-duplicate comparator strings (e.g. `"standard of care"` and `"standard of care (SOC) chemotherapy"`) before A11 ever sees them, and `resolve_identities()` correctly resolves them to one shared identity. But `apply_identities()` only copies the resolved `inn`, `atc_code`, `class_mechanism`, `class_source`, `is_combination`, and `components` back onto each variant — never `as_stated`:
```python
def apply_identities(records: Sequence[EvidenceRecord],
                     identities: Dict[str, Comparator]) -> None:
    for rec in records:
        if not rec.comparator:
            continue
        ident = identities.get(_normalise(rec.comparator.as_stated))
        if not ident:
            continue
        rec.comparator.inn = ident.inn
        rec.comparator.atc_code = ident.atc_code
        rec.comparator.class_mechanism = ident.class_mechanism
        rec.comparator.class_source = ident.class_source
        rec.comparator.is_combination = ident.is_combination or rec.comparator.is_combination
        if ident.components:
            rec.comparator.components = ident.components
```
`group_comparators()` (A14) groups on `identity_key(comp)`, which is `comp.inn` when known — but falls back to `_normalise(comp.as_stated)` whenever `inn` is empty, which is every category/generic comparator (no real substance to resolve to). Since `as_stated` was never unified, each variant keeps its own original wording and fragments back into separate A14 groups, separate A12 adjudications, and separate final rows — even though A11 correctly identified them as the same thing.

**Fix — one line, add it after the existing `class_source` assignment:**
```python
def apply_identities(records: Sequence[EvidenceRecord],
                     identities: Dict[str, Comparator]) -> None:
    for rec in records:
        if not rec.comparator:
            continue
        ident = identities.get(_normalise(rec.comparator.as_stated))
        if not ident:
            continue
        rec.comparator.inn = ident.inn
        rec.comparator.atc_code = ident.atc_code
        rec.comparator.class_mechanism = ident.class_mechanism
        rec.comparator.class_source = ident.class_source
        rec.comparator.is_combination = ident.is_combination or rec.comparator.is_combination
        if ident.components:
            rec.comparator.components = ident.components
        # Propagate the cluster's canonical display wording too, not just its
        # substance identity -- without this, a category/generic comparator
        # with no INN (e.g. "standard of care") still fragments at A14's
        # identity_key() fallback, because that key uses as_stated when inn
        # is empty, and as_stated was never unified across variants here.
        rec.comparator.as_stated = ident.as_stated or rec.comparator.as_stated
```

**Note for whoever implements this:** `ident.as_stated` is the representative's own resolved display wording (already set in `resolve_identities()`, e.g. via `comp.as_stated = display or name` for the LLM-resolved path, or `entry["inn"]`-derived for the vocabulary-matched path). This does not merge genuinely different specificity levels (e.g. `"chemotherapy"` vs. `"second-line chemotherapy"` will still stay separate, correctly, since `precluster_candidates()`'s token-set clustering already keeps them apart) — it only unifies wording for strings A11/precluster already agreed are the exact same identity.
