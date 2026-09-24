# Honest end-to-end validation — Run `RUN-20260924T130834-fbc492` (Tovorafenib)

This is back to Tovorafenib, so I have the GT ledger to score against. I pulled every comparator's evidence and sources, fetched three of the actual cited URLs live to check the extracted quotes against the real page content, and traced every anomaly back to its root cause in the data. `[XLSX]` = from this run's export, `[WEB]` = confirmed by fetching the live source just now.

---

## 1. The real headline: the line-of-therapy fix is genuinely working, and I verified it against the actual sources

This is the most important result in this run. `[XLSX]` — **Vinblastine, Carboplatin-Vincristine, Dabrafenib+Trametinib, and Trametinib monotherapy — 4 of your 7 GT comparators — are now surfacing**, up from as few as 0-1 in earlier rounds. This is the A12 line-of-therapy softening your SME shipped as a prompt change, and it's doing exactly what it was designed to do:

- **Vinblastine**: `in_scope` for both populations. `[XLSX]` reasoning: *"The Finnish HTA source explicitly places single-agent vinblastine as a comparator in the relapsed/refractory pLGG setting... matching the requested 2L+ population despite other sources describing first-line use."* I fetched the SIOPE guideline PDF and the FIREFLY-1 Nature Medicine paper directly. `[WEB]` — SIOPE genuinely only describes vinblastine as first-line/salvage, confirming the extraction is accurate, not hallucinated; and the Nature Medicine paper genuinely states *"the FIREFLY-1 trial met its primary endpoint by rejecting the null hypothesis ORR of 21% observed for single-agent vinblastine in this setting"* — and "this setting" is FIREFLY-1's own relapsed/refractory population. So the reasoning is correct and the source really does support vinblastine as a relapsed/refractory-setting comparator. This is a genuine, source-verified win.
- **Carboplatin-Vincristine**: `uncertain` (not silently dropped) for both populations, with an honest reason — *"All retrieved evidence describes carboplatin-vincristine exclusively as the first-line standard systemic treatment... no source addressing its use in the relapsed/refractory 2L+ setting requested"* — and it still appears in the final comparator list, flagged rather than hidden. This is exactly the intended behavior: real conflicting evidence → `in_scope`; only one-sided 1L evidence → `uncertain`, surfaced, not excluded.
- **Dabrafenib+Trametinib**: `in_scope` (intended-to-treat population) / `uncertain` (licensed population) — both population verdicts now weigh the prior-therapy-framed French sources against the active-comparator-framed Italian sources honestly, rather than excluding on the first prior-therapy mention.

Report this to your team as a real win — this is the single biggest quality improvement across the whole engagement so far, and I didn't just take the export's word for it; I checked it against the live sources myself.

---

## 2. A new, serious bug I found this run: `a11.comparator_identity`'s token cap was never raised, and it's truncating on every real run

`[XLSX]` — `Bedrock Calls` shows exactly **one** `a11.comparator_identity` call this run, with `output_tokens: 4000` — precisely at its cap. That's not a coincidence; it's truncation. I checked the code: every other prompt that hit this same problem (`a06.query_vocabulary`, `a10.claim_validation`, `a12.scope_adjudication`, `a14.outcome_harmonization`) had its cap centralized into `config.py`'s `LLMSettings` and raised. `a11.comparator_identity` alone still has a bare, hardcoded `max_tokens=4000` literal in `agents/a09_a13_validation.py`'s `resolve_identities()` function — it was missed.

**This is the direct root cause of the messiest part of this run's output.** When A11's response truncates mid-batch, every comparator past the truncation point falls back to "unresolved" — no INN, no translation, its own raw wording kept as-is. That's exactly what I found in `A11 Comparator Identity`: 18 entries with `class_source: unresolved`, including real, resolvable single-agent drugs that should never have ended up unresolved — **`trametinib en monothérapie`** is genuinely just trametinib, a well-known drug with a real INN, and it never got resolved to it. The result in the final `A16 Comparators` list is a mess of untranslated, unmerged fragments that are really only 2-3 underlying concepts: `chemioterapia` (Italian), `chimiothérapies`/`chimiothérapie` (French), `karboplatyna i winkrystyna` (Polish for "carboplatin and vincristine" — the SAME regimen as the separately-listed "Carboplatin-Vincristine regimen"), `conventional chemotherapy`, `current SOC chemotherapy`, `current standard of care (SoC) chemotherapy / investigator's choice of prespecified SoC chemotherapy regimens`, and `investigator's choice of prespecified SoC chemotherapy regimens` — seven separate rows that should very likely be far fewer.

**Fix — the same pattern already applied everywhere else, this is the one place it was skipped:**
```python
# config.py, LLMSettings — add:
comparator_identity_max_tokens: int = 8000   # was inline 4000 in resolve_identities(),
                                              # the one prompt this fix pattern missed

# agents/a09_a13_validation.py, resolve_identities() — replace max_tokens=4000 with:
max_tokens=C.LLM.comparator_identity_max_tokens
```
This is very likely the single highest-value fix available right now — it should both properly translate/merge the foreign-language fragments above AND correctly resolve real drugs like trametinib monotherapy to their actual INN, which also feeds into the `apply_identities()` display-wording fix I already gave you (that fix only helps once A11 successfully resolves something to propagate in the first place — this cap is the upstream blocker).

---

## 3. Still open, unchanged from before

- **`apply_identities()` doesn't propagate `as_stated`** — still not shipped. Combined with fix #2 above, this pair should meaningfully clean up the final list.
- **A15/A16 rationale loops still sequential** — `Stage Latency` shows A15 (282.9s) + A16 (97.5s) = 380.4s, 23.5% of this run's 1616.8s total, from 43+16+5=64 sequential LLM calls. Not yet parallelized.
- **Everolimus and Bevacizumab+chemo — GT comparators #5 and #6 — have zero hits anywhere in this run**, not in `A7 Retrieved Documents`, not in `A8 Extraction`. This has been a gap since early rounds and nothing in this run's data explains why; worth a dedicated retrieval-side investigation (are queries even being planned for these, or are they just not being found).
- **TPCV**: only a partial match this run (`PCV — prokarbazyna, lomustyna, winkrystyna`, missing the "T"), and it was correctly excluded at claim validation because the source itself doesn't confirm whether the recommendation applies to the paediatric population — a legitimate exclusion, not a bug, but TPCV remains effectively absent from the surfaced candidate pool.

## 4. One honest limitation in my own verification

I tried to fetch the HAS France committee transcript (`has-sante.fr/.../ojemda-24062026-transcription-ct-ap583`) to independently confirm the French dabrafenib+trametinib quotes — the fetch came back with only the page title, no body text, most likely because the page renders its content client-side (JavaScript) rather than serving it in the initial HTML. I can't independently confirm those specific quotes right now; I'm flagging that rather than implying I checked something I didn't. The other two spot-checks (SIOPE guideline, FIREFLY-1 paper) came back clean and accurate.

---

## Priority order for your team

1. **`a11.comparator_identity` max_tokens fix (§2)** — ML side, one config change + one line, highest expected impact, do this first.
2. **`apply_identities()` as_stated propagation** — ML side, already specified, pairs with #1.
3. **A15/A16 parallelization** — ML side, same pattern used 4 times already, ~24% of runtime.
4. **Everolimus/Bevacizumab retrieval gap** — needs investigation (ML side query construction, possibly also a question for your SME on whether these are expected to be findable from your curated source list at all).
5. Line-of-therapy fix (A12 prompt) — **no action needed, confirmed working correctly on real data.**
