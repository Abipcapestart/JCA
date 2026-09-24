# Implementation prompt — CT.gov structured-registry bundling, self-referential comparator, and pre-A11 deduplication

Paste this to whoever implements the fix. It's self-contained: I traced all three issues your SME team flagged to exact lines in `jca_phase1/`, confirmed the root cause of each by reading the actual logic (not guessing from symptoms), and Diwyashri's architectural point is correct and already has a concrete, low-risk landing spot in the code — I found it below.

All three issues share one root cause category: **the deterministic ClinicalTrials.gov ingestion path (`providers/registries.py` + `agents/a06_a08_retrieval.py`'s `records_from_trials()`) makes structural assumptions about trial-arm data that don't hold for every trial design, and it runs entirely outside A8's LLM extraction — so any prompt-level fix aimed at A8/A11/A12 literally cannot reach a bug here.** That's exactly what your team already noticed. Below is where each one lives and what to change.

---

## Issue 1 — Investigator's-choice / Standard-of-Care arms get bundled into a fake combination

**Where:** `agents/a06_a08_retrieval.py`, `records_from_trials()`, lines 705-729.

**The exact bug, read from the code:**
```python
name = ", ".join(arm.interventions) or arm.label
if not name:
    continue
is_combo = len(arm.interventions) > 1
rec = EvidenceRecord(
    ...
    comparator=Comparator(
        as_stated=name,
        role=(C.ROLE_ACTIVE_COMPARATOR if "COMPARATOR" in (arm.arm_type or "").upper() else C.ROLE_UNCLEAR),
        is_combination=is_combo,
        components=list(arm.interventions)),
    ...
```
`is_combo = len(arm.interventions) > 1` treats "this arm's intervention list has more than one entry" as sufficient evidence that it's a combination regimen given together. That's true for a genuine combination arm (e.g., dabrafenib + trametinib given concurrently) but false for a "Standard of Care" or "Investigator's Choice" arm, where ClinicalTrials.gov's `armGroups[].interventionNames[]` lists several *alternative* single-agent options a patient could receive — e.g. `["Lurbinectedin", "Topotecan", "Amrubicin"]` — not a regimen combining all three. The current code can't tell these apart and always picks "combination," which is wrong for the SoC case and produces a single fake multi-drug "comparator" instead of three real, distinct ones. Note this confirms exactly what your SME flagged: this path is entirely deterministic (`records_from_trials` bypasses `_extract_one`/A8 completely, per its own docstring: *"Typed trial arms straight into evidence records — no LLM"*), so no A8 prompt fix (the "MULTIPLE COMPARATORS NAMED TOGETHER" instruction) can ever run against it.

**The fix:**

1. **Add a detector for "these are alternatives, not a combination"** — based on `arm.label` and `arm.description` (both already captured in `TrialArm`, `providers/registries.py` line 53-58), not on the interventions list alone, since the interventions list is exactly the thing that's ambiguous. Look for markers CT.gov trial designers reliably use for this exact case:
   ```python
   _CHOICE_MARKERS = re.compile(
       r"\b(investigator'?s?\s+choice|physician'?s?\s+choice|treating\s+physician'?s?\s+choice|"
       r"at\s+the\s+discretion\s+of|per\s+investigator|one\s+of\s+the\s+following|"
       r"standard\s+of\s+care\b.*\bor\b|patient'?s?\s+choice)",
       re.IGNORECASE)
   ```
   This lives in `providers/registries.py`, near `_INTERVENTION_TYPE_PREFIX` (line 43), since it's the same kind of "typed-field-needs-a-deterministic-parse-rule" logic already established there.

2. **When the marker matches (or, as a second signal, when `arm.arm_type` is one of the comparator types AND the arm label itself contains "standard of care"/"SoC" alongside >1 intervention), split instead of join.** In `records_from_trials()`, replace the single-record-per-arm logic with: for a detected choice/SoC arm, emit **one `EvidenceRecord` per intervention** in `arm.interventions`, each with `is_combination=False`, a `comparator_scenario` of `"at_least_one"` (this value already exists as a concept in your `a08.extraction` prompt's `comparator.comparator_scenario` schema for exactly this "several, at least one acceptable" case — reuse it here rather than inventing a parallel vocabulary), and the same shared trial citation/evidence quote so a reviewer can see they came from the same arm. For a confirmed true combination (no choice marker, and ideally a positive signal like "in combination with" / "plus" / "+" in the label), keep the current join-as-one-record, `is_combination=True` behavior — that part isn't broken.

3. **Where no marker is found and it's genuinely ambiguous** (interventions.length > 1, no choice language, no combination language either): don't silently guess. Default to splitting (each in the JCA/comparator-scoping context is more useful evaluated as an individual candidate than merged into an uninterpretable joint name), but set `comparator_scenario` to a value that flags the ambiguity for the reviewer rather than asserting either one confidently — e.g. `"individualised"` if that fits your existing vocabulary's meaning, since your `a08.extraction` prompt already defines it as "a bundle of options chosen by patient characteristics," which is the honest default when you can't tell why the arm lists more than one drug.

**Test without a full run:** build 2-3 synthetic `TrialArm` fixtures — one clearly-labeled "Investigator's Choice of Lurbinectedin, Topotecan, or Amrubicin," one clearly a genuine combination ("Dabrafenib plus Trametinib"), one ambiguous multi-intervention arm with no marker language — and assert on `records_from_trials()`'s output directly:
```python
from jca_phase1.providers.registries import TrialRecord, TrialArm
from jca_phase1.agents.a06_a08_retrieval import records_from_trials
from jca_phase1.schema import Intervention  # adjust import to actual location

soc_arm = TrialArm(label="Standard of Care", arm_type="ACTIVE_COMPARATOR",
                    interventions=["Lurbinectedin", "Topotecan", "Amrubicin"],
                    description="Investigator's choice of lurbinectedin, topotecan, or amrubicin.")
trial = TrialRecord(identifier="NCT_TEST", arms=[soc_arm])
records = records_from_trials([trial], Intervention(product_name="Tovorafenib"))

names = sorted(r.comparator.as_stated for r in records if r.comparator)
assert names == ["Amrubicin", "Lurbinectedin", "Topotecan"], names
assert all(not r.comparator.is_combination for r in records)
print("PASS: investigator's-choice arm split into 3 standalone comparators, none flagged as combination")
```
No network, no LLM call, runs in under a second — the same style of isolated test as the fixes we validated last round.

---

## Issue 2 — Self-referential comparator: durable fix (the primary fix belongs in `comparator_arms()`, not just the A11/A12 backstop)

**Where:** `providers/registries.py`, `TrialRecord.comparator_arms()`, lines 75-101.

**The exact gap:**
```python
def comparator_arms(self, subject_drug: str) -> List[TrialArm]:
    tok = _tokens(subject_drug)
    out = []
    for a in self.arms:
        arm_type_upper = a.arm_type.upper()
        intervention_tokens = _tokens(" ".join(a.interventions))
        if (arm_type_upper not in _COMPARATOR_ARM_TYPES
                and tok and intervention_tokens == tok):
            continue            # exact match to the subject's own name
        ...
```
The exact-identity exclusion (`intervention_tokens == tok`) only runs `if arm_type_upper not in _COMPARATOR_ARM_TYPES` — i.e. it is **skipped entirely** whenever ClinicalTrials.gov happens to have typed the arm as `ACTIVE_COMPARATOR`, `PLACEBO_COMPARATOR`, `SHAM_COMPARATOR`, `NO_INTERVENTION`, or `OTHER`. The code's own comment defends this as intentional — "an arm explicitly typed as a comparator is never excluded by name, however similar it looks... a biosimilar, or an ADC built on the same root antibody, shares a token with the subject drug's name but is a genuinely different, legitimate comparator." That reasoning is sound for a *partial* token overlap (a biosimilar sharing a root name), but it's being applied to *exact, complete* token-set equality too — and an exact full-name match to the subject drug is never a legitimately different comparator, regardless of what `arm_type` the registry happened to assign it. This is exactly the class of defect your SME's message describes: "no check that a candidate's identity might equal the subject drug itself" — the check exists, but it's conditionally disabled precisely for the arm types where a sponsor's inconsistent CT.gov data entry (mistyping the drug's own arm as `OTHER` rather than `EXPERIMENTAL`, which happens) would most need it.

**The fix:** split the exclusion into two independent checks instead of one gated check:
```python
def comparator_arms(self, subject_drug: str) -> List[TrialArm]:
    tok = _tokens(subject_drug)
    out = []
    for a in self.arms:
        arm_type_upper = a.arm_type.upper()
        intervention_tokens = _tokens(" ".join(a.interventions))
        # Exact identity match is never a legitimate distinct comparator,
        # no matter how the registry typed the arm — this check is
        # unconditional, unlike the partial-overlap leniency below.
        if tok and intervention_tokens == tok:
            continue
        # Partial token overlap (a biosimilar, an ADC on the same root
        # antibody) IS allowed through when the registry itself calls the
        # arm a comparator type — that's the only place the leniency
        # from _COMPARATOR_ARM_TYPES should still apply.
        if (arm_type_upper not in _COMPARATOR_ARM_TYPES
                and tok and tok.issubset(intervention_tokens) and len(intervention_tokens) > len(tok)):
            # (only relevant if you want partial-superset cases excluded too;
            # otherwise this branch can be removed — the point is the exact-match
            # check above no longer depends on arm_type at all)
            pass
        if arm_type_upper.startswith("EXPERIMENTAL") and not a.interventions:
            continue
        out.append(a)
    return out
```
The essential change is just: **move the exact-match check out from behind the `arm_type_upper not in _COMPARATOR_ARM_TYPES` gate so it runs for every arm.** Keep the biosimilar/ADC leniency only for genuinely partial overlaps.

**Also apply this at the split point from Issue 1** — once a bundled SoC arm gets split into individual candidates (e.g., "Tovorafenib continuation, Topotecan, or Amrubicin"), re-check each individual split-out intervention against the subject drug identity, not just the joined string. Otherwise the subject drug could reappear as one of the split "comparator" options even after Issue 1 is fixed.

**Test without a full run:**
```python
from jca_phase1.providers.registries import TrialRecord, TrialArm

# The exact failure mode: subject drug's own arm mistyped as OTHER instead of EXPERIMENTAL
trial = TrialRecord(identifier="NCT_TEST2", arms=[
    TrialArm(label="Arm 1", arm_type="OTHER", interventions=["Tovorafenib"]),
    TrialArm(label="Arm 2", arm_type="ACTIVE_COMPARATOR", interventions=["Dabrafenib", "Trametinib"]),
])
result = trial.comparator_arms("Tovorafenib")
names = [", ".join(a.interventions) for a in result]
assert "Tovorafenib" not in names, f"Self-match leaked through despite arm_type=OTHER: {names}"
assert "Dabrafenib, Trametinib" in names, "Genuine comparator was wrongly excluded"
print("PASS: exact self-match excluded regardless of arm_type; genuine comparator kept")
```

---

## Issue 3 — Diwyashri's architectural point: deterministic pre-clustering before A11

**This is correct, and there's already good evidence for it in your own data** — Run 4's `a11.comparator_identity` call successfully grouped 4 near-duplicate name variants ("Dabrafenib + Trametinib," "Dabrafenib (Finlee) + Trametinib," "Trametinib (Spexotras) + Dabrafenib," "Trametinib + Dabrafenib") into one identity, purely through prompt instructions, in a single batch. That's the prompt-engineering approach working — for now, in one batch.

**Where the ceiling is, read directly from the code:** `agents/a09_a13_validation.py`, `resolve_identities()`, lines 345-383.
```python
distinct: List[str] = []
seen = set()
for rec in records:
    if rec.comparator and rec.comparator.as_stated:
        key = _normalise(rec.comparator.as_stated)
        if key not in seen:
            seen.add(key)
            distinct.append(rec.comparator.as_stated)
...
for start in range(0, len(unresolved), batch_size):   # batch_size=40
    batch = unresolved[start:start + batch_size]
    ...
    parsed = llm.call_json("a11.comparator_identity", payload, max_tokens=4000, default=None)
```
Two things worth being precise about, since this is exactly the mechanism Diwyashri is worried about:
1. **Dedup before A11 is exact-string-match only** (`_normalise(as_stated)` in a `seen` set) — "Dabrafenib + Trametinib" and "Dabrafenib (Finlee) + Trametinib" are two different strings after normalization, so they both enter `distinct` as separate entries and rely entirely on A11's prompt-level judgment to be recognized as the same thing.
2. **`batch_size=40` means A11 only ever sees one batch at a time.** As raw candidate volume grows past 40 (which your SME's message specifically flags — "batches keep growing past 20-25 raw candidates"), two near-duplicate variants can land in *different* batches. A11 has zero visibility across batches — it cannot deduplicate two things it's never shown together in the same call, no matter how good the prompt is. This isn't a prompt-quality ceiling, it's a structural one: the batching itself guarantees some future set of duplicates will be unrecoverable by A11 alone, once volume crosses the batch boundary at the wrong split point.

**The fix — insert a deterministic pre-clustering pass between building `distinct` and batching it (right between the current lines 362 and 379):**
```python
def _cluster_key(name: str) -> frozenset:
    """Canonical token set for near-duplicate clustering: strip brand-name
    parentheticals, punctuation, and connector words, leaving just the
    substantive tokens, order-independent."""
    # Drop parenthetical brand annotations first: "Dabrafenib (Finlee)" -> "Dabrafenib"
    stripped = re.sub(r"\([^)]*\)", " ", name)
    return frozenset(_substance_tokens(stripped))  # reuses the existing _STOP-filtered tokenizer

def precluster_candidates(distinct: List[str]) -> Dict[str, List[str]]:
    """Group near-duplicate candidate strings by identical canonical token
    set. Returns {representative: [all variants including representative]}.
    Runs before A11 sees anything, so duplicates can never split across
    batches regardless of how many raw candidates there are."""
    clusters: Dict[frozenset, List[str]] = {}
    for name in distinct:
        key = _cluster_key(name)
        clusters.setdefault(key, []).append(name)
    # Representative = shortest variant (usually the cleanest, un-annotated form)
    return {min(variants, key=len): variants for variants in clusters.values()}
```
Then in `resolve_identities()`, replace direct iteration over `distinct` with:
```python
clusters = precluster_candidates(distinct)
representatives = list(clusters.keys())
# ... batch `representatives` to A11 instead of `distinct` ...
# after resolving each representative's identity, apply the SAME resolved
# Comparator to every variant in clusters[representative] via `resolved[_normalise(variant)] = comp`
```
This guarantees near-duplicates are collapsed *before* the batch boundary exists, so growing past 40, past 100, past any number of raw candidates no longer risks splitting a duplicate pair across batches — the clustering pass sees the entire `distinct` list at once, in code, before any batching happens.

**Where to draw the line for "near-duplicate" vs "genuinely different comparator":** start conservative — exact canonical-token-set equality only (as coded above), which safely handles punctuation/brackets/brand-name annotations/word-order differences without risking merging two genuinely different regimens that happen to share tokens (e.g., don't fuzzy-match "Carboplatin + Vincristine" against "Carboplatin + Vincristine + Etoposide" — those are different regimens, and exact-set equality correctly keeps them apart since their token sets differ). If exact-set clustering isn't catching enough duplicates in practice, escalate to a similarity threshold (e.g., Jaccard ≥ 0.85) as a second pass — but ship the conservative version first and measure on real data before loosening it, exactly as Diwyashri's message suggests: try the strong, precise version first, escalate only if it doesn't hold.

**Test without a full run:**
```python
from jca_phase1.agents.a09_a13_validation import precluster_candidates

variants = ["Dabrafenib + Trametinib", "Dabrafenib (Finlee) + Trametinib",
            "Trametinib (Spexotras) + Dabrafenib", "Trametinib + Dabrafenib",
            "Carboplatin + Vincristine", "Carboplatin, Vincristine and Etoposide"]
clusters = precluster_candidates(variants)
assert len(clusters) == 3, f"Expected 3 clusters (D+T, CV, CVE), got {len(clusters)}: {clusters}"
dt_cluster = next(v for v in clusters.values() if len(v) == 4)
assert set(dt_cluster) == {"Dabrafenib + Trametinib", "Dabrafenib (Finlee) + Trametinib",
                           "Trametinib (Spexotras) + Dabrafenib", "Trametinib + Dabrafenib"}
print("PASS: 4 D+T variants clustered together; CV and CV+Etoposide correctly kept separate")
```
This is the single most valuable test here — it directly reproduces the exact 4-variant case you already saw in Run 4's real output and proves the code collapses them without needing any LLM call at all, which is the whole point of moving this out of prompt-engineering territory.

---

## Summary for the ML team (per Diwyashri's ask)

Flag this explicitly rather than letting it stay implicit: **the A11 prompt-level dedup is working today, on today's data volumes.** It is not a bug to leave alone, but it is also not the place to keep investing more prompt text if this resurfaces — the fix above (Issue 3) removes the batch-size dependency entirely, so it's worth doing proactively rather than waiting for a real cross-batch duplicate to slip through in production first.
