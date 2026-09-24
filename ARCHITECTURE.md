# JCA Phase 1 — Architecture

**Scope boundary.** This module implements **Phase 1 only**: Population + Intervention in,
confirmed Comparator and Outcome scope out. **PICO-set generation is Phase 2** and is absent by
design — `Phase1Output` has no `pico_sets` field, the output serialises `"pico_sets": null` with a
note so a consumer can never mistake absence for an error, and a structural assertion checks it.

---

## 1. Flow

```text
                     UI screen 1 — Define product
                                 │
   ┌─────────────────────────────▼──────────────────────────────┐
   │ A1  P&I VALIDATION                             [LLM]       │  SME Agent 1, as-is
   │     FIX blocks · CHECK advises                             │
   └─────────────────────────────┬──────────────────────────────┘
                                 │ zero FIX items
   ┌─────────────────────────────▼──────────────────────────────┐
   │ A2  INPUT STRUCTURING                          [LLM]       │  SME Agent 2, as-is
   │     every field confirmed / inferred / not_provided        │
   │     licensed population (+ ITT population if declared)     │
   └─────────────────────────────┬──────────────────────────────┘
                                 │  ← user confirms, in place on screen 1
   ┌─────────────────────────────▼──────────────────────────────┐
   │ A4  INDICATION LOCK                       [API + LLM]  NEW │
   │     EMA record → licensed indication, pivotal trials       │
   │     no authorisation yet → claimed indication wording      │
   │     emits the LEAKAGE GUARD                                │
   └─────────────────────────────┬──────────────────────────────┘
   ┌─────────────────────────────▼──────────────────────────────┐
   │ A3  POPULATION SCOPE BOUNDARY    [deterministic + 1 LLM] NEW│
   │     SME population fields → typed, checkable predicate     │
   └─────────────────────────────┬──────────────────────────────┘
   ┌─────────────────────────────▼──────────────────────────────┐
   │ A5  THERAPEUTIC AREA RESOLUTION    [keyword + LLM tiebreak] │  SME Agent 5 rules
   │     ── the result is CONSUMED, not discarded ──            │
   └─────────────────────────────┬──────────────────────────────┘
   ┌─────────────────────────────▼──────────────────────────────┐
   │ A6  QUERY PLANNING      [templates + 1 vocabulary LLM call] │
   │     vocabulary object has NO comparator field               │
   └─────────────────────────────┬──────────────────────────────┘
   ┌─────────────────────────────▼──────────────────────────────┐
   │ A7  SOURCE-ROUTED RETRIEVAL          (all in parallel)      │
   │   a HTA/regulatory     27 × domain-scoped search, localised │
   │   b Drug label         EMA resolver → full document         │
   │   c Guidelines         27 × AREA-FILTERED domains,          │
   │                        drug-anchored + LANDSCAPE passes     │
   │   d Trial registries   ClinicalTrials.gov API v2 (typed)    │
   │   e Literature         PubMed E-utilities (MeSH, pub-type)  │
   │   f Conference         search, tagged, never blended        │
   │   g General web        disabled by default (see §7 Q1)      │
   └─────────────────────────────┬──────────────────────────────┘
   ┌─────────────────────────────▼──────────────────────────────┐
   │ A8  EVIDENCE EXTRACTION            [LLM, one contract]      │
   │     typed EvidenceRecord · subject_drug mandatory           │
   │     structured API records bypass the LLM entirely          │
   └─────────────────────────────┬──────────────────────────────┘
   ┌─────────────────────────────▼──────────────────────────────┐
   │ A9  GROUNDING                      [deterministic]          │
   └─────────────────────────────┬──────────────────────────────┘
   ┌─────────────────────────────▼──────────────────────────────┐
   │ A10 CLAIM VALIDATION          [LLM + LIVE RE-FETCH]         │  SME Agent 10
   └─────────────────────────────┬──────────────────────────────┘
   ┌─────────────────────────────▼──────────────────────────────┐
   │ A11 COMPARATOR IDENTITY    [vocabulary, LLM residue]   NEW  │
   │     INN · ATC · the comparator's OWN class                  │
   └─────────────────────────────┬──────────────────────────────┘
   ┌─────────────────────────────▼──────────────────────────────┐
   │ A14 HARMONIZATION (grouping)                                │  SME Agent 11
   └─────────────────────────────┬──────────────────────────────┘
   ┌─────────────────────────────▼──────────────────────────────┐
   │ A12 SCOPE ADJUDICATION             [LLM]               NEW  │
   │     in / out / uncertain · evidence_for AND evidence_against│
   └─────────────────────────────┬──────────────────────────────┘
        ┌────────────────────────┴────────────────────────┐
   ┌────▼────────────────────────┐   ┌────────────────────▼─────┐
   │ A13 PER-STATE ASSIGNMENT NEW│   │ A15 OUTCOME CATALOG  NEW │
   │     a verdict for ALL 27    │   │     11-item catalog      │
   └────┬────────────────────────┘   └────────────────────┬─────┘
        └────────────────────────┬────────────────────────┘
   ┌─────────────────────────────▼──────────────────────────────┐
   │ A16 CONSOLIDATION      [deterministic + LLM rationale]      │  SME Agent 12
   └─────────────────────────────┬──────────────────────────────┘
   ┌─────────────────────────────▼──────────────────────────────┐
   │ A17 COMPLETENESS AUDIT             [deterministic]     NEW  │
   │     27 states × source classes × 4 outcome categories       │
   └─────────────────────────────┬──────────────────────────────┘
                                 ▼
          Phase1Output  →  UI screen 2  →  A18 user additions
                                 ▼
              handed to Phase 2 PICO-set consolidation
```

**18 components, 8 of which make LLM calls.** The SME's 12 agents do not map to 12 LLM agents,
because Agents 3–9 are seven *search configurations* feeding **one** extraction contract, and
Agents 11–12 are mostly deterministic assembly. Seven extraction prompts would be seven places
for one rule to drift.

---

## 2. Mapping to the SME base prompt

| Component | SME step | Disposition |
|---|---|---|
| A1 P&I Validation | Agent 1 | **Taken from SME — as-is** |
| A2 Input Structuring | Agent 2 | **Taken from SME — as-is** |
| A3 Scope Boundary | — | **New** |
| A4 Indication Lock | — | **New** |
| A5 Therapeutic Area | Agent 5 (first half) | **Modified** — rules kept; the result is now used |
| A6 Query Planning | — | **New, mostly deterministic** |
| A7a HTA/regulatory | Agent 3 | **Implemented inside the retrieval engine** |
| A7b Drug label | Agent 4 | **Implemented as API + retrieval** |
| A7c Guidelines | Agent 5 (second half) | **Implemented inside the retrieval engine, area-filtered** |
| A7d Trial registries | Agent 6 | **Implemented as API** |
| A7e PubMed | Agent 7 | **Implemented as API** |
| A7f Conference | Agent 8 | **Implemented inside the retrieval engine, kept separate** |
| A7g General web | Agent 9 | **Blocked pending an SME decision** — see §7 Q1 |
| A8 Extraction | implicit in Agents 3–9 | **New shared contract** |
| A9 Grounding | — | **New** (split out of what used to be "validation") |
| A10 Claim Validation | Agent 10 | **Modified — must re-fetch** |
| A11 Comparator Identity | — | **New** |
| A12 Scope Adjudication | — | **New** |
| A13 Per-State Assignment | Agents 3 + 12 | **Modified** — a verdict for all 27 |
| A14 Harmonization | Agent 11 | **Modified** — vocabulary-anchored |
| A15 Outcome Catalog | Agent 12 (outcome half) | **Modified + new catalog** |
| A16 Consolidation | Agent 12 | **Taken from SME — as-is** |
| A17 Completeness Audit | the agents' self-check lists | **Implemented as deterministic assertions** |
| A18 User Additions | step 14 | **Taken from SME — as-is** |

---

## 3. The invariants, and where they are enforced

Each of these corresponds to a defect observed in the delivered outputs.

| Invariant | Enforced at | Test |
|---|---|---|
| A comparator's class comes from its **own** INN, never the intervention's | `schema.ConsolidatedComparator.__post_init__` rejects an illegal `class_source`; `a06_a08_retrieval._extract_one` clears any class the model returns; `a09_a13_validation.resolve_identities` fills it from the ATC vocabulary | `test_class_comes_from_the_comparators_own_inn`, `test_extraction_never_sets_a_comparator_class`, `test_comparator_class_from_an_illegal_source_is_rejected` |
| Every claim carries a `subject_drug`, and another drug's comparator cannot survive | `validate_claims` deterministic pre-filter | `test_a_different_subject_drug_is_rejected_without_an_llm_call` |
| A comparator cannot be the intervention | same pre-filter | `test_the_intervention_cannot_be_its_own_comparator` |
| No comparator without grounded, validated evidence | `ConsolidatedComparator.__post_init__` | `test_comparator_without_evidence_cannot_be_constructed` |
| A facet the user never stated never rejects evidence | `ScopeFacet.is_bound`, `ScopeBoundary.unbounded_facets` | `test_unstated_facets_never_become_filters` |
| Exactly 27 Member-State entries; counts sum to 27 | `MemberStateSummary.__post_init__`, A17 assertions | `test_by_member_state_has_exactly_27_entries` |
| Every comparator carries 27 per-state verdicts | A13, A17 assertion | `test_every_comparator_gets_27_verdicts` |
| Every one of the 11 catalog outcomes has a status | A15 coverage loop, A17 assertion | `test_coverage_is_reported_for_every_catalog_item` |
| The query planner cannot name a comparator | `QueryVocabulary` has no such field; `_strip_possible_drug_names` | `test_vocabulary_cannot_carry_a_drug_name` |
| Validation re-opens the cited source | `_validate_batch` calls `fetch(..., use_cache=False)` | `test_validation_refetches_the_source` |
| A malformed validator response fails closed | `_validate_batch` | `test_unparseable_validator_response_fails_closed` |
| No PICO sets in Phase 1 | `Phase1Output`, A17 assertion | `test_phase1_output_has_no_pico_sets` |

---

## 4. Retrieval routing

```text
CURATED ENTRY THAT IS A DOCUMENT URL   (114 of 340 in the current workbook)
        └─> extract directly, full document. No search, no ranking luck.

CURATED ORGANISATION ROOT              (226 of 340 — agency and society homepages)
        └─> domain-scoped search with a localised query
            └─> balanced selection across domain groups
                └─> extract (FULL DOCUMENT for guidelines, HTA reports, labels:
                    the recommendation table is never in the top chunks)

STRUCTURED REGISTRY
        └─> API. armGroups[].interventions[] IS the comparator; extracting it
            from HTML with an LLM is the most expensive way to read a column.

LITERATURE
        └─> E-utilities with MeSH + publication-type filters, so
            "practice guidelines for this disease" is a retrievable CLASS
            rather than a lucky ranking.

DISEASE LANDSCAPE                      (not drug-anchored)
        └─> area-filtered guideline domains + PubMed "Practice Guideline"
            This is the pass that finds an option no drug-anchored query
            would surface.
```

### Why the area filter matters

Measured against the Data Team workbook via `GET /api/sources/preflight?areas=Oncology`:

| | Across 27 states |
|---|---|
| Guideline domains offered with no area filter | **297** |
| Guideline domains relevant to Oncology | **27** |
| Share of the search surface that is the right specialty | **9%** |

Without the filter, roughly nine in ten domains handed to each state's guideline search belong to
another specialty, competing for a small per-state retrieval budget.

---

## 5. Query enhancement — the decision

| Option | Verdict |
|---|---|
| LLM writes the queries | **Rejected.** A model that decides to search for a drug has decided that drug is a comparator |
| **LLM writes vocabulary only** | **Adopted.** One call per request; the output type has no comparator field |
| **Deterministic templates compose the queries** | **Adopted as primary.** Reviewable, diffable, testable, free |
| **Landscape pass alongside drug-anchored** | **Adopted — required** |
| Iterative refinement | **Adopted, bounded to one round**, fired only where a coverage assertion fails |

The deterministic half is what demonstrably fixes the missed-guideline-comparator failure. The
vocabulary call is plausible but unmeasured — **A/B it on the GT drugs before keeping it.**

---

## 6. Evidence handling — two separate checks

| | A9 Grounding | A10 Claim correctness |
|---|---|---|
| Question | Does this value appear in the retrieved text? | Does the source support this claim **for this population and this intervention**? |
| Method | Deterministic | LLM, against a **re-fetched full document** |
| Catches | Fabrication, invented numbers | Right drug wrong line; prior therapy posing as a comparator |
| Cost | ~0 | one call per source batch |

Conflating them produces a check that is both permissive and blind: a lexical match against a
short window looks like validation and is not.

Structured API records are exempt from re-fetch — the comparator *is* the API field, so there is
no citation to re-open. They still go through A12 scope adjudication, which is where a registry
arm from the wrong population is caught.

---

## 7. Open questions this code deliberately does not answer

These are marked in the code and surfaced in the output rather than silently decided.

1. **Tier 3 / non-peer-reviewed web.** The SME prompt and MAI-35277 mandate it; MAI-34392 says
   *"Non-peer-reviewed web content is excluded."* `config.ENABLE_TIER_3_GENERAL_WEB` defaults to
   **off**, flipped only by `JCA_ENABLE_TIER3=1`, so the conflict is a visible decision.
2. **Scope framing** — anticipate what any Member State might require, or reproduce what one JCA
   assessed? This sets the A12 adjudication prompt and makes precision numbers interpretable.
   Nothing downstream can be tuned until it is answered.
3. **The outcome catalog.** `data/outcome_catalog.json` is reconstructed from the Ground Truth
   Overview sheet and the mock UI, which agree item for item. **It is not defined in the SME base
   prompt** — the SME refers to "a standard endpoint list" it never enumerates. The file says so
   in its own provenance block, and a test asserts that it keeps saying so. Replace it with the
   authoritative list when the Data Team supplies one.
4. **Treatment-free / recurrence-free interval.** No field exists for it in the SME taxonomy, yet
   the Ground Truth shows it determining which comparator applies. A3 lifts it out of free text
   into a named facet as a workaround; a dedicated field would be better.
5. **Per-state verdict semantics.** The mock's legend is "standard of care / not used". This code
   emits `standard_of_care` / `not_established`, because asserting "not used" requires a source
   that says so and the pipeline has none. Confirm which claim the product intends.
6. **The INN/ATC vocabulary** in `data/inn_atc_seed.json` is a seed for testing. Production needs
   a licensed source (WHO INN list, WHOCC ATC, or RxNorm), versioned and recorded in the manifest.

---

## 8. What is deliberately NOT here

- **PICO-set generation** — Phase 2.
- **A resume-and-patch path.** The legacy codebase had scripts that rewrote part of a saved run
  without re-deriving what depended on it, which shipped a deliverable whose sections disagreed.
  There is one run path. Resume re-runs from a checkpoint through the same code, or it does not
  resume.
- **The target drug's own JCA report as a runtime source.** Previously blocked outright by
  `LeakageGuard`, on the reasoning that the drug's own JCA report is the answer key. Per explicit
  SME decision (2026-09-23), this is no longer the policy: `allow_jca_reports` now **defaults to
  `True`**, and `plan_queries()` runs a deliberate Tier 1 pass against the EU JCA domain, tagged
  `SRC_EU_JCA_REPORT` so a comparator sourced from it is clearly labeled, not blended silently into
  another Tier 1 class. `LeakageGuard` still targets documents about *this drug* specifically (generic
  EU HTA methodology guidance on the same domain stays reachable either way), and the flag is still
  recorded in the manifest either way, so which mode a given run used is always auditable. Setting
  `allow_jca_reports=False` restores the old anticipation-only behaviour for anyone who wants it.
- **A hardcoded source-workbook path.** `JCA_SOURCE_WORKBOOK` selects the file; the loader matches
  sheets fuzzily, finds columns by header name, reads cell hyperlinks, and **fails loudly** when a
  sheet cannot be matched or yields nothing.
