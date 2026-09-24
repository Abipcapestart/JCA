# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Scope boundary — read this first

This module implements **Phase 1 only**: Population + Intervention in, confirmed Comparator and
Outcome scope out. **PICO-set generation is Phase 2 and is deliberately absent.** `Phase1Output`
has no `pico_sets` field; the serialized output carries `"pico_sets": null` with an explanatory
note, and a structural assertion (`test_phase1_output_has_no_pico_sets`) enforces it. Do not add
PICO-set logic here — that belongs in a separate Phase 2 module.

## Commands

```bash
# Run the full test suite (72 tests, no network, no credentials)
python tests/test_phase1.py
# or
python -m pytest tests/ -v

# Run a single test class or test
python -m pytest tests/test_phase1.py::TestScopeAdjudication -v
python -m pytest tests/test_phase1.py::TestScopeAdjudication::test_prior_therapy_only_is_rejected_deterministically -v

# Dev server — credential-free demo providers by default (stdlib only, no install)
export JCA_SOURCE_WORKBOOK=fixtures/MadeAi_JCA_Source_List.xlsx
python -m jca_phase1.api.server                      # → http://127.0.0.1:8000

# Same, with real providers (Bedrock + Tavily + registry APIs)
python -m jca_phase1.api.server --production-providers
# or, with FastAPI/uvicorn installed:
JCA_PRODUCTION_PROVIDERS=1 uvicorn jca_phase1.api.server:app

# Source-workbook coverage report — run before any retrieval to catch a
# stale/mismatched workbook
curl "localhost:8000/api/sources/preflight?areas=Oncology"

# Credential-free, end-to-end pipeline exercise (deterministic fakes, in-process)
python -c "from jca_phase1.demo import run_demo; print(run_demo())"
```

Credentials are loaded from a `.env` file at the repo root via `python-dotenv`
(`config.py` calls `load_dotenv()` on import). Required for production providers:
`TAVILY_API_KEY`, AWS credentials, and **`AWS_REGION`** specifically for Bedrock —
note this is *not* the same variable boto3's default credential chain uses
(`AWS_DEFAULT_REGION`); set both if you rely on the AWS CLI/SDK default.

## Architecture

### One run path, no resume-and-patch scripts

`orchestrator.run_phase1()` is the single entry point, stages A1→A17 in a fixed
order (see `ARCHITECTURE.md` for the full flow diagram and the SME-prompt
mapping table). This is deliberate: the legacy codebase had scripts that
rewrote part of a saved run without re-deriving what depended on it, which is
how a deliverable shipped with sections that disagreed with each other. Resume
means re-running from a checkpoint through the same code — never patching
output in place.

Stage order: `A1 validate → A2 structure → [user confirms] → A4 indication lock →
A3 scope boundary → A5 areas → A6 query plan → A7 retrieve → A8 extract →
A9 ground → A10 validate claims → A11 identity → A12 adjudicate → A14 harmonise →
A13/A15 assign+catalog → A16 consolidate → A17 completeness`.

The 12 SME agents do not map 1:1 to the 18 components: A3–A9 are seven *search
configurations* feeding **one** shared extraction contract (`EvidenceRecord`),
and A11–A16 are mostly deterministic assembly. The point of the shared
contract is that seven separate extraction prompts would be seven places for
one rule to drift.

### Module layout and responsibilities

- `config.py` — EU-27 list, source classes, statuses, retrieval budget. Every
  constant is traceable to a requirement source (comments say which).
- `schema.py` — the typed data model. **This is where the invariants live**,
  enforced in `__post_init__` on the dataclasses (e.g. a comparator cannot be
  constructed without grounded evidence; an illegal `class_source` is
  rejected). Read the "invariants" table in `ARCHITECTURE.md` §3 before
  touching this file — most bugs this project cares about are prevented here,
  not by validation logic scattered downstream.
- `orchestrator.py` — the single run path described above.
- `demo.py` — a credential-free scenario with deterministic scripted
  providers, built to exercise real failure modes (wrong subject_drug, a
  comparator whose class must come from its own INN not the intervention's,
  prior-therapy-only rejection, a landscape-only comparator, empty outcome
  categories that must still be reported).
- `agents/a01_a05_input_context.py` — validation, structuring, indication
  lock, scope boundary, therapeutic area resolution.
- `agents/a06_a08_retrieval.py` — query planning (deterministic templates;
  the vocabulary LLM call **cannot** carry a drug name — enforced by the
  `QueryVocabulary` type having no such field), source-routed retrieval,
  extraction.
- `agents/a09_a13_validation.py` — grounding (deterministic lexical check),
  claim validation (LLM **against a re-fetched full document**, not the
  original retrieval window), comparator identity resolution, scope
  adjudication, per-state assignment.
- `agents/a14_a18_consolidation.py` — harmonization/grouping, outcome catalog
  coverage, consolidation, completeness audit, user additions.
- `providers/` — swappable backends behind small interfaces: `llm.py`
  (Bedrock + `ScriptedLLM` fake), `search.py` (Tavily + fixture fake; retry,
  cache, balanced domain-group selection), `registries.py`
  (ClinicalTrials.gov v2, PubMed E-utilities, EMA + fakes). Every provider is
  `Optional` on `Providers` — a missing provider degrades that pipeline
  capability explicitly (with a note in the output) rather than failing the
  run.
- `sources/workbook.py` — schema-tolerant, hyperlink-aware, area-aware loader
  for the curated source-list Excel workbook. Matches sheets fuzzily, reads
  cell hyperlinks (many curated URLs have no plain-text form), and **fails
  loudly** rather than silently returning nothing when a sheet can't be
  matched.
- `prompts/registry.py` — every prompt is versioned data (not a string buried
  in an agent module), with `register_override()` as the hook for an SME
  prompt-management UI. Prompt versions are written into the run manifest so
  a run can be traced back to the exact prompt text that produced it.
- `api/handlers.py` — framework-free handlers (`(payload) -> (status, dict)`),
  shared by both servers. `RunStore` is an **in-process, non-persistent**
  store — explicitly a placeholder to swap for real persistence before
  production. `_provider_factory` defaults to the credential-free demo
  providers so the UI works out of the box; `set_provider_factory()` swaps in
  `production_providers()` (real Bedrock/Tavily/registries).
- `api/server.py` — two servers over the same handlers: `create_fastapi_app()`
  (production, needs `fastapi`) and `run_dev_server()` (stdlib only, zero
  install, used for local demos and CI).

### Retrieval routing (why it's not a single search call)

Four distinct paths depending on what the curated source-list entry *is* —
see `ARCHITECTURE.md` §4 for the full breakdown and the measured 9% vs 91%
area-relevant search-surface numbers that motivate the therapeutic-area
filter:

1. Curated entry that's already a document URL → extract directly, no search.
2. Curated organisation root (agency/society homepage) → domain-scoped,
   localised search, then full-document extraction (recommendation tables are
   never in the top ranked chunks).
3. Structured registry (ClinicalTrials.gov, PubMed) → typed API fields, no
   LLM extraction — `armGroups[].interventions[]` *is* the comparator.
4. Disease-landscape pass (not drug-anchored) → area-filtered guideline
   domains + PubMed "Practice Guideline" type, the one pass that can surface a
   guideline-graded comparator no drug-anchored query would ever find.

### Two separate evidence checks — don't conflate them

`A9` grounding (deterministic: does this value appear in the retrieved text?)
and `A10` claim validation (LLM, against a **re-fetched** full source: does
this source support this claim for *this* population and *this*
intervention?) answer different questions. Conflating them produces a check
that is both permissive and blind — a lexical match against a short window
looks like validation and isn't. Structured API records skip A10 re-fetch
(there's no citation to re-open) but still go through A12 scope adjudication.

## Open questions the code deliberately leaves unresolved

These are live decisions, not oversights — see `ARCHITECTURE.md` §7 for full
detail before changing related behavior:

1. Tier 3 / non-peer-reviewed general web is off by default
   (`config.ENABLE_TIER_3_GENERAL_WEB`, flip with `JCA_ENABLE_TIER3=1`) — the
   SME prompt and one requirement doc mandate it, another says it's excluded.
2. Scope framing — anticipate what any Member State *might* require, vs.
   reproduce what one JCA actually assessed. This sets the A12 adjudication
   prompt; nothing downstream can be tuned until it's answered.
3. The outcome catalog (`data/outcome_catalog.json`) is reconstructed from the
   Ground Truth Overview sheet, not the SME base prompt — replace it with the
   Data Team's authoritative list when available.
4. Per-state verdict semantics: this code emits `standard_of_care` /
   `not_established` rather than the mock UI's "not used", because asserting
   "not used" would need a source that says so.
5. `data/inn_atc_seed.json` is a seed vocabulary for testing only — production
   needs a licensed INN/ATC source (WHO INN, WHOCC ATC, or RxNorm).

## Before production (from README)

- Replace `data/inn_atc_seed.json` with a licensed INN/ATC vocabulary.
- Confirm `data/outcome_catalog.json` with the Data Team.
- Answer the open questions in `ARCHITECTURE.md` §7 (Tier 3 and scope framing
  change behavior; the code refuses to decide them silently).
- Replace `RunStore` in `api/handlers.py` with real persistence.
- `allow_jca_reports` now **defaults to `True`** (2026-09-23, explicit SME decision) — the drug's
  own JCA report is a deliberate, labeled Tier 1 source (`SRC_EU_JCA_REPORT`), not evaluation-only.
  To still measure how much of accuracy is anticipation vs. reading the JCA report directly, run the
  three GT drugs with `allow_jca_reports=False` (the old default) and compare against the new default
  behaviour.
