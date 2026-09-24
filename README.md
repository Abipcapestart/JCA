# JCA Phase 1 — Comparator & Outcome Scoping

Given a Population and an Intervention, produce the comparators and outcomes each of the 27 EU
Member States is likely to expect, with the reasoning and the sources behind each one.

**Phase 1 only.** PICO-set generation is Phase 2 and is not produced here. The output object
carries `"pico_sets": null` with an explanatory note, and a structural assertion enforces it.

---

## Run it in 30 seconds, no credentials

```bash
cd jca_phase1
export JCA_SOURCE_WORKBOOK=fixtures/MadeAi_JCA_Source_List.xlsx

python -m jca_phase1.api.server          # → http://127.0.0.1:8000
python tests/test_phase1.py              # 72 tests, no network
```

The server runs on the standard library alone — no install step. The UI opens with a worked
example so the interface shows what it does before anything has been run.

### Run it for real

```bash
export TAVILY_API_KEY=...                # + AWS credentials for Bedrock
export JCA_SOURCE_WORKBOOK="/path/to/MadeAi_JCA_Source List 2.xlsx"
python -m jca_phase1.api.server --production-providers

# or, with FastAPI installed:
JCA_PRODUCTION_PROVIDERS=1 uvicorn jca_phase1.api.server:app
```

### Check the source list before anything else

```bash
curl "localhost:8000/api/sources/preflight?areas=Oncology"
```

This is the report that would have caught, on day one, that the workbook the Data Team maintains
was not the workbook the code read. It shows which sheets matched, how many URLs came out of each,
which Member States have no curated HTA source, and how much of each state's guideline search
surface is actually in the requested therapeutic area.

---

## Layout

```
jca_phase1/
├── config.py                  EU-27, tiers, source classes, statuses, retrieval budget
├── schema.py                  the typed model — where three invariants are enforced
├── orchestrator.py            the single run path, A1 → A17
├── demo.py                    a credential-free scenario that exercises the real failure modes
├── data/
│   ├── outcome_catalog.json   the 11-item catalog + its provenance and open question
│   ├── field_taxonomy.json    the SME population/intervention fields and their definitions
│   └── inn_atc_seed.json      seed substance vocabulary (replace for production)
├── prompts/registry.py        every prompt, versioned, with an override hook for an SME UI
├── providers/
│   ├── llm.py                 Bedrock + a scripted fake
│   ├── search.py              Tavily + a fixture fake; retry, cache, balanced selection
│   └── registries.py          ClinicalTrials.gov v2, PubMed E-utilities, EMA + fakes
├── sources/workbook.py        schema-tolerant, hyperlink-aware, area-aware source loader
├── agents/
│   ├── a01_a05_input_context.py    validation, structuring, indication lock, boundary, area
│   ├── a06_a08_retrieval.py        query planning, source-routed retrieval, extraction
│   ├── a09_a13_validation.py       grounding, claim validation, identity, adjudication, states
│   └── a14_a18_consolidation.py    harmonisation, catalog, consolidation, completeness, additions
└── api/
    ├── handlers.py            framework-free handlers
    └── server.py              FastAPI app + stdlib dev server over the same handlers
ui/index.html                  single-file interface
tests/test_phase1.py           72 tests
```

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/schema` | Field taxonomy, Member States, outcome catalog — the UI hardcodes nothing |
| POST | `/api/validate` | FIX / CHECK on the input, before a run |
| POST | `/api/runs` | Run Phase 1. `{"async_run": true}` returns immediately |
| GET | `/api/run?request_id=…` | Status, progress, output |
| POST | `/api/comparators` | Reviewer adds a comparator — drug name and Member State mandatory |
| POST | `/api/outcomes` | Reviewer adds an outcome |
| GET | `/api/sources/preflight` | Source-list coverage before any retrieval |
| GET | `/api/prompts` | Prompt registry with versions and which are SME-editable |

---

## What this changes, and why

Each item below traces to a specific defect in the delivered outputs.

| Change | The defect it fixes |
|---|---|
| Comparator is its own typed object; class resolved from its own INN | Every comparator in all three delivered files carried the assessed drug's mechanism |
| One run path; no resume-and-patch scripts | A workbook shipped with a Comparators sheet and a PICO sheet that disagreed |
| `subject_drug` on every record | A different drug's trial content appeared in another drug's outcomes |
| Indication lock + typed scope boundary | First-line populations survived into a relapsed/refractory request |
| Facets the user never stated never filter | Real comparators rejected on an axis nobody specified |
| Therapeutic-area filter actually applied | 9% of each state's guideline search surface was the right specialty |
| Landscape retrieval pass | A guideline-graded option no drug-anchored query would surface |
| Registry and literature APIs | Comparator arms re-derived from HTML instead of read as typed fields |
| Validation re-fetches the source | "Validation" compared a value to a short window of already-fetched text |
| Scope adjudication as its own stage, with evidence against | Exclusions were silent; precision and recall could not be tuned apart |
| A verdict for all 27 states per comparator | Only states with findings were listed; absence was ambiguous |
| Outcome catalog supplied in full + coverage report | A six-item `e.g.` list, two of whose items the GT marks *unlisted*; missing QoL/COA outcomes were invisible |
| Completeness assertions that fail the run | Completeness was a checklist an LLM ticked |
| Run manifest with prompt and catalog versions | A run could not be traced to the prompt text that produced it |
| Retry on the search provider | A transient rate-limit was indistinguishable from "nothing found" |

---

## Before production

1. **Replace `data/inn_atc_seed.json`** with a licensed INN/ATC vocabulary. It is a seed.
2. **Confirm `data/outcome_catalog.json`** with the Data Team. It is reconstructed from the GT
   Overview sheet and the mock UI, which agree — but it is not in the SME base prompt.
3. **Answer the open questions in `ARCHITECTURE.md` §7.** Two of them (Tier 3, and the scope
   framing) change behaviour, and the code deliberately refuses to decide them silently.
4. **Replace `RunStore`** in `api/handlers.py` with real persistence.
5. **Note: `allow_jca_reports` now defaults to `True`** (2026-09-23, explicit SME decision). The
   drug's own JCA report is a deliberate, labeled Tier 1 source (`SRC_EU_JCA_REPORT`) rather than
   an evaluation-only exception. To still measure anticipation vs. reading the JCA report directly,
   run the three GT drugs with `allow_jca_reports=False` and compare against the new default.
