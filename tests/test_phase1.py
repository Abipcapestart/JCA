"""
Tests for JCA Phase 1.

Every test here maps to a defect that was actually observed in the legacy
implementation or to an invariant the architecture claims to hold. They run with
no credentials and no network.

Run:  python -m pytest tests/ -v      (or: python tests/test_phase1.py)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import unittest.mock

import openpyxl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jca_phase1 import config as C
from jca_phase1 import orchestrator as orch
from jca_phase1.agents import a01_a05_input_context as inputs
from jca_phase1.agents import a06_a08_retrieval as retrieval
from jca_phase1.agents import a09_a13_validation as validation
from jca_phase1.agents import a14_a18_consolidation as consolidation
from jca_phase1.providers.llm import BedrockLLM, ScriptedLLM
from jca_phase1.prompts import registry as prompt_registry
from jca_phase1.providers.registries import (FixtureTrialRegistry, PublicationRecord,
                                             TrialArm, TrialRecord)
from jca_phase1.providers.search import FixtureSearchProvider, SearchHit, select_balanced
from jca_phase1.schema import (Comparator, ConsolidatedComparator, EvidenceRecord,
                               EvidenceRef, Field, FINDING_COMPARATOR, FINDING_OUTCOME,
                               Intervention,
                               LicensedIndicationRecord, MemberStateSummary,
                               ORIGIN_AGENT, POP_ITT, POP_LICENSED, Population,
                               PopulationContext, RetrievedDocument, ScopeAdjudication,
                               ScopeBoundary, ValidationOutcome)
from jca_phase1.sources.workbook import (is_document_url, load_source_inventory,
                                         normalise_domain)

FIXTURE_WORKBOOK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "fixtures", "MadeAi_JCA_Source_List.xlsx")


# ===========================================================================
# Source-list loading — the defect that blocked everything else
# ===========================================================================

class TestSourceWorkbook(unittest.TestCase):
    """The legacy loaders read the wrong file, matched sheet names exactly, and
    iterated with values_only=True so hyperlink URLs were invisible."""

    @classmethod
    def setUpClass(cls):
        cls.inv = load_source_inventory(FIXTURE_WORKBOOK, strict=False)

    def test_all_three_data_team_sheets_are_matched(self):
        for role in ("soc", "hta", "emea"):
            self.assertIn(role, self.inv.sheets_matched,
                          f"role {role} was not matched; sheets seen: {self.inv.sheets_seen}")

    def test_sheet_names_match_despite_whitespace_and_plurals(self):
        # 'Standard of Care  Guidelines' (two spaces) vs 'SoC';
        # 'HTA licensing sources' (plural) vs 'HTA licensing source'.
        self.assertEqual(self.inv.sheets_matched["soc"], "Standard of Care  Guidelines")
        self.assertEqual(self.inv.sheets_matched["hta"], "HTA licensing sources")
        self.assertEqual(self.inv.sheets_matched["emea"], "EMEA licensing sources")

    def test_hyperlink_only_urls_are_read(self):
        """Every Standard-of-Care URL is a cell hyperlink with no plain text.
        Reading with values_only=True returns zero, which is the legacy bug."""
        soc = [e for e in self.inv.entries if e.source_class == C.SRC_CLINICAL_GUIDELINE]
        self.assertGreater(len(soc), 200,
                           "hyperlink-only SoC URLs were not extracted")

    def test_all_three_sheets_contribute_entries(self):
        classes = {e.source_class for e in self.inv.entries}
        self.assertIn(C.SRC_CLINICAL_GUIDELINE, classes)
        self.assertIn(C.SRC_HTA_REGULATORY, classes)
        self.assertIn(C.SRC_DRUG_LABEL, classes)

    def test_pubmed_domain_never_surfaces_as_tier_1(self):
        """A curator can legitimately cite a PubMed-hosted paper as a
        guideline source (the SoC sheet does, for Croatia/Oncology). The
        citation itself is kept, but pubmed.ncbi.nlm.nih.gov must always be
        classified as Tier 2 SRC_PUBMED, never Tier 1, whichever sheet cited
        it -- retrieval has its own dedicated Tier 2 PubMed path and
        peer-review handling that a Tier 1 label would bypass."""
        pubmed_entries = [e for e in self.inv.entries if e.domain == "pubmed.ncbi.nlm.nih.gov"]
        self.assertTrue(pubmed_entries, "no pubmed.ncbi.nlm.nih.gov entry loaded at all")
        for e in pubmed_entries:
            self.assertEqual(e.source_class, C.SRC_PUBMED)
        self.assertNotIn("pubmed.ncbi.nlm.nih.gov",
                         self.inv.domains(C.SRC_CLINICAL_GUIDELINE))
        self.assertIn("pubmed.ncbi.nlm.nih.gov", self.inv.domains(C.SRC_PUBMED))

    def test_no_load_errors(self):
        errors = [p for p in self.inv.problems if p.severity == "ERROR"]
        self.assertEqual(errors, [], f"source list reported errors: {errors}")

    def test_therapeutic_area_filter_narrows_the_search_surface(self):
        """The single largest measured defect: without the area filter ~89% of
        the domains offered per state are the wrong specialty."""
        all_areas = self.inv.domains(C.SRC_CLINICAL_GUIDELINE, member_state="Austria")
        oncology = self.inv.domains(C.SRC_CLINICAL_GUIDELINE, member_state="Austria",
                                    areas=["Oncology"])
        self.assertGreater(len(all_areas), len(oncology))
        self.assertLessEqual(len(oncology), 3)

    def test_inahta_loads_as_a_shared_international_source(self):
        """Row 31's hyperlink was on the wrong column (HTA body full name,
        not Website) and was invisible to the loader. It must now load, and
        as a SHARED entry (member_state=None) -- it is not any one state's
        national body, and must never be attributed to one."""
        inahta = [e for e in self.inv.entries
                 if e.source_class == C.SRC_HTA_REGULATORY and e.domain == "inahta.org"]
        self.assertTrue(inahta, "inahta.org did not load at all")
        self.assertIsNone(inahta[0].member_state,
                          "INAHTA is international, not a national HTA body")

    def test_previously_gap_states_now_have_a_curated_hta_domain(self):
        for state, domain in (("Bulgaria", "ncpr.bg"), ("Cyprus", "moh.gov.cy"),
                              ("Greece", "eopyy.gov.gr"),
                              ("Malta", "pharmaceuticalaffairs.gov.mt")):
            got = self.inv.domains(C.SRC_HTA_REGULATORY, member_state=state,
                                   include_shared=False)
            self.assertEqual(got, [domain], f"{state} HTA domain mismatch")
        # Luxembourg is a deliberate, honest gap -- no confirmed dedicated
        # HTA/reimbursement agency was found, so nothing was force-added.
        self.assertEqual(
            self.inv.domains(C.SRC_HTA_REGULATORY, member_state="Luxembourg",
                             include_shared=False), [])

    def test_missing_workbook_fails_loudly(self):
        with self.assertRaises(FileNotFoundError):
            load_source_inventory("/nonexistent/workbook.xlsx", strict=True)

    def test_normalise_domain_accepts_bare_domains_and_urls(self):
        # A bare domain returning "" silently empties every domain-scoped search.
        self.assertEqual(normalise_domain("ema.europa.eu"), "ema.europa.eu")
        self.assertEqual(normalise_domain("www.ema.europa.eu"), "ema.europa.eu")
        self.assertEqual(normalise_domain("https://www.ema.europa.eu/en/x"), "ema.europa.eu")
        self.assertEqual(normalise_domain(""), "")

    def test_document_vs_hub_url_classification(self):
        self.assertTrue(is_document_url("https://x.org/a/b/c/guideline.pdf"))
        self.assertTrue(is_document_url("https://x.org/guidelines/lung/sclc"))
        self.assertFalse(is_document_url("https://kce.fgov.be"))
        self.assertFalse(is_document_url("https://www.ema.europa.eu/en/medicines"))

    def test_conference_sheet_role_is_recognised(self):
        """No production sheet is curated for this role yet -- this only
        proves the loader CAN pick one up once the data team adds it."""
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Conference Sources"
        ws.append(["Country", "Website"])
        ws.append(["EU-wide", "https://meetings.asco.org/abstracts-presentations"])
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            path = f.name
        try:
            wb.save(path)
            inv = load_source_inventory(path, strict=False)
            self.assertIn("conference", inv.sheets_matched)
            conf_entries = [e for e in inv.entries if e.source_class == C.SRC_CONFERENCE]
            self.assertTrue(conf_entries, "conference role matched but yielded zero URLs")
            # The exact condition plan_queries() gates the Tier 2 conference
            # query on -- today this is always [] because no sheet matches.
            self.assertTrue(inv.domains(C.SRC_CONFERENCE))
        finally:
            os.remove(path)


# ===========================================================================
# A1 / A2 — input handling
# ===========================================================================

class TestInputValidation(unittest.TestCase):

    def test_missing_mandatory_field_blocks(self):
        result = inputs.validate_pi({"indication_disease": ""}, {"product_name_inn": "X"})
        self.assertEqual(result.overall_status, "BLOCKED")
        self.assertFalse(result.passed)

    def test_check_items_do_not_block(self):
        result = inputs.validate_pi(
            {"indication_disease": "NSCLC"}, {"product_name_inn": "X"},
            added_fields=["age_group", "sex"])
        self.assertEqual(result.overall_status, "PASS")
        self.assertTrue(result.check_items)

    def test_field_never_added_is_not_a_check_item(self):
        result = inputs.validate_pi({"indication_disease": "NSCLC"},
                                    {"product_name_inn": "X"}, added_fields=[])
        self.assertEqual(result.check_items, [])


class TestInputStructuring(unittest.TestCase):

    def test_therapeutic_class_is_never_inferred(self):
        """SME Agent 2 rule 5, absolute. An inferred class is how a comparator
        ends up wearing the intervention's mechanism."""
        llm = ScriptedLLM({"a02.input_structuring": json.dumps({
            "populations": [],
            "intervention": {"fields": {"therapeutic_class_mechanism": {
                "value": "EGFR TKI", "provenance": "inferred",
                "inference_basis": "guessed from the drug name"}}}})})
        _, intervention = inputs.structure_pi(
            {"indication_disease": "NSCLC"}, {"product_name_inn": "X"}, "", llm)
        self.assertIsNone(intervention.fields["therapeutic_class_mechanism"].value)

    def test_user_statement_beats_inference(self):
        llm = ScriptedLLM({"a02.input_structuring": json.dumps({
            "populations": [{"population_id": "licensed", "fields": {
                "stage_severity": {"value": "WRONG", "provenance": "inferred"}}}],
            "intervention": {"fields": {}}})})
        pops, _ = inputs.structure_pi(
            {"indication_disease": "NSCLC", "stage_severity": "Stage IV"},
            {"product_name_inn": "X"}, "", llm)
        self.assertEqual(pops[0].value("stage_severity"), "Stage IV")

    def test_itt_declaration_creates_a_second_population(self):
        pops, _ = inputs.structure_pi(
            {"indication_disease": "SCLC",
             "itt_differs": "No, it differs — a broader relapsed population"},
            {"product_name_inn": "X"}, "", None)
        self.assertEqual(len(pops), 2)
        self.assertEqual(pops[1].population_id, "intended_to_treat")

    def test_itt_same_does_not_create_a_second_population(self):
        pops, _ = inputs.structure_pi(
            {"indication_disease": "SCLC", "itt_differs": "Yes, same"},
            {"product_name_inn": "X"}, "", None)
        self.assertEqual(len(pops), 1)


# ===========================================================================
# A3 — scope boundary
# ===========================================================================

class TestScopeBoundary(unittest.TestCase):

    def _population(self, **fields) -> Population:
        pop = Population()
        for k in inputs.POPULATION_FIELDS:
            pop.fields[k] = Field(value=fields.get(k), provenance="confirmed"
                                  if fields.get(k) else "not_provided")
        return pop

    def test_unstated_facets_never_become_filters(self):
        """A facet nobody specified must not silently reject evidence."""
        boundary = inputs.build_scope_boundary(
            self._population(indication_disease="SCLC"),
            Intervention(), LicensedIndicationRecord(), None)
        self.assertIn("age_group", boundary.unbounded_facets)
        self.assertNotIn("age_group", [f.name for f in boundary.must_match_facets()])

    def test_discriminating_stated_facets_are_must_match(self):
        boundary = inputs.build_scope_boundary(
            self._population(indication_disease="SCLC",
                             prior_therapy_line="progressed after platinum"),
            Intervention(), LicensedIndicationRecord(), None)
        names = [f.name for f in boundary.must_match_facets()]
        self.assertIn("prior_therapy_line", names)

    def test_free_text_interval_is_lifted_into_a_named_facet(self):
        """An axis with no SME field (a treatment-free interval) must survive as
        a comparable facet rather than being lost in prose."""
        llm = ScriptedLLM({"a03.scope_facet_normalise": json.dumps({"facets": [
            {"name": "treatment_free_interval", "value": ">= 90 days",
             "discriminating": True, "verbatim": "chemotherapy-free interval >= 90 days"}]})})
        boundary = inputs.build_scope_boundary(
            self._population(indication_disease="SCLC",
                             other_characteristics="chemotherapy-free interval >= 90 days"),
            Intervention(), LicensedIndicationRecord(), llm)
        facet = boundary.facets["treatment_free_interval"]
        self.assertTrue(facet.is_bound)
        self.assertTrue(facet.must_match)


# ===========================================================================
# A5 — therapeutic area
# ===========================================================================

class TestTherapeuticArea(unittest.TestCase):

    def _pop(self, indication, orphan=""):
        pop = Population()
        pop.fields["indication_disease"] = Field(value=indication, provenance="confirmed")
        pop.fields["orphan_rare_disease"] = Field(value=orphan or None,
                                                  provenance="confirmed" if orphan else "not_provided")
        for k in ("disease_subtype_histology", "icd_disease_code", "therapeutic_area"):
            pop.fields[k] = Field()
        return pop

    def test_single_area_resolves_without_an_llm_call(self):
        llm = ScriptedLLM({})
        result = inputs.resolve_therapeutic_areas(self._pop("small cell lung cancer"), llm)
        self.assertIn("Oncology", result.areas)
        self.assertEqual(len(llm.call_log), 0, "a confident keyword match must not spend a call")

    def test_orphan_flag_always_adds_rare_diseases(self):
        result = inputs.resolve_therapeutic_areas(
            self._pop("small cell lung cancer", orphan="Yes"), None)
        self.assertIn(C.RARE_DISEASES, result.areas)

    def test_area_is_always_resolved(self):
        result = inputs.resolve_therapeutic_areas(self._pop("a condition with no keywords"), None)
        self.assertTrue(result.areas, "the SME is explicit that there is no 'nothing fits'")


# ===========================================================================
# A6 — query planning
# ===========================================================================

class TestQueryPlanning(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.inv = load_source_inventory(FIXTURE_WORKBOOK, strict=False)

    def _plan(self, guard=None):
        pop = Population()
        pop.fields["indication_disease"] = Field(value="small cell lung cancer",
                                                 provenance="confirmed")
        for k in inputs.POPULATION_FIELDS:
            pop.fields.setdefault(k, Field())
        inter = Intervention()
        inter.fields["product_name_inn"] = Field(value="Examplimab", provenance="confirmed")
        for k in inputs.INTERVENTION_FIELDS:
            inter.fields.setdefault(k, Field())
        vocab = retrieval.build_vocabulary(pop, ["Oncology"], None)
        return retrieval.plan_queries(pop, inter, ["Oncology"], vocab, self.inv, guard), inter

    def test_jca_report_pass_only_planned_when_guard_allows_it(self):
        from jca_phase1.schema import LeakageGuard
        plan_no_guard, _ = self._plan(guard=None)
        plan_blocked, _ = self._plan(guard=LeakageGuard(allow_jca_reports=False))
        plan_open, _ = self._plan(guard=LeakageGuard(allow_jca_reports=True))
        self.assertFalse([i for i in plan_no_guard if i.source_class == C.SRC_EU_JCA_REPORT])
        self.assertFalse([i for i in plan_blocked if i.source_class == C.SRC_EU_JCA_REPORT])
        jca_items = [i for i in plan_open if i.source_class == C.SRC_EU_JCA_REPORT]
        self.assertEqual(len(jca_items), 1)
        self.assertEqual(jca_items[0].domains, ["health.ec.europa.eu"])
        self.assertEqual(jca_items[0].member_state, C.EU_WIDE)

    def test_every_member_state_is_planned(self):
        plan, _ = self._plan()
        states = {i.member_state for i in plan if i.member_state in C.EU_27_MEMBER_STATES}
        self.assertEqual(len(states), 27, "all 27 states must be assessed on every run")

    def test_states_without_a_curated_hta_source_still_get_searched(self):
        """Luxembourg has no curated HTA domain. 'No curated source' must not
        mean 'not searched' -- and it must not mean 'fully unrestricted' either,
        now that the international HTA database (INAHTA) is curated."""
        plan, _ = self._plan()
        items = [i for i in plan if i.member_state == "Luxembourg"
                 and i.source_class == C.SRC_HTA_REGULATORY]
        self.assertTrue(items, "Luxembourg has no HTA query at all")
        self.assertIn("Luxembourg", items[0].query)
        self.assertIn("inahta.org", items[0].domains,
                      "the shared/international HTA source should back-stop a state "
                      "with no curated national source")

    def test_states_with_a_newly_curated_hta_source_use_it(self):
        """Bulgaria, Cyprus, Greece and Malta previously had no curated HTA
        body; all four are now curated and must use their own domain, not
        the unscoped fallback."""
        plan, _ = self._plan()
        for state, domain in (("Bulgaria", "ncpr.bg"), ("Cyprus", "moh.gov.cy"),
                              ("Greece", "eopyy.gov.gr"),
                              ("Malta", "pharmaceuticalaffairs.gov.mt")):
            items = [i for i in plan if i.member_state == state
                     and i.source_class == C.SRC_HTA_REGULATORY]
            self.assertTrue(items, f"{state} has no HTA query at all")
            self.assertEqual(items[0].domains, [domain])
            self.assertNotIn(state, items[0].query,
                             "a curated-domain query is drug/indication-anchored, "
                             "not state-name-anchored like the unscoped fallback")

    def test_conference_domains_are_area_filtered(self):
        """An oncology-curated conference list must not be offered to a
        cardiovascular request -- the same area-filter guideline domains get."""
        plan, _ = self._plan()  # area = Oncology
        onc_conf = [i for i in plan if i.source_class == C.SRC_CONFERENCE]
        self.assertTrue(onc_conf, "no conference query for an Oncology request")

        pop = Population()
        pop.fields["indication_disease"] = Field(value="heart failure",
                                                 provenance="confirmed")
        for k in inputs.POPULATION_FIELDS:
            pop.fields.setdefault(k, Field())
        inter = Intervention()
        inter.fields["product_name_inn"] = Field(value="Examplimab", provenance="confirmed")
        for k in inputs.INTERVENTION_FIELDS:
            inter.fields.setdefault(k, Field())
        vocab = retrieval.build_vocabulary(pop, ["Cardiovascular"], None)
        cardio_plan = retrieval.plan_queries(pop, inter, ["Cardiovascular"], vocab, self.inv)
        cardio_conf = [i for i in cardio_plan if i.source_class == C.SRC_CONFERENCE]
        self.assertEqual(cardio_conf, [],
                         "no curated cardiovascular conference domain exists yet; "
                         "an unfiltered query would wrongly search oncology domains")

    def test_tier_3_is_enabled_by_default(self):
        self.assertTrue(C.ENABLE_TIER_3_GENERAL_WEB)

    def test_landscape_pass_exists_and_omits_the_drug(self):
        """A guideline listing the complete treatment-line landscape rarely
        ranks for a drug-anchored query. This is the confirmed cause of the
        missed guideline comparators."""
        plan, inter = self._plan()
        landscape = [i for i in plan if i.pass_type == retrieval.PASS_LANDSCAPE]
        self.assertTrue(landscape, "no landscape pass was planned")
        for item in landscape:
            self.assertNotIn(inter.product_name.lower(), item.query.lower())

    def test_query_vocabulary_is_batched_by_language_not_one_giant_call(self):
        """Run 4 post-fix validation: even at the raised 4000-token cap,
        a06.query_vocabulary hit 100% truncation because requesting
        localised_assessment_terms for all ~23 EU languages in one JSON
        response is naturally larger than any single-shot cap comfortably
        handles. Confirms build_vocabulary now makes multiple smaller calls,
        one per language batch, and merges their results."""
        pop = Population()
        pop.fields["indication_disease"] = Field(value="small cell lung cancer",
                                                 provenance="confirmed")
        for k in inputs.POPULATION_FIELDS:
            pop.fields.setdefault(k, Field())

        def fake_response(user_prompt):
            langs = user_prompt.split("LANGUAGES: ")[1].strip().split(", ")
            return json.dumps({
                "indication_synonyms": ["SCLC"],
                "indication_abbreviations": ["SCLC"],
                "disease_class_terms": ["lung cancer"],
                "outcome_requirement_terms": ["overall survival"],
                "localised_assessment_terms": {lang: [f"term-{lang}"] for lang in langs}})

        llm = ScriptedLLM({"a06.query_vocabulary": fake_response})
        vocab = retrieval.build_vocabulary(pop, ["Oncology"], llm)

        calls = [c for c in llm.seen if c[0] == "a06.query_vocabulary"]
        self.assertGreater(len(calls), 1,
                           "23 EU languages must not be requested in a single call")
        all_languages = sorted({lang for lang in retrieval.STATE_LANGUAGE.values()})
        self.assertEqual(set(vocab.localised_assessment_terms.keys()), set(all_languages),
                         "every language's terms must survive the merge across batches")
        for lang in all_languages:
            self.assertEqual(vocab.localised_assessment_terms[lang], [f"term-{lang}"])
        # Language-independent fields, requested redundantly in every batch,
        # must still come out deduplicated rather than repeated per batch.
        self.assertEqual(vocab.disease_class_terms, ["lung cancer"])

    def test_eu_wide_epar_query_carries_the_line_of_therapy_hint(self):
        """Fix 6: the EMA/EPAR pass is where the Vinblastine/CV evidence was
        actually retrieved in a real production run (drug_registry_label),
        not the per-state guideline landscape pass -- but only the landscape
        pass carried _line_hint(). Extend the same never-invented hint to
        the pass that actually found this evidence."""
        pop = Population()
        pop.fields["indication_disease"] = Field(value="small cell lung cancer",
                                                 provenance="confirmed")
        pop.fields["prior_therapy_line"] = Field(value="second-line or later",
                                                 provenance="confirmed")
        for k in inputs.POPULATION_FIELDS:
            pop.fields.setdefault(k, Field())
        inter = Intervention()
        inter.fields["product_name_inn"] = Field(value="Examplimab", provenance="confirmed")
        for k in inputs.INTERVENTION_FIELDS:
            inter.fields.setdefault(k, Field())
        vocab = retrieval.build_vocabulary(pop, ["Oncology"], None)
        plan = retrieval.plan_queries(pop, inter, ["Oncology"], vocab, self.inv, None)
        epar = [i for i in plan if i.source_class == C.SRC_DRUG_LABEL
                and i.member_state == C.EU_WIDE]
        self.assertTrue(epar, "no EU-wide EPAR query was planned")
        self.assertIn("second-line or later", epar[0].query)

    def test_eu_wide_epar_query_has_no_stray_artifact_when_no_line_is_stated(self):
        plan, _ = self._plan()  # no prior_therapy_line/line_of_therapy stated
        epar = [i for i in plan if i.source_class == C.SRC_DRUG_LABEL
                and i.member_state == C.EU_WIDE]
        self.assertTrue(epar, "no EU-wide EPAR query was planned")
        self.assertEqual(epar[0].query, "Examplimab EPAR summary of product characteristics")
        self.assertNotIn("  ", epar[0].query)

    def test_guideline_domains_are_area_filtered(self):
        plan, _ = self._plan()
        onc = set(self.inv.domains(C.SRC_CLINICAL_GUIDELINE, member_state="Austria",
                                   areas=["Oncology"]))
        item = next(i for i in plan if i.member_state == "Austria"
                    and i.source_class == C.SRC_CLINICAL_GUIDELINE)
        self.assertTrue(set(item.domains).issubset(onc))

    def test_vocabulary_cannot_carry_a_drug_name(self):
        """The query planner must not be able to name the answer."""
        llm = ScriptedLLM({"a06.query_vocabulary": json.dumps({
            "indication_synonyms": ["SCLC", "topotecan therapy"],
            "disease_class_terms": ["osimertinib regimens"],
            "indication_abbreviations": [], "localised_assessment_terms": {},
            "outcome_requirement_terms": []})})
        pop = Population()
        pop.fields["indication_disease"] = Field(value="SCLC", provenance="confirmed")
        pop.fields["disease_subtype_histology"] = Field()
        vocab = retrieval.build_vocabulary(pop, ["Oncology"], llm)
        joined = " ".join(vocab.indication_synonyms + vocab.disease_class_terms).lower()
        self.assertNotIn("topotecan", joined)
        self.assertNotIn("osimertinib", joined)


# ===========================================================================
# A7 — PubMed publications must not be discarded after retrieval
# ===========================================================================

class TestPubMedConversion(unittest.TestCase):
    """retrieve_structured() fetches real PublicationRecords; before this fix
    they went into RetrievalResult.publications and nowhere else. Every other
    Tier 2/3 source class reaches extraction -- PubMed silently didn't."""

    def test_publication_with_an_abstract_becomes_a_document(self):
        pub = PublicationRecord(
            pmid="1", title="A relapsed SCLC review", abstract="Topotecan is used.",
            journal="Annals of Oncology", year="2025",
            url="https://pubmed.ncbi.nlm.nih.gov/1/")
        docs = retrieval.documents_from_publications([pub])
        self.assertEqual(len(docs), 1)
        doc = docs[0]
        self.assertEqual(doc.source_class, C.SRC_PUBMED)
        self.assertEqual(doc.text, "Topotecan is used.")
        self.assertTrue(doc.ok)
        self.assertEqual(doc.member_state, C.GENERAL_EVIDENCE)

    def test_publication_types_are_carried_onto_the_document(self):
        """The metadata that lets A8 tell a preprint from a journal article —
        previously fetched from NCBI and then dropped at this exact
        conversion step."""
        pub = PublicationRecord(
            pmid="3", title="A preprint", abstract="Topotecan is used.",
            publication_types=["Preprint"], url="https://pubmed.ncbi.nlm.nih.gov/3/")
        docs = retrieval.documents_from_publications([pub])
        self.assertEqual(docs[0].publication_types, ["Preprint"])

    def test_publication_without_an_abstract_is_dropped(self):
        """Nothing for the shared extraction prompt to read -- correctly
        skipped rather than sent through as an empty document."""
        pub = PublicationRecord(pmid="2", title="No abstract available", abstract="",
                                url="https://pubmed.ncbi.nlm.nih.gov/2/")
        self.assertEqual(retrieval.documents_from_publications([pub]), [])


class _RecordingLiteratureClient:
    """Records every query string it's asked to search, returning nothing --
    exercises retrieve_structured()'s query CONSTRUCTION in isolation, not
    a real PubMed call."""

    def __init__(self):
        self.queries: list = []

    def search(self, query, publication_types=None, max_results=20):
        self.queries.append((query, publication_types))
        return []


class TestPubMedQuerySynonyms(unittest.TestCase):
    """indication_synonyms almost never reached the actual PubMed query
    string: query 1 (drug+disease) never referenced it at all, and query 2
    only used vocab.indication_synonyms[0] -- so a query-vocabulary fix that
    correctly returns several good alternate phrasings had nowhere for most
    of them to go. Confirmed real gap, fixed by OR-joining every synonym
    into both queries (PubMed/Entrez's term syntax already supports OR)."""

    def _run(self, disease_class_terms=None, indication_synonyms=None):
        from jca_phase1.schema import LicensedIndicationRecord, QueryVocabulary
        pop = Population(fields={"indication_disease": Field(value="paediatric low-grade glioma",
                                                              provenance="confirmed")})
        inter = Intervention(fields={"product_name_inn": Field(value="Tovorafenib",
                                                               provenance="confirmed")})
        vocab = QueryVocabulary(disease_class_terms=disease_class_terms or [],
                                indication_synonyms=indication_synonyms or [])
        client = _RecordingLiteratureClient()
        retrieval.retrieve_structured(pop, inter, LicensedIndicationRecord(),
                                      None, client, vocab)
        return client.queries

    def test_main_query_now_includes_every_synonym_not_just_the_raw_condition(self):
        queries = self._run(indication_synonyms=["pediatric low-grade glioma", "pLGG"])
        main_query, main_filters = queries[0]
        self.assertIsNone(main_filters)
        self.assertIn("Tovorafenib", main_query)
        self.assertIn('"paediatric low-grade glioma"', main_query,
                     "the raw condition string must still be included")
        self.assertIn('"pediatric low-grade glioma"', main_query,
                     "a synonym that used to be discarded entirely must now reach the query")
        self.assertIn("pLGG", main_query)

    def test_single_word_synonym_is_not_wrapped_in_quotes(self):
        queries = self._run(indication_synonyms=["pLGG"])
        main_query, _ = queries[0]
        self.assertIn(" OR pLGG" if "OR" in main_query else "pLGG", main_query)
        self.assertNotIn('"pLGG"', main_query)

    def test_no_synonyms_falls_back_to_the_original_simple_query(self):
        queries = self._run(indication_synonyms=[])
        main_query, _ = queries[0]
        self.assertEqual(main_query, "Tovorafenib paediatric low-grade glioma")

    def test_guideline_query_no_longer_drops_all_but_the_first_synonym(self):
        queries = self._run(disease_class_terms=[],
                            indication_synonyms=["pediatric low-grade glioma", "pLGG",
                                                 "childhood low-grade glioma"])
        broad_query, filters = queries[1]
        self.assertEqual(filters, ["Practice Guideline", "Guideline"])
        for term in ("pediatric low-grade glioma", "pLGG", "childhood low-grade glioma"):
            self.assertIn(term, broad_query,
                         f"{term!r} must survive into the guideline query, not just index 0")

    def test_disease_class_terms_are_combined_with_synonyms_not_just_first_element(self):
        queries = self._run(disease_class_terms=["glioma"],
                            indication_synonyms=["pediatric low-grade glioma"])
        broad_query, _ = queries[1]
        self.assertIn("glioma", broad_query)
        self.assertIn('"pediatric low-grade glioma"', broad_query)

    def test_disease_only_query_anchors_one_required_term_instead_of_pure_or(self):
        """Real production defect: every disease term was OR'd together with
        no requirement that the actual disease name appear in a match, so
        unrelated documents (gout, UTI, Cushing's syndrome) matched purely on
        generic co-occurring words. Fix: the shortest/core term becomes a
        REQUIRED (field-tagged, ANDed) anchor, with the rest only OR'd inside
        that AND clause -- never satisfiable by an unrelated term alone."""
        queries = self._run(disease_class_terms=["glioma"],
                            indication_synonyms=["pediatric low-grade glioma", "pLGG"])
        broad_query, _ = queries[1]
        self.assertIn("[Title/Abstract]", broad_query,
                     "the anchor term must use PubMed's field-tag syntax")
        self.assertIn(" AND (", broad_query,
                     "the rest of the terms must be ANDed against the anchor, not just OR'd in")
        # "pLGG" is the shortest candidate, so it must be the anchor.
        self.assertTrue(broad_query.startswith("pLGG[Title/Abstract]"))

    def test_empty_vocab_falls_back_to_the_condition_as_the_anchor_not_a_bare_sentence(self):
        """When A6 returns no disease_class_terms or indication_synonyms at
        all, the query must still anchor on the raw condition string via
        field-tag syntax -- not submit it as an unstructured OR-free sentence
        the way the old fallback did."""
        queries = self._run(disease_class_terms=[], indication_synonyms=[])
        broad_query, _ = queries[1]
        self.assertEqual(broad_query, '"paediatric low-grade glioma"[Title/Abstract]')


class TestPubMedLandscapePass(unittest.TestCase):
    """SME follow-up: query 1 is always drug-anchored, and query 2 -- the
    only disease-only query -- is restricted to publication_types=
    ["Practice Guideline", "Guideline"]. A comparator's own regular clinical
    study or review (not tagged that way by PubMed) was unreachable by
    either query, however good the synonym coverage became -- exactly why
    Everolimus kept not surfacing even after fetch volume and synonyms
    improved. Fixed by adding a third, genuinely unrestricted disease-
    landscape pass, mirroring plan_queries()'s clinical_guideline landscape
    pass (which also deliberately omits the drug)."""

    def _run(self, disease_class_terms=None, indication_synonyms=None):
        from jca_phase1.schema import LicensedIndicationRecord, QueryVocabulary
        pop = Population(fields={"indication_disease": Field(value="paediatric low-grade glioma",
                                                              provenance="confirmed")})
        inter = Intervention(fields={"product_name_inn": Field(value="Tovorafenib",
                                                               provenance="confirmed")})
        vocab = QueryVocabulary(disease_class_terms=disease_class_terms or [],
                                indication_synonyms=indication_synonyms or [])
        client = _RecordingLiteratureClient()
        retrieval.retrieve_structured(pop, inter, LicensedIndicationRecord(),
                                      None, client, vocab)
        return client.queries

    def test_a_third_query_is_issued_with_no_publication_type_restriction(self):
        queries = self._run(indication_synonyms=["pediatric low-grade glioma"])
        self.assertEqual(len(queries), 3,
                         "expected drug-anchored, guideline-restricted, and landscape passes")
        landscape_query, landscape_filters = queries[2]
        self.assertIsNone(landscape_filters,
                          "the landscape pass must not be restricted to guideline-type "
                          "publications -- that's exactly the restriction that hid a "
                          "comparator's own regular clinical study or review")

    def test_landscape_query_is_disease_only_not_drug_anchored(self):
        queries = self._run(indication_synonyms=["pediatric low-grade glioma"])
        landscape_query, _ = queries[2]
        self.assertNotIn("Tovorafenib", landscape_query,
                         "the landscape pass must omit the drug entirely, the same way "
                         "the clinical_guideline landscape pass does")

    def test_landscape_query_uses_the_same_disease_terms_as_the_guideline_pass(self):
        queries = self._run(disease_class_terms=["glioma"],
                            indication_synonyms=["pediatric low-grade glioma"])
        guideline_query, _ = queries[1]
        landscape_query, _ = queries[2]
        self.assertEqual(guideline_query, landscape_query,
                         "same disease-only term set; only the publication-type "
                         "restriction differs between the two passes")

    def test_all_three_query_strings_reach_the_attempt_detail_for_excel_export(self):
        """PubMed/CT.gov never appear in the 'A6 Query Plan' export sheet at
        all (they bypass plan_queries() by design) -- this was the one
        retrieval pass with zero visibility anywhere into what was actually
        searched for. The 'A7 Retrieval Attempts' sheet already has a
        'detail' column (populated for every other source class's retrieval
        failures); the PubMed attempt now uses it to carry the actual query
        strings, so a reviewer can see them without reading raw bedrock/
        tavily call logs."""
        from jca_phase1.schema import LicensedIndicationRecord, QueryVocabulary
        pop = Population(fields={"indication_disease": Field(value="paediatric low-grade glioma",
                                                              provenance="confirmed")})
        inter = Intervention(fields={"product_name_inn": Field(value="Tovorafenib",
                                                               provenance="confirmed")})
        vocab = QueryVocabulary(indication_synonyms=["pediatric low-grade glioma"])
        client = _RecordingLiteratureClient()
        result = retrieval.retrieve_structured(pop, inter, LicensedIndicationRecord(),
                                               None, client, vocab)
        pubmed_attempts = [a for a in result.attempts if a.source_class == C.SRC_PUBMED]
        self.assertEqual(len(pubmed_attempts), 1)
        detail = pubmed_attempts[0].detail
        self.assertIn("Tovorafenib", detail)
        self.assertIn("pediatric low-grade glioma", detail)
        self.assertIn("guideline-restricted", detail)
        self.assertIn("landscape", detail)


class TestPubMedXMLParsing(unittest.TestCase):
    """_parse_pubmed_xml used .text/findtext(), which stops at the first
    child element -- any inline markup (<i>, <sup>, common for gene names
    and p-values) silently truncated the abstract/title right there."""

    def test_inline_markup_does_not_truncate_the_abstract(self):
        from jca_phase1.providers.registries import _parse_pubmed_xml
        xml = """<?xml version="1.0"?>
<PubmedArticleSet><PubmedArticle><MedlineCitation>
<PMID>123</PMID>
<Article>
<ArticleTitle>Effect of <i>KRAS</i> mutation on outcomes</ArticleTitle>
<Journal><Title>Annals of Oncology</Title></Journal>
<Abstract><AbstractText>The response rate was <i>51%</i> in <i>KRAS</i>-mutant patients.</AbstractText></Abstract>
</Article>
</MedlineCitation></PubmedArticle></PubmedArticleSet>"""
        pubs = _parse_pubmed_xml(xml)
        self.assertEqual(len(pubs), 1)
        self.assertIn("51%", pubs[0].abstract,
                      "text after an inline <i> tag must not be dropped")
        self.assertIn("KRAS-mutant patients", pubs[0].abstract)
        self.assertEqual(pubs[0].title, "Effect of KRAS mutation on outcomes")


class TestNextIdThreadSafety(unittest.TestCase):

    def test_concurrent_calls_never_produce_duplicate_ids(self):
        """_next_id's counter increment is called from inside
        extract_from_documents()'s ThreadPoolExecutor on every real run.
        A tight concurrent loop reliably reproduces the race without the
        lock; with it, every id must be unique."""
        import concurrent.futures
        ids = []
        lock = __import__("threading").Lock()

        def worker():
            local = [retrieval._next_id("x") for _ in range(50)]
            with lock:
                ids.extend(local)

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(lambda _: worker(), range(16)))
        self.assertEqual(len(ids), len(set(ids)), "duplicate finding_id under concurrency")


# ===========================================================================
# A8 / A9 / A10 — extraction, grounding, validation
# ===========================================================================

def _record(**kw) -> EvidenceRecord:
    base = dict(finding_id="cmp-1", finding_type=FINDING_COMPARATOR,
                subject_drug="Examplimab", source_id="src-1",
                source_url="https://example.org/doc", source_class=C.SRC_HTA_REGULATORY,
                tier=1, member_state="Germany", grounded=True,
                evidence_quote="topotecan is the appropriate comparator",
                comparator=Comparator(as_stated="Topotecan",
                                      role=C.ROLE_ACTIVE_COMPARATOR),
                population_context=PopulationContext(disease="SCLC"))
    base.update(kw)
    return EvidenceRecord(**base)


class TestNormalise(unittest.TestCase):
    """_normalise() used NFKC only, which does not strip accents -- 'Tovorafénib'
    and 'Tovorafenib' tokenized as different words, so every accented mention
    of the subject drug from a French-language source was treated as a
    different drug entirely."""

    def test_accented_and_plain_spelling_tokenize_identically(self):
        self.assertEqual(validation._substance_tokens("Tovorafénib"),
                         validation._substance_tokens("Tovorafenib"))

    def test_is_other_drug_no_longer_fooled_by_the_accent(self):
        self.assertFalse(validation._is_other_drug("Tovorafénib", "Tovorafenib"))

    def test_same_substance_recognises_the_accented_spelling(self):
        self.assertTrue(validation._same_substance("Tovorafénib", "Tovorafenib"))

    def test_genuinely_different_drugs_still_do_not_match(self):
        """Regression guard: the fix must not start conflating real
        different-drug cases just because it's now more lenient about accents."""
        self.assertTrue(validation._is_other_drug("Dabrafenib plus trametinib", "Tovorafenib"))


class TestGrounding(unittest.TestCase):

    def test_value_present_in_text_is_grounded(self):
        ok, _ = validation.is_grounded("Topotecan", "the comparator is topotecan")
        self.assertTrue(ok)

    def test_invented_value_is_not_grounded(self):
        ok, _ = validation.is_grounded("Pembrolizumab plus lenvatinib",
                                       "the comparator is topotecan")
        self.assertFalse(ok)

    def test_quote_window_prefers_numerals(self):
        text = ("Irrelevant preamble. " * 20) + "median overall survival was 13.6 months" + (
                " trailing text." * 20)
        window = validation.best_quote_window("overall survival 13.6 months", text)
        self.assertIn("13.6", window)


class TestClaimValidation(unittest.TestCase):

    def _providers(self, verdicts):
        search = FixtureSearchProvider(documents={"https://example.org/doc": "topotecan " * 50})
        llm = ScriptedLLM({"a10.claim_validation": json.dumps({"results": verdicts})})
        return search, llm

    def test_a_different_subject_drug_is_rejected_without_an_llm_call(self):
        """A document about extensive-stage SCLC discusses several drugs. Without
        subject_drug, one drug's comparator silently becomes another's."""
        rec = _record(subject_drug="Otherdrugimab")
        search, llm = self._providers([])
        inter = Intervention(fields={"product_name_inn": Field(value="Examplimab",
                                                               provenance="confirmed")})
        validation.validate_claims([rec], "SCLC", inter, search, llm)
        self.assertEqual(rec.validation.verdict, C.V_WRONG_SUBJECT_DRUG)
        self.assertEqual(len(llm.call_log), 0)

    def test_role_unclear_defers_the_subject_drug_check_to_the_llm(self):
        """Fix 5: extraction itself wasn't sure how this mention relates to
        the subject drug (role: unclear). Hard-rejecting it deterministically
        makes that uncertainty final before a10's real LLM call -- which has
        the full re-fetched document text -- ever gets to judge it. Real
        production case: a genuinely relevant comparator mention was thrown
        out as WRONG_SUBJECT_DRUG this way."""
        rec = _record(subject_drug="Otherdrugimab",
                      comparator=Comparator(as_stated="Topotecan",
                                            role=C.ROLE_UNCLEAR))
        search, llm = self._providers([{"index": 1, "verdict": "SUPPORTED", "reason": "ok"}])
        inter = Intervention(fields={"product_name_inn": Field(value="Examplimab",
                                                               provenance="confirmed")})
        validation.validate_claims([rec], "SCLC", inter, search, llm)
        self.assertNotEqual(rec.validation.verdict, C.V_WRONG_SUBJECT_DRUG)
        self.assertEqual(len(llm.call_log), 1,
                         "an unclear-role record must survive to the real a10 LLM call")

    def test_confident_role_with_a_different_subject_drug_is_still_hard_rejected(self):
        """Regression guard: Fix 5 must not weaken the check for records
        where extraction WAS confident about the role."""
        rec = _record(subject_drug="Otherdrugimab",
                      comparator=Comparator(as_stated="Topotecan",
                                            role=C.ROLE_ACTIVE_COMPARATOR))
        search, llm = self._providers([])
        inter = Intervention(fields={"product_name_inn": Field(value="Examplimab",
                                                               provenance="confirmed")})
        validation.validate_claims([rec], "SCLC", inter, search, llm)
        self.assertEqual(rec.validation.verdict, C.V_WRONG_SUBJECT_DRUG)
        self.assertEqual(len(llm.call_log), 0)

    def test_confident_comparator_claim_naming_itself_as_subject_drug_is_not_pre_rejected(self):
        """The SME's a10 prompt (`ABSOLUTE RULE FOR COMPARATOR CLAIMS` in
        prompts.registry._CLAIM_VALIDATION) explicitly says subject_drug naming
        the comparator itself is the normal, correct shape of this data -- not
        grounds for rejection. But the deterministic pre-filter only compared
        subject_drug against the requested intervention, so a confidently
        role-tagged comparator claim (the common case) was hard-rejected as
        WRONG_SUBJECT_DRUG before the LLM -- and the SME's prompt rule -- ever
        saw it. Real production case: subject_drug='Vinblastine' on a
        confidently-tagged Vinblastine comparator claim."""
        rec = _record(subject_drug="Vinblastine",
                      comparator=Comparator(as_stated="Vinblastine",
                                            role=C.ROLE_ACTIVE_COMPARATOR))
        search, llm = self._providers([{"index": 1, "verdict": "SUPPORTED", "reason": "ok"}])
        inter = Intervention(fields={"product_name_inn": Field(value="Tovorafenib",
                                                               provenance="confirmed")})
        validation.validate_claims([rec], "LGG", inter, search, llm)
        self.assertNotEqual(rec.validation.verdict, C.V_WRONG_SUBJECT_DRUG)
        self.assertEqual(len(llm.call_log), 1,
                         "must reach the real a10 LLM call, where the SME's prompt rule applies")

    def test_accented_subject_drug_spelling_is_not_rejected_via_the_pipeline(self):
        """Real production bug: a French HAS document spells the subject drug
        'Tovorafénib'. NFKC alone doesn't strip the accent, so _WORD's
        [a-z0-9]+ regex splits the word around it, giving zero token overlap
        with the plain 'Tovorafenib' -- every French-sourced claim about the
        subject drug itself was being rejected as a DIFFERENT drug."""
        rec = _record(subject_drug="Tovorafénib",
                      comparator=Comparator(as_stated="Dabrafenib plus trametinib",
                                            role=C.ROLE_ACTIVE_COMPARATOR))
        rec.source_url = "https://example.org/doc"
        search, llm = self._providers([{"index": 1, "verdict": "SUPPORTED", "reason": "ok"}])
        inter = Intervention(fields={"product_name_inn": Field(value="Tovorafenib",
                                                               provenance="confirmed")})
        validation.validate_claims([rec], "LGG", inter, search, llm)
        self.assertNotEqual(rec.validation.verdict, C.V_WRONG_SUBJECT_DRUG,
                           "the accented spelling of the SAME drug must not be "
                           "treated as evidence about a different one")

    def test_the_intervention_cannot_be_its_own_comparator(self):
        rec = _record(comparator=Comparator(as_stated="Examplimab",
                                            role=C.ROLE_ACTIVE_COMPARATOR))
        search, llm = self._providers([])
        inter = Intervention(fields={"product_name_inn": Field(value="Examplimab",
                                                               provenance="confirmed")})
        validation.validate_claims([rec], "SCLC", inter, search, llm)
        self.assertEqual(rec.validation.verdict, C.V_WRONG_INTERVENTION)

    def test_ungrounded_record_never_reaches_the_validator(self):
        rec = _record(grounded=False)
        search, llm = self._providers([])
        inter = Intervention(fields={"product_name_inn": Field(value="Examplimab",
                                                               provenance="confirmed")})
        validation.validate_claims([rec], "SCLC", inter, search, llm)
        self.assertFalse(rec.validation.passed)

    def test_unparseable_validator_response_fails_closed(self):
        """A malformed batch response must never pass a finding through."""
        rec = _record()
        search = FixtureSearchProvider(documents={"https://example.org/doc": "topotecan " * 50})
        llm = ScriptedLLM({"a10.claim_validation": "not json at all"})
        inter = Intervention(fields={"product_name_inn": Field(value="Examplimab",
                                                               provenance="confirmed")})
        validation.validate_claims([rec], "SCLC", inter, search, llm)
        self.assertEqual(rec.validation.verdict, C.V_NOT_SUPPORTED)

    def test_inaccessible_source_is_distinct_from_unsupported(self):
        rec = _record()
        search = FixtureSearchProvider(documents={})  # URL not present
        llm = ScriptedLLM({})
        inter = Intervention(fields={"product_name_inn": Field(value="Examplimab",
                                                               provenance="confirmed")})
        validation.validate_claims([rec], "SCLC", inter, search, llm)
        self.assertEqual(rec.validation.verdict, C.V_SOURCE_INACCESSIBLE)

    def test_pubmed_records_reuse_the_already_fetched_abstract_instead_of_refetching(self):
        """Real production defect: Tavily's fetch() fails on 100% of
        pubmed.ncbi.nlm.nih.gov URLs ("extract returned no results"), so every
        PubMed-sourced claim was lost as SOURCE_INACCESSIBLE even though the
        actual abstract text (from NCBI's own E-utilities API, method=
        'pubmed_api') was already sitting in `documents`. validate_claims()
        must reuse that text for a PubMed record instead of calling
        search.fetch() at all, while still running the real a10 LLM
        judgment -- not a self-evidencing pass-through."""
        rec = _record(source_url="https://pubmed.ncbi.nlm.nih.gov/12345/",
                      retrieval_method="pubmed_api")
        pubmed_doc = RetrievedDocument(
            url="https://pubmed.ncbi.nlm.nih.gov/12345/",
            resolved_url="https://pubmed.ncbi.nlm.nih.gov/12345/",
            text="Topotecan is the recommended second-line treatment.",
            ok=True, method="pubmed_api")
        # No entry for this URL in FixtureSearchProvider's documents -- a real
        # fetch attempt would fail with SOURCE_INACCESSIBLE, exactly like the
        # production case.
        search = FixtureSearchProvider(documents={})
        llm = ScriptedLLM({"a10.claim_validation": json.dumps(
            {"results": [{"index": 1, "verdict": "SUPPORTED", "reason": "ok"}]})})
        inter = Intervention(fields={"product_name_inn": Field(value="Examplimab",
                                                               provenance="confirmed")})
        validation.validate_claims([rec], "SCLC", inter, search, llm,
                                   documents=[pubmed_doc])
        self.assertEqual(len(search.call_log), 0,
                         "must not attempt a Tavily fetch for an already-fetched PubMed URL")
        self.assertEqual(rec.validation.verdict, C.V_SUPPORTED)
        self.assertEqual(len(llm.call_log), 1,
                         "must still run the real a10 LLM judgment, not a pass-through")

    def test_a_source_that_fails_live_refetch_falls_back_to_the_already_fetched_a7_copy(self):
        """Non-PubMed sources still attempt a genuine independent re-fetch
        first (the SME's explicit requirement), unlike the PubMed case above.
        But if that live re-fetch itself fails -- a page down, a Tavily
        extract hiccup -- falling back to the copy A7 already fetched this
        pass keeps the claim judged against real text instead of losing it
        as SOURCE_INACCESSIBLE. This is a fallback for a flaky live source,
        not a substitute for the independent re-fetch attempt: refetched
        must record False here (was NOT independently re-opened)."""
        rec = _record(source_url="https://has.example/guideline")
        a7_doc = RetrievedDocument(
            url="https://has.example/guideline", resolved_url="https://has.example/guideline",
            text="Topotecan is recommended second-line.", ok=True, method="tavily")
        # Not in FixtureSearchProvider's documents -- the live re-fetch attempt
        # at A10 fails, exactly like a page that's down right now.
        search = FixtureSearchProvider(documents={})
        llm = ScriptedLLM({"a10.claim_validation": json.dumps(
            {"results": [{"index": 1, "verdict": "SUPPORTED", "reason": "ok"}]})})
        inter = Intervention(fields={"product_name_inn": Field(value="Examplimab",
                                                               provenance="confirmed")})
        validation.validate_claims([rec], "SCLC", inter, search, llm, documents=[a7_doc])
        fetch_calls = [c for c in search.call_log if c.op == "fetch"]
        self.assertEqual(len(fetch_calls), 1,
                         "a live re-fetch must still be attempted first for a non-PubMed source")
        self.assertEqual(rec.validation.verdict, C.V_SUPPORTED)
        self.assertFalse(rec.validation.refetched,
                         "fallback to the A7 copy is not an independent re-fetch")

    def test_a_source_with_no_fallback_and_a_failed_live_refetch_is_still_inaccessible(self):
        """Regression guard: the fallback must not mask a genuine total
        failure -- no A7 copy and a failed live re-fetch must still fail
        closed as SOURCE_INACCESSIBLE, exactly as before this change."""
        rec = _record(source_url="https://has.example/guideline")
        search = FixtureSearchProvider(documents={})
        llm = ScriptedLLM({})
        inter = Intervention(fields={"product_name_inn": Field(value="Examplimab",
                                                               provenance="confirmed")})
        validation.validate_claims([rec], "SCLC", inter, search, llm, documents=[])
        self.assertEqual(rec.validation.verdict, C.V_SOURCE_INACCESSIBLE)

    def test_validation_refetches_the_source(self):
        rec = _record()
        search = FixtureSearchProvider(documents={"https://example.org/doc": "topotecan " * 50})
        llm = ScriptedLLM({"a10.claim_validation": json.dumps(
            {"results": [{"index": 1, "verdict": "SUPPORTED", "reason": "ok"}]})})
        inter = Intervention(fields={"product_name_inn": Field(value="Examplimab",
                                                               provenance="confirmed")})
        validation.validate_claims([rec], "SCLC", inter, search, llm)
        self.assertTrue(rec.validation.refetched,
                        "the SME requires the cited source to be re-opened")

    def test_a_dense_source_is_split_into_bounded_sub_batches_with_one_fetch(self):
        """A10/runtime fix: a source with many claims used to go into ONE
        unbounded a10.claim_validation call -- if that response truncated at
        the token cap, EVERY claim in the batch failed closed, including
        genuinely on-target ones. Confirms a source with more than
        _MAX_CLAIMS_PER_VALIDATION_BATCH claims is now split into multiple
        bounded calls, while the document itself is still fetched only ONCE
        (not once per sub-batch)."""
        n = validation._MAX_CLAIMS_PER_VALIDATION_BATCH + 3
        recs = [_record(finding_id=f"cmp-{i}", source_url="https://example.org/dense-doc")
               for i in range(n)]
        search = FixtureSearchProvider(
            documents={"https://example.org/dense-doc": "topotecan " * 200})

        def fake_response(user_prompt):
            # Each sub-batch only ever contains up to the bound -- count how
            # many claims THIS call was actually given.
            num_claims = user_prompt.count("\n   population context stated")
            return json.dumps({"results": [
                {"index": i + 1, "verdict": "SUPPORTED", "reason": "ok"}
                for i in range(num_claims)]})

        llm = ScriptedLLM({"a10.claim_validation": fake_response})
        inter = Intervention(fields={"product_name_inn": Field(value="Examplimab",
                                                               provenance="confirmed")})
        validation.validate_claims(recs, "SCLC", inter, search, llm)

        validation_calls = [c for c in llm.seen if c[0] == "a10.claim_validation"]
        self.assertEqual(len(validation_calls), 2,
                         f"{n} claims at a bound of "
                         f"{validation._MAX_CLAIMS_PER_VALIDATION_BATCH} must split into 2 calls")
        fetch_calls = [c for c in search.call_log if c.op == "fetch"]
        self.assertEqual(len(fetch_calls), 1,
                         "the same dense source must be fetched only ONCE, not once per "
                         "sub-batch")
        self.assertTrue(all(r.validation.verdict == C.V_SUPPORTED for r in recs))


# ===========================================================================
# A11 — comparator identity. The class defect.
# ===========================================================================

class TestPreclusterCandidates(unittest.TestCase):
    """resolve_identities() batches candidates to A11 at batch_size=40 --
    A11 has zero visibility across batches, so two near-duplicate variants
    landing in different batches could never be recognised as the same
    candidate no matter how good the prompt is. precluster_candidates()
    collapses exact canonical-token-set duplicates BEFORE batching, so a
    duplicate pair can never be split by a batch boundary regardless of raw
    candidate volume. This is the same 4-variant Dabrafenib+Trametinib case
    already seen resolving correctly within a single A11 batch in Run 4 --
    pinned here as pure deterministic code, with no LLM call at all."""

    def test_four_dt_variants_cluster_together_cv_and_cve_stay_separate(self):
        variants = ["Dabrafenib + Trametinib", "Dabrafenib (Finlee) + Trametinib",
                   "Trametinib (Spexotras) + Dabrafenib", "Trametinib + Dabrafenib",
                   "Carboplatin + Vincristine", "Carboplatin, Vincristine and Etoposide"]
        clusters = validation.precluster_candidates(variants)
        self.assertEqual(len(clusters), 3,
                         f"expected 3 clusters (D+T, CV, CV+Etoposide), got {len(clusters)}: {clusters}")
        dt_cluster = next(v for v in clusters.values() if len(v) == 4)
        self.assertEqual(set(dt_cluster), {
            "Dabrafenib + Trametinib", "Dabrafenib (Finlee) + Trametinib",
            "Trametinib (Spexotras) + Dabrafenib", "Trametinib + Dabrafenib"})

    def test_cv_and_cv_plus_etoposide_are_not_fuzzy_merged(self):
        """Conservative by design: sharing tokens is not enough -- the sets
        must be IDENTICAL. A genuinely different (larger) regimen must never
        collapse into a smaller one just because it shares two ingredients."""
        clusters = validation.precluster_candidates(
            ["Carboplatin + Vincristine", "Carboplatin, Vincristine and Etoposide"])
        self.assertEqual(len(clusters), 2)

    def test_representative_is_the_shortest_variant(self):
        clusters = validation.precluster_candidates(
            ["Dabrafenib (Finlee) + Trametinib", "Dabrafenib + Trametinib"])
        self.assertEqual(list(clusters.keys()), ["Dabrafenib + Trametinib"])

    def test_cross_batch_duplicates_still_resolve_identically(self):
        """The actual structural fix: with batch_size=1, each cluster
        REPRESENTATIVE gets its own separate LLM call (no chance for the LLM
        itself to notice the duplication across calls) -- yet every original
        variant must still end up with the identical resolved identity,
        because clustering happens before any batching, not because the LLM
        happened to see them together."""
        records = [_record(finding_id=f"cmp-{i}", comparator=Comparator(as_stated=name))
                  for i, name in enumerate([
                      "Dabrafenib + Trametinib", "Dabrafenib (Finlee) + Trametinib",
                      "Trametinib (Spexotras) + Dabrafenib", "Trametinib + Dabrafenib"])]

        def fake_response(user_prompt):
            return json.dumps({
                "items": [{"index": 1, "display_name": "Dabrafenib + Trametinib",
                          "is_combination": True, "is_category": False,
                          "inn": "", "atc_code": "", "class_mechanism": "",
                          "components": [
                              {"name": "Dabrafenib", "inn": "dabrafenib",
                               "class_mechanism": "BRAF kinase inhibitor"},
                              {"name": "Trametinib", "inn": "trametinib",
                               "class_mechanism": "MEK inhibitor"}]}],
                "excluded": []})

        llm = ScriptedLLM({"a11.comparator_identity": fake_response})
        identities, excluded = validation.resolve_identities(records, llm, batch_size=1)
        self.assertEqual(len(llm.seen), 1,
                         "all 4 variants must collapse to ONE representative, ONE LLM call")
        validation.apply_identities(records, identities)
        for rec in records:
            self.assertEqual(rec.comparator.inn, "dabrafenib+trametinib", rec.comparator.as_stated)
            self.assertEqual(rec.comparator.class_source, "source_stated")
            # Fix: the cluster's canonical display wording must propagate to
            # every variant too, not just its substance identity -- otherwise
            # a no-INN category comparator still fragments at A14 grouping,
            # since identity_key() falls back to as_stated when inn is empty.
            self.assertEqual(rec.comparator.as_stated, "Dabrafenib + Trametinib")

    def test_raw_as_stated_preserves_each_variants_original_wording(self):
        """The unified as_stated wins for grouping, but build_comparator()
        still needs each record's ORIGINAL wording to populate
        aliases_merged -- raw_as_stated must capture it before it's
        overwritten, and must never be clobbered on a second apply_identities
        call (e.g. a re-run)."""
        records = [_record(finding_id=f"cmp-{i}", comparator=Comparator(as_stated=name))
                  for i, name in enumerate([
                      "Dabrafenib + Trametinib", "Dabrafenib (Finlee) + Trametinib"])]

        def fake_response(user_prompt):
            return json.dumps({
                "items": [{"index": 1, "display_name": "Dabrafenib + Trametinib",
                          "is_combination": True, "is_category": False,
                          "inn": "", "atc_code": "", "class_mechanism": "",
                          "components": [
                              {"name": "Dabrafenib", "inn": "dabrafenib",
                               "class_mechanism": "BRAF kinase inhibitor"},
                              {"name": "Trametinib", "inn": "trametinib",
                               "class_mechanism": "MEK inhibitor"}]}],
                "excluded": []})

        llm = ScriptedLLM({"a11.comparator_identity": fake_response})
        identities, _ = validation.resolve_identities(records, llm, batch_size=1)
        validation.apply_identities(records, identities)
        self.assertEqual(records[0].comparator.raw_as_stated, "Dabrafenib + Trametinib")
        self.assertEqual(records[1].comparator.raw_as_stated, "Dabrafenib (Finlee) + Trametinib")
        # Re-applying must not overwrite the already-captured original.
        validation.apply_identities(records, identities)
        self.assertEqual(records[1].comparator.raw_as_stated, "Dabrafenib (Finlee) + Trametinib")

    def test_no_inn_category_comparator_variants_share_one_identity_key_after_apply(self):
        """Direct proof the A14-grouping-fragmentation bug is closed: a
        category comparator with no INN (e.g. "standard of care") falls back
        to as_stated for identity_key(). Before the as_stated-propagation
        fix, two near-duplicate wordings A11 correctly clustered together
        still produced two DIFFERENT identity_key()s and fragmented into
        separate A14 groups despite being the same resolved identity."""
        records = [_record(finding_id="cmp-0",
                           comparator=Comparator(as_stated="standard of care")),
                  _record(finding_id="cmp-1",
                         comparator=Comparator(as_stated="standard of care (SOC) chemotherapy"))]
        llm = ScriptedLLM({"a11.comparator_identity": lambda _: json.dumps({
            "items": [{"index": 1, "display_name": "Standard of care",
                      "is_combination": False, "is_category": True,
                      "inn": "", "atc_code": "", "class_mechanism": ""}],
            "excluded": []})})
        identities, _ = validation.resolve_identities(records, llm, batch_size=1)
        validation.apply_identities(records, identities)
        keys = {validation.identity_key(rec.comparator) for rec in records}
        self.assertEqual(len(keys), 1,
                         f"both variants must share one identity_key after apply_identities, "
                         f"got {keys}")

    def test_cross_batch_exclusion_also_propagates_to_every_variant(self):
        records = [_record(finding_id=f"cmp-{i}", comparator=Comparator(as_stated=name))
                  for i, name in enumerate(["Surgery and chemotherapy",
                                            "surgery and chemotherapy (generic)"])]
        # These two normalise to DIFFERENT canonical token sets (an extra
        # "generic" token), so this also confirms exclusion propagation works
        # per-cluster, not just globally -- only genuinely clustered variants
        # share an exclusion.
        llm = ScriptedLLM({"a11.comparator_identity": lambda _: json.dumps({
            "items": [], "excluded": [{"index": 1, "reason": "not a treatment identity"}]})})
        identities, excluded = validation.resolve_identities(records, llm, batch_size=1)
        excluded_values = {e["value"] for e in excluded}
        self.assertIn("Surgery and chemotherapy", excluded_values)
        self.assertIn("surgery and chemotherapy (generic)", excluded_values)


class TestComparatorIdentity(unittest.TestCase):

    def test_class_comes_from_the_comparators_own_inn(self):
        """THE defect: every comparator in all three delivered files carried the
        assessed drug's mechanism."""
        rec = _record(comparator=Comparator(as_stated="Topotecan"))
        identities, _ = validation.resolve_identities([rec], None)
        validation.apply_identities([rec], identities)
        self.assertEqual(rec.comparator.inn, "topotecan")
        self.assertEqual(rec.comparator.class_mechanism, "topoisomerase I inhibitor")
        self.assertEqual(rec.comparator.class_source, "atc_vocabulary")

    def test_rich_component_objects_from_a11_are_coerced_to_names(self):
        """Real production crash: the a11 JSON schema shows an empty
        components array with no worked example, and a sibling top-level
        brand_names field makes it a reasonable model inference that each
        component should be a rich object (inn/display_name/brand_names/...)
        rather than a bare string. Claude genuinely returned that shape for
        "Dabrafenib + Trametinib" in a real run -- an un-coerced dict
        reaching identity_key()'s unicodedata.normalize() crashed the whole
        pipeline with a TypeError. Both shapes must resolve to plain names."""
        rec = _record(comparator=Comparator(as_stated="Dabrafenib + Trametinib"))
        llm = ScriptedLLM({"a11.comparator_identity": json.dumps({
            "items": [{
                "index": 1, "inn": "", "display_name": "Dabrafenib + Trametinib",
                "brand_names": [], "is_combination": True, "is_category": False,
                "class_mechanism": "", "atc_code": "",
                "components": [
                    {"inn": "dabrafenib", "display_name": "Dabrafenib",
                     "brand_names": [], "class_mechanism": "BRAF kinase inhibitor",
                     "atc_code": "L01XE23"},
                    {"inn": "trametinib", "display_name": "Trametinib",
                     "brand_names": [], "class_mechanism": "MEK inhibitor",
                     "atc_code": "L01XE25"},
                ]}],
            "excluded": []})})
        identities, _ = validation.resolve_identities([rec], llm)
        validation.apply_identities([rec], identities)
        self.assertEqual(rec.comparator.components, ["dabrafenib", "trametinib"])
        # Must not raise -- this is exactly what crashed in production.
        key = validation.identity_key(rec.comparator)
        self.assertEqual(key, "dabrafenib+trametinib")

    def test_combination_resolves_class_mechanism_from_per_component_data(self):
        """Fix 4: the a11 schema's flat inn/atc_code/class_mechanism fields
        couldn't hold a per-component answer for a combination, so every
        combination in three real production runs came out class_source:
        unresolved even when the model correctly identified each component's
        own class. Same "Dabrafenib + Trametinib" fixture as the crash test
        above -- this pins that class_mechanism/class_source now roll up
        from the per-component data instead of staying empty."""
        rec = _record(comparator=Comparator(as_stated="Dabrafenib + Trametinib"))
        llm = ScriptedLLM({"a11.comparator_identity": json.dumps({
            "items": [{
                "index": 1, "display_name": "Dabrafenib + Trametinib",
                "brand_names": [], "is_combination": True, "is_category": False,
                "inn": "", "atc_code": "", "class_mechanism": "",
                "components": [
                    {"name": "Dabrafenib", "inn": "dabrafenib",
                     "class_mechanism": "BRAF kinase inhibitor", "atc_code": "L01XE23"},
                    {"name": "Trametinib", "inn": "trametinib",
                     "class_mechanism": "MEK inhibitor", "atc_code": "L01XE25"},
                ]}],
            "excluded": []})})
        identities, _ = validation.resolve_identities([rec], llm)
        validation.apply_identities([rec], identities)
        self.assertEqual(rec.comparator.class_source, "source_stated",
                         "must resolve from per-component data, not stay 'unresolved'")
        self.assertIn("BRAF kinase inhibitor", rec.comparator.class_mechanism)
        self.assertIn("MEK inhibitor", rec.comparator.class_mechanism)
        self.assertEqual(rec.comparator.inn, "dabrafenib+trametinib")

    def test_coerce_component_names_accepts_strings_dicts_and_junk(self):
        self.assertEqual(
            validation._coerce_component_names(["topotecan", "cav"]),
            ["topotecan", "cav"])
        self.assertEqual(
            validation._coerce_component_names([{"display_name": "Dabrafenib"},
                                                 {"inn": "trametinib"}]),
            ["Dabrafenib", "trametinib"])
        self.assertEqual(validation._coerce_component_names([{}, "", None]), [])
        self.assertEqual(validation._coerce_component_names(None), [])

    def test_coerce_component_names_prefers_inn_over_a_brand_annotated_display_name(self):
        """A second real finding from the same production run: one source's
        mention carried a clean inn AND a brand-annotated display_name
        ("Dabrafenib (SPEXOTRAS)") for the same combination three OTHER
        sources named without the brand. Preferring display_name here would
        silently stop this mention from merging into the same identity_key()
        group as the rest -- inn is the canonical merge key and must win."""
        self.assertEqual(
            validation._coerce_component_names(
                [{"inn": "dabrafenib", "display_name": "Dabrafenib (SPEXOTRAS)"}]),
            ["dabrafenib"])

    def test_brand_annotated_mention_merges_with_plain_mentions_of_the_same_combo(self):
        plain = Comparator(as_stated="Dabrafenib plus Trametinib", is_combination=True,
                           components=validation._coerce_component_names(
                               [{"inn": "dabrafenib", "display_name": "Dabrafenib"},
                                {"inn": "trametinib", "display_name": "Trametinib"}]))
        branded = Comparator(as_stated="Trametinib (FINLEE) + Dabrafenib (SPEXOTRAS)",
                             is_combination=True, components=validation._coerce_component_names(
                                 [{"inn": "trametinib", "display_name": "Trametinib (FINLEE)"},
                                  {"inn": "dabrafenib", "display_name": "Dabrafenib (SPEXOTRAS)"}]))
        self.assertEqual(validation.identity_key(plain), validation.identity_key(branded))

    def test_extraction_never_sets_a_comparator_class(self):
        """Even if the model returns one, the extraction boundary clears it."""
        doc_text = "The comparator was topotecan in the second line."
        llm = ScriptedLLM({"a08.extraction": json.dumps([{
            "finding_type": "comparator", "subject_drug": "Examplimab",
            "comparator": {"as_stated": "Topotecan", "role": "active_comparator",
                           "class_mechanism": "bispecific T-cell engager"},
            "population_context": {"disease": "SCLC"},
            "evidence_quote": "The comparator was topotecan"}])})
        from jca_phase1.schema import RetrievedDocument
        doc = RetrievedDocument(url="https://x.org/a/b/c", resolved_url="https://x.org/a/b/c",
                                text=doc_text, ok=True, source_class=C.SRC_HTA_REGULATORY)
        pop = Population(fields={"indication_disease": Field(value="SCLC", provenance="confirmed")})
        inter = Intervention(fields={"product_name_inn": Field(value="Examplimab",
                                                               provenance="confirmed")})
        recs = retrieval._extract_one(doc, pop, inter, llm)
        self.assertTrue(recs)
        self.assertEqual(recs[0].comparator.class_mechanism, "",
                         "extraction must never populate a comparator class")

    def test_extraction_also_coerces_rich_component_objects(self):
        """Same defensive fix applied at the a08.extraction boundary, since
        it constructs Comparator.components from LLM JSON the same way."""
        from jca_phase1.schema import RetrievedDocument
        llm = ScriptedLLM({"a08.extraction": json.dumps([{
            "finding_type": "comparator", "subject_drug": "Examplimab",
            "comparator": {"as_stated": "Dabrafenib + Trametinib",
                           "role": "active_comparator", "is_combination": True,
                           "components": [{"inn": "dabrafenib", "display_name": "Dabrafenib"},
                                          {"inn": "trametinib", "display_name": "Trametinib"}]},
            "population_context": {"disease": "pLGG"},
            "evidence_quote": "Dabrafenib plus trametinib was given"}])})
        doc = RetrievedDocument(url="https://x.org/a", resolved_url="https://x.org/a",
                                text="Dabrafenib plus trametinib was given.",
                                ok=True, source_class=C.SRC_HTA_REGULATORY)
        pop = Population(fields={"indication_disease": Field(value="pLGG", provenance="confirmed")})
        inter = Intervention(fields={"product_name_inn": Field(value="Examplimab",
                                                               provenance="confirmed")})
        recs = retrieval._extract_one(doc, pop, inter, llm)
        self.assertTrue(recs)
        self.assertEqual(recs[0].comparator.components, ["dabrafenib", "trametinib"])

    def test_brand_name_resolves_to_inn(self):
        rec = _record(comparator=Comparator(as_stated="Tagrisso"))
        identities, _ = validation.resolve_identities([rec], None)
        validation.apply_identities([rec], identities)
        self.assertEqual(rec.comparator.inn, "osimertinib")

    def test_combinations_key_on_the_complete_component_set(self):
        """Two regimens sharing one ingredient are two regimens, never a third
        synthesised one."""
        a = Comparator(as_stated="A + B + C", is_combination=True,
                       components=["a", "b", "c"])
        b = Comparator(as_stated="D + C", is_combination=True, components=["d", "c"])
        self.assertNotEqual(validation.identity_key(a), validation.identity_key(b))

    def test_same_combination_worded_differently_merges(self):
        a = Comparator(as_stated="carboplatin + etoposide", is_combination=True,
                       components=["carboplatin", "etoposide"])
        b = Comparator(as_stated="etoposide plus carboplatin", is_combination=True,
                       components=["etoposide", "carboplatin"])
        self.assertEqual(validation.identity_key(a), validation.identity_key(b))

    def test_unresolved_identity_is_marked_not_guessed(self):
        rec = _record(comparator=Comparator(as_stated="Some unlisted regimen"))
        identities, _ = validation.resolve_identities([rec], None)
        validation.apply_identities([rec], identities)
        self.assertEqual(rec.comparator.class_source, "unresolved")
        self.assertEqual(rec.comparator.class_mechanism, "")

    def test_a11_excluded_candidate_is_not_resurrected_as_unresolved(self):
        """Real production defect (Run 4): a11 explicitly excluded "surgery
        and chemotherapy" with a reason ("not resolvable as a single
        substance-based comparator identity"), but the final `distinct`
        resurrection loop re-added it anyway as class_source: unresolved,
        so it reached A12 adjudication (uncertain) and the final export --
        directly contradicting A11's own verdict."""
        rec = _record(comparator=Comparator(as_stated="Surgery and chemotherapy"))
        llm = ScriptedLLM({"a11.comparator_identity": json.dumps({
            "items": [],
            "excluded": [{"index": 1, "reason": "not a resolvable substance identity"}]})})
        identities, excluded = validation.resolve_identities([rec], llm)
        self.assertNotIn(validation._normalise("Surgery and chemotherapy"),
                         {validation._normalise(k) for k in identities})
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0]["value"], "Surgery and chemotherapy")

    def test_filter_a11_excluded_drops_the_record_before_grouping(self):
        rec_excluded = _record(finding_id="cmp-x",
                               comparator=Comparator(as_stated="Surgery and chemotherapy"))
        rec_kept = _record(finding_id="cmp-y", comparator=Comparator(as_stated="Topotecan"))
        excluded = [{"value": "Surgery and chemotherapy", "reason": "not a treatment identity"}]
        survivors = validation.filter_a11_excluded([rec_excluded, rec_kept], excluded)
        self.assertEqual([r.finding_id for r in survivors], ["cmp-y"])

    def test_filter_a11_excluded_is_a_no_op_when_nothing_was_excluded(self):
        rec = _record(comparator=Comparator(as_stated="Topotecan"))
        self.assertEqual(validation.filter_a11_excluded([rec], []), [rec])


# ===========================================================================
# A8 — deterministic peer-review classification and conference data_maturity.
# [CONFIRMED FROM SME] Preprints and unconfirmed-status publications are
# never excluded, only clearly marked; this applies to PubMed AND general-web
# sources.
# ===========================================================================

def _extract_with(source_class, publication_types=None, parsed_item=None):
    from jca_phase1.schema import RetrievedDocument
    item = parsed_item or {
        "finding_type": "comparator", "subject_drug": "Examplimab",
        "comparator": {"as_stated": "Topotecan", "role": "active_comparator"},
        "population_context": {"disease": "SCLC"},
        "evidence_quote": "The comparator was topotecan"}
    llm = ScriptedLLM({"a08.extraction": json.dumps([item])})
    doc = RetrievedDocument(url="https://x.org/a", resolved_url="https://x.org/a",
                            text="The comparator was topotecan in the second line.",
                            ok=True, source_class=source_class,
                            publication_types=publication_types or [])
    pop = Population(fields={"indication_disease": Field(value="SCLC", provenance="confirmed")})
    inter = Intervention(fields={"product_name_inn": Field(value="Examplimab",
                                                           provenance="confirmed")})
    recs = retrieval._extract_one(doc, pop, inter, llm)
    return recs[0]


class TestPeerReviewClassification(unittest.TestCase):

    def test_pubmed_preprint_is_marked_not_excluded(self):
        rec = _extract_with(C.SRC_PUBMED, publication_types=["Preprint"])
        self.assertEqual(rec.peer_review_status, C.PEER_REVIEW_PREPRINT)

    def test_pubmed_journal_article_is_confirmed_peer_reviewed(self):
        rec = _extract_with(C.SRC_PUBMED, publication_types=["Journal Article"])
        self.assertEqual(rec.peer_review_status, C.PEER_REVIEW_CONFIRMED)

    def test_pubmed_with_no_publication_types_is_unconfirmed(self):
        """The API returned nothing usable -- honestly unconfirmed, not
        assumed peer-reviewed just because the source class is 'pubmed'."""
        rec = _extract_with(C.SRC_PUBMED, publication_types=[])
        self.assertEqual(rec.peer_review_status, C.PEER_REVIEW_UNCONFIRMED)

    def test_general_web_is_always_unconfirmed(self):
        """No API, no metadata -- a flat deterministic default, never a guess."""
        rec = _extract_with(C.SRC_GENERAL_WEB)
        self.assertEqual(rec.peer_review_status, C.PEER_REVIEW_UNCONFIRMED)

    def test_non_literature_sources_are_not_applicable(self):
        """Peer-review status is not a meaningful concept for an HTA report;
        forcing 'unconfirmed' onto it would mislead, not inform."""
        rec = _extract_with(C.SRC_HTA_REGULATORY)
        self.assertEqual(rec.peer_review_status, "")

    def test_known_preprint_domain_is_the_fallback_when_types_are_missing(self):
        """publication_types is only populated on the direct PubMed-API path
        (documents_from_publications). A pubmed-tagged document reaching
        extraction some other way, with no such metadata, must still be
        caught if its own URL names a known preprint server -- not silently
        defaulted to 'unconfirmed' just because the metadata is absent."""
        from jca_phase1.schema import RetrievedDocument
        item = {"finding_type": "comparator", "subject_drug": "Examplimab",
                "comparator": {"as_stated": "Topotecan", "role": "active_comparator"},
                "population_context": {"disease": "SCLC"},
                "evidence_quote": "The comparator was topotecan"}
        llm = ScriptedLLM({"a08.extraction": json.dumps([item])})
        doc = RetrievedDocument(url="https://www.biorxiv.org/content/x",
                                resolved_url="https://www.biorxiv.org/content/x",
                                text="The comparator was topotecan in the second line.",
                                ok=True, source_class=C.SRC_PUBMED, publication_types=[])
        pop = Population(fields={"indication_disease": Field(value="SCLC", provenance="confirmed")})
        inter = Intervention(fields={"product_name_inn": Field(value="Examplimab",
                                                               provenance="confirmed")})
        rec = retrieval._extract_one(doc, pop, inter, llm)[0]
        self.assertEqual(rec.peer_review_status, C.PEER_REVIEW_PREPRINT)


class TestDataMaturity(unittest.TestCase):

    def test_data_maturity_is_captured_from_the_model_for_conference_sources(self):
        item = {"finding_type": "outcome", "subject_drug": "Examplimab",
                "outcome": {"measure": "Overall survival"},
                "population_context": {"disease": "SCLC"},
                "evidence_quote": "median OS not yet reached, interim analysis",
                "data_maturity": "interim"}
        rec = _extract_with(C.SRC_CONFERENCE, parsed_item=item)
        self.assertEqual(rec.data_maturity, "interim")
        self.assertEqual(rec.evidence_status, "conference_abstract")

    def test_data_maturity_defaults_to_empty_when_not_stated(self):
        rec = _extract_with(C.SRC_CONFERENCE)
        self.assertEqual(rec.data_maturity, "")

    def test_data_maturity_is_never_set_for_non_conference_sources(self):
        """The field is documented as conference-only; a guideline document
        that happens to mention 'interim analysis' must not pick it up just
        because the model returned it -- evidence_status is gated the same
        way, and data_maturity must be too."""
        item = {"finding_type": "outcome", "subject_drug": "Examplimab",
                "outcome": {"measure": "Overall survival"},
                "population_context": {"disease": "SCLC"},
                "evidence_quote": "interim analysis reported",
                "data_maturity": "interim"}
        rec = _extract_with(C.SRC_CLINICAL_GUIDELINE, parsed_item=item)
        self.assertEqual(rec.data_maturity, "")


# ===========================================================================
# A12 — scope adjudication
# ===========================================================================

class TestScopeAdjudication(unittest.TestCase):

    def test_prior_therapy_only_is_rejected_deterministically(self):
        """The induction-phase backbone trap: treatment history, not a
        comparator for the later phase."""
        rec = _record(comparator=Comparator(as_stated="Carboplatin + etoposide",
                                            role=C.ROLE_PRIOR_THERAPY))
        adj = validation.adjudicate_scope("k", rec.comparator, [rec],
                                          __import__("jca_phase1.schema", fromlist=["x"]).ScopeBoundary(),
                                          Intervention(), None)
        self.assertEqual(adj.verdict, C.SCOPE_OUT)
        self.assertEqual(adj.decisive_facet, "role")

    def test_evidence_against_is_retained(self):
        rec = _record(comparator=Comparator(as_stated="X", role=C.ROLE_BACKGROUND))
        from jca_phase1.schema import ScopeBoundary
        adj = validation.adjudicate_scope("k", rec.comparator, [rec], ScopeBoundary(),
                                          Intervention(), None)
        self.assertTrue(adj.evidence_against,
                        "an exclusion must be reviewable, not a silent drop")

    def test_no_adjudicator_yields_uncertain_not_in_scope(self):
        rec = _record()
        from jca_phase1.schema import ScopeBoundary
        adj = validation.adjudicate_scope("k", rec.comparator, [rec], ScopeBoundary(),
                                          Intervention(), None)
        self.assertEqual(adj.verdict, C.SCOPE_UNCERTAIN)


# ===========================================================================
# A13 — per-state verdicts
# ===========================================================================

class TestMemberStateAssignment(unittest.TestCase):

    def test_every_comparator_gets_27_verdicts(self):
        """The UI renders all 27 states with a two-value legend, so 'absent'
        must never be ambiguous."""
        verdicts = validation.assign_member_states([_record()], C.EU_27_MEMBER_STATES)
        self.assertEqual(len(verdicts), 27)
        self.assertEqual({v.member_state for v in verdicts}, set(C.EU_27_MEMBER_STATES))

    def test_state_with_evidence_is_standard_of_care(self):
        verdicts = validation.assign_member_states([_record(member_state="Germany")],
                                                   C.EU_27_MEMBER_STATES)
        germany = next(v for v in verdicts if v.member_state == "Germany")
        self.assertEqual(germany.verdict, C.STATE_STANDARD_OF_CARE)

    def test_states_without_evidence_are_not_established_not_not_used(self):
        """Asserting 'not used' would be a claim no source supports."""
        verdicts = validation.assign_member_states([_record(member_state="Germany")],
                                                   C.EU_27_MEMBER_STATES)
        malta = next(v for v in verdicts if v.member_state == "Malta")
        self.assertEqual(malta.verdict, C.STATE_NOT_ESTABLISHED)
        self.assertTrue(malta.reason)

    def test_terminated_trial_loses_to_a_same_tier_non_terminated_record(self):
        """[CONFIRMED FROM SME] Terminated-trial evidence is flagged and given
        less importance, never excluded, when it competes with non-terminated
        evidence at the same tier for the per-state representative pick."""
        terminated = _record(member_state="Germany", tier=2,
                             source_id="src-terminated", trial_status="TERMINATED",
                             evidence_quote="a very long and detailed terminated-trial quote")
        active = _record(member_state="Germany", tier=2,
                         source_id="src-active", trial_status="",
                         evidence_quote="short")
        verdicts = validation.assign_member_states([terminated, active], C.EU_27_MEMBER_STATES)
        germany = next(v for v in verdicts if v.member_state == "Germany")
        self.assertEqual(germany.source_id, "src-active",
                         "non-terminated evidence must win even though the "
                         "terminated record has the longer quote")

    def test_terminated_trial_still_wins_when_it_is_the_only_evidence(self):
        """Not excluded: still selected when there is nothing better."""
        terminated = _record(member_state="Germany", trial_status="TERMINATED")
        verdicts = validation.assign_member_states([terminated], C.EU_27_MEMBER_STATES)
        germany = next(v for v in verdicts if v.member_state == "Germany")
        self.assertEqual(germany.verdict, C.STATE_STANDARD_OF_CARE)


# ===========================================================================
# A8/A14 — TrialRecord.status -> EvidenceRecord.trial_status, and terminated-
# trial de-prioritisation inside consolidation. [CONFIRMED FROM SME]
# ===========================================================================

class TestTrialConversion(unittest.TestCase):

    def test_trial_status_is_propagated_onto_evidence_records(self):
        trial = TrialRecord(identifier="NCT999", status="TERMINATED",
                            arms=[TrialArm(label="Arm A", arm_type="ACTIVE_COMPARATOR",
                                          interventions=["Topotecan"])],
                            primary_outcomes=["Overall survival"])
        recs = retrieval.records_from_trials(
            [trial], Intervention(fields={"product_name_inn":
                                          Field(value="Examplimab", provenance="confirmed")}))
        comparator_recs = [r for r in recs if r.finding_type == FINDING_COMPARATOR]
        outcome_recs = [r for r in recs if r.finding_type == FINDING_OUTCOME]
        self.assertTrue(comparator_recs and outcome_recs)
        self.assertTrue(all(r.trial_status == "TERMINATED" for r in comparator_recs))
        self.assertTrue(all(r.trial_status == "TERMINATED" for r in outcome_recs))
        self.assertTrue(comparator_recs[0].is_terminated_trial)

    def test_non_terminated_trial_is_not_flagged(self):
        trial = TrialRecord(identifier="NCT1000", status="RECRUITING",
                            arms=[TrialArm(label="Arm A", arm_type="ACTIVE_COMPARATOR",
                                          interventions=["Topotecan"])])
        recs = retrieval.records_from_trials(
            [trial], Intervention(fields={"product_name_inn":
                                          Field(value="Examplimab", provenance="confirmed")}))
        self.assertFalse(recs[0].is_terminated_trial)


class TestComparatorArms(unittest.TestCase):
    """comparator_arms() used to exclude any arm sharing ANY token with the
    subject drug's name -- wrongly dropping a biosimilar or an ADC built on
    the same root antibody as a genuinely different comparator."""

    def test_name_similar_but_explicitly_typed_comparator_arm_is_kept(self):
        trial = TrialRecord(arms=[
            TrialArm(label="Experimental", arm_type="EXPERIMENTAL",
                     interventions=["Trastuzumab deruxtecan"]),
            TrialArm(label="Comparator", arm_type="ACTIVE_COMPARATOR",
                     interventions=["Trastuzumab"]),
        ])
        arms = trial.comparator_arms("Trastuzumab deruxtecan")
        self.assertEqual([a.label for a in arms], ["Comparator"],
                         "an ACTIVE_COMPARATOR-typed arm must never be excluded by "
                         "name, however similar it looks to the subject drug")

    def test_exact_name_match_on_an_untyped_arm_is_still_excluded(self):
        trial = TrialRecord(arms=[
            TrialArm(label="Arm 1", arm_type="EXPERIMENTAL", interventions=["Examplimab"]),
            TrialArm(label="Arm 2", arm_type="ACTIVE_COMPARATOR", interventions=["Topotecan"]),
        ])
        arms = trial.comparator_arms("Examplimab")
        self.assertEqual([a.label for a in arms], ["Arm 2"])

    def test_clinicaltrials_gov_intervention_type_prefix_does_not_defeat_exact_match(self):
        """Real production bug, confirmed against a real run: ClinicalTrials.gov's
        v2 API prefixes EVERY interventionNames entry with its type category --
        "Drug: Tovorafenib", never bare "Tovorafenib". The exact-token-set match
        added for the ADC/biosimilar fix (above) broke on this prefix
        ({"drug","tovorafenib"} != {"tovorafenib"}), letting a single-arm
        trial's own EXPERIMENTAL arm through as a fake "comparator" -- on
        every trial this client fetches, not just this one. Must be stripped
        at parse time so comparator_arms() sees the clean name."""
        from jca_phase1.providers.registries import ClinicalTrialsGovClient
        payload = {"studies": [{"protocolSection": {
            "identificationModule": {"nctId": "NCT04775485", "briefTitle": "x"},
            "armsInterventionsModule": {"armGroups": [
                {"label": "Arm 1: Low-Grade Glioma", "type": "EXPERIMENTAL",
                 "interventionNames": ["Drug: Tovorafenib"]},
            ]},
            "designModule": {}, "statusModule": {}, "conditionsModule": {},
            "eligibilityModule": {}, "outcomesModule": {},
        }}]}
        trials = ClinicalTrialsGovClient._parse(payload)
        self.assertEqual(trials[0].arms[0].interventions, ["Tovorafenib"])
        self.assertEqual(trials[0].comparator_arms("Tovorafenib"), [],
                         "the subject drug's own single arm must not survive as a "
                         "\"comparator\" just because ClinicalTrials.gov prefixed "
                         "its intervention name with \"Drug: \"")

    def test_strip_intervention_type_prefix_covers_the_common_categories(self):
        from jca_phase1.providers.registries import _strip_intervention_type_prefix
        cases = [
            ("Drug: Topotecan", "Topotecan"),
            ("Biological: Tarlatamab", "Tarlatamab"),
            ("Combination Product: CAV", "CAV"),
            ("Placebo", "Placebo"),  # no prefix -- unchanged
        ]
        for raw, expected in cases:
            self.assertEqual(_strip_intervention_type_prefix(raw), expected, raw)

    def test_exact_self_match_is_excluded_even_when_arm_type_is_a_comparator_type(self):
        """Real gap: the exact-identity exclusion only ran when arm_type was
        NOT one of _COMPARATOR_ARM_TYPES -- so a sponsor's inconsistent
        CT.gov data entry (mistyping the drug's own arm as OTHER rather than
        EXPERIMENTAL) let the subject drug's own arm through as a fake
        "comparator". An exact, complete token-set match to the subject drug
        is never a legitimately different comparator, whatever arm_type the
        registry assigned it."""
        trial = TrialRecord(arms=[
            TrialArm(label="Arm 1", arm_type="OTHER", interventions=["Tovorafenib"]),
            TrialArm(label="Arm 2", arm_type="ACTIVE_COMPARATOR",
                     interventions=["Dabrafenib", "Trametinib"]),
        ])
        result = trial.comparator_arms("Tovorafenib")
        names = [", ".join(a.interventions) for a in result]
        self.assertNotIn("Tovorafenib", names,
                         "self-match leaked through despite arm_type=OTHER")
        self.assertIn("Dabrafenib, Trametinib", names,
                      "genuine comparator must not be wrongly excluded")

    def test_partial_overlap_on_a_comparator_typed_arm_is_still_kept(self):
        """Regression guard: the fix must stay scoped to EXACT matches --
        a biosimilar/ADC sharing one token with the subject drug, on an arm
        the registry itself types as a comparator, must still survive."""
        trial = TrialRecord(arms=[
            TrialArm(label="Comparator", arm_type="ACTIVE_COMPARATOR",
                     interventions=["Trastuzumab"]),
        ])
        arms = trial.comparator_arms("Trastuzumab deruxtecan")
        self.assertEqual([a.label for a in arms], ["Comparator"])


class TestSoCArmBundling(unittest.TestCase):
    """records_from_trials() used `is_combo = len(arm.interventions) > 1` --
    true for a genuine combination, but also true for a "Standard of Care"/
    "Investigator's Choice" arm listing several ALTERNATIVE single-agent
    options, which CT.gov represents identically as a flat list. Confirmed
    real bundling defect: ["Lurbinectedin", "Topotecan", "Amrubicin"] joined
    into one fake three-drug "comparator" instead of three real, distinct
    ones."""

    def _inter(self, name="Tovorafenib"):
        return Intervention(fields={"product_name_inn": Field(value=name,
                                                              provenance="confirmed")})

    def test_investigators_choice_arm_is_split_into_standalone_comparators(self):
        soc_arm = TrialArm(
            label="Standard of Care", arm_type="ACTIVE_COMPARATOR",
            interventions=["Lurbinectedin", "Topotecan", "Amrubicin"],
            description="Investigator's choice of lurbinectedin, topotecan, or amrubicin.")
        trial = TrialRecord(identifier="NCT_TEST", arms=[soc_arm])
        records = retrieval.records_from_trials([trial], self._inter())
        names = sorted(r.comparator.as_stated for r in records if r.comparator)
        self.assertEqual(names, ["Amrubicin", "Lurbinectedin", "Topotecan"])
        self.assertTrue(all(not r.comparator.is_combination for r in records))
        self.assertTrue(all(r.comparator.comparator_scenario == "at_least_one" for r in records))

    def test_genuine_combination_arm_is_kept_joined(self):
        combo_arm = TrialArm(
            label="Dabrafenib plus Trametinib", arm_type="EXPERIMENTAL",
            interventions=["Dabrafenib", "Trametinib"],
            description="Dabrafenib in combination with trametinib.")
        trial = TrialRecord(identifier="NCT_TEST3", arms=[combo_arm])
        records = retrieval.records_from_trials([trial], self._inter())
        comparator_recs = [r for r in records if r.comparator]
        self.assertEqual(len(comparator_recs), 1,
                         "a genuine combination must stay one record, not split")
        self.assertEqual(comparator_recs[0].comparator.as_stated, "Dabrafenib, Trametinib")
        self.assertTrue(comparator_recs[0].comparator.is_combination)

    def test_ambiguous_multi_intervention_arm_defaults_to_split_not_a_guess(self):
        """No choice marker, no combination marker -- don't assert either
        confidently; split (more useful for scoping than an uninterpretable
        joint name) and flag the ambiguity via comparator_scenario."""
        ambiguous_arm = TrialArm(
            label="Arm B", arm_type="ACTIVE_COMPARATOR",
            interventions=["Vinblastine", "Carboplatin"], description="")
        trial = TrialRecord(identifier="NCT_TEST4", arms=[ambiguous_arm])
        records = retrieval.records_from_trials([trial], self._inter())
        names = sorted(r.comparator.as_stated for r in records if r.comparator)
        self.assertEqual(names, ["Carboplatin", "Vinblastine"])
        self.assertTrue(all(not r.comparator.is_combination for r in records))
        self.assertTrue(all(r.comparator.comparator_scenario == "individualised"
                            for r in records))

    def test_subject_drug_listed_as_one_of_the_choice_options_is_excluded(self):
        """The joined-string exact-match check on comparator_arms() cannot
        catch this -- the joined set {"tovorafenib","topotecan"} never
        equals the subject drug's own token set. Only re-checking each
        SPLIT-OUT candidate individually catches it."""
        soc_arm = TrialArm(
            label="Investigator's Choice", arm_type="ACTIVE_COMPARATOR",
            interventions=["Tovorafenib", "Topotecan"],
            description="Investigator's choice to continue tovorafenib or switch to topotecan.")
        trial = TrialRecord(identifier="NCT_TEST5", arms=[soc_arm])
        records = retrieval.records_from_trials([trial], self._inter("Tovorafenib"))
        names = [r.comparator.as_stated for r in records if r.comparator]
        self.assertEqual(names, ["Topotecan"],
                         "the subject drug's own name must not survive as one of the "
                         "split choice options")


class TestComparatorGrouping(unittest.TestCase):
    """group_comparators() had a filter comparing finding_type (a
    comparator/outcome value) against C.SRC_HTA_REGULATORY (a source-class
    constant) -- always true, silently redundant with the very next check.
    Removed; this pins the actual, intended behaviour going forward."""

    def test_outcome_only_records_are_excluded(self):
        outcome_rec = _record(finding_type="outcome", comparator=None,
                              outcome=None)
        groups = consolidation.group_comparators([outcome_rec])
        self.assertEqual(groups, {})

    def test_comparator_records_are_grouped_by_identity(self):
        rec = _record()
        groups = consolidation.group_comparators([rec])
        self.assertEqual(len(groups), 1)


class TestEvidenceRefFieldPropagation(unittest.TestCase):
    """The four fields that used to be dropped (or never existed) at the
    EvidenceRecord -> EvidenceRef conversion inside consolidation."""

    def _consolidate(self, records):
        from jca_phase1.schema import ScopeAdjudication
        return consolidation.build_comparator(
            "k", Comparator(as_stated="Topotecan"), records,
            ScopeAdjudication(), C.EU_27_MEMBER_STATES, None)

    def test_evidence_status_and_data_maturity_survive_consolidation(self):
        rec = _record(source_class=C.SRC_CONFERENCE, evidence_status="conference_abstract",
                     data_maturity="interim")
        consolidated = self._consolidate([rec])
        self.assertEqual(consolidated.evidence[0].evidence_status, "conference_abstract")
        self.assertEqual(consolidated.evidence[0].data_maturity, "interim")

    def test_peer_review_status_survives_consolidation(self):
        rec = _record(source_class=C.SRC_PUBMED, peer_review_status=C.PEER_REVIEW_PREPRINT)
        consolidated = self._consolidate([rec])
        self.assertEqual(consolidated.evidence[0].peer_review_status, C.PEER_REVIEW_PREPRINT)

    def test_terminated_trial_evidence_is_kept_not_excluded(self):
        """[CONFIRMED FROM SME] Never excluded — both records must still be
        present in the final evidence list."""
        terminated = _record(source_id="src-terminated", source_class=C.SRC_TRIAL_REGISTRY,
                             tier=2, trial_status="TERMINATED",
                             evidence_quote="terminated trial arm quote")
        active = _record(source_id="src-active", source_class=C.SRC_TRIAL_REGISTRY,
                         tier=2, trial_status="", evidence_quote="active trial arm quote")
        consolidated = self._consolidate([terminated, active])
        trial_statuses = {e.trial_status for e in consolidated.evidence}
        self.assertEqual(trial_statuses, {"TERMINATED", ""})

    def test_terminated_trial_loses_the_recommendation_strength_pick(self):
        """A terminated record placed FIRST in encounter order must not win
        the representative pick purely by list position."""
        terminated = _record(source_id="src-terminated", source_class=C.SRC_TRIAL_REGISTRY,
                             tier=2, trial_status="TERMINATED",
                             recommendation_strength=C.REC_PREFERRED,
                             evidence_quote="terminated trial arm quote")
        active = _record(source_id="src-active", source_class=C.SRC_TRIAL_REGISTRY,
                         tier=2, trial_status="", recommendation_strength=C.REC_CONDITIONAL,
                         evidence_quote="active trial arm quote")
        consolidated = self._consolidate([terminated, active])
        self.assertEqual(consolidated.recommendation_strength, C.REC_CONDITIONAL,
                         "non-terminated evidence must win the representative "
                         "pick even though the terminated record came first")


class TestMultiPopulationVerdictVisibility(unittest.TestCase):
    """Run 4 post-fix validation, defect 2e: Dabrafenib+Trametinib was
    adjudicated once per population (in_scope for one, uncertain for
    another, for a genuinely different reason -- GT itself splits this by
    population). Only the 'best' verdict reached the final export; the
    discarded uncertain verdict and its reason were invisible anywhere.
    Confirms every population's own verdict now survives onto the
    ConsolidatedComparator, not just the representative one."""

    def test_every_populations_verdict_survives_consolidation(self):
        from jca_phase1.schema import ScopeAdjudication
        rec = _record()
        in_verdict = ScopeAdjudication(verdict=C.SCOPE_IN, reason="French evidence supports it")
        uncertain_verdict = ScopeAdjudication(
            verdict=C.SCOPE_UNCERTAIN,
            reason="framed as prior/ineligible therapy, not a comparator")
        consolidated = consolidation.build_comparator(
            "k", Comparator(as_stated="Dabrafenib + Trametinib"), [rec], in_verdict,
            C.EU_27_MEMBER_STATES, None, population_ids=["licensed", "intended_to_treat"],
            adjudications_by_population={"licensed": in_verdict,
                                         "intended_to_treat": uncertain_verdict})
        self.assertEqual(consolidated.scope_adjudication.verdict, C.SCOPE_IN,
                         "the single representative verdict is unchanged")
        self.assertEqual(set(consolidated.adjudications_by_population.keys()),
                         {"licensed", "intended_to_treat"})
        self.assertEqual(
            consolidated.adjudications_by_population["intended_to_treat"].verdict,
            C.SCOPE_UNCERTAIN,
            "the discarded uncertain verdict for the other population must survive, "
            "not just the winning one")

    def test_defaults_to_empty_when_not_provided(self):
        from jca_phase1.schema import ScopeAdjudication
        rec = _record()
        consolidated = consolidation.build_comparator(
            "k", Comparator(as_stated="Topotecan"), [rec], ScopeAdjudication(),
            C.EU_27_MEMBER_STATES, None)
        self.assertEqual(consolidated.adjudications_by_population, {})


class TestComparatorAliasesMerged(unittest.TestCase):
    """apply_identities() unifies as_stated for grouping and preserves each
    record's original wording in raw_as_stated (see TestPreclusterCandidates).
    build_comparator() must surface those preserved originals as
    aliases_merged, mirroring ConsolidatedOutcome.aliases_merged -- otherwise
    a reviewer has no visible trace in Excel of which raw wordings (language/
    brand-name/citation-format variants) were folded into one comparator."""

    def test_distinct_raw_wordings_are_listed_excluding_the_final_display_name(self):
        from jca_phase1.schema import ScopeAdjudication
        records = [
            _record(finding_id="cmp-0",
                   comparator=Comparator(as_stated="Topotecan", raw_as_stated="Topotecan",
                                         inn="topotecan")),
            _record(finding_id="cmp-1",
                   comparator=Comparator(as_stated="Topotecan",
                                         raw_as_stated="monoterapia con topotecan",
                                         inn="topotecan")),
            _record(finding_id="cmp-2",
                   comparator=Comparator(as_stated="Topotecan", raw_as_stated="Topotecan (1997)",
                                         inn="topotecan")),
        ]
        consolidated = consolidation.build_comparator(
            "k", Comparator(as_stated="Topotecan", inn="topotecan"), records,
            ScopeAdjudication(), C.EU_27_MEMBER_STATES, None)
        self.assertEqual(set(consolidated.aliases_merged),
                         {"monoterapia con topotecan", "Topotecan (1997)"},
                         "the final display name itself must not appear in aliases_merged")

    def test_no_variation_leaves_aliases_merged_empty(self):
        from jca_phase1.schema import ScopeAdjudication
        rec = _record(comparator=Comparator(as_stated="Topotecan", raw_as_stated="Topotecan"))
        consolidated = consolidation.build_comparator(
            "k", Comparator(as_stated="Topotecan"), [rec], ScopeAdjudication(),
            C.EU_27_MEMBER_STATES, None)
        self.assertEqual(consolidated.aliases_merged, [])


# ===========================================================================
# A15 — the outcome catalog
# ===========================================================================

class TestTokenCapCentralization(unittest.TestCase):
    """Four prompts hit their max_tokens cap 100% of the time across three
    real production runs (a06, a12, a14, a16.indication_synthesis) -- the
    caps were inline literals scattered across agent files, which is exactly
    what let this drift unnoticed for three runs. Centralized onto C.LLM and
    raised; this test pins both facts: the caps exist with adequate headroom,
    and each call site actually references them rather than a literal."""

    def test_new_caps_exist_with_real_headroom_over_the_old_ones(self):
        self.assertGreaterEqual(C.LLM.query_vocabulary_max_tokens, 3000)
        self.assertGreaterEqual(C.LLM.scope_adjudication_max_tokens, 4000)
        self.assertGreaterEqual(C.LLM.outcome_harmonization_max_tokens, 8000)
        self.assertGreaterEqual(C.LLM.indication_synthesis_max_tokens, 1500)

    def test_extraction_cap_raised_with_matching_read_timeout_headroom(self):
        """The extraction cap was raised for headroom on one unusually dense
        chunk (chunking already bounds the common case) -- but BedrockLLM
        calls invoke_model synchronously, so raising max_tokens alone (no
        matching read_timeout increase) would just trade a fast truncation
        failure for a slow read-timeout-then-retry failure. Pin both moved
        together, not just the token cap."""
        import inspect
        import re as re_module
        from jca_phase1.providers import llm as llm_module
        self.assertGreaterEqual(C.LLM.extraction_max_tokens, 20000)
        client_src = inspect.getsource(llm_module.BedrockLLM._lazy_client)
        match = re_module.search(r"read_timeout=(\d+)", client_src)
        self.assertIsNotNone(match, "read_timeout must still be an explicit, inspectable value")
        self.assertGreaterEqual(int(match.group(1)), 450)

    def test_call_sites_reference_the_config_not_a_literal(self):
        import inspect
        from jca_phase1.agents import a06_a08_retrieval, a09_a13_validation, a14_a18_consolidation
        sources = {
            "a06.query_vocabulary": inspect.getsource(a06_a08_retrieval.build_vocabulary),
            "a12.scope_adjudication": inspect.getsource(a09_a13_validation.adjudicate_scope),
        }
        for prompt_id, src in sources.items():
            self.assertIn("C.LLM.", src, f"{prompt_id} still hardcodes max_tokens")
        consolidation_src = inspect.getsource(a14_a18_consolidation)
        self.assertIn("C.LLM.outcome_harmonization_max_tokens", consolidation_src)
        self.assertIn("C.LLM.indication_synthesis_max_tokens", consolidation_src)


class TestGenericAndForwardLookingMentionGuidance(unittest.TestCase):
    """Run 4 post-fix validation, defect 2d: a future confirmatory-trial
    commitment ("must submit results from an ongoing study comparing X with
    chemotherapy") and generic background scene-setting ("limited options,
    including surgery and chemotherapy") were both extracted as role:
    active_comparator and then adjudicated in_scope -- neither is a real,
    source-recommended comparator. Pins that the prompt guidance addressing
    this exists at the bumped versions."""

    def test_extraction_prompt_addresses_forward_looking_and_generic_mentions(self):
        text, version = prompt_registry.get("a08.extraction")
        self.assertEqual(version, "v5")
        self.assertIn("FORWARD-LOOKING TRIAL COMMITMENTS", text)
        self.assertIn("GENERIC BACKGROUND SCENE-SETTING", text)

    def test_scope_adjudication_prompt_downweights_unclear_role_evidence(self):
        text, version = prompt_registry.get("a12.scope_adjudication")
        self.assertEqual(version, "v4")
        self.assertIn("role: unclear", text)
        self.assertIn("lean uncertain", text)


class TestSMEv8Reconciliation(unittest.TestCase):
    """SME base-prompt doc v8 reconciliation. Pins that every SME paragraph
    we were missing (some going back to v7, never previously applied) is now
    present, AND that our own engineering-side additions (Fix 4's a11
    per-component schema, the Run 4 a08/a12 fixes) survived the merge
    untouched rather than being silently overwritten."""

    def test_a08_has_all_three_sme_paragraphs_and_our_own_addition(self):
        text, version = prompt_registry.get("a08.extraction")
        self.assertEqual(version, "v5")
        self.assertIn("MULTIPLE COMPARATORS NAMED TOGETHER", text)
        self.assertIn("MULTIPLE OUTCOMES NAMED TOGETHER", text)
        self.assertIn("COMPARATOR AND OUTCOME INFORMATION IN STRUCTURED FORMAT", text)
        # Our own Run 4 addition must survive the merge untouched.
        self.assertIn("FORWARD-LOOKING TRIAL COMMITMENTS", text)
        self.assertIn("GENERIC BACKGROUND SCENE-SETTING", text)

    def test_a10_has_both_missing_sme_paragraphs(self):
        text, version = prompt_registry.get("a10.claim_validation")
        self.assertEqual(version, "v3")
        self.assertIn("This same distinction applies to OUTCOME claims", text)
        self.assertIn("SCOPE_ONLY_NO_DATA rather than WRONG_SUBJECT_DRUG", text)
        self.assertIn("ABSOLUTE RULE FOR COMPARATOR CLAIMS", text)

    def test_a11_has_self_reference_step1_and_third_example_plus_our_schema(self):
        text, version = prompt_registry.get("a11.comparator_identity")
        self.assertEqual(version, "v4")
        self.assertIn("Does this candidate resolve to the SAME substance as the "
                     "requested intervention itself?", text)
        self.assertIn("STEP 1 - NORMALISE BEFORE RESOLVING", text)
        self.assertIn("bracketed abbreviation", text)
        self.assertIn("atezolizumab", text)
        # Our own Fix 4 per-component schema must survive the merge untouched.
        self.assertIn('"components": [{"name": "", "inn": "", "atc_code": "", '
                     '"class_mechanism": ""}]', text)

    def test_a12_has_self_reference_absolute_rule_plus_our_addition(self):
        text, version = prompt_registry.get("a12.scope_adjudication")
        self.assertEqual(version, "v4")
        self.assertIn("ABSOLUTE RULE: the requested intervention itself is NEVER "
                     "a valid comparator", text)
        # Our own Run 4 addition must survive the merge untouched.
        self.assertIn("lean uncertain", text)

    def test_a14_has_both_missing_sme_paragraphs(self):
        text, version = prompt_registry.get("a14.outcome_harmonization")
        self.assertEqual(version, "v3")
        self.assertIn("A time-point qualifier attached to a survival/response "
                     "concept", text)
        self.assertIn("Differences in capitalization or whitespace alone never "
                     "indicate a different concept", text)

    def test_identical_prompts_are_untouched(self):
        """Prompts confirmed identical between our registry and SME's v7/v8
        doc must show no version churn from this reconciliation."""
        for prompt_id, expected_version in [
            ("a01.pi_validation", "v2"), ("a02.input_structuring", "v2"),
            ("a03.scope_facet_normalise", "v2"), ("a04.indication_lock", "v2"),
            ("a05.area_adjudication", "v2"), ("a06.query_vocabulary", "v2"),
            ("a16.comparator_rationale", "v2"), ("a16.outcome_rationale", "v2"),
            ("a16.indication_synthesis", "v2"),
        ]:
            _, version = prompt_registry.get(prompt_id)
            self.assertEqual(version, expected_version, prompt_id)


class TestOutcomeCatalog(unittest.TestCase):

    def test_catalog_has_eleven_items(self):
        self.assertEqual(len(consolidation.CATALOG_ITEMS), 11)

    def test_catalog_provenance_is_recorded_as_not_sme(self):
        """The catalog is a MadeAI platform artifact, not an SME requirement.
        The data file must keep saying so."""
        prov = consolidation.CATALOG["provenance"]
        self.assertIn("NOT defined in the SME base prompt", prov["IMPORTANT"])

    def test_synonyms_match_deterministically(self):
        self.assertEqual(consolidation.match_catalog("OS")["catalog_id"], "OS")
        self.assertEqual(consolidation.match_catalog("overall survival")["catalog_id"], "OS")
        self.assertEqual(
            consolidation.match_catalog("serious adverse event")["catalog_id"], "AE_SERIOUS")

    def test_duration_of_response_is_not_a_catalog_item(self):
        """The legacy prompt named Duration of response and Complete response
        rate as catalog terms; the Ground Truth marks both UNLISTED."""
        self.assertIsNone(consolidation.match_catalog("Duration of response"))
        self.assertIsNone(consolidation.match_catalog("Complete response rate"))

    def test_catalog_block_lists_every_item_for_the_model(self):
        """A model cannot map to a catalog it has never seen."""
        block = consolidation.catalog_prompt_block()
        for item in consolidation.CATALOG_ITEMS:
            self.assertIn(item["catalog_id"], block)

    def test_coverage_is_reported_for_every_catalog_item(self):
        view = consolidation.build_outcomes([], None)
        self.assertEqual(len(view.catalog_coverage), 11)
        self.assertTrue(all(c.status == C.EV_NONE for c in view.catalog_coverage))

    def test_missing_outcome_is_a_reported_status_not_an_absence(self):
        view = consolidation.build_outcomes([("Overall survival", "OS", [
            _record(finding_type="outcome", comparator=None,
                    outcome=__import__("jca_phase1.schema", fromlist=["x"]).OutcomeMention(
                        measure="Overall survival", unit="months"))])], None)
        qol = [c for c in view.catalog_coverage if c.category == C.CAT_QOL]
        self.assertTrue(qol)
        self.assertTrue(all(c.status == C.EV_NONE for c in qol))
        self.assertTrue(all(c.detail for c in qol), "a gap must carry an explanation")


class TestArmPrefixStripping(unittest.TestCase):
    """ClinicalTrials.gov's own API puts a literal 'Arm N:' prefix in
    outcomesModule measure text for multi-arm trials. Deterministic, not
    prompt-instruction-dependent, per SME request."""

    def test_strips_every_observed_variant(self):
        cases = [
            ("Arm 1: Overall response rate", "Overall response rate"),
            ("Arm 1 and 3: Number of participants reporting adverse events",
             "Number of participants reporting adverse events"),
            ("Arm 1, Arm 2 and Arm 3: Duration of response (DOR)",
             "Duration of response (DOR)"),
            ("Arm 1 and Arm 2: Duration of overall survival",
             "Duration of overall survival"),
            ("Overall survival", "Overall survival"),  # no prefix -- unchanged
        ]
        for raw, expected in cases:
            self.assertEqual(retrieval._strip_arm_prefix(raw), expected, raw)

    def test_stripped_measure_now_matches_the_catalog(self):
        """This is the actual defect: before the fix, the prefix alone kept
        this from matching ORR via the deliberately-exact _catalog_key()."""
        stripped = retrieval._strip_arm_prefix("Arm 1: Overall response rate")
        self.assertEqual(consolidation.match_catalog(stripped)["catalog_id"], "ORR")

    def test_records_from_trials_applies_the_strip_but_keeps_the_raw_quote(self):
        trial = TrialRecord(identifier="NCT999",
                            primary_outcomes=["Arm 1: Overall response rate"])
        recs = retrieval.records_from_trials(
            [trial], Intervention(fields={"product_name_inn":
                                          Field(value="Examplimab", provenance="confirmed")}))
        self.assertEqual(recs[0].outcome.measure, "Overall response rate")
        self.assertIn("Arm 1: Overall response rate", recs[0].evidence_quote,
                      "the raw wording must survive in the audit trail")

    def test_rano_and_rapno_lgg_are_not_collapsed_by_the_fix(self):
        """A genuine methodological distinction (RANO vs. RAPNO-LGG, a
        purpose-built instrument for paediatric low-grade glioma) must not
        be erased by a fix aimed at a purely mechanical registry artifact."""
        self.assertIsNone(consolidation.match_catalog("ORR by RANO criteria"))
        self.assertIsNone(consolidation.match_catalog("ORR by RAPNO-LGG criteria"))


class TestExtractionChunking(unittest.TestCase):
    """Run 4 post-fix validation: a real 76KB clinical-guideline PDF (GT's own
    cited source for TPCV, Carboplatin+Vincristine, and Vinblastine) hit
    a08.extraction's 16000-output-token cap and returned nothing usable --
    its real content needed more JSON to describe than one response could
    hold, even though the input easily fit the context window. Confirms the
    document is now split into extraction-sized chunks before the LLM call,
    each chunk's results merged, rather than one call reading the whole
    document and risking a truncated, unusable response."""

    def test_short_document_is_a_single_chunk(self):
        text = "short document text"
        self.assertEqual(retrieval._chunk_document_text(text), [text])

    def test_long_document_is_split_into_multiple_overlapping_chunks(self):
        """Use a position-marked string (not repeated filler) so each
        chunk's exact start/end offset in the original text can be verified
        directly, proving there's no gap between consecutive chunks."""
        text = "".join(str(i % 10) for i in range(100000))
        chunks = retrieval._chunk_document_text(text)
        self.assertGreater(len(chunks), 1)
        pos = 0
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk, text[pos:pos + len(chunk)])
            if i > 0:
                self.assertLessEqual(len(chunk), retrieval._EXTRACTION_CHUNK_CHARS)
            pos += len(chunk) - retrieval._EXTRACTION_CHUNK_OVERLAP_CHARS
        # The last chunk must reach the end of the text -- no trailing gap.
        last_start = sum(len(c) - retrieval._EXTRACTION_CHUNK_OVERLAP_CHARS
                         for c in chunks[:-1])
        self.assertEqual(last_start + len(chunks[-1]), len(text))

    def test_chunk_count_is_bounded(self):
        text = "x" * 10_000_000
        chunks = retrieval._chunk_document_text(text)
        self.assertLessEqual(len(chunks), retrieval._EXTRACTION_MAX_CHUNKS)

    def test_extraction_merges_records_across_chunks(self):
        """A comparator only extractable from the SECOND half of a long
        document must still surface -- this is the actual production defect:
        with no chunking, the model saw the whole document but the response
        describing everything it found was too large and got cut off before
        this comparator (whichever one the model happened to describe last)
        was ever written out."""
        long_text = ("Background filler about the disease and its epidemiology. " * 700
                    + "Vinblastine monotherapy is the recommended second-line comparator. "
                    + "More trailing background filler text repeated many times over. " * 700)
        self.assertGreater(len(long_text), retrieval._EXTRACTION_CHUNK_CHARS)

        def fake_response(user_prompt):
            if "Vinblastine" in user_prompt:
                return json.dumps([{
                    "finding_type": "comparator", "subject_drug": "Tovorafenib",
                    "comparator": {"as_stated": "Vinblastine", "role": "active_comparator"},
                    "population_context": {"disease": "LGG"},
                    "evidence_quote": "Vinblastine monotherapy is the recommended "
                                     "second-line comparator."}])
            return json.dumps([])

        from jca_phase1.schema import RetrievedDocument
        llm = ScriptedLLM({"a08.extraction": fake_response})
        doc = RetrievedDocument(url="https://x.org/a", resolved_url="https://x.org/a",
                                text=long_text, ok=True, source_class=C.SRC_CLINICAL_GUIDELINE)
        pop = Population(fields={"indication_disease": Field(value="LGG", provenance="confirmed")})
        inter = Intervention(fields={"product_name_inn": Field(value="Tovorafenib",
                                                               provenance="confirmed")})
        recs = retrieval._extract_one(doc, pop, inter, llm)
        self.assertTrue(any(r.comparator and r.comparator.as_stated == "Vinblastine"
                            for r in recs),
                        "a comparator only present in a later chunk must still be extracted")

    def test_chunk_boundary_prefers_a_paragraph_break_over_a_hard_cutoff(self):
        """A hard character cutoff can split a recommendation-table row or a
        sentence exactly in half; breaking at a nearby blank line instead
        reduces (the overlap is what covers the rest) that risk."""
        # A paragraph break sits just after the 80%-of-chunk-size search
        # start, well before the hard cutoff -- the break must land there,
        # not at the hard _EXTRACTION_CHUNK_CHARS boundary.
        para_break_at = int(retrieval._EXTRACTION_CHUNK_CHARS * 0.85)
        text = ("a" * para_break_at) + "\n\n" + ("b" * (retrieval._EXTRACTION_CHUNK_CHARS * 2))
        chunks = retrieval._chunk_document_text(text)
        self.assertEqual(len(chunks[0]), para_break_at,
                         "first chunk must end exactly at the paragraph break")

    def test_chunk_boundary_falls_back_to_hard_cutoff_with_no_paragraph_break(self):
        text = "a" * (retrieval._EXTRACTION_CHUNK_CHARS * 2)
        chunks = retrieval._chunk_document_text(text)
        self.assertEqual(len(chunks[0]), retrieval._EXTRACTION_CHUNK_CHARS)

    def test_extraction_deduplicates_a_claim_repeated_in_the_overlap_window(self):
        """The same sentence can legitimately appear in two consecutive
        chunks' overlap region and get extracted twice -- once per chunk. An
        exact-verbatim evidence_quote repeat is the same claim, not two."""
        long_text = "Padding text repeated many times over to force chunking. " * 1500
        llm = ScriptedLLM({"a08.extraction": lambda _: json.dumps([{
            "finding_type": "comparator", "subject_drug": "Tovorafenib",
            "comparator": {"as_stated": "Everolimus", "role": "active_comparator"},
            "population_context": {"disease": "LGG"},
            "evidence_quote": "Everolimus is an established alternative."}])})
        from jca_phase1.schema import RetrievedDocument
        doc = RetrievedDocument(url="https://x.org/b", resolved_url="https://x.org/b",
                                text=long_text, ok=True, source_class=C.SRC_CLINICAL_GUIDELINE)
        pop = Population(fields={"indication_disease": Field(value="LGG", provenance="confirmed")})
        inter = Intervention(fields={"product_name_inn": Field(value="Tovorafenib",
                                                               provenance="confirmed")})
        recs = retrieval._extract_one(doc, pop, inter, llm)
        self.assertEqual(len(recs), 1,
                         "an identical evidence_quote from every chunk must collapse to one record")

    def test_unparseable_chunk_response_is_logged_not_silent(self):
        """This took manual cross-tabulation across hundreds of raw LLM
        calls to discover once -- must be visible without that."""
        long_text = "Padding text repeated many times over to force chunking. " * 1500
        # call_json's default for a08.extraction is [] (a list), so a genuine
        # parse failure already returns that default and is NOT what trips
        # this log line -- what trips it is the model returning VALID JSON
        # that isn't a list (e.g. a bare object), which the isinstance check
        # below the log line is guarding against.
        llm = ScriptedLLM({"a08.extraction": lambda _: "{}"})
        from jca_phase1.schema import RetrievedDocument
        doc = RetrievedDocument(url="https://x.org/c", resolved_url="https://x.org/c",
                                text=long_text, ok=True, source_class=C.SRC_CLINICAL_GUIDELINE)
        pop = Population(fields={"indication_disease": Field(value="LGG", provenance="confirmed")})
        inter = Intervention(fields={"product_name_inn": Field(value="Tovorafenib",
                                                               provenance="confirmed")})
        with self.assertLogs("jca_phase1.agents.a06_a08_retrieval", level="WARNING") as cm:
            recs = retrieval._extract_one(doc, pop, inter, llm)
        self.assertEqual(recs, [])
        self.assertTrue(any("chunk" in msg and "x.org/c" in msg for msg in cm.output))

    def test_document_level_extraction_crash_is_logged_not_silent(self):
        def raising_llm(*args, **kwargs):
            raise RuntimeError("simulated Bedrock failure")
        llm = unittest.mock.Mock()
        llm.call_json.side_effect = raising_llm
        from jca_phase1.schema import RetrievedDocument
        doc = RetrievedDocument(url="https://x.org/d", resolved_url="https://x.org/d",
                                text="short document text", ok=True,
                                source_class=C.SRC_CLINICAL_GUIDELINE)
        pop = Population(fields={"indication_disease": Field(value="LGG", provenance="confirmed")})
        inter = Intervention(fields={"product_name_inn": Field(value="Tovorafenib",
                                                               provenance="confirmed")})
        with self.assertLogs("jca_phase1.agents.a06_a08_retrieval", level="WARNING") as cm:
            recs = retrieval.extract_from_documents([doc], pop, inter, llm)
        self.assertEqual(recs, [])
        self.assertTrue(any("x.org/d" in msg and "RuntimeError" in msg for msg in cm.output))

    def test_chunks_from_multiple_documents_are_parallelized_without_cross_contamination(self):
        """A real run showed extraction dominating wall-clock time (59% of a
        44-minute run) because a document needing several chunks processed
        them SEQUENTIALLY inside its own worker thread -- one worker stuck on
        an 8-chunk PDF blocked that slot while the pool's other workers
        cycled through simple 1-chunk documents. extract_from_documents() now
        flattens every (document, chunk) pair across ALL documents into one
        pool, so a chunk-heavy document's calls interleave with everyone
        else's. This pins the CORRECTNESS side of that change: results must
        still land on the right document even when chunks from different
        documents are dispatched to the pool interleaved, not grouped."""
        from jca_phase1.schema import RetrievedDocument

        long_text = ("Padding text repeated many times over to force chunking. " * 1500)
        doc_a = RetrievedDocument(url="https://x.org/multi-chunk", resolved_url="https://x.org/multi-chunk",
                                  text=long_text, ok=True, source_class=C.SRC_CLINICAL_GUIDELINE)
        doc_b = RetrievedDocument(url="https://x.org/single-chunk", resolved_url="https://x.org/single-chunk",
                                  text="short document text", ok=True, source_class=C.SRC_CLINICAL_GUIDELINE)
        self.assertGreater(len(retrieval._chunk_document_text(long_text)), 1)

        def fake_response(user_prompt):
            if "multi-chunk" in user_prompt:
                return json.dumps([{
                    "finding_type": "comparator", "subject_drug": "Tovorafenib",
                    "comparator": {"as_stated": "Vinblastine", "role": "active_comparator"},
                    "population_context": {"disease": "LGG"},
                    "evidence_quote": "Padding text repeated many times over to force chunking."}])
            return json.dumps([{
                "finding_type": "comparator", "subject_drug": "Tovorafenib",
                "comparator": {"as_stated": "Everolimus", "role": "active_comparator"},
                "population_context": {"disease": "LGG"},
                "evidence_quote": "short document text"}])

        llm = ScriptedLLM({"a08.extraction": fake_response})
        pop = Population(fields={"indication_disease": Field(value="LGG", provenance="confirmed")})
        inter = Intervention(fields={"product_name_inn": Field(value="Tovorafenib",
                                                               provenance="confirmed")})
        recs = retrieval.extract_from_documents([doc_a, doc_b], pop, inter, llm, max_workers=4)

        a_names = {r.comparator.as_stated for r in recs if r.source_url == "https://x.org/multi-chunk"}
        b_names = {r.comparator.as_stated for r in recs if r.source_url == "https://x.org/single-chunk"}
        self.assertEqual(a_names, {"Vinblastine"})
        self.assertEqual(b_names, {"Everolimus"})
        self.assertTrue(all(r.source_id == (doc_a.source_id if r.source_url == doc_a.url
                                            else doc_b.source_id) for r in recs))


# ===========================================================================
# Schema invariants
# ===========================================================================

class TestInvariants(unittest.TestCase):

    def test_comparator_without_evidence_cannot_be_constructed(self):
        with self.assertRaises(ValueError):
            ConsolidatedComparator(generic_name="Topotecan", origin=ORIGIN_AGENT)

    def test_user_added_comparator_needs_no_evidence(self):
        c = ConsolidatedComparator(generic_name="Topotecan", origin="user_added")
        self.assertEqual(c.generic_name, "Topotecan")

    def test_comparator_class_from_an_illegal_source_is_rejected(self):
        with self.assertRaises(ValueError):
            ConsolidatedComparator(
                generic_name="Topotecan", origin="user_added",
                class_or_mechanism="bispecific T-cell engager",
                class_source="intervention")

    def test_member_state_summary_must_sum_to_27(self):
        with self.assertRaises(ValueError):
            MemberStateSummary(identified_count=10, not_identified_count=10)

    def test_phase1_output_has_no_pico_sets(self):
        from jca_phase1.schema import Phase1Output
        d = Phase1Output().to_dict()
        self.assertIsNone(d["pico_sets"])
        self.assertIn("Phase 2", d["pico_sets_note"])


# ===========================================================================
# Leakage guard
# ===========================================================================

class TestLeakageGuard(unittest.TestCase):

    def test_target_drug_jca_report_is_blocked(self):
        from jca_phase1.schema import LeakageGuard
        guard = LeakageGuard(drug_tokens=["tarlatamab"])
        self.assertTrue(guard.is_blocked(
            "https://health.ec.europa.eu/document/hta_jca_mp_202417_tarlatamab_report_en.pdf"))

    def test_generic_eu_guidance_on_the_same_domain_is_not_blocked(self):
        """Blocking the whole domain would also block legitimate EU HTA
        methodology guidance, which is not an answer key."""
        from jca_phase1.schema import LeakageGuard
        guard = LeakageGuard(drug_tokens=["tarlatamab"])
        self.assertFalse(guard.is_blocked(
            "https://health.ec.europa.eu/document/hta-methodological-guidance-outcomes_en.pdf"))

    def test_guard_can_be_disabled_for_evaluation_and_is_recorded(self):
        from jca_phase1.schema import LeakageGuard
        guard = LeakageGuard(drug_tokens=["tarlatamab"], allow_jca_reports=True)
        self.assertFalse(guard.is_blocked(
            "https://health.ec.europa.eu/document/hta_jca_mp_202417_tarlatamab_report_en.pdf"))
        self.assertTrue(guard.to_dict()["allow_jca_reports"])

    def test_record_deduplicates_the_same_url_logged_twice(self):
        """record() always appends (race-free under concurrent callers);
        to_dict() must still report each blocked URL only once."""
        from jca_phase1.schema import LeakageGuard
        guard = LeakageGuard()
        guard.record("https://example.org/a")
        guard.record("https://example.org/a")
        guard.record("https://example.org/b")
        self.assertEqual(guard.to_dict()["blocked_urls"],
                         sorted(["https://example.org/a", "https://example.org/b"]))

    def test_the_bare_class_still_blocks_by_default(self):
        """LeakageGuard's OWN dataclass default stays conservative (blocks)
        -- only the pipeline's entry points (RunOptions, lock_indication)
        changed what they pass in. This is what makes the change reversible
        without touching the guard's own definition."""
        from jca_phase1.schema import LeakageGuard
        self.assertFalse(LeakageGuard().allow_jca_reports)

    def test_lock_indication_now_defaults_to_allowing_jca_reports(self):
        """Per explicit SME decision (2026-09-23): the drug's own JCA report
        is now a deliberate Tier 1 source by default, not evaluation-only."""
        _, guard = inputs.lock_indication(
            Intervention(fields={"product_name_inn":
                                 Field(value="Tarlatamab", provenance="confirmed")}))
        self.assertTrue(guard.allow_jca_reports)
        self.assertFalse(guard.is_blocked(
            "https://health.ec.europa.eu/document/hta_jca_mp_202417_tarlatamab_report_en.pdf"))

    def test_run_options_now_defaults_to_allowing_jca_reports(self):
        self.assertTrue(orch.RunOptions().allow_jca_reports)


class TestStreamlitBenchmarkingModeCheckbox(unittest.TestCase):
    """Fix 8: 'remember to edit config for GT-benchmarking runs' is not a
    real guarantee -- the Streamlit UI now has a checkbox whose state must
    actually reach RunOptions.allow_jca_reports, not just exist cosmetically."""

    @classmethod
    def setUpClass(cls):
        cls._streamlit_app_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "streamlit_app")
        sys.path.insert(0, cls._streamlit_app_dir)
        import workflow_runner
        cls.workflow_runner = workflow_runner

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(cls._streamlit_app_dir)

    def _run(self, allow_jca_reports):
        captured = {}

        def fake_run_phase1(population_input, intervention_input, providers, options=None,
                            debug_capture=None):
            captured["allow_jca_reports"] = options.allow_jca_reports
            out = unittest.mock.Mock()
            out.comparators = []
            out.to_dict.return_value = {}
            return out

        with unittest.mock.patch.object(self.workflow_runner, "run_phase1", fake_run_phase1), \
             unittest.mock.patch.object(self.workflow_runner, "_demo_providers",
                                        lambda: unittest.mock.Mock(llm=None, search=None)), \
             unittest.mock.patch.object(self.workflow_runner.telemetry, "summarize_bedrock",
                                        lambda llm: ([], {"total_cost_usd": 0.0})), \
             unittest.mock.patch.object(self.workflow_runner.telemetry, "build_stage_timeline",
                                        lambda *a, **kw: {}):
            self.workflow_runner.run_full_workflow(
                "run-1", {}, {}, "demo", {}, {}, [], "2026-01-01T00:00:00Z",
                allow_jca_reports=allow_jca_reports)
        return captured["allow_jca_reports"]

    def test_unchecked_benchmarking_mode_allows_jca_reports(self):
        self.assertTrue(self._run(allow_jca_reports=True))

    def test_checked_benchmarking_mode_blocks_jca_reports(self):
        self.assertFalse(self._run(allow_jca_reports=False))


# ===========================================================================
# Balanced selection
# ===========================================================================

class TestPromptCaching(unittest.TestCase):
    """BedrockLLM._invoke uses invoke_model (raw Anthropic Messages body), so
    caching means a cache_control content block on `system`, not Bedrock's
    Converse-only cachePoint. Tested by faking the boto3 client -- no real
    AWS call."""

    class _FakeBody:
        def __init__(self, payload):
            self._data = json.dumps(payload).encode()

        def read(self):
            return self._data

    class _FakeBedrockClient:
        def __init__(self, usage=None):
            self.last_body = None
            self.usage = usage or {"input_tokens": 1, "output_tokens": 1}

        def invoke_model(self, modelId, body):
            self.last_body = json.loads(body)
            return {"body": TestPromptCaching._FakeBody(
                {"content": [{"type": "text", "text": "ok"}], "usage": self.usage})}

    def test_cacheable_prompt_wraps_system_in_a_cache_control_block(self):
        llm = BedrockLLM(model_id="test-model")
        fake = self._FakeBedrockClient()
        llm._client = fake
        llm._invoke("SYSTEM TEXT", "user text", 100, "a08.extraction", cacheable=True)
        self.assertEqual(fake.last_body["system"],
                         [{"type": "text", "text": "SYSTEM TEXT",
                           "cache_control": {"type": "ephemeral"}}])

    def test_non_cacheable_prompt_sends_a_plain_system_string(self):
        llm = BedrockLLM(model_id="test-model")
        fake = self._FakeBedrockClient()
        llm._client = fake
        llm._invoke("SYSTEM TEXT", "user text", 100, "a01.pi_validation", cacheable=False)
        self.assertEqual(fake.last_body["system"], "SYSTEM TEXT")

    def test_cache_usage_is_parsed_from_the_response(self):
        llm = BedrockLLM(model_id="test-model")
        fake = self._FakeBedrockClient(usage={
            "input_tokens": 5, "output_tokens": 2,
            "cache_creation_input_tokens": 900, "cache_read_input_tokens": 0})
        llm._client = fake
        _, usage = llm._invoke("SYSTEM TEXT", "user text", 100, "a08.extraction",
                               cacheable=True)
        self.assertEqual(usage["cache_creation_input_tokens"], 900)
        self.assertEqual(usage["cache_read_input_tokens"], 0)

    def test_registry_flags_match_the_documented_call_frequency_rule(self):
        # Called many times per run against the same static system text.
        for pid in ("a08.extraction", "a10.claim_validation", "a12.scope_adjudication",
                   "a16.comparator_rationale", "a16.outcome_rationale",
                   "a16.indication_synthesis"):
            self.assertTrue(prompt_registry.is_cacheable(pid), pid)
        # Called once (or 0-2 times) per run -- nothing to amortize a cache
        # write against.
        for pid in ("a01.pi_validation", "a02.input_structuring",
                   "a03.scope_facet_normalise", "a04.indication_lock",
                   "a05.area_adjudication", "a06.query_vocabulary",
                   "a11.comparator_identity", "a14.outcome_harmonization"):
            self.assertFalse(prompt_registry.is_cacheable(pid), pid)

    def test_unknown_prompt_id_is_not_cacheable(self):
        self.assertFalse(prompt_registry.is_cacheable("not.a.real.prompt"))


class TestTavilyExtractDepth(unittest.TestCase):
    """extract_depth was hardcoded to "basic" for every fetch, regardless of
    full_document -- confirmed in production: a real guideline PDF fetched
    with full_document=True came back as ~3KB of template boilerplate,
    because "basic" depth handles PDFs/tables poorly. full_document=True is
    exactly the signal FULL_DOCUMENT_SOURCE_CLASSES already uses for "this
    source's real content is a table, read it whole" -- escalating to
    "advanced" for that case is a direct application of a distinction the
    code already draws, not a rule about any specific document."""

    class _FakeTavilyClient:
        def __init__(self):
            self.last_params = None

        def extract(self, **params):
            self.last_params = params
            return {"results": [{"raw_content": "x" * 600, "url": params["urls"][0],
                                 "title": "t"}], "usage": {"credits": 1}}

    def test_full_document_uses_advanced_extract_depth(self):
        from jca_phase1.providers.search import TavilySearchProvider
        provider = TavilySearchProvider(api_key="test")
        fake = self._FakeTavilyClient()
        provider._client = fake
        provider._fetch("https://siope.eu/media/documents/escp-low-grade-gliomas-lgg.pdf",
                        "", full_document=True)
        self.assertEqual(fake.last_params["extract_depth"], "advanced")

    def test_scoped_fetch_still_uses_basic_extract_depth(self):
        from jca_phase1.providers.search import TavilySearchProvider
        provider = TavilySearchProvider(api_key="test")
        fake = self._FakeTavilyClient()
        provider._client = fake
        provider._fetch("https://example.org/a", "some query", full_document=False)
        self.assertEqual(fake.last_params["extract_depth"], "basic")


class TestShortContentThreshold(unittest.TestCase):
    """Retrieval deep dive, issue 4: a hard 500-char floor marked any short
    page as source_inaccessible, identical to a genuinely broken fetch --
    including a legitimately short, single-fact page (a brief national HTA
    decision notice, a "no assessment on file" confirmation that is itself
    informative). Lowered to still catch truly empty/error responses
    without discarding real short content."""

    class _FakeTavilyClient:
        def __init__(self, content: str):
            self.content = content

        def extract(self, **params):
            return {"results": [{"raw_content": self.content, "url": params["urls"][0],
                                 "title": "t"}], "usage": {"credits": 1}}

    def test_a_short_but_real_page_is_no_longer_marked_inaccessible(self):
        """Between the old (500) and new (100) threshold -- previously
        wrongly marked inaccessible, now correctly treated as real content."""
        from jca_phase1.providers.search import TavilySearchProvider
        content = ("No health technology assessment has been filed for this "
                  "product in this Member State as of the most recent review cycle.")
        self.assertGreater(len(content), 100)
        self.assertLess(len(content), 500)
        provider = TavilySearchProvider(api_key="test")
        provider._client = self._FakeTavilyClient(content)
        doc = provider._fetch("https://example.org/short", "", full_document=False)
        self.assertTrue(doc.ok, "a real, if short, page must not be treated "
                               "identically to a broken fetch")

    def test_a_genuinely_empty_response_is_still_marked_inaccessible(self):
        from jca_phase1.providers.search import TavilySearchProvider
        provider = TavilySearchProvider(api_key="test")
        provider._client = self._FakeTavilyClient("")
        doc = provider._fetch("https://example.org/empty", "", full_document=False)
        self.assertFalse(doc.ok)
        self.assertEqual(doc.status, C.EV_SOURCE_INACCESSIBLE)


class TestBalancedSelection(unittest.TestCase):

    def test_no_category_is_starved(self):
        hits = [SearchHit(url=f"https://a.org/{i}") for i in range(5)]
        hits += [SearchHit(url="https://b.org/1")]
        selected = select_balanced(hits, 4, {"a": ["a.org"], "b": ["b.org"]})
        self.assertIn("https://b.org/1", selected)

    def test_execute_item_call_site_now_balances_across_curated_domains(self):
        """Retrieval deep dive, issue 2: select_balanced()'s own
        anti-starvation guarantee only activates with more than one domain
        group, but the call site collapsed a query's whole curated domain
        list into a single "primary" group -- so it never once activated in
        production, despite existing specifically for this. A well-ranked
        domain with many hits must not crowd out a curated domain with few."""
        from jca_phase1.schema import LeakageGuard, QueryPlanItem
        hits_a = [SearchHit(url=f"https://esmo.org/page{i}", score=0.9) for i in range(5)]
        hits_b = [SearchHit(url="https://siope.eu/media/documents/escp-low-grade-gliomas-lgg.pdf",
                            score=0.5)]
        search = FixtureSearchProvider(
            index={"esmo.org": hits_a, "siope.eu": hits_b},
            documents={h.url: "some fetched text" for h in hits_a + hits_b})
        item = QueryPlanItem(query="q", source_class=C.SRC_CLINICAL_GUIDELINE,
                             member_state="Austria", pass_type=retrieval.PASS_DRUG_ANCHORED,
                             domains=["esmo.org", "siope.eu"], max_urls=3)
        docs, _attempt, _blocked = retrieval._execute_item(item, search, LeakageGuard())
        urls = [d.url for d in docs]
        self.assertTrue(any("siope.eu" in u for u in urls),
                        f"siope.eu was crowded out entirely despite being a curated "
                        f"domain in this query: {urls}")


# ===========================================================================
# ITT: a second population must actually be retrieved for and adjudicated,
# not just structured and then ignored
# ===========================================================================

class TestITTMultiPopulation(unittest.TestCase):

    def test_retrieval_pass_runs_once_per_population(self):
        """Before the fix, every retrieval call used populations[0] only,
        however many populations existed. With an ITT population declared,
        _run_retrieval_pass must be invoked once per population. The two
        populations' passes now run CONCURRENTLY (runtime optimization: no
        correctness reason two independent passes needed to be sequential),
        so completion order is no longer guaranteed -- assert the SET of
        populations invoked, not a strict order."""
        import threading
        calls = []
        calls_lock = threading.Lock()
        original = orch._run_retrieval_pass

        def spy(pop, *args, **kwargs):
            with calls_lock:
                calls.append(pop.population_id)
            return original(pop, *args, **kwargs)

        with unittest.mock.patch.object(orch, "_run_retrieval_pass", side_effect=spy):
            orch.run_phase1(
                {"indication_disease": "SCLC",
                 "itt_differs": "No, it differs — a broader relapsed population"},
                {"product_name_inn": "Examplimab"},
                orch.Providers(),  # every provider None -- exercises the degraded
                                   # path safely, no network/LLM needed for this check
                options=orch.RunOptions(strict_sources=False,
                                        source_workbook=FIXTURE_WORKBOOK))
        self.assertEqual(set(calls), {POP_LICENSED, POP_ITT})
        self.assertEqual(len(calls), 2, "each population must run exactly once, not more")

    def test_populations_genuinely_overlap_in_wall_clock_time(self):
        """Runtime optimization: the two populations' A6-A10 passes used to
        run fully sequentially, paying the full single-population wall-clock
        time twice back to back -- confirmed the single largest contributor
        to a real 40-minute run. Proves actual concurrency (not just that
        both eventually run): two slow passes finish in close to ONE pass's
        duration, not the sum of both."""
        import threading
        import time
        original = orch._run_retrieval_pass
        barrier = threading.Barrier(2, timeout=5)

        def slow_spy(pop, *args, **kwargs):
            # Every population's pass must reach this point at roughly the
            # same time -- if they ran sequentially, the SECOND call
            # wouldn't start until the first finished sleeping, and this
            # barrier would time out.
            barrier.wait()
            return original(pop, *args, **kwargs)

        with unittest.mock.patch.object(orch, "_run_retrieval_pass", side_effect=slow_spy):
            started = time.time()
            orch.run_phase1(
                {"indication_disease": "SCLC",
                 "itt_differs": "No, it differs — a broader relapsed population"},
                {"product_name_inn": "Examplimab"},
                orch.Providers(),
                options=orch.RunOptions(strict_sources=False,
                                        source_workbook=FIXTURE_WORKBOOK))
        self.assertLess(time.time() - started, 4,
                        "both populations must reach the barrier concurrently, "
                        "not one after the other")

    def test_single_population_run_calls_the_pass_exactly_once(self):
        """Regression guard: the common, single-population case must not
        start calling the pass twice just because the loop now exists."""
        original = orch._run_retrieval_pass
        calls = []

        def spy(pop, *args, **kwargs):
            calls.append(pop.population_id)
            return original(pop, *args, **kwargs)

        with unittest.mock.patch.object(orch, "_run_retrieval_pass", side_effect=spy):
            orch.run_phase1(
                {"indication_disease": "SCLC"},
                {"product_name_inn": "Examplimab"},
                orch.Providers(),
                options=orch.RunOptions(strict_sources=False,
                                        source_workbook=FIXTURE_WORKBOOK))
        self.assertEqual(calls, [POP_LICENSED])

    def test_comparator_tagged_with_only_the_population_that_confirmed_it(self):
        """A candidate in scope for one population's boundary but not the
        other's must be tagged with only the population(s) that actually
        confirmed it -- exercising the same adjudicate_scope-per-boundary +
        build_comparator(population_ids=...) sequence run_phase1's A12/A16
        now use."""
        rec = _record()
        licensed = ScopeBoundary(population_id=POP_LICENSED,
                                 licensed_indication_wording="licensed population wording")
        itt = ScopeBoundary(population_id=POP_ITT,
                            licensed_indication_wording="broader itt population wording")

        def scope_llm(prompt: str) -> str:
            if "broader itt population wording" in prompt:
                return json.dumps({"verdict": "out_of_scope", "decisive_facet": "x",
                                   "reason": "not applicable to the itt population",
                                   "evidence_for": [], "evidence_against": []})
            return json.dumps({"verdict": "in_scope", "decisive_facet": "x",
                               "reason": "applies to the licensed population",
                               "evidence_for": [], "evidence_against": []})

        llm = ScriptedLLM({"a12.scope_adjudication": scope_llm})
        adjs = {b.population_id: validation.adjudicate_scope(
                    "k", rec.comparator, [rec], b, Intervention(), llm)
                for b in (licensed, itt)}
        self.assertEqual(adjs[POP_LICENSED].verdict, C.SCOPE_IN)
        self.assertEqual(adjs[POP_ITT].verdict, C.SCOPE_OUT)

        pop_ids_in = [pid for pid, a in adjs.items()
                     if a.verdict in (C.SCOPE_IN, C.SCOPE_UNCERTAIN)]
        self.assertEqual(pop_ids_in, [POP_LICENSED])

        consolidated = consolidation.build_comparator(
            "k", rec.comparator, [rec], adjs[POP_LICENSED], C.EU_27_MEMBER_STATES,
            None, population_ids=pop_ids_in)
        self.assertEqual(consolidated.population_ids, [POP_LICENSED])

    def test_single_population_comparator_defaults_to_licensed(self):
        rec = _record()
        consolidated = consolidation.build_comparator(
            "k", rec.comparator, [rec], ScopeAdjudication(), C.EU_27_MEMBER_STATES, None)
        self.assertEqual(consolidated.population_ids, [POP_LICENSED])


class TestCrossPopulationExtractionCache(unittest.TestCase):
    """Retrieval deep dive, issue 1: a08.extraction is population-agnostic by
    design (extracts every claim a document states; population RELEVANCE is
    decided later at A12) -- so re-fetching the SAME document per population
    and re-extracting it independently pays for, and independently re-risks
    truncation on, a call whose answer cannot legitimately differ. Confirmed
    real waste in a production run: the richest guideline PDF was
    independently extracted (and independently truncated) once per
    population, and its comparator's resulting split verdicts were mistaken
    for an A11-grouping artifact before being traced to this root cause."""

    def _fixture_pop_intervention(self):
        pop1 = Population(population_id=POP_LICENSED,
                          fields={"indication_disease": Field(value="LGG", provenance="confirmed")})
        pop2 = Population(population_id=POP_ITT,
                          fields={"indication_disease": Field(value="Broader LGG",
                                                              provenance="confirmed")})
        for p in (pop1, pop2):
            for k in inputs.POPULATION_FIELDS:
                p.fields.setdefault(k, Field())
        inter = Intervention(fields={"product_name_inn": Field(value="Tovorafenib",
                                                               provenance="confirmed")})
        for k in inputs.INTERVENTION_FIELDS:
            inter.fields.setdefault(k, Field())
        return pop1, pop2, inter

    def test_same_document_across_two_populations_extracts_only_once(self):
        from jca_phase1.schema import LeakageGuard, RetrievedDocument
        doc = RetrievedDocument(url="https://x.org/shared", resolved_url="https://x.org/shared",
                                text="Vinblastine is a recommended second-line option.",
                                ok=True, source_class=C.SRC_CLINICAL_GUIDELINE)

        class _FakeExecuteResult:
            def __init__(self, documents):
                self.documents = documents
                self.attempts = []
                self.blocked_by_guard = []

        class _FakeStructured:
            attempts: list = []
            trials: list = []
            publications: list = []

        class _FakeSearch:
            def fetch(self, *a, **kw):
                return doc

        llm = ScriptedLLM({"a08.extraction": json.dumps([{
            "finding_type": "comparator", "subject_drug": "Tovorafenib",
            "comparator": {"as_stated": "Vinblastine", "role": "active_comparator"},
            "population_context": {"disease": "LGG"},
            "evidence_quote": "Vinblastine is a recommended second-line option."}])})
        providers = orch.Providers(llm=llm, search=_FakeSearch())
        pop1, pop2, inter = self._fixture_pop_intervention()
        areas = type("Areas", (), {"areas": ["Oncology"]})()
        inv = load_source_inventory(FIXTURE_WORKBOOK, strict=False)
        guard = LeakageGuard()

        with unittest.mock.patch.object(orch.retrieval, "execute_plan",
                                        return_value=_FakeExecuteResult([doc])), \
             unittest.mock.patch.object(orch.retrieval, "retrieve_structured",
                                        return_value=_FakeStructured()):
            cache: dict = {}
            r1 = orch._run_retrieval_pass(pop1, inter, areas, inv, None, guard, providers,
                                          orch.RunOptions(), cache)
            r2 = orch._run_retrieval_pass(pop2, inter, areas, inv, None, guard, providers,
                                          orch.RunOptions(), cache)

        extraction_calls = [c for c in llm.seen if c[0] == "a08.extraction"]
        self.assertEqual(len(extraction_calls), 1,
                         "the second population must reuse the cached extraction, "
                         "not re-call the LLM for the identical document")
        self.assertEqual(len(r1.records), 1)
        self.assertEqual(len(r2.records), 1)
        self.assertNotEqual(r1.records[0].finding_id, r2.records[0].finding_id,
                            "each population's copy must get its own finding_id")

        # Downstream grounding/claim-validation mutate a record's fields in
        # place -- population 1's mutations must never bleed into
        # population 2's independent copy, at either the top-level record or
        # a nested object (Comparator) reached through it.
        r1.records[0].grounded = True
        r2.records[0].grounded = False
        self.assertTrue(r1.records[0].grounded)
        self.assertFalse(r2.records[0].grounded)
        r1.records[0].comparator.inn = "mutated-by-pop1"
        self.assertNotEqual(r2.records[0].comparator.inn, "mutated-by-pop1",
                            "deep copy must isolate nested Comparator objects too")

    def test_document_yielding_zero_records_is_still_cached_as_such(self):
        """A document that genuinely has nothing extractable must not be
        re-extracted every population either -- caching an empty result is
        still a cache hit, not 'try again next time'."""
        from jca_phase1.schema import LeakageGuard, RetrievedDocument
        doc = RetrievedDocument(url="https://x.org/empty", resolved_url="https://x.org/empty",
                                text="Irrelevant background text with nothing extractable.",
                                ok=True, source_class=C.SRC_CLINICAL_GUIDELINE)

        class _FakeExecuteResult:
            def __init__(self, documents):
                self.documents = documents
                self.attempts = []
                self.blocked_by_guard = []

        class _FakeStructured:
            attempts: list = []
            trials: list = []
            publications: list = []

        class _FakeSearch:
            def fetch(self, *a, **kw):
                return doc

        llm = ScriptedLLM({"a08.extraction": "[]"})
        providers = orch.Providers(llm=llm, search=_FakeSearch())
        pop1, pop2, inter = self._fixture_pop_intervention()
        areas = type("Areas", (), {"areas": ["Oncology"]})()
        inv = load_source_inventory(FIXTURE_WORKBOOK, strict=False)
        guard = LeakageGuard()

        with unittest.mock.patch.object(orch.retrieval, "execute_plan",
                                        return_value=_FakeExecuteResult([doc])), \
             unittest.mock.patch.object(orch.retrieval, "retrieve_structured",
                                        return_value=_FakeStructured()):
            cache: dict = {}
            orch._run_retrieval_pass(pop1, inter, areas, inv, None, guard, providers,
                                     orch.RunOptions(), cache)
            orch._run_retrieval_pass(pop2, inter, areas, inv, None, guard, providers,
                                     orch.RunOptions(), cache)

        extraction_calls = [c for c in llm.seen if c[0] == "a08.extraction"]
        self.assertEqual(len(extraction_calls), 1,
                         "a document with zero extractable records must still be "
                         "cached as such, not re-extracted every population")


class TestRefinementRoundWiring(unittest.TestCase):
    """Retrieval deep dive, issue 3: plan_refinement() was fully built,
    documented, and enabled by default (config.py's enable_refinement_round)
    -- but nothing anywhere ever called it, so every genuine coverage gap
    from the first retrieval pass (a state/source-class that came back empty
    or inaccessible) just stayed a gap for the rest of the run."""

    def _fixture(self):
        pop = Population(population_id=POP_LICENSED,
                         fields={"indication_disease": Field(value="LGG", provenance="confirmed")})
        for k in inputs.POPULATION_FIELDS:
            pop.fields.setdefault(k, Field())
        inter = Intervention(fields={"product_name_inn": Field(value="Tovorafenib",
                                                               provenance="confirmed")})
        for k in inputs.INTERVENTION_FIELDS:
            inter.fields.setdefault(k, Field())
        areas = type("Areas", (), {"areas": ["Oncology"]})()
        inv = load_source_inventory(FIXTURE_WORKBOOK, strict=False)
        return pop, inter, areas, inv

    def test_a_genuine_gap_triggers_a_refinement_round(self):
        from jca_phase1.schema import (LeakageGuard, RetrievedDocument, SourceClassAttempt)

        gap_attempt = SourceClassAttempt(member_state="Austria",
                                         source_class=C.SRC_CLINICAL_GUIDELINE,
                                         attempted=True, status=C.EV_NONE)
        first_result = type("R", (), {"documents": [], "attempts": [gap_attempt],
                                      "blocked_by_guard": []})()

        refined_doc = RetrievedDocument(
            url="https://x.org/refined", resolved_url="https://x.org/refined",
            text="refined document text", ok=True, source_class=C.SRC_CLINICAL_GUIDELINE)
        refined_attempt = SourceClassAttempt(member_state="Austria",
                                             source_class=C.SRC_CLINICAL_GUIDELINE,
                                             attempted=True, status=C.EV_FOUND)
        refined_result = type("R", (), {"documents": [refined_doc], "attempts": [refined_attempt],
                                        "blocked_by_guard": []})()

        execute_plan_calls = []

        def fake_execute_plan(plan, *a, **kw):
            execute_plan_calls.append(plan)
            return first_result if len(execute_plan_calls) == 1 else refined_result

        class _FakeStructured:
            attempts: list = []
            trials: list = []
            publications: list = []

        llm = ScriptedLLM({"a08.extraction": "[]"})
        providers = orch.Providers(llm=llm, search=object())  # non-None sentinel
        pop, inter, areas, inv = self._fixture()
        guard = LeakageGuard()

        with unittest.mock.patch.object(orch.retrieval, "execute_plan",
                                        side_effect=fake_execute_plan), \
             unittest.mock.patch.object(orch.retrieval, "retrieve_structured",
                                        return_value=_FakeStructured()):
            result = orch._run_retrieval_pass(pop, inter, areas, inv, None, guard,
                                              providers, orch.RunOptions())

        self.assertEqual(len(execute_plan_calls), 2,
                         "a genuine coverage gap must trigger a second, refinement "
                         "execute_plan() call")
        self.assertTrue(any(d.url == "https://x.org/refined" for d in result.documents),
                        "the refinement round's document must reach the final result")

    def test_no_gaps_means_no_refinement_call(self):
        """Regression guard: a clean first pass must not trigger a second
        execute_plan() call -- refinement is for genuine gaps only."""
        from jca_phase1.schema import LeakageGuard, SourceClassAttempt

        found_attempt = SourceClassAttempt(member_state="Austria",
                                           source_class=C.SRC_CLINICAL_GUIDELINE,
                                           attempted=True, status=C.EV_FOUND)
        clean_result = type("R", (), {"documents": [], "attempts": [found_attempt],
                                      "blocked_by_guard": []})()
        execute_plan_calls = []

        def fake_execute_plan(plan, *a, **kw):
            execute_plan_calls.append(plan)
            return clean_result

        class _FakeStructured:
            attempts: list = []
            trials: list = []
            publications: list = []

        llm = ScriptedLLM({"a08.extraction": "[]"})
        providers = orch.Providers(llm=llm, search=object())
        pop, inter, areas, inv = self._fixture()
        guard = LeakageGuard()

        with unittest.mock.patch.object(orch.retrieval, "execute_plan",
                                        side_effect=fake_execute_plan), \
             unittest.mock.patch.object(orch.retrieval, "retrieve_structured",
                                        return_value=_FakeStructured()):
            orch._run_retrieval_pass(pop, inter, areas, inv, None, guard,
                                     providers, orch.RunOptions())

        self.assertEqual(len(execute_plan_calls), 1)


# ===========================================================================
# End to end
# ===========================================================================

class TestEndToEnd(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.environ["JCA_SOURCE_WORKBOOK"] = FIXTURE_WORKBOOK
        from jca_phase1.demo import run_demo
        cls.out = run_demo()

    def test_run_completes_and_all_assertions_pass(self):
        failed = [a.name for a in self.out.completeness.assertions if not a.passed]
        self.assertEqual(failed, [], f"failed assertions: {failed}")

    def test_comparators_were_found(self):
        self.assertTrue(self.out.comparators)

    def test_no_comparator_carries_the_intervention_class(self):
        intervention_class = self.out.intervention.value("therapeutic_class_mechanism")
        for c in self.out.comparators:
            self.assertNotEqual(c.class_or_mechanism, intervention_class)

    def test_by_member_state_has_exactly_27_entries(self):
        self.assertEqual(len(self.out.by_member_state), 27)

    def test_unidentified_states_use_the_exact_sme_wording(self):
        for entry in self.out.by_member_state:
            if entry.status == "not_identified":
                self.assertEqual(entry.finding, C.NOT_IDENTIFIED_COMPARATOR_TEXT)

    def test_induction_regimen_was_excluded_with_a_reason(self):
        reasons = " ".join(e["reason"] for e in self.out.validation.to_dict()["excluded"])
        self.assertIn("prior therapy", reasons.lower())

    def test_pubmed_publication_reaches_the_final_output(self):
        """DEMO_PUBLICATIONS is wired into the demo's literature provider.
        Before the retrieve_structured() -> documents fix, its finding never
        reached extraction, so no comparator ever cited a pubmed source."""
        pubmed_sources = [s for c in self.out.comparators for s in c.sources
                          if s.source_class == C.SRC_PUBMED]
        self.assertTrue(pubmed_sources, "no comparator cites a PubMed source")

    def test_run_manifest_records_what_is_needed_to_reproduce_the_run(self):
        m = self.out.run_manifest
        for key in ("prompt_versions", "outcome_catalog_version", "source_workbook",
                    "leakage_guard", "therapeutic_areas_applied", "model"):
            self.assertIn(key, m)
        self.assertFalse(m["phase2_included"])

    def test_output_serialises_to_json(self):
        json.dumps(self.out.to_dict(), default=str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
