"""
Tests for streamlit_app/excel_export.py covering the SME's latest ask:
print the PubMed/tier-1/2/3 query strings and the evidence_status /
data_maturity / trial_status / peer_review_status fields in the Excel
export -- both in the final-output "Evidence" sheets (already covered
before this change) and in the shared per-stage debug-trail sheets that
use _record_row()/_RECORD_HEADERS (A8/A9/A10/A10-Usable/A14/A15-Groups),
which previously dropped those four fields.
"""
import io
import sys
import unittest

sys.path.insert(0, "streamlit_app")

import openpyxl

from jca_phase1 import config as C
from streamlit_app import excel_export


def _load(run):
    wb = openpyxl.load_workbook(io.BytesIO(excel_export.build_excel(run)))
    return wb


def _headers(ws):
    return [c.value for c in ws[1]]


class TestRecordRowEvidenceFields(unittest.TestCase):
    def test_record_row_includes_the_four_sme_requested_fields(self):
        row = excel_export._record_row({
            "finding_id": "f1", "finding_type": "comparator", "subject_drug": "DrugX",
            "member_state": "DE", "source_class": C.SRC_CONFERENCE, "tier": 2,
            "evidence_status": "conference_abstract", "data_maturity": "interim",
            "trial_status": "TERMINATED", "peer_review_status": "preprint",
        })
        self.assertEqual(row[-4:], ["conference_abstract", "interim", "TERMINATED", "preprint"])
        self.assertEqual(len(row), len(excel_export._RECORD_HEADERS))

    def test_record_headers_name_the_four_fields_at_the_end(self):
        self.assertEqual(excel_export._RECORD_HEADERS[-4:],
                         ["evidence_status", "data_maturity", "trial_status",
                          "peer_review_status"])


class TestQueryPlanTierColumn(unittest.TestCase):
    def _run_with_plan_and_attempts(self):
        return {
            "output": {"comparators": [], "outcomes": {}},
            "debug_capture": {
                "A6_query_plan": {"plan": [
                    {"query": "drug AND disease", "source_class": C.SRC_HTA_REGULATORY,
                     "member_state": "DE", "pass_type": "primary", "domains": [],
                     "max_urls": 5, "language": "en", "note": ""},
                    {"query": "guideline query", "source_class": C.SRC_CLINICAL_GUIDELINE,
                     "member_state": "EU-wide", "pass_type": "primary", "domains": [],
                     "max_urls": 5, "language": "en", "note": ""},
                ]},
                "A7_retrieval": {"attempts": [
                    {"member_state": "EU-wide", "source_class": C.SRC_PUBMED,
                     "attempted": True, "documents_retrieved": 10, "status": "ok",
                     "detail": "[1 drug-anchored] q1 | [2 guideline-restricted] q2 | "
                              "[3 landscape, unrestricted] q3"},
                ], "documents": []},
            },
        }

    def test_a6_query_plan_sheet_has_a_tier_column_matching_source_class(self):
        wb = _load(self._run_with_plan_and_attempts())
        ws = wb["A6 Query Plan (Mix)"]
        headers = _headers(ws)
        self.assertIn("tier", headers)
        tier_idx = headers.index("tier") + 1
        source_idx = headers.index("source_class") + 1
        for row in ws.iter_rows(min_row=2, values_only=False):
            source_class = row[source_idx - 1].value
            tier = row[tier_idx - 1].value
            self.assertEqual(tier, C.TIER_OF_SOURCE_CLASS[source_class])

    def test_a7_attempts_sheet_carries_pubmed_query_strings_and_tier(self):
        wb = _load(self._run_with_plan_and_attempts())
        ws = wb["A7 Retrieval Attempts (Det)"]
        headers = _headers(ws)
        self.assertIn("query_sent", headers)
        self.assertIn("tier", headers)
        row = list(ws.iter_rows(min_row=2, values_only=True))[0]
        detail = row[headers.index("query_sent")]
        self.assertIn("drug-anchored", detail)
        self.assertIn("guideline-restricted", detail)
        self.assertIn("landscape", detail)
        self.assertEqual(row[headers.index("tier")], C.TIER_OF_SOURCE_CLASS[C.SRC_PUBMED])


if __name__ == "__main__":
    unittest.main()
