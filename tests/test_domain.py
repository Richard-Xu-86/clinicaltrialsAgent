"""Normalization and drug matching."""

import pytest

from ctviz.domain.drugs import canonical_drug_names, find_drug_evidence
from ctviz.domain.normalize import normalize_studies, normalize_study, parse_partial_date, phase_bucket
from tests.conftest import make_study


class TestNormalize:
    def test_full_record_keeps_source_paths(self):
        rec = normalize_study(make_study(
            "NCT00000001", title="A trial", phases=["PHASE2", "PHASE1"], start="2019-03",
            status="COMPLETED", sponsor="Acme", sponsor_class="INDUSTRY", enrollment=120,
            interventions=[{"name": "Drug X", "type": "DRUG", "otherNames": ["X-100"]}],
            conditions=["Melanoma"], sites=[("Canada", "RECRUITING"), ("Canada", None)],
        ))
        assert rec.phase.value == "PHASE1/PHASE2"
        assert rec.phase.path == "protocolSection.designModule.phases"
        assert rec.phase.raw == "PHASE2, PHASE1"  # verbatim order, as in the API
        assert rec.start.value.year == 2019 and rec.start.value.month == 3 and rec.start.value.day is None
        assert rec.interventions[0].other_names[0].path.endswith("interventions[0].otherNames[0]")
        assert rec.sites[1].country.path.endswith("locations[1].country")
        assert rec.enrollment.value == 120

    def test_missing_modules_do_not_raise(self):
        rec = normalize_study(make_study("NCT00000002"))
        assert rec.phase is None and rec.start is None and rec.interventions == [] and rec.sites == []

    @pytest.mark.parametrize("bad", [{}, {"protocolSection": {}}, None, "x"])
    def test_records_without_id_are_skipped(self, bad):
        assert normalize_study(bad) is None

    def test_duplicates_removed(self):
        studies = [make_study("NCT1"), make_study("NCT1"), make_study("NCT2")]
        assert [r.nct_id for r in normalize_studies(studies)] == ["NCT1", "NCT2"]

    def test_negative_or_bool_enrollment_ignored(self):
        assert normalize_study(make_study("NCT3", enrollment=-5)).enrollment is None

    @pytest.mark.parametrize("raw,expected", [
        ("2017-10-03", (2017, 10, 3)), ("2026-09", (2026, 9, None)), ("2020", (2020, None, None)),
    ])
    def test_partial_dates(self, raw, expected):
        d = parse_partial_date(raw)
        assert (d.year, d.month, d.day) == expected

    def test_unparseable_date(self):
        assert parse_partial_date("September 2020") is None

    def test_phase_bucket_order_independent(self):
        assert phase_bucket(["PHASE3", "PHASE2"]) == phase_bucket(["PHASE2", "PHASE3"]) == "PHASE2/PHASE3"


class TestCanonicalDrugNames:
    # Every left-hand value below was observed in real ClinicalTrials.gov records.
    @pytest.mark.parametrize("raw,expected", [
        ("Pembrolizumab", ["pembrolizumab"]),
        ("MK-3475", ["pembrolizumab"]),
        ("Pembrolizumab Injection [Keytruda]", ["pembrolizumab"]),
        ("KEYTRUDA ®( Pembrolizumab)", ["pembrolizumab"]),
        ("Pembrolizumab 25 MG/ML [KEYTRUDA®]", ["pembrolizumab"]),
        ("Neoadjuvant Pembrolizumab", ["pembrolizumab"]),
        ("lenvatinib plus pembrolizumab", ["lenvatinib", "pembrolizumab"]),
        ("Nivolumab, Pembrolizumab", ["nivolumab", "pembrolizumab"]),
        ("Pembrolizumab/Quavonlimab", ["pembrolizumab", "quavonlimab"]),
        ("Combination Therapy: PY314 + Pembrolizumab", ["py314", "pembrolizumab"]),
        ("Pembrolizumab (+) Berahyaluronidase alfa", ["pembrolizumab", "berahyaluronidase alfa"]),
        ("Cisplatin 75 mg/m2", ["cisplatin"]),
        ("Carboplatin AUC 5", ["carboplatin"]),
        ("Semaglutide 2.4 mg s.c.", ["semaglutide"]),
        ("Placebo for pembrolizumab", []),
        ("Placebo", []),
        ("Standard of care chemotherapy", []),
    ])
    def test_observed_variants(self, raw, expected):
        assert canonical_drug_names(raw) == expected


class TestDrugEvidence:
    def _rec(self, **kw):
        return normalize_study(make_study("NCT9", **kw))

    def test_matches_intervention_name(self):
        rec = self._rec(interventions=[{"name": "Radiation", "type": "RADIATION"},
                                       {"name": "Pembrolizumab 200 mg", "type": "DRUG"}])
        ev = find_drug_evidence(rec, "pembrolizumab")
        assert ev.path.endswith("interventions[1].name") and ev.raw == "Pembrolizumab 200 mg"

    def test_brand_name_query_matches_generic(self):
        rec = self._rec(interventions=[{"name": "Pembrolizumab", "type": "DRUG"}])
        assert find_drug_evidence(rec, "Keytruda") is not None

    def test_matches_registered_other_name(self):
        rec = self._rec(interventions=[{"name": "Study drug", "type": "DRUG", "otherNames": ["LY3298176"]}])
        ev = find_drug_evidence(rec, "tirzepatide")
        assert ev is not None and ev.path.endswith("otherNames[0]")

    def test_placebo_arm_is_not_evidence(self):
        rec = self._rec(interventions=[{"name": "Placebo for pembrolizumab", "type": "DRUG"}])
        assert find_drug_evidence(rec, "pembrolizumab") is None

    def test_hyphenated_arm_label(self):
        rec = self._rec(interventions=[{"name": "LSG-Tirzepatide group", "type": "DRUG"}])
        assert find_drug_evidence(rec, "tirzepatide") is not None

    def test_title_fallback_only_without_intervention_match(self):
        rec = self._rec(title="Semaglutide Effectiveness in MASLD", interventions=[])
        ev = find_drug_evidence(rec, "semaglutide")
        assert ev.path == "protocolSection.identificationModule.briefTitle"

    def test_class_level_study_excluded(self):
        rec = self._rec(title="GLP-1 receptor agonists and lactation",
                        interventions=[{"name": "GLP-1 receptor agonist therapy", "type": "DRUG"}])
        assert find_drug_evidence(rec, "semaglutide") is None

    def test_no_substring_false_positive(self):
        rec = self._rec(interventions=[{"name": "Pembrolizumab2x (fictional)", "type": "DRUG"}])
        assert find_drug_evidence(rec, "pembrolizumab") is None
