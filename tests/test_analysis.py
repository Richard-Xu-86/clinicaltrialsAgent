"""Aggregation and chart builders on small synthetic record sets."""

import pytest

from ctviz.analysis.charts import (
    BuildOptions,
    CohortData,
    build_comparison,
    build_distribution,
    build_histogram,
    build_network,
    build_scatter,
    build_trend,
)
from ctviz.analysis.dimensions import DIMENSIONS, ExtractContext
from ctviz.domain.normalize import normalize_studies
from ctviz.schemas import Dimension, NetworkSpec, NumericField, PlanFilters
from tests.conftest import make_study

OPTS = BuildOptions(citation_limit=100, top_n=None, ctx=ExtractContext())


def cohort(studies, label="c", **filters):
    return CohortData(label, PlanFilters(**filters), normalize_studies(studies))


def test_phase_distribution_sums_to_trial_count():
    c = cohort([
        make_study("NCT1", phases=["PHASE1", "PHASE2"]),
        make_study("NCT2", phases=["PHASE2"]),
        make_study("NCT3", phases=["PHASE3"]),
        make_study("NCT4"),  # observational: no phase
    ])
    rows = build_distribution(c, DIMENSIONS[Dimension.PHASE], OPTS).visualization.data
    by_phase = {r["phase"]: r["trial_count"] for r in rows}
    assert by_phase == {"Phase 1/2": 1, "Phase 2": 1, "Phase 3": 1, "No phase (e.g. observational)": 1}
    assert sum(by_phase.values()) == 4
    assert [r["phase"] for r in rows][:3] == ["Phase 1/2", "Phase 2", "Phase 3"]  # ordinal order


def test_citations_carry_exact_evidence_and_respect_limit():
    c = cohort([make_study(f"NCT{i}", phases=["PHASE2"]) for i in range(5)])
    opts = BuildOptions(citation_limit=2, top_n=None, ctx=ExtractContext())
    row = build_distribution(c, DIMENSIONS[Dimension.PHASE], opts).visualization.data[0]
    assert row["trial_count"] == row["citation_count"] == 5
    assert len(row["citations"]) == 2
    ev = row["citations"][0].evidence[0]
    assert (ev.field, ev.value) == ("protocolSection.designModule.phases", "PHASE2")


def test_country_counts_trials_not_sites_and_respects_recruiting_sites():
    c = cohort([
        make_study("NCT1", sites=[("France", "RECRUITING"), ("France", "RECRUITING"), ("Spain", "COMPLETED")]),
        make_study("NCT2", sites=[("France", "RECRUITING")]),
    ])
    dim = DIMENSIONS[Dimension.COUNTRY]
    all_sites = {r["country"]: r["trial_count"] for r in build_distribution(c, dim, OPTS).visualization.data}
    assert all_sites == {"France": 2, "Spain": 1}
    rec_opts = BuildOptions(citation_limit=5, top_n=None, ctx=ExtractContext(recruiting_sites_only=True))
    recruiting = {r["country"]: r["trial_count"] for r in build_distribution(c, dim, rec_opts).visualization.data}
    assert recruiting == {"France": 2}


def test_top_n_warns():
    c = cohort([make_study(f"NCT{i}", conditions=[f"cond {i}"]) for i in range(20)])
    result = build_distribution(c, DIMENSIONS[Dimension.CONDITION],
                                BuildOptions(citation_limit=1, top_n=5, ctx=ExtractContext()))
    assert len(result.visualization.data) == 5 and "top 5 of 20" in result.warnings[0]


def test_trend_zero_fills_and_counts_missing():
    c = cohort([make_study("NCT1", start="2016-01"), make_study("NCT2", start="2018-05-02"),
                make_study("NCT3")], start_year=2015, end_year=2018)
    result = build_trend([c], None, None, OPTS)
    counts = {r["start_year"]: r["trial_count"] for r in result.visualization.data}
    assert counts == {2015: 0, 2016: 1, 2017: 0, 2018: 1}
    assert result.time_granularity == "year"
    assert "1 of 3 trials have no start date" in result.warnings[0]


def test_comparison_rows_align_across_cohorts():
    a = cohort([make_study("NCT1", phases=["PHASE3"]), make_study("NCT2", phases=["PHASE3"])], label="A")
    b = cohort([make_study("NCT3", phases=["PHASE1"])], label="B")
    rows = build_comparison([a, b], PlanFilters(), DIMENSIONS[Dimension.PHASE], "Drug", OPTS).visualization.data
    table = {(r["phase"], r["cohort"]): (r["trial_count"], r["share_of_cohort"]) for r in rows}
    assert table == {("Phase 1", "A"): (0, 0.0), ("Phase 1", "B"): (1, 1.0),
                     ("Phase 3", "A"): (2, 1.0), ("Phase 3", "B"): (0, 0.0)}


def test_histogram_bins_partition_values():
    c = cohort([make_study(f"NCT{i}", enrollment=n) for i, n in enumerate([5, 19, 20, 49, 300, 12000])])
    rows = build_histogram([c], NumericField.ENROLLMENT, OPTS).visualization.data
    assert sum(r["trial_count"] for r in rows) == 6
    assert rows[0]["bin_label"] == "0–19" and rows[0]["trial_count"] == 2
    assert rows[-1]["bin_end"] is None and rows[-1]["trial_count"] == 1  # open-ended top bin


def test_scatter_one_point_per_trial_with_both_fields_cited():
    c = cohort([make_study("NCT1", enrollment=40, start="2020-01", completion="2021-07"),
                make_study("NCT2", enrollment=10)])  # no dates -> dropped
    result = build_scatter([c], NumericField.ENROLLMENT, NumericField.DURATION_MONTHS, OPTS)
    (point,) = result.visualization.data
    assert point["duration_months"] == 18 and point["enrollment"] == 40
    assert len(point["citations"][0].evidence) == 3  # enrollment + start + completion
    assert result.warnings


def test_drug_cooccurrence_network():
    drugs = lambda *names: [{"name": n, "type": "DRUG"} for n in names]  # noqa: E731
    c = cohort([
        make_study("NCT1", interventions=drugs("Nivolumab", "Ipilimumab")),
        make_study("NCT2", interventions=drugs("Opdivo + Yervoy")),  # brand names, one entry
        make_study("NCT3", interventions=drugs("Nivolumab", "Relatlimab")),
        make_study("NCT4", interventions=drugs("Placebo", "Nivolumab")),
    ])
    viz = build_network([c], NetworkSpec(source="drug", target="drug", min_edge_weight=2), OPTS).visualization
    assert [(e.source, e.target, e.trial_count) for e in viz.data.edges] == [
        ("drug:ipilimumab", "drug:nivolumab", 2)]
    nodes = {n.id: n.trial_count for n in viz.data.nodes}
    assert nodes == {"drug:nivolumab": 4, "drug:ipilimumab": 2}
    assert {c.nct_id for c in viz.data.edges[0].citations} == {"NCT1", "NCT2"}


def test_bipartite_network_orientation():
    c = cohort([make_study("NCT1", sponsor="Acme", interventions=[{"name": "Drug A", "type": "DRUG"}])])
    viz = build_network([c], NetworkSpec(source="sponsor", target="drug", min_edge_weight=1), OPTS).visualization
    (edge,) = viz.data.edges
    assert edge.source == "sponsor:acme" and edge.target == "drug:drug a"
    assert {n.entity_type for n in viz.data.nodes} == {"sponsor", "drug"}


# --------------------------------------------------------------------------- regressions from review


def _star_and_pair():
    """A hub drug used by five sponsors once each, plus one sponsor-drug pair seen 3 times."""
    studies = [make_study(f"NCT1{i}", sponsor=f"Sponsor {i}", interventions=[{"name": "Hubdrug", "type": "DRUG"}])
               for i in range(5)]
    studies += [make_study(f"NCT2{i}", sponsor="Pair Co", interventions=[{"name": "Pairdrug", "type": "DRUG"}])
                for i in range(3)]
    return cohort(studies)


def test_network_small_budget_is_never_empty():
    spec = NetworkSpec(source="sponsor", target="drug", top_n_nodes=2, min_edge_weight=1)
    viz = build_network([_star_and_pair()], spec, OPTS).visualization
    assert len(viz.data.nodes) == 2 and len(viz.data.edges) == 1
    assert viz.data.edges[0].trial_count == 3  # strongest link wins


def test_network_keeps_hub_with_many_weak_links():
    spec = NetworkSpec(source="sponsor", target="drug", top_n_nodes=5, min_edge_weight=1)
    viz = build_network([_star_and_pair()], spec, OPTS).visualization
    ids = {n.id for n in viz.data.nodes}
    assert "drug:hubdrug" in ids
    assert all(any(n.id in (e.source, e.target) for e in viz.data.edges) for n in viz.data.nodes)


def test_missing_phase_bucket_cites_the_absent_field():
    rows = build_distribution(cohort([make_study("NCT1")]), DIMENSIONS[Dimension.PHASE], OPTS).visualization.data
    ev = rows[0]["citations"][0].evidence[0]
    assert (ev.field, ev.value) == ("protocolSection.designModule.phases", None)


@pytest.mark.parametrize("values,labels", [
    ([3, 7, 15], ["0–19"]),                             # small values keep a bounded bin
    ([10, 300], ["0–19", "20–49", "50–99", "100–249", "250–499"]),
    ([10, 9000], ["0–19", "20–49", "50–99", "100–249", "250–499", "500–999", "1,000–4,999", "5,000+"]),
])
def test_histogram_bins(values, labels):
    c = cohort([make_study(f"NCT{i}", enrollment=v) for i, v in enumerate(values)])
    viz = build_histogram([c], NumericField.ENROLLMENT, OPTS).visualization
    assert [r["bin_label"] for r in viz.data] == labels == viz.encoding.x.sort
    assert sum(r["trial_count"] for r in viz.data) == len(values)


def test_year_histogram_labels_have_no_thousands_separator():
    c = cohort([make_study(f"NCT{i}", start=f"{y}-01") for i, y in enumerate([2010, 2012, 2019, 2024])])
    labels = [r["bin_label"] for r in build_histogram([c], NumericField.START_YEAR, OPTS).visualization.data]
    assert all("," not in label for label in labels)


def test_trend_series_are_capped_and_missing_reported():
    studies = [make_study(f"NCT{i}", start="2020-01", conditions=[f"cond {i}"]) for i in range(12)]
    studies.append(make_study("NCT99", conditions=["cond 0"]))  # no start date
    result = build_trend([cohort(studies)], DIMENSIONS[Dimension.CONDITION], "Condition", OPTS)
    assert len({r["series"] for r in result.visualization.data}) == 8
    assert any("top 8 of 12" in w for w in result.warnings)
    assert any("no start date" in w for w in result.warnings)


def test_every_cartesian_row_has_trial_count():
    c = cohort([make_study("NCT1", enrollment=40, start="2020-01", completion="2021-07")])
    rows = build_scatter([c], NumericField.ENROLLMENT, NumericField.DURATION_MONTHS, OPTS).visualization.data
    assert rows[0]["trial_count"] == 1


def test_top_n_on_fixed_dimension_warns():
    c = cohort([make_study("NCT1", phases=["PHASE1"])])
    result = build_distribution(c, DIMENSIONS[Dimension.PHASE], BuildOptions(5, top_n=1, ctx=ExtractContext()))
    assert any("top_n ignored" in w for w in result.warnings)
