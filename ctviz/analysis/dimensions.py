"""Registry of the dimensions and numeric fields the service can chart.

Adding a new way to slice trials means adding one entry here: which API fields it
needs, how to pull categories (with evidence) out of a TrialRecord, and how to
label and order them. The aggregation and chart code never special-cases a
dimension.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from ..domain.drugs import canonical_drugs, display_name
from ..domain.records import TrialRecord
from ..schemas import Dimension, Evidence, NumericField


@dataclass(frozen=True)
class Category:
    key: str
    label: str
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True)
class ExtractContext:
    """Knobs that change how a dimension is read, derived from the plan."""

    # When the question is about recruiting trials, a trial's country should come
    # from its recruiting sites, not from sites that closed years ago.
    recruiting_sites_only: bool = False


Extractor = Callable[[TrialRecord, ExtractContext], list[Category]]


@dataclass(frozen=True)
class DimensionDef:
    key: Dimension
    row_field: str                      # key used in response rows
    title: str                          # axis title
    api_fields: tuple[str, ...]         # ClinicalTrials.gov field names to request
    extract: Extractor
    channel_type: Literal["nominal", "ordinal", "temporal"] = "nominal"
    order: tuple[str, ...] | None = None  # fixed display order (category keys); else by count
    multi_valued: bool = False            # one trial can land in several categories
    missing_label: str | None = None      # bucket for trials without a value; None = drop them
    missing_path: str | None = None       # field whose absence puts a trial in that bucket
    counting_rule: str | None = None      # explanation surfaced in response meta


# ----------------------------------------------------------------------------- labels

PHASE_LABELS = {
    "EARLY_PHASE1": "Early Phase 1",
    "PHASE1": "Phase 1",
    "PHASE1/PHASE2": "Phase 1/2",
    "PHASE2": "Phase 2",
    "PHASE2/PHASE3": "Phase 2/3",
    "PHASE3": "Phase 3",
    "PHASE4": "Phase 4",
    "NA": "Not applicable",
}
PHASE_ORDER = tuple(PHASE_LABELS) + ("__missing__",)

SPONSOR_CLASS_LABELS = {
    "INDUSTRY": "Industry",
    "NIH": "NIH",
    "FED": "Other U.S. federal",
    "OTHER_GOV": "Other government",
    "NETWORK": "Network",
    "INDIV": "Individual",
    "OTHER": "Other (academic, hospital, non-profit)",
    "UNKNOWN": "Unknown",
}

MISSING = "__missing__"


def _humanize_code(code: str) -> str:
    return code.replace("_", " ").capitalize()


def _single(sourced, labels: dict[str, str] | None = None) -> list[Category]:
    if sourced is None:
        return []
    label = (labels or {}).get(sourced.value) or _humanize_code(sourced.value)
    return [Category(sourced.value, label, (sourced.evidence(),))]


# ----------------------------------------------------------------------------- extractors

def _phase(rec: TrialRecord, _: ExtractContext) -> list[Category]:
    if rec.phase is None:
        return []
    label = PHASE_LABELS.get(rec.phase.value, rec.phase.value.replace("PHASE", "Phase ").replace("/", " / "))
    return [Category(rec.phase.value, label, (rec.phase.evidence(),))]


def _status(rec, _):
    return _single(rec.status)


def _study_type(rec, _):
    return _single(rec.study_type)


def _sponsor_class(rec, _):
    return _single(rec.sponsor_class, SPONSOR_CLASS_LABELS)


def _lead_sponsor(rec: TrialRecord, _) -> list[Category]:
    if rec.lead_sponsor is None:
        return []
    return [Category(rec.lead_sponsor.value.casefold(), rec.lead_sponsor.value, (rec.lead_sponsor.evidence(),))]


def _intervention_type(rec: TrialRecord, _) -> list[Category]:
    by_type: dict[str, list[Evidence]] = {}
    for intervention in rec.interventions:
        if intervention.type:
            by_type.setdefault(intervention.type, []).append(
                Evidence(field=intervention.name.path.replace(".name", ".type"), value=intervention.type)
            )
    return [Category(t, _humanize_code(t), (ev[0],)) for t, ev in by_type.items()]


def _drug(rec: TrialRecord, _) -> list[Category]:
    found: dict[str, Category] = {}
    for intervention in rec.interventions:
        for name in canonical_drugs(intervention):
            if name not in found:
                found[name] = Category(name, display_name(name), (intervention.name.evidence(),))
    return list(found.values())


def _condition(rec: TrialRecord, _) -> list[Category]:
    found: dict[str, Category] = {}
    for cond in rec.conditions:
        key = " ".join(cond.value.casefold().split())
        found.setdefault(key, Category(key, cond.value, (cond.evidence(),)))
    return list(found.values())


def _country(rec: TrialRecord, ctx: ExtractContext) -> list[Category]:
    sites = rec.sites
    if ctx.recruiting_sites_only and any(s.status for s in sites):
        sites = [s for s in sites if s.status == "RECRUITING"]
    found: dict[str, Category] = {}
    for site in sites:
        found.setdefault(site.country.value, Category(site.country.value, site.country.value, (site.country.evidence(),)))
    return list(found.values())


def _start_year(rec: TrialRecord, _) -> list[Category]:
    if rec.start is None:
        return []
    year = str(rec.start.value.year)
    return [Category(year, year, (rec.start.evidence(),))]


# ----------------------------------------------------------------------------- registry

INTERVENTION_FIELDS = ("InterventionName", "InterventionType", "InterventionOtherName")

DIMENSIONS: dict[Dimension, DimensionDef] = {
    d.key: d
    for d in [
        DimensionDef(
            Dimension.PHASE, "phase", "Phase", ("Phase",), _phase,
            channel_type="ordinal", order=PHASE_ORDER, missing_label="No phase (e.g. observational)",
            missing_path="protocolSection.designModule.phases",
            counting_rule="Trials registered under two phases (e.g. Phase 1 and Phase 2) are counted "
                          "once, in a combined 'Phase 1/2' bucket, so bars sum to the number of trials.",
        ),
        DimensionDef(Dimension.STATUS, "status", "Overall status", ("OverallStatus",), _status),
        DimensionDef(Dimension.STUDY_TYPE, "study_type", "Study type", ("StudyType",), _study_type),
        DimensionDef(
            Dimension.SPONSOR_CLASS, "sponsor_class", "Lead sponsor category",
            ("LeadSponsorClass",), _sponsor_class,
            counting_rule="Sponsor category is the lead sponsor's class as registered "
                          "(collaborators are not counted).",
        ),
        DimensionDef(
            Dimension.LEAD_SPONSOR, "lead_sponsor", "Lead sponsor", ("LeadSponsorName",), _lead_sponsor,
            counting_rule="Sponsors are grouped by exact lead-sponsor name (case-insensitive); "
                          "subsidiaries are not merged.",
        ),
        DimensionDef(
            Dimension.INTERVENTION_TYPE, "intervention_type", "Intervention type",
            ("InterventionName", "InterventionType"), _intervention_type, multi_valued=True,
            counting_rule="A trial with several intervention types (e.g. Drug and Radiation) is "
                          "counted once under each type.",
        ),
        DimensionDef(
            Dimension.DRUG, "drug", "Drug", INTERVENTION_FIELDS, _drug, multi_valued=True,
            counting_rule="Drug names are normalized (brand/code names mapped to generics, doses and "
                          "placebos removed, combinations split); a trial counts once per drug.",
        ),
        DimensionDef(
            Dimension.CONDITION, "condition", "Condition", ("Condition",), _condition, multi_valued=True,
            counting_rule="Conditions are grouped by exact registered text (case-insensitive); a trial "
                          "listing several conditions counts once under each.",
        ),
        DimensionDef(
            Dimension.COUNTRY, "country", "Country", ("LocationCountry", "LocationStatus"), _country,
            multi_valued=True,
            counting_rule="A trial counts once per country in which it has at least one site "
                          "(not once per site).",
        ),
        DimensionDef(
            Dimension.START_YEAR, "start_year", "Start year", ("StartDate",), _start_year,
            channel_type="temporal",
            counting_rule="Trials are bucketed by registered start date, which for future trials is "
                          "an estimate.",
        ),
    ]
}


def missing_description(dim: DimensionDef, ctx: ExtractContext) -> str:
    """Wording for the 'N trials have no ...' warning."""
    if dim.key == Dimension.COUNTRY and ctx.recruiting_sites_only:
        return "recruiting site (with a country)"
    return dim.title.lower()


# ----------------------------------------------------------------------------- numeric fields

@dataclass(frozen=True)
class NumericValue:
    value: float
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True)
class NumericDef:
    key: NumericField
    row_field: str
    title: str
    unit: str | None
    api_fields: tuple[str, ...]
    extract: Callable[[TrialRecord], NumericValue | None]
    bin_edges: tuple[float, ...] | None = None  # preset histogram bins; else equal width
    notes: tuple[str, ...] = field(default_factory=tuple)
    thousands: bool = True  # thousands separators in labels ("5,000+"), off for years


def _enrollment(rec: TrialRecord) -> NumericValue | None:
    if rec.enrollment is None:
        return None
    return NumericValue(float(rec.enrollment.value), (rec.enrollment.evidence(),))


def _duration(rec: TrialRecord) -> NumericValue | None:
    if rec.start is None or rec.primary_completion is None:
        return None
    start, end = rec.start.value.month_index, rec.primary_completion.value.month_index
    if start is None or end is None or end < start:
        return None
    return NumericValue(float(end - start), (rec.start.evidence(), rec.primary_completion.evidence()))


def _start_year_numeric(rec: TrialRecord) -> NumericValue | None:
    if rec.start is None:
        return None
    return NumericValue(float(rec.start.value.year), (rec.start.evidence(),))


NUMERIC_FIELDS: dict[NumericField, NumericDef] = {
    NumericField.ENROLLMENT: NumericDef(
        NumericField.ENROLLMENT, "enrollment", "Enrollment", "participants", ("EnrollmentCount",),
        _enrollment,
        # Enrollment is extremely skewed (a few trials enroll 100k+); fixed, human-meaningful
        # bins read better than equal-width ones.
        bin_edges=(0, 20, 50, 100, 250, 500, 1000, 5000, float("inf")),
        notes=("Enrollment is the registered count: actual for completed trials, anticipated otherwise.",),
    ),
    NumericField.DURATION_MONTHS: NumericDef(
        NumericField.DURATION_MONTHS, "duration_months", "Duration (start to primary completion)",
        "months", ("StartDate", "PrimaryCompletionDate"), _duration,
        bin_edges=(0, 6, 12, 24, 36, 48, 60, 84, 120, float("inf")),
        notes=("Duration = primary completion date minus start date, at month precision; "
               "estimated dates are used for ongoing trials.",),
    ),
    NumericField.START_YEAR: NumericDef(
        NumericField.START_YEAR, "start_year", "Start year", None, ("StartDate",), _start_year_numeric,
        thousands=False,
    ),
}
