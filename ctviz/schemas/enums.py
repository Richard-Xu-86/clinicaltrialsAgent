"""Controlled vocabularies shared by the request, the query plan and the response.

Keeping these as enums (rather than free strings) is what lets us hand the LLM a
closed schema: it can only choose dimensions and filters we know how to compute.
"""

from enum import Enum


class AnalysisType(str, Enum):
    TREND = "trend"                # count of trials over start year
    DISTRIBUTION = "distribution"  # count of trials per category of one dimension
    COMPARISON = "comparison"      # distribution, split into 2+ cohorts (drug A vs drug B)
    NETWORK = "network"            # co-occurrence of entities within trials
    HISTOGRAM = "histogram"        # distribution of a numeric field
    SCATTER = "scatter"            # one point per trial, two numeric fields


class Dimension(str, Enum):
    """Categorical fields a chart can be grouped by."""

    PHASE = "phase"
    STATUS = "status"
    SPONSOR_CLASS = "sponsor_class"
    LEAD_SPONSOR = "lead_sponsor"
    INTERVENTION_TYPE = "intervention_type"
    DRUG = "drug"
    CONDITION = "condition"
    COUNTRY = "country"
    STUDY_TYPE = "study_type"
    START_YEAR = "start_year"


class NumericField(str, Enum):
    ENROLLMENT = "enrollment"
    DURATION_MONTHS = "duration_months"  # start date -> primary completion date
    START_YEAR = "start_year"


class EntityType(str, Enum):
    """Node types available for network graphs."""

    DRUG = "drug"
    SPONSOR = "sponsor"
    CONDITION = "condition"
    COUNTRY = "country"


class CompareField(str, Enum):
    DRUG = "drug"
    CONDITION = "condition"
    SPONSOR = "sponsor"
    COUNTRY = "country"


class Phase(str, Enum):
    """Phase codes exactly as ClinicalTrials.gov returns them."""

    EARLY_PHASE1 = "EARLY_PHASE1"
    PHASE1 = "PHASE1"
    PHASE2 = "PHASE2"
    PHASE3 = "PHASE3"
    PHASE4 = "PHASE4"
    NA = "NA"


class OverallStatus(str, Enum):
    NOT_YET_RECRUITING = "NOT_YET_RECRUITING"
    RECRUITING = "RECRUITING"
    ENROLLING_BY_INVITATION = "ENROLLING_BY_INVITATION"
    ACTIVE_NOT_RECRUITING = "ACTIVE_NOT_RECRUITING"
    SUSPENDED = "SUSPENDED"
    TERMINATED = "TERMINATED"
    COMPLETED = "COMPLETED"
    WITHDRAWN = "WITHDRAWN"
    UNKNOWN = "UNKNOWN"


class StudyType(str, Enum):
    INTERVENTIONAL = "INTERVENTIONAL"
    OBSERVATIONAL = "OBSERVATIONAL"
    EXPANDED_ACCESS = "EXPANDED_ACCESS"


class VisualizationType(str, Enum):
    BAR_CHART = "bar_chart"
    GROUPED_BAR_CHART = "grouped_bar_chart"
    TIME_SERIES = "time_series"
    HISTOGRAM = "histogram"
    SCATTER_PLOT = "scatter_plot"
    NETWORK_GRAPH = "network_graph"
