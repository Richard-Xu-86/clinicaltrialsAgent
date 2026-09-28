from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ctviz.config import Settings

ROOT = Path(__file__).resolve().parents[1]
RECORDINGS = ROOT / "examples" / "recordings"


def make_study(
    nct_id: str,
    *,
    title: str | None = None,
    phases: list[str] | None = None,
    start: str | None = None,
    start_type: str | None = None,
    completion: str | None = None,
    status: str | None = None,
    sponsor: str | None = None,
    sponsor_class: str | None = None,
    interventions: list[dict[str, Any]] | None = None,
    conditions: list[str] | None = None,
    sites: list[tuple[str, str | None]] | None = None,
    enrollment: int | None = None,
    study_type: str | None = None,
) -> dict[str, Any]:
    """Build a study dict shaped exactly like a ClinicalTrials.gov v2 response record."""
    ps: dict[str, Any] = {"identificationModule": {"nctId": nct_id}}
    if title:
        ps["identificationModule"]["briefTitle"] = title
    design: dict[str, Any] = {}
    if phases is not None:
        design["phases"] = phases
    if enrollment is not None:
        design["enrollmentInfo"] = {"count": enrollment}
    if study_type:
        design["studyType"] = study_type
    if design:
        ps["designModule"] = design
    status_mod: dict[str, Any] = {}
    if status:
        status_mod["overallStatus"] = status
    if start:
        status_mod["startDateStruct"] = {"date": start, **({"type": start_type} if start_type else {})}
    if completion:
        status_mod["primaryCompletionDateStruct"] = {"date": completion}
    if status_mod:
        ps["statusModule"] = status_mod
    if sponsor or sponsor_class:
        ps["sponsorCollaboratorsModule"] = {"leadSponsor": {k: v for k, v in
                                                            (("name", sponsor), ("class", sponsor_class)) if v}}
    if interventions is not None:
        ps["armsInterventionsModule"] = {"interventions": interventions}
    if conditions is not None:
        ps["conditionsModule"] = {"conditions": conditions}
    if sites is not None:
        ps["contactsLocationsModule"] = {"locations": [
            {"country": c, **({"status": s} if s else {})} for c, s in sites
        ]}
    return {"protocolSection": ps}


@pytest.fixture
def offline_settings(tmp_path) -> Settings:
    return Settings(cache_dir=RECORDINGS, ctgov_offline=True, openai_api_key=None, ctgov_min_interval_s=0)


@pytest.fixture
def fast_settings(tmp_path) -> Settings:
    """Live-mode settings with no spacing/backoff delays, for mocked HTTP tests."""
    return Settings(cache_dir=tmp_path / "cache", openai_api_key=None, ctgov_min_interval_s=0,
                    ctgov_max_retries=2)
