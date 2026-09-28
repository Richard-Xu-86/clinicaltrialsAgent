"""Raw ClinicalTrials.gov study JSON -> TrialRecord.

The API omits modules and fields freely (observational studies have no phases,
~7% of pembrolizumab trials list no locations, some dates are month-only). The
normalizer never raises on missing data; it leaves the attribute empty and the
aggregation layer decides how to bucket "not reported".
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .records import Intervention, PartialDate, Site, Sourced, TrialRecord

log = logging.getLogger(__name__)

PS = "protocolSection"
_DATE_RE = re.compile(r"^(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?$")


def _get(obj: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def _str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def parse_partial_date(raw: str | None) -> PartialDate | None:
    if not raw:
        return None
    match = _DATE_RE.match(raw.strip())
    if not match:
        return None
    year, month, day = (int(g) if g else None for g in match.groups())
    return PartialDate(year=year, month=month, day=day)


def phase_bucket(phases: list[str]) -> str:
    """Multi-phase trials (e.g. PHASE1 + PHASE2) get their own combined bucket.

    Counting such a trial once under "Phase 1/2" keeps bar totals equal to the
    number of distinct trials, which is what people expect a phase chart to sum to.
    """
    order = ["EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4", "NA"]
    known = sorted({p for p in phases if p in order}, key=order.index)
    return "/".join(known) if known else "/".join(sorted(set(phases)))


def normalize_study(study: dict[str, Any]) -> TrialRecord | None:
    ps = study.get(PS) if isinstance(study, dict) else None
    nct_id = _str(_get(ps, "identificationModule", "nctId"))
    if not nct_id:
        log.warning("skipping study without nctId")
        return None

    rec = TrialRecord(nct_id=nct_id, title=_str(_get(ps, "identificationModule", "briefTitle")))

    # --- design
    phases = _get(ps, "designModule", "phases")
    if isinstance(phases, list) and phases:
        phases = [p for p in phases if isinstance(p, str)]
        if phases:
            rec.phase = Sourced(phase_bucket(phases), f"{PS}.designModule.phases", ", ".join(phases))

    study_type = _str(_get(ps, "designModule", "studyType"))
    if study_type:
        rec.study_type = Sourced(study_type, f"{PS}.designModule.studyType", study_type)

    enrollment = _get(ps, "designModule", "enrollmentInfo", "count")
    if isinstance(enrollment, int) and not isinstance(enrollment, bool) and enrollment >= 0:
        rec.enrollment = Sourced(enrollment, f"{PS}.designModule.enrollmentInfo.count", str(enrollment))

    # --- status and dates
    status = _str(_get(ps, "statusModule", "overallStatus"))
    if status:
        rec.status = Sourced(status, f"{PS}.statusModule.overallStatus", status)

    for attr, key in (("start", "startDateStruct"), ("primary_completion", "primaryCompletionDateStruct")):
        raw = _str(_get(ps, "statusModule", key, "date"))
        parsed = parse_partial_date(raw)
        if parsed:
            setattr(rec, attr, Sourced(parsed, f"{PS}.statusModule.{key}.date", raw))

    # --- sponsor
    lead = _get(ps, "sponsorCollaboratorsModule", "leadSponsor") or {}
    name = _str(lead.get("name")) if isinstance(lead, dict) else None
    if name:
        rec.lead_sponsor = Sourced(name, f"{PS}.sponsorCollaboratorsModule.leadSponsor.name", name)
    klass = _str(lead.get("class")) if isinstance(lead, dict) else None
    if klass:
        rec.sponsor_class = Sourced(klass, f"{PS}.sponsorCollaboratorsModule.leadSponsor.class", klass)

    # --- interventions
    for i, item in enumerate(_get(ps, "armsInterventionsModule", "interventions") or []):
        iname = _str(item.get("name")) if isinstance(item, dict) else None
        if not iname:
            continue
        others = tuple(
            Sourced(o.strip(), f"{PS}.armsInterventionsModule.interventions[{i}].otherNames[{j}]", o.strip())
            for j, o in enumerate(item.get("otherNames") or [])
            if isinstance(o, str) and o.strip()
        )
        rec.interventions.append(
            Intervention(
                name=Sourced(iname, f"{PS}.armsInterventionsModule.interventions[{i}].name", iname),
                type=_str(item.get("type")),
                other_names=others,
            )
        )

    # --- conditions
    for i, cond in enumerate(_get(ps, "conditionsModule", "conditions") or []):
        text = _str(cond)
        if text:
            rec.conditions.append(Sourced(text, f"{PS}.conditionsModule.conditions[{i}]", text))

    # --- sites (country + site status)
    for i, loc in enumerate(_get(ps, "contactsLocationsModule", "locations") or []):
        if not isinstance(loc, dict):
            continue
        country = _str(loc.get("country"))
        if country:
            rec.sites.append(
                Site(
                    country=Sourced(country, f"{PS}.contactsLocationsModule.locations[{i}].country", country),
                    status=_str(loc.get("status")),
                )
            )

    return rec


def normalize_studies(studies: list[dict[str, Any]]) -> list[TrialRecord]:
    """Normalize and de-duplicate by NCT ID (defensive; pages should not overlap)."""
    seen: set[str] = set()
    out: list[TrialRecord] = []
    for study in studies:
        rec = normalize_study(study)
        if rec and rec.nct_id not in seen:
            seen.add(rec.nct_id)
            out.append(rec)
    return out
