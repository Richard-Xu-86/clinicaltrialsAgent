"""Planning: question (+ optional structured fields) -> validated, executable QueryPlan."""

from __future__ import annotations

import logging

from ..config import Settings
from ..schemas import PlannerInfo, VisualizeRequest
from .llm import OpenAIPlanner, PlannerError
from .resolve import resolve_plan
from .rules import RuleBasedPlanner

log = logging.getLogger(__name__)

__all__ = ["plan_request", "OpenAIPlanner", "RuleBasedPlanner", "PlannerError"]


def plan_request(request: VisualizeRequest, settings: Settings, llm: OpenAIPlanner | None = None) -> PlannerInfo:
    """Try the LLM planner; fall back to rules if it is unconfigured or fails."""
    fallback_reason = None
    raw_plan = None
    method, model = "rule_based", None

    if llm is None:
        try:
            llm = OpenAIPlanner(settings)
        except PlannerError as exc:
            fallback_reason = str(exc)

    if llm is not None:
        try:
            raw_plan = llm.plan(request)
            method, model = "llm", llm.model
        except PlannerError as exc:
            log.warning("LLM planner failed, using rules: %s", exc)
            fallback_reason = str(exc)

    if raw_plan is None:
        raw_plan = RuleBasedPlanner().plan(request.query, explicit_drug=request.drug_name)

    resolution = resolve_plan(raw_plan, request)
    return PlannerInfo(
        method=method,
        model=model,
        fallback_reason=fallback_reason,
        plan=resolution.plan,
        overrides=resolution.overrides,
        adjustments=resolution.adjustments,
    )
