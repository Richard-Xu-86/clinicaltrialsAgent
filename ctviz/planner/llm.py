"""LLM planner: natural-language question -> QueryPlan, via OpenAI function calling.

The model sees the question and the plan schema, and must answer by calling a
single function (tool_choice is forced) whose parameters schema *is* the QueryPlan
JSON schema. We then validate the arguments with pydantic. If validation fails we
send the error back once and let the model correct itself; after that the caller
falls back to rules.

We use plain function calling rather than strict structured outputs: strict mode
requires every field to be required with no defaults, which would make the plan
schema worse for no gain, since pydantic validates the arguments either way.

The model never sees trial data and never produces numbers for the chart.
"""

from __future__ import annotations

import copy
import json
import logging
from datetime import date
from typing import Any

from pydantic import ValidationError

from ..config import Settings
from ..schemas import QueryPlan, VisualizeRequest

log = logging.getLogger(__name__)

TOOL_NAME = "submit_query_plan"


class PlannerError(RuntimeError):
    pass


SYSTEM_PROMPT = """\
You translate questions about clinical trials into a query plan for a service that \
queries ClinicalTrials.gov and draws a chart. You never answer the question yourself \
and never invent numbers; you only describe what to fetch and how to group it.

Choose `analysis`:
- trend: counts over time ("per year", "since 2015", "over time"). group_by = start_year. \
Use series_by for "... by phase over time".
- distribution: counts per category of one dimension ("by phase", "which countries", \
"most common intervention types").
- comparison: the same distribution for 2-5 cohorts ("drug A vs drug B", "across two \
conditions"). Put the cohorts in `compare`, not in filters.
- network: relationships between entities ("network of sponsors and drugs", \
"drugs that co-occur"). Same source and target = co-occurrence within trials.
- histogram: distribution of a numeric field (enrollment size, trial duration).
- scatter: one point per trial, two numeric fields.

Filters narrow the trial set. Only set a filter the question (or the caller's \
structured fields) actually implies. Geographic questions ("which countries") group by \
country; they do not filter by country. "Recruiting" means statuses=[RECRUITING]. \
Use generic drug names when you know them (Keytruda -> pembrolizumab) and plain \
condition names ("non-small cell lung cancer"). Relative dates are relative to {today}.

If the question refers to "this drug" or similar, the caller's structured fields say \
which one. Keep `rationale` to one sentence."""


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve pydantic's $defs/$ref into a self-contained schema.

    Some tool-schema consumers handle $ref poorly; a flat schema is also easier to
    read when debugging what the model was shown.
    """
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                target = defs[node["$ref"].split("/")[-1]]
                merged = {**walk(copy.deepcopy(target)), **{k: v for k, v in node.items() if k != "$ref"}}
                return merged
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


def plan_tool_schema() -> dict[str, Any]:
    return _inline_refs(QueryPlan.model_json_schema())


def _explicit_fields(request: VisualizeRequest) -> dict[str, Any]:
    fields = request.model_dump(
        exclude={"query", "max_records", "max_citations_per_datum"}, exclude_none=True, mode="json"
    )
    return fields


class OpenAIPlanner:
    method = "llm"

    def __init__(self, settings: Settings, client: Any = None):
        if client is None:
            if not settings.openai_api_key:
                raise PlannerError("OPENAI_API_KEY is not set")
            try:
                import openai  # imported lazily so the service runs without the SDK configured
            except ImportError:
                raise PlannerError("the 'openai' package is not installed in this environment") from None

            # Our own loop already retries once with feedback; keep the SDK's transport retries
            # low so an OpenAI outage falls back to the rule-based planner quickly.
            client = openai.OpenAI(api_key=settings.openai_api_key, timeout=settings.planner_timeout_s,
                                   max_retries=1)
        self.client = client
        self.model = settings.openai_model
        self.max_attempts = settings.planner_max_attempts
        self._tool = {
            "type": "function",
            "function": {
                "name": TOOL_NAME,
                "description": "Submit the query plan for the user's question.",
                "parameters": plan_tool_schema(),
            },
        }

    def plan(self, request: VisualizeRequest) -> QueryPlan:
        user_text = f"Question: {request.query}"
        explicit = _explicit_fields(request)
        if explicit:
            user_text += f"\nCaller-supplied structured fields (authoritative): {json.dumps(explicit)}"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT.format(today=date.today().isoformat())},
            {"role": "user", "content": user_text},
        ]

        last_error = "no attempt made"
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    # No temperature: GPT-5-family reasoning models only accept the default.
                    max_completion_tokens=4096,
                    messages=messages,
                    tools=[self._tool],
                    tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
                )
            except Exception as exc:  # network, auth, rate limit, overload: all mean "use the fallback"
                raise PlannerError(f"OpenAI call failed: {type(exc).__name__}: {exc}") from exc

            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None) or []
            if not tool_calls:
                last_error = "model did not call the planning function"
                messages += [_echo(message), {"role": "user", "content": f"Call the {TOOL_NAME} function."}]
                continue

            call = tool_calls[0]
            try:
                return QueryPlan.model_validate(json.loads(call.function.arguments))
            except (ValidationError, json.JSONDecodeError) as exc:
                last_error = _compact_errors(exc) if isinstance(exc, ValidationError) else f"invalid JSON: {exc}"
                log.info("plan attempt %d invalid: %s", attempt, last_error)
                messages += [
                    _echo(message),
                    {"role": "tool", "tool_call_id": call.id,
                     "content": f"The plan did not validate: {last_error}. Submit a corrected plan."},
                ]

        raise PlannerError(f"no valid plan after {self.max_attempts} attempts: {last_error}")


def _echo(message: Any) -> dict[str, Any]:
    """Minimal assistant message to send back. Built by hand rather than dumping the SDK
    object, whose extra fields (refusal, annotations, ...) are not valid request input."""
    out: dict[str, Any] = {"role": "assistant", "content": getattr(message, "content", None) or ""}
    calls = getattr(message, "tool_calls", None) or []
    if calls:
        out["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in calls
        ]
    return out


def _compact_errors(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:8])
