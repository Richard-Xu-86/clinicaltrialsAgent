"""Run the example requests and write each response to examples/<name>.json. This part is only for testing and proof of concept

    python scripts/run_examples.py                 # live API (+ LLM planner if OPENAI_API_KEY is set)
    python scripts/run_examples.py --offline       # replay recorded API responses in examples/recordings
    python scripts/run_examples.py --print-urls    # list the first-page API URLs each example needs

Note: outputs contain a generated_at timestamp and, when live, whatever the
registry holds today, so re-running will change counts slightly over time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ctviz.config import Settings  # noqa: E402
from ctviz.ctgov.client import build_url  # noqa: E402
from ctviz.ctgov.query import build_search_params  # noqa: E402
from ctviz.pipeline import VisualizationService, expand_cohorts, required_fields  # noqa: E402
from ctviz.planner import plan_request  # noqa: E402
from ctviz.schemas import VisualizeRequest  # noqa: E402

EXAMPLES = ROOT / "examples"
RECORDINGS = EXAMPLES / "recordings"


def load_examples() -> list[dict]:
    return json.loads((EXAMPLES / "requests.json").read_text())


def print_urls(settings: Settings) -> None:
    """First-page URL for each cohort, with the pageToken slot marked for recorders."""
    urls = [build_url(settings.ctgov_base_url, "version", {})]
    for example in load_examples():
        request = VisualizeRequest(**example["request"])
        plan = plan_request(request, settings).plan
        cohorts = expand_cohorts(plan)
        fields = required_fields(plan, cohorts)
        for cohort in cohorts:
            params = build_search_params(cohort.filters, fields)
            base = {**params, "pageSize": settings.ctgov_page_size, "countTotal": "true", "format": "json"}
            urls.append(build_url(settings.ctgov_base_url, "studies", {**base, "pageToken": "__TOKEN__"}))
    print(json.dumps(urls, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="replay examples/recordings only")
    parser.add_argument("--print-urls", action="store_true")
    parser.add_argument("--only", help="run a single example by name")
    args = parser.parse_args()

    overrides = {"cache_dir": RECORDINGS}
    if args.offline:
        overrides["ctgov_offline"] = True
        # Replay only: the recordings match the rule-based plans, and an LLM might phrase
        # a filter differently ("Huntington disease"), which would miss the recording.
        overrides["openai_api_key"] = None
    settings = Settings(**overrides)

    if args.print_urls:
        print_urls(settings)
        return

    service = VisualizationService(settings)
    for example in load_examples():
        if args.only and example["name"] != args.only:
            continue
        request = VisualizeRequest(**example["request"])
        response = service.visualize(request)
        out = EXAMPLES / f"{example['name']}.json"
        payload = {"request": request.model_dump(mode="json", exclude_defaults=True),
                   "response": response.model_dump(mode="json")}
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        viz = response.visualization
        size = len(viz.data.nodes) if viz.type.value == "network_graph" else len(viz.data)
        print(f"{example['name']}: {viz.type.value}, {size} data items, planner={response.meta.planner.method}")


if __name__ == "__main__":
    main()
