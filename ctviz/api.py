"""HTTP layer. Kept thin: validation, error mapping, and nothing else."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from .ctgov.client import OfflineCacheMiss
from .pipeline import UpstreamError, VisualizationService
from .schemas import QueryPlan, VisualizeRequest, VisualizeResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="ClinicalTrials.gov Query-to-Visualization Agent",
    version="1.0.0",
    description="Turns questions about clinical trials into renderer-ready visualization specs, "
                "backed by the ClinicalTrials.gov v2 API, with per-datum citations.",
)


@lru_cache
def get_service() -> VisualizationService:
    return VisualizationService()


STATIC = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def demo() -> FileResponse:
    """Minimal demo client (Vega-Lite + d3) that renders the spec returned by /v1/visualize."""
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/visualize", response_model=VisualizeResponse, response_model_exclude_none=False)
def visualize(request: VisualizeRequest, service: Annotated[VisualizationService, Depends(get_service)]):
    # Declared sync on purpose: the pipeline does blocking I/O and FastAPI runs it in a threadpool.
    try:
        return service.visualize(request)
    except OfflineCacheMiss as exc:
        raise HTTPException(status_code=503, detail=f"offline mode: {exc}") from exc
    except UpstreamError as exc:
        raise HTTPException(status_code=502, detail=f"ClinicalTrials.gov request failed: {exc}") from exc


@app.get("/v1/schema")
def schemas() -> JSONResponse:
    """JSON Schemas for the request, the plan and the response (also in /docs)."""
    return JSONResponse({
        "request": VisualizeRequest.model_json_schema(),
        "plan": QueryPlan.model_json_schema(),
        "response": VisualizeResponse.model_json_schema(),
    })
