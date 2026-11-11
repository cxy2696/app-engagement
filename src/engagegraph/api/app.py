"""FastAPI application for EngageGraph.

The service is stateless apart from the loaded artefact and an append-only
feedback log.  It is designed to run behind a reverse proxy: health endpoints,
request IDs, structured access logs, optional API-key protection for writes, and
Prometheus-style counters are included out of the box.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engagegraph import __version__
from engagegraph.artifacts import ModelRegistry, load_artifact
from engagegraph.recommend import RecommendationService


class FeedbackEvent(BaseModel):
    impression_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(min_length=1, max_length=256)
    item_id: str = Field(min_length=1, max_length=512)
    event: str = Field(pattern="^(impression|click|install|dwell|dismiss|view|purchase|like|skip)$")
    surface: str = Field(default="api", max_length=64)
    dwell_seconds: float | None = Field(default=None, ge=0, le=86_400)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchRecommendationRequest(BaseModel):
    user_ids: list[str] = Field(min_length=1, max_length=100)
    k: int = Field(default=10, ge=1, le=100)
    explain: bool = True


class _Counters:
    """Thread-safe-enough counters for a single-process demo deployment."""

    def __init__(self) -> None:
        self.requests: dict[tuple[str, int], int] = {}
        self.latency_sum = 0.0
        self.latency_count = 0
        self.feedback = 0
        self.started = time.time()

    def observe(self, path: str, status: int, latency: float) -> None:
        key = (path, status)
        self.requests[key] = self.requests.get(key, 0) + 1
        self.latency_sum += latency
        self.latency_count += 1

    def render(self) -> str:
        lines = [
            "# HELP engagegraph_requests_total HTTP requests by route and status.",
            "# TYPE engagegraph_requests_total counter",
        ]
        for (path, status), count in sorted(self.requests.items()):
            lines.append(f'engagegraph_requests_total{{route="{path}",status="{status}"}} {count}')
        lines += [
            "# HELP engagegraph_request_latency_seconds_mean Mean request latency.",
            "# TYPE engagegraph_request_latency_seconds_mean gauge",
            f"engagegraph_request_latency_seconds_mean "
            f"{self.latency_sum / max(self.latency_count, 1):.6f}",
            "# HELP engagegraph_feedback_events_total Feedback events accepted.",
            "# TYPE engagegraph_feedback_events_total counter",
            f"engagegraph_feedback_events_total {self.feedback}",
            "# HELP engagegraph_uptime_seconds Process uptime.",
            "# TYPE engagegraph_uptime_seconds gauge",
            f"engagegraph_uptime_seconds {time.time() - self.started:.1f}",
        ]
        return "\n".join(lines) + "\n"


def _resolve_artifact(explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    env = os.environ.get("ENGAGEGRAPH_ARTIFACT")
    if env:
        return Path(env)
    try:
        return ModelRegistry().resolve().path
    except FileNotFoundError:
        demo = Path("artifacts/demo")
        return demo if (demo / "manifest.json").exists() else None


def _hash_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def create_app(artifact_path: str | Path | None = None) -> FastAPI:
    counters = _Counters()
    state: dict[str, Any] = {"service": None, "error": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        path = _resolve_artifact(str(artifact_path) if artifact_path else None)
        if path is None:
            state["error"] = "no artefact configured"
        else:
            try:
                state["service"] = RecommendationService(load_artifact(path))
            except Exception as exc:  # pragma: no cover - startup failure path
                state["error"] = f"{type(exc).__name__}: {exc}"
        yield

    app = FastAPI(
        title="EngageGraph API",
        version=__version__,
        description=(
            "Domain-agnostic next-item recommendations and item-similarity search "
            "from a reproducible temporal benchmark. Domain labels come from the "
            "loaded artefact. See /docs for the OpenAPI schema."
        ),
        lifespan=lifespan,
    )
    app.state.counters = counters
    app.state.engagement = state

    origins = [
        origin.strip()
        for origin in os.environ.get("ENGAGEGRAPH_CORS_ORIGINS", "").split(",")
        if origin.strip()
    ]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    def service() -> RecommendationService:
        if state["service"] is None:
            raise HTTPException(status_code=503, detail=state["error"] or "model not loaded")
        return state["service"]

    @app.middleware("http")
    async def _observe(request: Request, call_next: Any) -> Response:
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:16])
        started = time.perf_counter()
        response = await call_next(request)
        latency = time.perf_counter() - started
        route = request.scope.get("route")
        path_label = getattr(route, "path", request.url.path)
        counters.observe(path_label, response.status_code, latency)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-ms"] = f"{latency * 1000:.2f}"
        return response

    @app.exception_handler(KeyError)
    async def _not_found(_: Request, exc: KeyError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": f"not found: {exc.args[0]}"})

    @app.exception_handler(ValueError)
    async def _bad_request(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/healthz", tags=["ops"])
    async def healthz() -> dict[str, Any]:
        loaded = state["service"] is not None
        return {
            "status": "ok" if loaded else "degraded",
            "model_loaded": loaded,
            "version": __version__,
            "error": state["error"],
        }

    @app.get("/readyz", tags=["ops"])
    async def readyz() -> Response:
        if state["service"] is None:
            return JSONResponse(status_code=503, content={"ready": False, "error": state["error"]})
        return JSONResponse(content={"ready": True})

    @app.get("/version", tags=["ops"])
    async def version() -> dict[str, str]:
        return {"name": "engagegraph", "version": __version__}

    @app.get("/metrics", tags=["ops"], response_class=PlainTextResponse)
    async def metrics() -> str:
        return counters.render()

    @app.get("/v1/meta", tags=["catalog"])
    async def meta() -> dict[str, Any]:
        return service().metadata()

    @app.get("/v1/popular", tags=["catalog"])
    async def popular(k: int = Query(default=10, ge=1, le=100)) -> dict[str, Any]:
        items = service().popular(k)
        return {"items": [item.as_dict() for item in items]}

    @app.get("/v1/items/similar", tags=["catalog"])
    @app.get("/v1/apps/similar", tags=["catalog"], include_in_schema=False)
    async def similar(
        item_id: str = Query(min_length=1, max_length=512),
        k: int = Query(default=10, ge=1, le=100),
    ) -> dict[str, Any]:
        items = service().similar(item_id, k)
        return {"item_id": item_id, "items": [item.as_dict() for item in items]}

    @app.get("/v1/users/{user_id}/recommendations", tags=["recommendations"])
    async def recommendations(
        user_id: str,
        k: int = Query(default=10, ge=1, le=100),
        explain: bool = True,
        diversify: bool = False,
    ) -> dict[str, Any]:
        return service().recommend(user_id, k=k, explain=explain, diversify=diversify).as_dict()

    @app.get("/v1/users/{user_id}", tags=["users"])
    async def profile(user_id: str) -> dict[str, Any]:
        return service().profile(user_id)

    @app.post("/v1/recommendations", tags=["recommendations"])
    async def batch(request: BatchRecommendationRequest) -> dict[str, Any]:
        svc = service()
        responses = [
            svc.recommend(user_id, k=request.k, explain=request.explain).as_dict()
            for user_id in request.user_ids
        ]
        return {"results": responses, "count": len(responses)}

    @app.post("/v1/feedback", tags=["feedback"], status_code=202)
    async def feedback(event: FeedbackEvent, request: Request) -> dict[str, Any]:
        required_key = os.environ.get("ENGAGEGRAPH_API_KEY")
        if required_key and request.headers.get("X-API-Key") != required_key:
            raise HTTPException(status_code=401, detail="invalid or missing API key")
        directory = Path(os.environ.get("ENGAGEGRAPH_FEEDBACK_DIR", "var/feedback"))
        directory.mkdir(parents=True, exist_ok=True)
        record = event.model_dump()
        record["received_at"] = datetime.now(timezone.utc).isoformat()
        record["server_user_hash"] = _hash_id(event.user_id)
        path = directory / f"{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        counters.feedback += 1
        return {"accepted": True, "stored": path.name}

    return app


app = create_app()
