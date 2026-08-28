"""FastAPI service.

One endpoint that matters: POST /ask. Two that make it operable: /health, which
reports whether the corpus and credentials are actually usable rather than just
whether the process is alive, and GET /traces/{id}, so a returned trace id can
be inspected.

The retrievers are built once at startup, not per request. Rebuilding the BM25
index on every call would dominate the latency figures reported in the README
and would be an odd thing to ship.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..config import DEFAULT_PROFILE, settings
from ..llm import have_credentials
from ..retrieval import build_retrievers

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Built once. A failure here should stop the service starting rather than
    # surface as a 500 on the first question.
    try:
        _state["retrievers"] = build_retrievers(DEFAULT_PROFILE)
        _state["ready"] = True
        _state["error"] = ""
    except Exception as exc:  # noqa: BLE001
        _state["ready"] = False
        _state["error"] = str(exc)
    yield
    _state.clear()


app = FastAPI(
    title="aml-guidance-agent",
    description=(
        "Retrieval-augmented agent over public AML regulatory guidance. "
        "Answers are validated against retrieved passages before they are "
        "returned, and the service refuses when the corpus does not support "
        "an answer."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    retriever: str = Field(
        default="hybrid",
        description="bm25 | dense | hybrid. Defaults to hybrid.",
    )


class Citation(BaseModel):
    chunk_id: int
    title: str
    publisher: str
    page: int | None = None
    source_url: str
    section_heading: str | None = None


class AskResponse(BaseModel):
    outcome: str = Field(..., description="answered | refused")
    answer: str | None = None
    citations: list[Citation] = []
    refusal_reason: str | None = None
    trace_id: str
    attempts: int
    searches: int
    latency_ms: float


@app.get("/health")
def health() -> dict[str, Any]:
    """Readiness, not liveness.

    Reports what a caller actually needs to know: is the corpus loaded, and can
    the LLM-backed steps run. A service that returns 200 while the corpus is
    empty is lying about being healthy.
    """
    corpus: dict[str, Any] = {"ready": bool(_state.get("ready")), "error": _state.get("error", "")}

    if _state.get("ready"):
        try:
            from ..db import connect, corpus_stats

            with connect() as conn:
                stats = corpus_stats(conn)
            corpus["documents"] = stats["documents"]
            corpus["chunks"] = sum(p["chunks"] for p in stats["profiles"])
        except Exception as exc:  # noqa: BLE001
            corpus["ready"] = False
            corpus["error"] = str(exc)

    return {
        "status": "ok" if corpus["ready"] else "degraded",
        "corpus": corpus,
        "credentials": have_credentials(),
        "profile": DEFAULT_PROFILE,
        "model": settings.anthropic_model,
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    if not _state.get("ready"):
        raise HTTPException(
            status_code=503,
            detail=(
                "Corpus is not loaded. Run `make ingest` (or `.\\run.ps1 ingest`). "
                f"Startup error: {_state.get('error', 'unknown')}"
            ),
        )

    retrievers = _state["retrievers"]
    if request.retriever not in retrievers:
        raise HTTPException(
            status_code=400,
            detail=f"unknown retriever {request.retriever!r}; choose from {sorted(retrievers)}",
        )

    # Imported here so that the module, and therefore /health, still works on a
    # machine with no credentials configured.
    from ..agent.pipeline import Pipeline, write_trace
    from ..llm import MissingCredentials

    try:
        result = Pipeline(retrievers[request.retriever]).ask(request.question)
    except MissingCredentials as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    write_trace(result)

    if result.outcome == "error":
        raise HTTPException(status_code=502, detail=result.refusal_reason or "agent error")

    return AskResponse(
        outcome=result.outcome,
        answer=result.summary or None,
        citations=[Citation(**c) for c in result.citations],
        refusal_reason=result.refusal_reason or None,
        trace_id=result.trace_id,
        attempts=result.attempts,
        searches=result.total_searches,
        latency_ms=round(result.latency_ms, 1),
    )


@app.get("/traces/{trace_id}")
def get_trace(trace_id: str) -> dict[str, Any]:
    """Return a stored trace.

    The trace includes drafts that validation rejected, which is the point:
    a citation-validating system that cannot show you a rejection has not
    demonstrated anything.
    """
    if not trace_id.isalnum():
        raise HTTPException(status_code=400, detail="invalid trace id")

    path = settings.traces_dir / f"{trace_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"no trace {trace_id}")

    return json.loads(path.read_text(encoding="utf-8"))
