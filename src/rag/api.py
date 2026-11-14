"""FastAPI service for the RAG pipeline."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from .pipeline import RAGPipeline

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000, examples=["How much does a BI-FAST transfer cost?"])
    debug: bool = False


class CitationOut(BaseModel):
    n: int
    source: str
    section: str
    chunk_id: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    grounded: bool
    citations: list[CitationOut]
    latency_ms: float
    retrieved: list[dict] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = RAGPipeline()
    docs = os.getenv("RAG_DOCS_DIR", "data/docs")
    if os.getenv("RAG_INGEST_ON_START", "true").lower() == "true" and os.path.isdir(docs):
        app.state.pipeline.ingest_directory(docs)
    yield


app = FastAPI(title="Production RAG Pipeline", version="0.1.0", lifespan=lifespan,
              description="Hybrid retrieval (dense + BM25/FTS, RRF), re-ranking and cited, grounded answers.")


@app.get("/healthz")
def healthz(request: Request) -> dict:
    p: RAGPipeline = request.app.state.pipeline
    return {"status": "ok", "chunks": p.store.count(), "embeddings": p.embeddings.name, "llm": p.llm.name}


@app.post("/v1/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request) -> AskResponse:
    ans = await run_in_threadpool(request.app.state.pipeline.ask, body.question)
    return AskResponse(
        answer=ans.answer,
        refused=ans.refused,
        grounded=ans.grounded,
        citations=[CitationOut(**c.__dict__) for c in ans.citations],
        latency_ms=ans.latency_ms,
        retrieved=[{"chunk_id": s.chunk.id, "section": s.chunk.section, "score": s.score,
                    "signals": {k: v for k, v in s.signals.items() if k != "ranks"}} for s in ans.retrieved]
        if body.debug else None,
    )


@app.post("/v1/ingest")
async def ingest(request: Request) -> dict:
    docs = os.getenv("RAG_DOCS_DIR", "data/docs")
    return await run_in_threadpool(request.app.state.pipeline.ingest_directory, docs)
