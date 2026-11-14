"""End-to-end RAG pipeline: ingest documents and answer questions with citations."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from .embeddings import Embeddings, get_embeddings
from .generation import LLM, REFUSAL, SYSTEM_PROMPT, build_prompt, get_llm, pack_context, validate_citations
from .ingest import load_directory
from .models import Answer, Citation
from .retrieval import HybridRetriever, RetrievalConfig
from .stores import InMemoryStore, PgVectorStore, VectorStore

log = logging.getLogger("rag")


def get_store(embeddings: Embeddings, kind: str | None = None) -> VectorStore:
    kind = (kind or os.getenv("RAG_STORE") or "memory").lower()
    if kind == "memory":
        return InMemoryStore()
    if kind == "pgvector":
        dsn = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")
        return PgVectorStore(dsn, dim=embeddings.dim, table=os.getenv("RAG_TABLE", f"rag_chunks_{embeddings.name}"))
    raise ValueError(f"Unknown RAG_STORE '{kind}'")


class RAGPipeline:
    def __init__(self, store: VectorStore | None = None, embeddings: Embeddings | None = None,
                 llm: LLM | None = None, retrieval: RetrievalConfig | None = None, max_context_tokens: int = 1200):
        self.embeddings = embeddings or get_embeddings()
        self.store = store or get_store(self.embeddings)
        self.llm = llm or get_llm()
        self.retriever = HybridRetriever(self.store, self.embeddings, config=retrieval)
        self.max_context_tokens = max_context_tokens

    # ----------------------------------------------------------------- ingest
    def ingest_directory(self, path: str | Path) -> dict[str, dict[str, int]]:
        report = {}
        for doc_id, chunks in load_directory(path).items():
            vectors = self.embeddings.embed_documents([c.text_for_embedding for c in chunks])
            for chunk, vector in zip(chunks, vectors):
                chunk.embedding = vector
            report[doc_id] = self.store.upsert_document(doc_id, chunks)
        log.info("ingested %s", report)
        return report

    # ----------------------------------------------------------------- ask
    def ask(self, question: str) -> Answer:
        start = time.perf_counter()
        retrieved = self.retriever.retrieve(question)
        best = retrieved[0].score if retrieved else 0.0

        if best < self.retriever.config.min_relevance:
            # Refuse *before* calling the model: cheaper, and no chance to hallucinate.
            return Answer(question, REFUSAL, True, [], retrieved, True, _ms(start), {})

        context = pack_context(retrieved, self.max_context_tokens)
        raw, usage = self.llm.complete(SYSTEM_PROMPT, build_prompt(question, context))
        if REFUSAL.split(".")[0] in raw:
            return Answer(question, REFUSAL, True, [], retrieved, True, _ms(start), usage)

        text, cited, grounded = validate_citations(raw, len(context))
        citations = [Citation(n, context[n - 1].chunk.id, context[n - 1].chunk.source, context[n - 1].chunk.section)
                     for n in cited]
        return Answer(question, text, False, citations, retrieved, grounded, _ms(start), usage)


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)
