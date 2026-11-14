"""Hybrid retrieval: dense + lexical candidates -> Reciprocal Rank Fusion -> re-ranking."""

from __future__ import annotations

from dataclasses import dataclass

from .embeddings import Embeddings
from .models import ScoredChunk
from .stores import VectorStore
from .text import tokens


def reciprocal_rank_fusion(result_lists: list[list[ScoredChunk]], k: int = 60) -> list[ScoredChunk]:
    """Fuse ranked lists without needing comparable scores (cosine vs BM25 live on different scales)."""
    fused: dict[str, ScoredChunk] = {}
    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            entry = fused.setdefault(item.chunk.id, ScoredChunk(item.chunk, 0.0, {}))
            entry.score += 1.0 / (k + rank)
            entry.signals.update(item.signals)
            entry.signals.setdefault("ranks", []).append(rank)
    return sorted(fused.values(), key=lambda s: -s.score)


class OverlapReranker:
    """Cheap, deterministic re-ranker: how much of the question does a chunk actually cover?

    Combines query-term coverage (with IDF-like weighting by term rarity in the candidate set),
    section-title matches and the fused retrieval score. Swap for a cross-encoder or an LLM
    re-ranker in production; the interface is the same.
    """

    name = "overlap"

    def rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        q_terms = set(tokens(query))
        if not q_terms or not candidates:
            return candidates
        doc_terms = [set(tokens(c.chunk.content)) for c in candidates]
        sec_terms = [set(tokens(c.chunk.section)) for c in candidates]
        df = {t: sum(t in d or t in s for d, s in zip(doc_terms, sec_terms)) for t in q_terms}
        weight = {t: 1.0 / (1 + df[t]) for t in q_terms}
        total = sum(weight.values())
        max_fused = max(c.score for c in candidates) or 1.0
        out = []
        for c, d, s in zip(candidates, doc_terms, sec_terms):
            coverage = sum(weight[t] for t in q_terms if t in d or t in s) / total
            title = sum(weight[t] for t in q_terms if t in s) / total
            relevance = round(0.65 * coverage + 0.2 * title + 0.15 * (c.score / max_fused), 4)
            out.append(ScoredChunk(c.chunk, relevance, {**c.signals, "coverage": round(coverage, 3), "rerank": relevance}))
        return sorted(out, key=lambda s: -s.score)


@dataclass
class RetrievalConfig:
    dense_k: int = 12
    lexical_k: int = 12
    top_n: int = 4
    min_relevance: float = 0.28   # below this the pipeline refuses instead of guessing (calibrated with `rag eval`)


class HybridRetriever:
    def __init__(self, store: VectorStore, embeddings: Embeddings, reranker=None, config: RetrievalConfig | None = None):
        self.store, self.embeddings = store, embeddings
        self.reranker = reranker or OverlapReranker()
        self.config = config or RetrievalConfig()

    def retrieve(self, query: str) -> list[ScoredChunk]:
        cfg = self.config
        dense = self.store.dense_search(self.embeddings.embed_query(query), cfg.dense_k)
        lexical = self.store.lexical_search(query, cfg.lexical_k)
        fused = reciprocal_rank_fusion([dense, lexical])
        return self.reranker.rerank(query, fused)[: cfg.top_n]
