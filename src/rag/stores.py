"""Vector stores with dense + lexical search.

* ``InMemoryStore``: numpy cosine + Okapi BM25. Great for tests and small corpora.
* ``PgVectorStore``: PostgreSQL + pgvector (HNSW, cosine) + full-text search (tsvector, GIN).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

import numpy as np

from .models import Chunk, ScoredChunk
from .text import tokens


class VectorStore(Protocol):
    def upsert_document(self, doc_id: str, chunks: list[Chunk]) -> dict[str, int]: ...

    def dense_search(self, embedding: list[float], k: int) -> list[ScoredChunk]: ...

    def lexical_search(self, query: str, k: int) -> list[ScoredChunk]: ...

    def count(self) -> int: ...


# --------------------------------------------------------------------------- in-memory
class InMemoryStore:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.chunks: dict[str, Chunk] = {}
        self.k1, self.b = k1, b
        self._dirty = True

    def upsert_document(self, doc_id: str, chunks: list[Chunk]) -> dict[str, int]:
        existing = {cid for cid, c in self.chunks.items() if c.doc_id == doc_id}
        incoming = {c.id for c in chunks}
        for cid in existing - incoming:
            del self.chunks[cid]
        for c in chunks:
            self.chunks[c.id] = c
        self._dirty = True
        return {"inserted": len(incoming - existing), "unchanged": len(incoming & existing),
                "deleted": len(existing - incoming)}

    def count(self) -> int:
        return len(self.chunks)

    def _index(self) -> None:
        if not self._dirty:
            return
        self._ids = list(self.chunks)
        self._matrix = np.array([self.chunks[i].embedding for i in self._ids], dtype=np.float32) \
            if self._ids else np.zeros((0, 1))
        self._tf = [Counter(tokens(self.chunks[i].text_for_embedding)) for i in self._ids]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avgdl = (sum(self._len) / len(self._len)) if self._len else 1.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(self._ids)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self._dirty = False

    def dense_search(self, embedding: list[float], k: int) -> list[ScoredChunk]:
        self._index()
        if not self._ids:
            return []
        scores = self._matrix @ np.asarray(embedding, dtype=np.float32)
        top = np.argsort(-scores)[:k]
        return [ScoredChunk(self.chunks[self._ids[i]], float(scores[i]), {"dense": float(scores[i])}) for i in top]

    def lexical_search(self, query: str, k: int) -> list[ScoredChunk]:
        self._index()
        q = set(tokens(query))
        results = []
        for i, tf in enumerate(self._tf):
            score = 0.0
            for t in q & tf.keys():
                f = tf[t]
                score += self._idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self._len[i] / self._avgdl))
            if score > 0:
                results.append((score, i))
        results.sort(reverse=True)
        return [ScoredChunk(self.chunks[self._ids[i]], s, {"bm25": s}) for s, i in results[:k]]


# --------------------------------------------------------------------------- pgvector
SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS {table} (
    id           text PRIMARY KEY,
    doc_id       text NOT NULL,
    source       text NOT NULL,
    section      text NOT NULL,
    content      text NOT NULL,
    content_hash text NOT NULL,
    position     int  NOT NULL,
    embedding    vector({dim}) NOT NULL,
    tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', section || ' ' || content)) STORED,
    updated_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS {table}_embedding_hnsw ON {table} USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS {table}_tsv_gin ON {table} USING gin (tsv);
CREATE INDEX IF NOT EXISTS {table}_doc_id ON {table} (doc_id);
"""

_COLS = "id, doc_id, source, section, content, content_hash, position"


class PgVectorStore:
    def __init__(self, dsn: str, dim: int, table: str = "rag_chunks"):
        import psycopg
        from pgvector.psycopg import register_vector

        if not re.fullmatch(r"[a-z_][a-z0-9_]*", table):
            raise ValueError("invalid table name")
        self.table, self.dim = table, dim
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self.conn.execute(SCHEMA.format(table=table, dim=dim))

    def _row(self, row, score: float, signal: str) -> ScoredChunk:
        chunk = Chunk(*row[:7])
        return ScoredChunk(chunk, float(score), {signal: float(score)})

    def upsert_document(self, doc_id: str, chunks: list[Chunk]) -> dict[str, int]:
        with self.conn.transaction():
            existing = {r[0] for r in self.conn.execute(f"SELECT id FROM {self.table} WHERE doc_id = %s", (doc_id,))}
            incoming = {c.id for c in chunks}
            stale = list(existing - incoming)
            if stale:
                self.conn.execute(f"DELETE FROM {self.table} WHERE id = ANY(%s)", (stale,))
            new = [c for c in chunks if c.id not in existing]
            with self.conn.cursor() as cur:
                cur.executemany(
                    f"INSERT INTO {self.table} ({_COLS}, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (id) DO NOTHING",
                    [(c.id, c.doc_id, c.source, c.section, c.content, c.content_hash, c.position,
                      np.asarray(c.embedding, dtype=np.float32)) for c in new],
                )
        return {"inserted": len(new), "unchanged": len(incoming & existing), "deleted": len(stale)}

    def count(self) -> int:
        return self.conn.execute(f"SELECT count(*) FROM {self.table}").fetchone()[0]

    def dense_search(self, embedding: list[float], k: int) -> list[ScoredChunk]:
        vec = np.asarray(embedding, dtype=np.float32)
        rows = self.conn.execute(
            f"SELECT {_COLS}, 1 - (embedding <=> %s) AS score FROM {self.table} ORDER BY embedding <=> %s LIMIT %s",
            (vec, vec, k),
        ).fetchall()
        return [self._row(r, r[7], "dense") for r in rows]

    def lexical_search(self, query: str, k: int) -> list[ScoredChunk]:
        words = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 1]
        if not words:
            return []
        # OR the terms: natural-language questions rarely contain *every* word of a passage.
        tsquery = " | ".join(dict.fromkeys(words))
        rows = self.conn.execute(
            f"SELECT {_COLS}, ts_rank_cd(tsv, q) AS score FROM {self.table}, to_tsquery('english', %s) q "
            "WHERE tsv @@ q ORDER BY score DESC LIMIT %s",
            (tsquery, k),
        ).fetchall()
        return [self._row(r, r[7], "fts") for r in rows]

    def clear(self) -> None:
        self.conn.execute(f"TRUNCATE {self.table}")
