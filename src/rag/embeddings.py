"""Embedding providers.

* ``HashEmbeddings``: deterministic feature-hashing (words + char trigrams). Zero cost, offline;
  used for tests and CI. It captures lexical similarity, not true semantics.
* ``GeminiEmbeddings``: Google ``gemini-embedding-001`` via REST (free tier available).
"""

from __future__ import annotations

import hashlib
import itertools
import os
from typing import Protocol

import httpx
import numpy as np

from .text import tokens


class Embeddings(Protocol):
    dim: int
    name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


class HashEmbeddings:
    name = "hash"

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _features(self, text: str) -> list[tuple[str, float]]:
        words = tokens(text)
        feats = [(f"w:{w}", 1.0) for w in words]
        feats += [(f"b:{a}_{b}", 0.7) for a, b in itertools.pairwise(words)]
        for w in words:
            padded = f"#{w}#"
            feats += [(f"c:{padded[i:i + 3]}", 0.3) for i in range(len(padded) - 2)]
        return feats

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for feat, weight in self._features(text):
            h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
            vec[h % self.dim] += weight if (h >> 63) & 1 else -weight
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return _normalise(np.stack([self._embed(t) for t in texts])).tolist() if texts else []

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class GeminiEmbeddings:
    name = "gemini"
    url = "https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"

    def __init__(self, model: str = "gemini-embedding-001", dim: int = 768, *, api_key: str | None = None,
                 client: httpx.Client | None = None, batch_size: int = 50):
        self.model, self.dim, self.batch_size = model, dim, batch_size
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=60)

    def _call(self, texts: list[str], task: str) -> list[list[float]]:
        key = self._api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Set GOOGLE_API_KEY (or GEMINI_API_KEY) to use Gemini embeddings")
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            body = {"requests": [{
                "model": f"models/{self.model}",
                "content": {"parts": [{"text": t}]},
                "taskType": task,
                "outputDimensionality": self.dim,
            } for t in batch]}
            resp = self._client.post(self.url.format(model=self.model), json=body, headers={"x-goog-api-key": key})
            resp.raise_for_status()
            vectors = np.array([e["values"] for e in resp.json()["embeddings"]], dtype=np.float32)
            out.extend(_normalise(vectors).tolist())  # truncated dims must be re-normalised for cosine
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._call(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._call([text], "RETRIEVAL_QUERY")[0]


def get_embeddings(kind: str | None = None) -> Embeddings:
    kind = (kind or os.getenv("RAG_EMBEDDINGS") or "hash").lower()
    if kind == "hash":
        return HashEmbeddings(int(os.getenv("RAG_EMBED_DIM", "384")))
    if kind == "gemini":
        return GeminiEmbeddings(dim=int(os.getenv("RAG_EMBED_DIM", "768")))
    raise ValueError(f"Unknown RAG_EMBEDDINGS '{kind}'")
