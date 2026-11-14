from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    id: str
    doc_id: str
    source: str
    section: str
    content: str
    content_hash: str
    position: int = 0
    embedding: list[float] | None = field(default=None, repr=False)

    @property
    def text_for_embedding(self) -> str:
        # Prepending the section title gives short chunks the context they need to be found.
        return f"{self.section}\n{self.content}"


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    signals: dict[str, Any] = field(default_factory=dict)


@dataclass
class Citation:
    n: int
    chunk_id: str
    source: str
    section: str


@dataclass
class Answer:
    question: str
    answer: str
    refused: bool
    citations: list[Citation]
    retrieved: list[ScoredChunk]
    grounded: bool
    latency_ms: float
    usage: dict[str, int] = field(default_factory=dict)
