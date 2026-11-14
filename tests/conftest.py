from pathlib import Path

import pytest

from rag.embeddings import HashEmbeddings
from rag.generation import ExtractiveLLM
from rag.pipeline import RAGPipeline
from rag.stores import InMemoryStore

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "data" / "docs"


@pytest.fixture
def pipeline() -> RAGPipeline:
    p = RAGPipeline(store=InMemoryStore(), embeddings=HashEmbeddings(), llm=ExtractiveLLM())
    p.ingest_directory(DOCS)
    return p
