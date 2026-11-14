"""Integration tests against a real PostgreSQL + pgvector (run in CI via a service container)."""

import os
import uuid

import pytest

from rag.embeddings import HashEmbeddings
from rag.generation import ExtractiveLLM
from rag.pipeline import RAGPipeline
from tests.conftest import DOCS

pytestmark = pytest.mark.pg
DSN = os.getenv("DATABASE_URL")


@pytest.fixture
def pg_pipeline():
    if not DSN:
        pytest.skip("DATABASE_URL not set")
    from rag.stores import PgVectorStore

    table = f"test_{uuid.uuid4().hex[:8]}"
    store = PgVectorStore(DSN, dim=384, table=table)
    yield RAGPipeline(store=store, embeddings=HashEmbeddings(384), llm=ExtractiveLLM())
    store.conn.execute(f"DROP TABLE IF EXISTS {table}")


def test_pgvector_ingest_is_idempotent_and_hybrid_search_works(pg_pipeline):
    first = pg_pipeline.ingest_directory(DOCS)
    second = pg_pipeline.ingest_directory(DOCS)
    assert sum(r["inserted"] for r in first.values()) == pg_pipeline.store.count()
    assert all(r["inserted"] == 0 for r in second.values())

    store = pg_pipeline.store
    assert store.lexical_search("RTGS fee", 3)[0].chunk.section.endswith("RTGS transfers")
    assert store.dense_search(pg_pipeline.embeddings.embed_query("replacement card fee"), 3)[0].chunk.doc_id == "cards"

    ans = pg_pipeline.ask("How much does a BI-FAST transfer cost?")
    assert "2,500" in ans.answer and ans.citations
