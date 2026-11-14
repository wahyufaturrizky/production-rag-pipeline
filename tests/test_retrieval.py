import numpy as np

from rag.embeddings import HashEmbeddings
from rag.models import Chunk, ScoredChunk
from rag.retrieval import OverlapReranker, reciprocal_rank_fusion
from rag.stores import InMemoryStore
from rag.text import stem, tokens


def _chunk(cid, text, section="S"):
    return Chunk(cid, "d", "d.md", section, text, cid)


def test_stemming_groups_word_forms():
    assert stem("disputes") == stem("disputed") == stem("dispute")
    assert stem("investigation") == stem("investigated")
    assert "same" in tokens("same-day delivery") and "day" in tokens("same-day delivery")


def test_hash_embeddings_are_normalised_and_similarity_tracks_overlap():
    emb = HashEmbeddings(dim=256)
    a, b, c = (np.array(v) for v in emb.embed_documents(["block my card", "how to block a card", "deposit insurance"]))
    assert abs(np.linalg.norm(a) - 1) < 1e-5
    assert a @ b > a @ c


def test_rrf_rewards_agreement_between_retrievers():
    x, y, z = _chunk("x", "x"), _chunk("y", "y"), _chunk("z", "z")
    dense = [ScoredChunk(x, 0.9), ScoredChunk(y, 0.8)]
    lexical = [ScoredChunk(y, 7.0), ScoredChunk(z, 5.0)]
    assert next(s.chunk.id for s in reciprocal_rank_fusion([dense, lexical])) == "y"


def test_bm25_and_upsert_lifecycle():
    store, emb = InMemoryStore(), HashEmbeddings()
    chunks = [_chunk("a", "BI-FAST transfers cost IDR 2,500"), _chunk("b", "Savings interest is 3.25%")]
    for c in chunks:
        c.embedding = emb.embed_query(c.text_for_embedding)
    assert store.upsert_document("d", chunks) == {"inserted": 2, "unchanged": 0, "deleted": 0}
    assert store.lexical_search("BI-FAST fee", 5)[0].chunk.id == "a"
    assert store.upsert_document("d", chunks[:1]) == {"inserted": 0, "unchanged": 1, "deleted": 1}
    assert store.count() == 1


def test_reranker_prefers_chunk_that_answers_the_question():
    good = ScoredChunk(_chunk("g", "Disputes are investigated within 14 working days.", "Dispute timelines"), 0.01)
    noisy = ScoredChunk(_chunk("n", "Open the app and tap Dispute.", "Disputing a transaction"), 0.02)
    ranked = OverlapReranker().rerank("How long does a dispute investigation take?", [noisy, good])
    assert ranked[0].chunk.id == "g"


def test_gemini_adapters_build_correct_requests(monkeypatch):
    import json

    import httpx

    from rag.embeddings import GeminiEmbeddings
    from rag.generation import GeminiLLM

    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = json.loads(request.content)
        if "batchEmbedContents" in str(request.url):
            return httpx.Response(200, json={"embeddings": [{"values": [3.0, 4.0]} for _ in body["requests"]]})
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "Fee is IDR 2,500. [1]"}]}}],
                                         "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 8}})

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    vecs = GeminiEmbeddings(dim=2, client=client).embed_documents(["a", "b"])
    assert np.allclose(vecs, [[0.6, 0.8], [0.6, 0.8]])  # re-normalised after dimension truncation
    req = json.loads(seen[0].content)["requests"][0]
    assert req["taskType"] == "RETRIEVAL_DOCUMENT" and req["outputDimensionality"] == 2

    text, usage = GeminiLLM("gemini-2.5-flash", client=client).complete("sys", "prompt")
    assert text.startswith("Fee is") and usage == {"input_tokens": 50, "output_tokens": 8}
    assert seen[-1].headers["x-goog-api-key"] == "test-key"
