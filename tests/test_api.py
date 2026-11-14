from fastapi.testclient import TestClient


def test_ask_endpoint(monkeypatch):
    monkeypatch.setenv("RAG_STORE", "memory")
    monkeypatch.setenv("RAG_EMBEDDINGS", "hash")
    monkeypatch.setenv("RAG_LLM", "extractive")
    monkeypatch.chdir(__import__("tests.conftest", fromlist=["ROOT"]).ROOT)
    from rag.api import app

    with TestClient(app) as client:
        assert client.get("/healthz").json()["chunks"] > 0
        body = client.post("/v1/ask", json={"question": "What is the RTGS fee?", "debug": True}).json()
        assert "25,000" in body["answer"] and body["citations"] and body["retrieved"]
        assert client.post("/v1/ask", json={"question": "Who is the CEO of NusaBank?"}).json()["refused"] is True
        assert client.post("/v1/ask", json={"question": "x"}).status_code == 422
