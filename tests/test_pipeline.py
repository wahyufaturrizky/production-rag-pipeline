from rag.evaluate import check_gates, evaluate, load_cases
from rag.generation import REFUSAL, validate_citations
from tests.conftest import ROOT


def test_answer_is_cited_and_grounded(pipeline):
    ans = pipeline.ask("How much does a BI-FAST transfer cost?")
    assert not ans.refused and ans.grounded
    assert "2,500" in ans.answer and "[1]" in ans.answer
    assert ans.citations[0].source == "transfers.md"


def test_unanswerable_question_is_refused_without_calling_the_llm(pipeline):
    class Exploding:
        name = "boom"

        def complete(self, *a):
            raise AssertionError("LLM must not be called for low-relevance questions")

    pipeline.llm = Exploding()
    ans = pipeline.ask("What is NusaBank's current stock price?")
    assert ans.refused and ans.answer == REFUSAL and not ans.citations


def test_llm_refusal_is_respected(pipeline):
    class Refuser:
        name = "refuser"

        def complete(self, system, prompt):
            return REFUSAL, {}

    pipeline.llm = Refuser()
    assert pipeline.ask("How much does a BI-FAST transfer cost?").refused


def test_validate_citations_drops_hallucinated_references():
    text, cited, grounded = validate_citations("Fee is IDR 2,500 per transfer [1]. Arrives instantly every day [7].", 2)
    assert cited == [1] and "[7]" not in text and grounded is False


def test_reingest_is_idempotent(pipeline):
    report = pipeline.ingest_directory(ROOT / "data" / "docs")
    assert all(r["inserted"] == 0 and r["deleted"] == 0 for r in report.values())


def test_golden_set_meets_release_thresholds(pipeline):
    metrics, _ = evaluate(pipeline, load_cases(ROOT / "eval" / "golden_qa.jsonl"))
    gates = check_gates(metrics, ROOT / "eval" / "thresholds.yaml")
    assert all(g[3] for g in gates), [g for g in gates if not g[3]]
    assert metrics["refusal_accuracy"] == 1.0
