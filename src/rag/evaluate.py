"""Offline evaluation of retrieval and answers against a golden QA set."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .pipeline import RAGPipeline


@dataclass
class CaseResult:
    id: str
    question: str
    answerable: bool
    rank: int | None
    refused: bool
    contains_facts: bool
    cites_expected_doc: bool
    grounded: bool
    answer: str
    top_sources: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        if not self.answerable:
            return self.refused
        return (not self.refused) and self.contains_facts and self.cites_expected_doc and self.grounded


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(pipeline: RAGPipeline, cases: list[dict[str, Any]]) -> tuple[dict[str, float], list[CaseResult]]:
    results: list[CaseResult] = []
    for case in cases:
        answerable = case.get("answerable", True)
        ans = pipeline.ask(case["question"])
        rank = None
        if answerable:
            for i, sc in enumerate(ans.retrieved, 1):
                if sc.chunk.doc_id == case["expected_doc"] and case["expected_section"].lower() in sc.chunk.section.lower():
                    rank = i
                    break
        text = ans.answer.lower()
        results.append(CaseResult(
            id=case["id"],
            question=case["question"],
            answerable=answerable,
            rank=rank,
            refused=ans.refused,
            contains_facts=(not ans.refused) and all(f.lower() in text for f in case.get("must_contain", [])),
            cites_expected_doc=any(c.chunk_id.startswith(f"{case.get('expected_doc')}:") for c in ans.citations),
            grounded=ans.grounded,
            answer=ans.answer,
            top_sources=[f"{s.chunk.doc_id} > {s.chunk.section.split(' > ')[-1]} ({s.score:.2f})" for s in ans.retrieved[:3]],
        ))

    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    answered = [r for r in answerable if not r.refused]
    ratio = lambda xs: round(sum(xs) / len(xs), 4) if xs else 1.0
    metrics = {
        "hit_rate_at_k": ratio([r.rank is not None for r in answerable]),
        "mrr": ratio([1 / r.rank if r.rank else 0 for r in answerable]),
        "answer_accuracy": ratio([(not r.refused) and r.contains_facts for r in answerable]),
        "citation_accuracy": ratio([r.cites_expected_doc for r in answered]),
        "grounded_rate": ratio([r.grounded for r in answered]),
        "refusal_accuracy": ratio([r.refused for r in unanswerable]),
        "false_refusal_rate": ratio([r.refused for r in answerable]),
        "cases": len(results),
        "passed": sum(r.passed for r in results),
    }
    return metrics, results


def check_gates(metrics: dict[str, float], thresholds_path: str | Path) -> list[tuple[str, float, float, bool]]:
    thresholds = yaml.safe_load(Path(thresholds_path).read_text(encoding="utf-8"))
    gates = []
    for group in thresholds.values():
        for name, minimum in group.items():
            gates.append((name, metrics[name], minimum, metrics[name] >= minimum))
    return gates


def render_report(metrics: dict[str, float], results: list[CaseResult], gates) -> str:
    ok = all(g[3] for g in gates)
    lines = [f"# RAG Eval Report: {'✅ PASSED' if ok else '❌ FAILED'}", "",
             f"{metrics['passed']}/{metrics['cases']} cases fully passed.", "",
             "| Metric | Value | Threshold | |", "|---|---|---|---|"]
    lines += [f"| {n} | {v:.1%} | ≥ {t:.0%} | {'✅' if p else '❌'} |" for n, v, t, p in gates]
    lines += [f"| false_refusal_rate | {metrics['false_refusal_rate']:.1%} | (info) | |", "", "## Failing cases", ""]
    failing = [r for r in results if not r.passed]
    if not failing:
        lines.append("None 🎉")
    for r in failing:
        lines.append(f"- **{r.id}** {r.question}\n  - rank={r.rank} refused={r.refused} facts={r.contains_facts} "
                     f"cites_doc={r.cites_expected_doc} grounded={r.grounded}\n  - top: {', '.join(r.top_sources)}\n"
                     f"  - answer: {r.answer[:200]}")
    return "\n".join(lines) + "\n"


def to_json(metrics, results, gates) -> str:
    return json.dumps({"metrics": metrics,
                       "gates": [{"name": n, "value": v, "threshold": t, "passed": p} for n, v, t, p in gates],
                       "cases": [{**asdict(r), "passed": r.passed} for r in results]}, indent=2, ensure_ascii=False)
