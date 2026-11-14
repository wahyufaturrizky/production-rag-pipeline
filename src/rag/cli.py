"""CLI: ``rag ingest``, ``rag ask`` and ``rag eval``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .evaluate import check_gates, evaluate, load_cases, render_report, to_json
from .pipeline import RAGPipeline


def _load_dotenv(path: Path = Path(".env")) -> None:
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.split(" #")[0].strip().strip('"').strip("'"))


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(prog="rag", description="Production RAG pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_ing = sub.add_parser("ingest", help="Ingest a folder of .md/.txt documents")
    p_ing.add_argument("path", nargs="?", default="data/docs")
    p_ask = sub.add_parser("ask", help="Ask a question")
    p_ask.add_argument("question")
    p_ask.add_argument("--docs", default="data/docs", help="Ingest first (needed for the in-memory store)")
    p_ask.add_argument("--debug", action="store_true", help="Show retrieved chunks and scores")
    p_eval = sub.add_parser("eval", help="Run the golden QA set and enforce thresholds")
    p_eval.add_argument("--cases", default="eval/golden_qa.jsonl")
    p_eval.add_argument("--thresholds", default="eval/thresholds.yaml")
    p_eval.add_argument("--docs", default="data/docs")
    p_eval.add_argument("--out", default="reports")
    args = parser.parse_args(argv)

    pipeline = RAGPipeline()
    if args.cmd == "ingest":
        print(json.dumps(pipeline.ingest_directory(args.path), indent=2))
        return 0

    pipeline.ingest_directory(args.docs)  # idempotent: unchanged chunks are skipped
    if args.cmd == "ask":
        ans = pipeline.ask(args.question)
        print(ans.answer)
        for c in ans.citations:
            print(f"  [{c.n}] {c.source} > {c.section}")
        if args.debug:
            for sc in ans.retrieved:
                print(f"  - {sc.score:.3f} {sc.chunk.id} {sc.signals}")
        return 0

    metrics, results = evaluate(pipeline, load_cases(args.cases))
    gates = check_gates(metrics, args.thresholds)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = render_report(metrics, results, gates)
    (out / "rag_eval.md").write_text(report, encoding="utf-8")
    (out / "rag_eval.json").write_text(to_json(metrics, results, gates), encoding="utf-8")
    print(report)
    return 0 if all(g[3] for g in gates) else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
