"""Grounded answer generation with citations, context budgeting and refusal."""

from __future__ import annotations

import os
import re
from typing import Protocol

import httpx

from .models import ScoredChunk
from .text import split_sentences, tokens

REFUSAL = "I don't have information about that in NusaBank's documentation. Please contact the 24/7 hotline 1500-321."

SYSTEM_PROMPT = f"""You are NusaBank's documentation assistant (NusaBank is a fictional bank).
Answer ONLY using the numbered context passages. After every sentence, cite the passage(s) it uses, like [1] or [1][3].
If the passages do not contain the answer, reply exactly: "{REFUSAL}"
Be concise (at most 4 sentences). Do not mention the word "context" or "passage"."""


class LLM(Protocol):
    name: str

    def complete(self, system: str, prompt: str) -> tuple[str, dict[str, int]]: ...


def pack_context(chunks: list[ScoredChunk], max_tokens: int = 1200) -> list[ScoredChunk]:
    """Keep the best chunks that fit the token budget (~4 chars per token), dropping near-duplicates."""
    packed, used, seen = [], 0, set()
    for sc in chunks:
        fingerprint = frozenset(tokens(sc.chunk.content))
        if any(len(fingerprint & s) / max(1, len(fingerprint | s)) > 0.9 for s in seen):
            continue
        cost = len(sc.chunk.content) // 4 + 10
        if used + cost > max_tokens and packed:
            break
        packed.append(sc)
        used += cost
        seen.add(fingerprint)
    return packed


def build_prompt(question: str, chunks: list[ScoredChunk]) -> str:
    blocks = "\n\n".join(f"[{i}] ({c.chunk.source} > {c.chunk.section})\n{c.chunk.content}" for i, c in enumerate(chunks, 1))
    return f"{blocks}\n\nQuestion: {question}\nAnswer:"


class ExtractiveLLM:
    """Offline stand-in: picks the sentences that best cover the question and cites them.

    Fully grounded by construction, which makes it a good CI baseline for the pipeline mechanics.
    """

    name = "extractive"

    def complete(self, system: str, prompt: str) -> tuple[str, dict[str, int]]:
        question = prompt.rsplit("Question:", 1)[1].split("\nAnswer:")[0]
        q = set(tokens(question))
        passages = re.findall(r"\[(\d+)\] \([^)]*\)\n(.*?)(?=\n\n\[\d+\] \(|\n\nQuestion:)", prompt, flags=re.DOTALL)
        scored = []
        for n, text in passages:
            for pos, sentence in enumerate(split_sentences(text)):
                overlap = len(q & set(tokens(sentence)))
                if overlap:
                    scored.append((overlap, -int(n), -pos, sentence, n))
        if not scored:
            return REFUSAL, {"input_tokens": len(prompt) // 4, "output_tokens": 20}
        scored.sort(reverse=True)
        best = scored[0][0]
        top_passage = scored[0][4]
        # At most two sentences, both from the strongest passage: avoids stitching unrelated facts together.
        picked = [s for s in scored if s[0] >= best and s[4] == top_passage][:2]
        picked.sort(key=lambda s: (-s[1], -s[2]))  # keep document order for readability
        answer = " ".join(f"{s[3]} [{s[4]}]" for s in picked)
        return answer, {"input_tokens": len(prompt) // 4, "output_tokens": len(answer) // 4}


class GeminiLLM:
    name = "gemini"

    def __init__(self, model: str | None = None, client: httpx.Client | None = None):
        self.model = model or os.getenv("RAG_LLM_MODEL", "gemini-2.5-flash")
        self._client = client or httpx.Client(timeout=60)

    def complete(self, system: str, prompt: str) -> tuple[str, dict[str, int]]:
        key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Set GOOGLE_API_KEY (or GEMINI_API_KEY) to use Gemini")
        config = {"temperature": 0, "maxOutputTokens": 400}
        if "2.5-flash" in self.model:
            config["thinkingConfig"] = {"thinkingBudget": 0}
        resp = self._client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            headers={"x-goog-api-key": key},
            json={"systemInstruction": {"parts": [{"text": system}]},
                  "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                  "generationConfig": config},
        )
        resp.raise_for_status()
        data = resp.json()
        parts = (data.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
        usage = data.get("usageMetadata", {})
        return "".join(p.get("text", "") for p in parts).strip(), {
            "input_tokens": usage.get("promptTokenCount", 0), "output_tokens": usage.get("candidatesTokenCount", 0)}


def get_llm(kind: str | None = None) -> LLM:
    kind = (kind or os.getenv("RAG_LLM") or "extractive").lower()
    if kind == "extractive":
        return ExtractiveLLM()
    if kind == "gemini":
        return GeminiLLM()
    raise ValueError(f"Unknown RAG_LLM '{kind}'")


def validate_citations(answer: str, n_passages: int) -> tuple[str, list[int], bool]:
    """Drop citations to passages that don't exist; report whether every sentence is cited."""
    cited = []

    def keep(m: re.Match) -> str:
        n = int(m.group(1))
        if 1 <= n <= n_passages:
            cited.append(n)
            return m.group(0)
        return ""

    cleaned = re.sub(r"\[(\d+)\]", keep, answer).strip()
    sentences = [s for s in split_sentences(cleaned) if len(s) > 15]
    grounded = bool(sentences) and all(re.search(r"\[\d+\]", s) for s in sentences)
    return cleaned, sorted(set(cited)), grounded
