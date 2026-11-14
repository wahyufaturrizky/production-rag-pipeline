"""Shared text utilities: tokenisation, stopwords and a tiny stemmer."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does", "for", "from", "how", "i", "if", "in", "into", "is", "it", "its", "may", "my", "of", "on", "or", "our", "so", "than", "that", "the", "their", "them", "then", "there", "these", "this", "to", "up", "was", "we", "what", "when", "where", "which", "who", "why", "will", "with", "you", "your", "yours", "me", "about", "after", "before", "any", "all", "also", "have", "has", "had", "not", "no", "yes", "per", "please", "tell", "much", "many", "get"]
)

_TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")


def stem(word: str) -> str:
    """Crude but effective stemming: strip a common suffix, then keep a 6-character prefix.

    'disputes' / 'disputed' / 'dispute' -> 'disput'; 'investigation' / 'investigated' -> 'invest'.
    """
    if word[:1].isdigit():
        return word
    for suffix in ("ing", "ed", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix) and not word.endswith("ss"):
            word = word[: -len(suffix)]
            break
    return word[:6]


def tokens(text: str, *, drop_stopwords: bool = True) -> list[str]:
    out = []
    for word in _TOKEN.findall(text.lower()):
        # Keep compounds ("bi-fast") and also their parts ("same-day" -> "same", "day").
        parts = [word, *word.split("-")] if "-" in word and not word[:1].isdigit() else [word]
        out.extend(stem(w) for w in parts if not (drop_stopwords and w in STOPWORDS))
    return out


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip())
    return [p.strip() for p in parts if p.strip()]
