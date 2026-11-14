"""Document ingestion: load -> clean -> split by headings -> size-bounded chunks -> stable ids."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from .models import Chunk

SUPPORTED = {".md", ".txt"}


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def chunk_document(text: str, source: str, *, chunk_size: int = 600, chunk_overlap: int = 80) -> list[Chunk]:
    """Split on Markdown headings first (keeps sections coherent), then cap chunk size with overlap."""
    doc_id = Path(source).stem
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "title"), ("##", "section"), ("###", "subsection")], strip_headers=True
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks: list[Chunk] = []
    for section_doc in header_splitter.split_text(clean(text)):
        meta = section_doc.metadata
        section = " > ".join(v for k in ("title", "section", "subsection") if (v := meta.get(k)))
        for piece in size_splitter.split_text(section_doc.page_content):
            content = piece.strip()
            if len(content) < 20:  # drop fragments with no retrievable meaning
                continue
            digest = _sha(f"{section}\n{content}")
            chunks.append(Chunk(
                id=f"{doc_id}:{digest[:12]}",  # stable: unchanged text -> same id -> idempotent upserts
                doc_id=doc_id,
                source=source,
                section=section or doc_id,
                content=content,
                content_hash=digest,
                position=len(chunks),
            ))
    return chunks


def load_directory(path: str | Path, **chunk_kwargs) -> dict[str, list[Chunk]]:
    """Return {doc_id: chunks} for every supported file under ``path``."""
    out: dict[str, list[Chunk]] = {}
    for file in sorted(Path(path).rglob("*")):
        if file.suffix.lower() in SUPPORTED and file.is_file():
            chunks = chunk_document(file.read_text(encoding="utf-8"), file.name, **chunk_kwargs)
            if chunks:
                out[chunks[0].doc_id] = chunks
    return out
