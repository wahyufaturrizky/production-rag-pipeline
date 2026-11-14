from rag.ingest import chunk_document, clean, load_directory
from tests.conftest import DOCS

DOC = """# Title

## Section A
First paragraph about cards. Second sentence here.

## Section B
Something about transfers and limits that is long enough.
"""


def test_chunks_follow_headings_and_carry_metadata():
    chunks = chunk_document(DOC, "demo.md")
    assert [c.section for c in chunks] == ["Title > Section A", "Title > Section B"]
    assert all(c.doc_id == "demo" and c.id.startswith("demo:") for c in chunks)


def test_chunk_ids_are_stable_and_content_addressed():
    a, b = chunk_document(DOC, "demo.md"), chunk_document(DOC, "demo.md")
    assert [c.id for c in a] == [c.id for c in b]
    changed = chunk_document(DOC.replace("limits", "fees"), "demo.md")
    assert changed[0].id == a[0].id and changed[1].id != a[1].id


def test_long_sections_are_split_with_size_cap():
    long_doc = "# T\n\n## S\n" + " ".join(f"Sentence number {i} about savings." for i in range(200))
    chunks = chunk_document(long_doc, "long.md", chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 5 and max(len(c.content) for c in chunks) <= 300


def test_clean_and_load_directory():
    assert clean("a  b\r\n\n\n\nc") == "a b\n\nc"
    docs = load_directory(DOCS)
    assert set(docs) == {"cards", "disputes", "loans", "savings", "security", "transfers"}
