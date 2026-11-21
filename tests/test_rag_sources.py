"""Document loading, chunking, source metadata, language detection, de-duplication and retrieval."""

from __future__ import annotations

import re

import pytest

from intake_assistant.rag.chunking import split_text
from intake_assistant.rag.loaders import DocumentError, load_document
from intake_assistant.rag.qa import ask

from .conftest import SAMPLE_DOCS

PDF_EN = SAMPLE_DOCS / "DEMO_preparing_for_a_consultation.pdf"
PDF_RU = SAMPLE_DOCS / "DEMO_ru_tests_glossary.pdf"
MD = SAMPLE_DOCS / "DEMO_ru_headache_notes.md"
CHUNK_ID = re.compile(r"^[0-9a-f]{12}-(p\d+|s)-c\d{3}$")


def test_pdf_pages_and_title():
    doc = load_document(PDF_EN.name, PDF_EN.read_bytes())
    assert doc.title == "DEMO material: Preparing for a consultation"
    assert [s.page for s in doc.sections] == [1, 2]


def test_russian_pdf_text_and_title_extract_correctly():
    doc = load_document(PDF_RU.name, PDF_RU.read_bytes())
    assert doc.title == "ДЕМО-материал: словарь терминов и анализов"
    assert [s.page for s in doc.sections] == [1, 2]
    assert "Ферритин — белок" in doc.sections[0].text


def test_markdown_title_and_sections():
    doc = load_document(MD.name, MD.read_bytes())
    assert doc.title.startswith("ДЕМО-материал")
    assert "Изменения со временем" in [s.heading for s in doc.sections]
    assert all(s.page is None for s in doc.sections)


def test_txt_loading_cp1251():
    doc = load_document("notes.txt", "Заголовок\n\nТекст про сон.".encode("cp1251"))
    assert doc.title == "Заголовок" and doc.file_type == "text"


@pytest.mark.parametrize(
    "name,data,msg",
    [("x.docx", b"abc", "не поддерживается"), ("empty.txt", b"", "пуст"), ("bad.pdf", b"not a pdf", "не удалось прочитать")],
)
def test_loader_errors(name, data, msg):
    with pytest.raises(DocumentError, match=msg):
        load_document(name, data)


def test_chunk_size_and_overlap():
    text = " ".join(f"Предложение номер {i} рассказывает о дневнике симптомов." for i in range(120))
    chunks = split_text(text, 400, 80)
    assert len(chunks) > 5 and all(len(c) <= 400 for c in chunks)
    for a, b in zip(chunks, chunks[1:]):
        assert a[-40:].split()[-1] in b[:120]
    with pytest.raises(ValueError):
        split_text(text, 400, 300)


def test_index_metadata_with_language(indexed_store):
    docs = {d.filename: d for d in indexed_store.list_documents()}
    assert PDF_EN.name in docs and PDF_RU.name in docs and MD.name in docs
    assert docs[PDF_RU.name].pages == 2
    got = indexed_store.collection.get(include=["metadatas"])
    for cid, md in zip(got["ids"], got["metadatas"]):
        assert CHUNK_ID.match(cid), cid
        assert md["title"] and md["filename"] and md["doc_hash"]
        assert md["language"] == ("en" if md["filename"] == PDF_EN.name else "ru")
        if md["filename"].endswith(".pdf"):
            assert md["page"] in (1, 2)
        else:
            assert md["page"] == -1 and md["heading"]


def test_duplicate_upload_is_not_reindexed(indexed_store):
    before = indexed_store.count()
    r = indexed_store.add_file("копия.md", MD.read_bytes(), 500, 50)
    assert r.status == "duplicate" and r.n_chunks == 0 and indexed_store.count() == before


def test_delete_and_reindex(indexed_store):
    md_doc = next(d for d in indexed_store.list_documents() if d.filename == MD.name)
    indexed_store.delete_document(md_doc.doc_hash)
    assert MD.name not in {d.filename for d in indexed_store.list_documents()}
    results = indexed_store.reindex_all(400, 50)
    assert all(r.status == "indexed" for r in results)
    assert all(d.chunk_size == 400 for d in indexed_store.list_documents())


def test_persistence_across_instances(indexed_store, demo_settings, demo_provider):
    from intake_assistant.rag.store import DocumentStore

    again = DocumentStore(demo_settings.chroma_path, demo_settings.library_path, "test_docs", demo_provider)
    assert again.count() == indexed_store.count() > 0


def test_russian_question_answer_cites_real_page(indexed_store, demo_provider):
    res = ask("Что такое референсный интервал?", indexed_store, demo_provider, 4, 0.12, is_demo=True)
    assert res.status == "answered"
    retrieved_ids = {c["chunk_id"] for c in res.retrieved}
    assert res.cited and {c["chunk_id"] for c in res.cited} <= retrieved_ids
    src = next(s for s in res.sources if s.filename == PDF_RU.name)
    assert src.page == 1 and src.title == "ДЕМО-материал: словарь терминов и анализов"
    assert src.quote and src.quote.replace("\n", " ") in " ".join(src.excerpt.split())
    assert f"[{src.chunk_id}]" in res.answer and "Демо-режим" in res.answer


def test_english_source_is_labelled_as_original_in_demo(indexed_store, demo_provider):
    res = ask("What should I include in my medication list?", indexed_store, demo_provider, 4, 0.12, is_demo=True)
    assert res.status == "answered"
    en = [s for s in res.sources if s.language == "en"]
    assert en and all(s.translation_ru is None for s in en)
    assert "на английском" in res.language_note
