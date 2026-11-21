"""Patient data, appointment records and the medical knowledge base stay separate; nothing sensitive is logged.

Also: patient statements, doctor instructions and AI explanations stay distinguishable.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

from intake_assistant.records import ORIGIN_LABELS, RecordStore
from intake_assistant.report import build_report
from intake_assistant.sample import build_fictional_record, build_fictional_session

PKG = Path(__file__).resolve().parent.parent / "intake_assistant"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mods.add(("." * node.level) + (node.module or ""))
        elif isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
    return mods


def test_patient_modules_do_not_import_rag():
    for name in ("report.py", "intake.py", "schemas.py", "pdf_export.py", "report_edit.py", "records.py",
                 "actions.py", "labs.py", "timeline.py", "urgent.py"):
        assert not any("rag" in m for m in _imports(PKG / name)), name


def _full_record(demo_settings, demo_provider, indexed_store, tmp_path):
    from intake_assistant.urgent import load_pathway

    rs = RecordStore(tmp_path / "appointments")
    rec = build_fictional_record(demo_provider, indexed_store, rs, 4, demo_settings.min_relevance, True,
                                 load_pathway(demo_settings.urgent_rules_path))
    return rs, rec


def test_patient_data_never_reaches_vector_store(indexed_store, demo_provider, demo_settings, tmp_path):
    before = set(indexed_store.collection.get()["ids"])
    _, rec = _full_record(demo_settings, demo_provider, indexed_store, tmp_path)
    stored = indexed_store.collection.get(include=["documents"])
    assert set(stored["ids"]) == before
    blob = " ".join(stored["documents"])
    for e in rec.intake.entries:
        assert e.text not in blob
    for c in rec.consultation_items:
        if len(c.text) > 30:  # a bare term such as "МРТ" legitimately occurs in reference documents
            assert c.text[:60] not in blob
    for d in rec.instructions:
        assert d.text[:60] not in blob
    # Uploads live in the record folder, never in the knowledge-base library folder.
    lib_files = {p.name for p in Path(demo_settings.library_path).glob("*")}
    assert not any(u.filename in " ".join(lib_files) for u in rec.uploads)


def test_records_are_stored_apart_from_knowledge_base(indexed_store, demo_provider, demo_settings, tmp_path):
    rs, rec = _full_record(demo_settings, demo_provider, indexed_store, tmp_path)
    rec_dir = rs.root / rec.id
    assert (rec_dir / "record.json").exists()
    assert Path(demo_settings.chroma_path).resolve() not in rec_dir.resolve().parents


def test_origins_are_distinguishable_in_record(indexed_store, demo_provider, demo_settings, tmp_path):
    _, rec = _full_record(demo_settings, demo_provider, indexed_store, tmp_path)
    assert {d.origin for d in rec.instructions} == {"doctor"}
    assert {q.origin for q in rec.patient_questions} == {"patient"}
    assert {c.origin for c in rec.consultation_items} == {"patient", "patient_document"}
    assert {a.origin for a in rec.actions} == {"doctor"}
    for x in rec.explanations:
        assert {s.origin for s in x.sources} <= {"knowledge_base"}
        # AI background never contains the patient's own document lines (kept in document_states instead).
        for row in x.lab_rows:
            assert row.line not in x.background
    assert set(ORIGIN_LABELS) == {"patient", "patient_document", "doctor", "ai", "knowledge_base", "rules"}


def test_report_contains_no_rag_content(indexed_store, demo_provider):
    session = build_fictional_session(demo_provider)
    dumped = build_report(session, session.extraction, mode="demo", generator="t").model_dump_json()
    for doc in indexed_store.collection.get(include=["documents"])["documents"]:
        first = doc.split(".")[0]
        if len(first) > 40:
            assert first not in dumped


def test_no_patient_text_logged(caplog, demo_provider, demo_settings, indexed_store, tmp_path):
    caplog.set_level(logging.DEBUG)
    _, rec = _full_record(demo_settings, demo_provider, indexed_store, tmp_path)
    logged = " ".join(r.getMessage() for r in caplog.records)
    for e in rec.intake.entries:
        assert e.text not in logged
    assert rec.instructions[0].text[:40] not in logged


def test_api_key_not_in_settings_repr(demo_settings):
    from dataclasses import replace

    assert "sk-very-secret" not in repr(replace(demo_settings, openai_api_key="sk-very-secret"))
