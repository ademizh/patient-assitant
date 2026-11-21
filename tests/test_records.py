"""One coherent appointment record: persistence, carried context, corrections, export and deletion."""

from __future__ import annotations

import io

import pytest
from pypdf import PdfReader

from intake_assistant.pdf_export import checklist_to_pdf, record_to_pdf
from intake_assistant.records import AppointmentRecord, PatientQuestion, RecordStore, next_id
from intake_assistant.sample import SAMPLE_DIR, build_fictional_record
from intake_assistant.urgent import load_pathway


@pytest.fixture()
def full(demo_settings, demo_provider, indexed_store, tmp_path):
    rs = RecordStore(tmp_path / "appointments")
    rec = build_fictional_record(demo_provider, indexed_store, rs, 4, demo_settings.min_relevance, True,
                                 load_pathway(demo_settings.urgent_rules_path))
    return rs, rec


def test_full_record_round_trip(full):
    rs, rec = full
    loaded = rs.load(rec.id)
    assert loaded == rec.model_copy(update={"updated_at": loaded.updated_at})
    assert loaded.fictional and loaded.title.startswith("ВЫМЫШЛЕННЫЙ")
    # every stage is present in the single record
    assert loaded.intake.entries and loaded.report and loaded.specialist_guidance
    assert loaded.consultation_items and loaded.explanations
    assert loaded.instructions and loaded.actions and loaded.instruction_issues
    assert {u.kind for u in loaded.uploads} == {"test_result", "instructions"}
    for u in loaded.uploads:
        assert rs.upload_path(loaded, u).exists()


def test_context_is_carried_between_stages(full):
    _, rec = full
    # Questions stated during intake are available during the appointment without re-asking.
    intake_qs = [q for q in rec.patient_questions if q.source == "intake"]
    assert [q.text for q in intake_qs] == ["Нужно ли сдавать какие-то анализы?",
                                           "Можно ли продолжать принимать ибупрофен так часто?"]
    assert intake_qs[0].entry_ids == ["E11"]
    assert any(q.source == "consultation" for q in rec.patient_questions)
    # The specialist search used the patient's own words from the intake.
    assert "головные боли" in rec.specialist_guidance.query
    assert rec.specialist_guidance.status == "suggested"


def test_correction_marks_summaries_stale_but_keeps_originals(full):
    _, rec = full
    assert not rec.report_is_stale() and not rec.actions_are_stale()
    original = rec.intake.entries[4].original_text
    rec.intake.update_entry_text("E5", "Справа во лбу.")
    assert rec.report_is_stale() and rec.intake.entries[4].original_text == original
    rec.instructions[0].text += "\nПить больше воды."
    assert rec.actions_are_stale() and "Пить больше воды" not in rec.instructions[0].original_text


def test_delete_record_removes_json_and_uploads(full):
    rs, rec = full
    folder = rs.root / rec.id
    upload_files = [rs.upload_path(rec, u) for u in rec.uploads]
    assert folder.exists() and all(p.exists() for p in upload_files)
    assert rs.delete(rec.id) is True
    assert not folder.exists() and not any(p.exists() for p in upload_files)
    assert rec.id not in [r.id for r in rs.list_records()]
    assert rs.delete(rec.id) is False


def test_delete_single_upload(full):
    rs, rec = full
    up = rec.uploads[0]
    path = rs.upload_path(rec, up)
    rs.delete_upload(rec, up.id)
    assert not path.exists() and rec.find_upload(up.id) is None


def test_record_ids_are_validated(tmp_path):
    rs = RecordStore(tmp_path)
    for bad in ("../etc", "abc", "ABCDEF123456", "a" * 13):
        with pytest.raises(ValueError):
            rs.load(bad)
    assert rs.exists("../../x") is False


def test_empty_record_and_ids():
    rec = AppointmentRecord()
    assert rec.is_empty()
    rec.patient_questions.append(PatientQuestion(id="Q1", text="?"))
    rec.patient_questions.append(PatientQuestion(id="Q3", text="?"))
    assert next_id("Q", rec.patient_questions) == "Q4" and not rec.is_empty()


def test_record_pdf_and_checklist_pdf_in_russian(full):
    _, rec = full
    text = "\n".join(p.extract_text() for p in PdfReader(io.BytesIO(record_to_pdf(rec))).pages)
    for needle in ["1. До приёма", "2. Во время приёма", "3. После приёма", "Указание врача", "Объяснение ИИ",
                   "Со слов пациента", "Из документа пациента", "Источник: «ДЕМО-материал: словарь терминов и анализов»",
                   "стр. 1", "референсный интервал в бланке НЕ указан", "Ибупрофен 400 мг", "Список дел после приёма",
                   "ВЫМЫШЛЕННЫЙ ПРИМЕР"]:
        assert needle in text, needle
    ch = "\n".join(p.extract_text() for p in PdfReader(io.BytesIO(checklist_to_pdf(rec))).pages)
    for needle in ["Список дел после приёма", "срок от врача: не указан", "ваша целевая дата (не от врача)",
                   "Требует уточнения у врача", "не чаще 2 раз в сутки"]:
        assert needle in ch, needle


def test_sample_files_are_labelled_fictional():
    assert (SAMPLE_DIR / "FICTIONAL_doctor_instructions.txt").read_text(encoding="utf-8").startswith("ВЫМЫШЛЕННЫЙ ПРИМЕР")
    lab = PdfReader(SAMPLE_DIR / "FICTIONAL_lab_results.pdf").pages[0].extract_text()
    assert "ВЫМЫШЛЕННЫЙ ПРИМЕР" in lab
