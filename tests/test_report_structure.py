"""Doctor report: structure, provenance, Russian content and exports (existing functionality, now in Russian)."""

from __future__ import annotations

import io
import json

from pypdf import PdfReader

from intake_assistant.pdf_export import report_to_pdf
from intake_assistant.report import LIST_SECTIONS, build_report, report_to_json
from intake_assistant.sample import build_fictional_session, load_script
from intake_assistant.schemas import DoctorReport


def _fictional_report(provider):
    session = build_fictional_session(provider)
    return session, build_report(session, session.extraction, mode="demo", generator=provider.name)


def test_report_has_all_required_sections(demo_provider):
    _, report = _fictional_report(demo_provider)
    data = json.loads(report_to_json(report))
    for key in [
        "meta", "main_concern", "symptom_details", "chronology", "progression", "health_events", "medications",
        "allergies", "medical_history", "negative_findings", "missing_information", "uncertainties",
        "contradictions", "patient_questions", "ai_generated", "source_entries",
    ]:
        assert key in data, key
    assert data["meta"]["schema_version"] == "2.0"
    assert data["meta"]["fictional_example"] is True
    assert "не является клинически проверенной" in data["meta"]["disclaimer"]


def test_main_concern_and_sections_populated(demo_provider):
    _, r = _fictional_report(demo_provider)
    assert r.main_concern and "головные боли" in r.main_concern.text
    assert any("ибупрофен 400 мг" in i.text for i in r.medications)
    assert any("во лбу" in i.text for i in r.symptom_details)
    assert any("новую работу" in i.text for i in r.health_events)
    assert any("стали чаще" in i.text for i in r.progression)
    assert [n.text for n in r.negative_findings] == ["Рвоты и температуры нет.", "Аллергии нет."]
    assert [q.text for q in r.patient_questions] == [
        "Нужно ли сдавать какие-то анализы?", "Можно ли продолжать принимать ибупрофен так часто?"]
    assert r.contradictions and set(r.contradictions[0].source_entry_ids) == {"E10", "E12"}


def test_chronology_is_ordered_oldest_first_with_verbatim_times(demo_provider):
    _, r = _fictional_report(demo_provider)
    times = [i.time_reference for i in r.chronology]
    # "в детстве" (childhood) < "В прошлом месяце" < "Три недели назад" < "примерно две недели назад" < "За последнюю неделю"
    assert times.index("в детстве") < times.index("В прошлом месяце") < times.index("Три недели назад")
    assert times.index("Три недели назад") < times.index("примерно две недели назад") < times.index("За последнюю неделю")


def test_every_item_cites_existing_entries(demo_provider):
    _, r = _fictional_report(demo_provider)
    ids = {e.id for e in r.source_entries}
    items = [r.main_concern] + [i for key, _ in LIST_SECTIONS for i in getattr(r, key)]
    assert items
    for item in items:
        assert item.source_entry_ids, item
        assert set(item.source_entry_ids) <= ids, item


def test_original_entries_preserved_verbatim(demo_provider):
    _, r = _fictional_report(demo_provider)
    script_texts = [s["text"] for s in load_script()["intake_steps"] if s["action"] == "entry" and s["source"] == "text"]
    originals = [e.original_text for e in r.source_entries if e.source == "text"]
    assert originals == script_texts
    audio = [e for e in r.source_entries if e.source == "audio"]
    assert len(audio) == 1 and audio[0].audio_filename.endswith("_ru.wav")
    assert r.source_entries[1].question_slot == "progression"


def test_ai_summary_is_separate_and_labelled(demo_provider):
    _, r = _fictional_report(demo_provider)
    assert r.ai_generated.summary.startswith("[Демо-шаблон")
    assert r.ai_generated.generator.startswith("демо")
    assert all(r.ai_generated.summary not in i.text for key, _ in LIST_SECTIONS for i in getattr(r, key))


def test_json_round_trip(demo_provider):
    _, r = _fictional_report(demo_provider)
    assert DoctorReport.model_validate_json(report_to_json(r)) == r


def test_pdf_export_in_russian_with_cyrillic(demo_provider):
    _, r = _fictional_report(demo_provider)
    pdf = report_to_pdf(r)
    assert pdf[:4] == b"%PDF"
    text = "\n".join(p.extract_text() for p in PdfReader(io.BytesIO(pdf)).pages)
    for needle in [
        "Сведения пациента перед приёмом",
        "ВЫМЫШЛЕННЫЙ ПРИМЕР",
        "Резюме, составленное ИИ",
        "Явно отрицаемые пациентом симптомы",
        "Не хватает информации",
        "Противоречия",
        "Вопросы пациента к врачу",
        "исходные записи пациента",
        "не является клинически проверенной",
        "ибупрофен 400 мг",
    ]:
        assert needle in text, needle
