"""End-to-end UI tests with Streamlit's AppTest (demo mode, temporary storage), fully in Russian."""

from __future__ import annotations

import re

import pytest
from streamlit.testing.v1 import AppTest

from intake_assistant.records import RecordStore

from .conftest import ROOT

APP = str(ROOT / "app.py")


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_MODE", "demo")
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("LIBRARY_PATH", str(tmp_path / "lib"))
    monkeypatch.setenv("APPOINTMENTS_PATH", str(tmp_path / "appointments"))
    monkeypatch.setenv("APP_TODAY", "2026-10-11")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return tmp_path


@pytest.fixture()
def app(env):
    return AppTest.from_file(APP, default_timeout=120).run()


def btn(at, label, n=0):
    matches = [b for b in at.button if b.label == label]
    assert matches, f"button {label!r} not found; have {[b.label for b in at.button]}"
    return matches[n]


def area(at, label, n=0):
    matches = [t for t in at.text_area if t.label == label]
    assert matches, f"text_area {label!r} not found; have {[t.label for t in at.text_area]}"
    return matches[n]


def test_intake_flow_one_question_at_a_time(app, env):
    at = app
    assert not at.exception
    area(at, "Ваш ответ").input("Болит левое ухо.")
    btn(at, "Отправить").click().run()
    assert not at.exception
    assert at.chat_message[-1].markdown[-1].value == "**Когда это началось? Можно указать примерно.**"
    btn(at, "⏭️ Пропустить вопрос").click().run()
    assert "менялись симптомы" in at.chat_message[-1].markdown[-1].value
    btn(at, "✅ Завершить рассказ").click().run()
    assert any("Рассказ завершён" in s.value for s in at.success)
    rec = at.session_state["record"]
    assert rec.intake.skipped_slots == ["onset"]
    # The record is persisted locally as soon as it has content.
    assert RecordStore(env / "appointments").exists(rec.id)


def test_ui_is_in_russian(app):
    at = btn(app, "👤 Пример").click().run()
    labels = [b.label for b in at.button] + [t.label for t in at.tabs] + [t.label for t in at.text_area] + \
             [t.label for t in at.text_input] + [c.label for c in at.checkbox]
    english = [l for l in labels if re.search(r"[A-Za-z]{4,}", l) and not re.search(r"[А-Яа-яЁё]", l)]
    assert english == [], english
    tab_labels = [t.label for t in at.tabs]
    for label in ("🗓️ 1 · До приёма", "🩺 2 · Во время приёма", "✅ 3 · После приёма", "📁 Вся запись", "📚 База знаний"):
        assert label in tab_labels


def test_fictional_end_to_end_record(app):
    at = btn(app, "👤 Пример").click().run()
    assert not at.exception
    rec = at.session_state["record"]
    assert rec.fictional and len(rec.intake.entries) == 12
    assert rec.report is not None and rec.specialist_guidance.status == "suggested"
    assert len(rec.explanations) == 3 and rec.actions and rec.instruction_issues
    # Report regenerate + edit + download (before)
    btn(at, "🔄 Обновить отчёт").click().run()
    assert not at.exception
    summary = next(t for t in at.text_area if t.label == "Резюме ИИ")
    summary.input("Резюме, исправленное при проверке.")
    btn(at, "💾 Сохранить правки").click().run()
    r2 = at.session_state["record"].report
    assert r2.ai_generated.summary == "Резюме, исправленное при проверке." and r2.ai_generated.edited_by_reviewer
    dl = [b.label for b in at.get("download_button")]
    for label in ("⬇️ PDF для врача", "⬇️ Вся запись (PDF)", "⬇️ Вся запись (JSON)", "⬇️ Список дел (PDF)"):
        assert label in dl, label
    # Reminder for the overdue item set by the patient's own date is shown in-app.
    assert any("Напоминания" in w.value for w in at.warning)


def test_during_explain_term_and_save(app):
    at = btn(app, "Загрузить демонстрационные документы").click().run()
    at.radio(key="consult_kind").set_value("term").run()
    area(at, "Текст").input("ферритин")
    btn(at, "Добавить", 0).click().run()  # 0 = consultation form, 1 = instructions form
    rec = at.session_state["record"]
    assert rec.consultation_items and rec.consultation_items[-1].text == "ферритин"
    cid = rec.consultation_items[-1].id
    at.button(key=f"explain_{cid}").click().run()
    assert not at.exception
    x = at.session_state["pending_expl"][cid]
    assert x.background_status == "answered" and x.sources and "Ферритин — белок" in x.background
    at.button(key=f"save_x_{cid}").click().run()
    saved = at.session_state["record"].explanations
    assert saved and saved[0].id == "X1" and saved[0].sources[0].page == 1


def test_during_test_result_values_preserved(app):
    at = btn(app, "Загрузить демонстрационные документы").click().run()
    at.radio(key="consult_kind").set_value("test_result").run()
    area(at, "Текст").input("Ферритин 9 нг/мл ↓\nГемоглобин (HGB) 128 г/л 120–140")
    btn(at, "Добавить", 0).click().run()
    cid = at.session_state["record"].consultation_items[-1].id
    at.button(key=f"explain_{cid}").click().run()
    x = at.session_state["pending_expl"][cid]
    row = x.lab_rows[0]
    assert (row.name, row.value, row.unit, row.reference_range, row.flag) == ("Ферритин", "9", "нг/мл", None, "↓")


def test_after_checklist_completion_persists(app, env):
    at = app
    area(at, "Указания врача").input("Сдать анализ крови до 20.10.2026.\nИбупрофен 400 мг 2 раза в день 5 дней.")
    btn(at, "Добавить", 1).click().run()  # 0 = consultation form, 1 = instructions form
    rec = at.session_state["record"]
    assert rec.instructions and rec.instructions[0].origin == "doctor"
    btn(at, "📋 Составить список дел").click().run()
    rec = at.session_state["record"]
    assert len(rec.actions) == 2
    med = next(a for a in rec.actions if a.category == "medication")
    assert med.medication.dose == "400 мг" and med.medication.frequency == "2 раза в день"
    test = next(a for a in rec.actions if a.category == "test")
    at.checkbox(key=f"done_p_{test.id}_pending").check().run()
    assert not at.exception
    stored = RecordStore(env / "appointments").load(rec.id)
    assert next(a for a in stored.actions if a.id == test.id).status == "done"


def test_delete_record_and_uploads(app, env):
    at = btn(app, "👤 Пример").click().run()
    rec = at.session_state["record"]
    folder = env / "appointments" / rec.id
    assert (folder / "uploads").exists() and any((folder / "uploads").iterdir())
    btn(at, "Да, удалить запись и файлы").click().run()
    assert not folder.exists()
    assert at.session_state["record"].is_empty()
    assert any("удалены" in s.value for s in at.success)


def test_knowledge_base_answer_with_sources_and_refusal(app):
    at = btn(app, "Загрузить демонстрационные документы").click().run()
    btn(at, "Что такое референсный интервал?").click().run()
    q, res = at.session_state["qa_history"][0]
    assert res.status == "answered" and res.sources
    btn(at, "Какая максимальная суточная доза ибупрофена для детей?").click().run()
    q, res = at.session_state["qa_history"][0]
    assert res.status == "insufficient_evidence" and not res.cited
    assert any("недостаточно сведений" in w.value for w in at.warning)


def test_review_correction_preserves_original(app):
    at = btn(app, "👤 Пример").click().run()
    box = next(t for t in at.text_area if t.label == "Текст E5")
    box.input("Справа во лбу.")
    btn(at, "Сохранить исправления и обработать заново").click().run()
    e5 = at.session_state["record"].intake.entries[4]
    assert e5.text == "Справа во лбу." and e5.original_text == "В основном справа во лбу, за глазом."
    assert at.session_state["record"].report_is_stale()


def test_record_selector_is_stable(app, env):
    """Regression: saving must never silently switch to another record (same titles, re-ordered list)."""
    at = app
    for text in ("Первая запись: болит спина.", "Вторая запись: болит горло."):
        btn(at, "➕ Новая").click().run()
        area(at, "Ваш ответ").input(text)
        btn(at, "Отправить").click().run()
    current = at.session_state["record"].id
    for answer in ("Неделю назад.", "Стало хуже.", "Постоянно."):
        area(at, "Ваш ответ").input(answer)
        btn(at, "Отправить").click().run()
        assert at.session_state["record"].id == current
    rec = at.session_state["record"]
    assert rec.intake.entries[0].text == "Вторая запись: болит горло." and len(rec.intake.entries) == 4
    # Explicitly choosing the other record works.
    other = next(r.id for r in RecordStore(env / "appointments").list_records() if r.id != current)
    at.selectbox[0].set_value(other).run()
    assert at.session_state["record"].id == other
    assert at.session_state["record"].intake.entries[0].text == "Первая запись: болит спина."
