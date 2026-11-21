"""Russian audio (demo mode), transcription-correction provenance and report editing."""

from __future__ import annotations

import wave

from intake_assistant.intake import IntakeSession
from intake_assistant.report import build_report
from intake_assistant.report_edit import CERTAINTY_RU, items_to_rows, rows_to_items
from intake_assistant.sample import SAMPLE_DIR

from .conftest import ROOT


def test_bundled_russian_clips_are_real_speech_audio():
    for stem in ("fictional_severity_answer_ru", "fictional_consult_question_ru"):
        with wave.open(str(SAMPLE_DIR / f"{stem}.wav")) as w:
            seconds = w.getnframes() / w.getframerate()
            frames = w.readframes(w.getnframes())
        assert seconds > 1.5
        # not silence and not a single tone: many distinct sample values
        assert len(set(frames[::2])) > 50


def test_demo_transcribes_bundled_russian_clip(demo_provider):
    wav = SAMPLE_DIR / "fictional_severity_answer_ru.wav"
    res = demo_provider.transcribe(wav.read_bytes(), wav.name)
    assert not res.is_placeholder and res.text == "Примерно шесть из десяти. Вечером иногда сильнее."
    assert "Демо-режим" in res.note


def test_demo_unknown_audio_returns_labelled_placeholder(demo_provider):
    res = demo_provider.transcribe(b"RIFF" + b"\x00" * 100, "other.wav")
    assert res.is_placeholder and "ДЕМО-РЕЖИМ" in res.text and "НЕ расшифровано" in res.text


def test_corrected_russian_transcription_keeps_original(demo_provider):
    s = IntakeSession()
    raw = "Боль в груди уже два дня, усилевается при ходьбе"
    e = s.add_entry("Боль в груди уже два дня, усиливается при ходьбе", source="audio", original_text=raw,
                    audio_filename="r.wav")
    assert e.original_text == raw and e.was_edited and e.source == "audio"
    s.extraction = demo_provider.extract(s.entries)
    r = build_report(s, s.extraction, mode="demo", generator="t")
    assert r.source_entries[0].original_text == raw and r.main_concern.source_entry_ids == ["E1"]


def test_report_edits_are_marked(demo_provider):
    s = IntakeSession()
    s.add_entry("Кашель. Начался примерно неделю назад.")
    s.extraction = demo_provider.extract(s.entries)
    r = build_report(s, s.extraction, mode="demo", generator="t")
    rows = items_to_rows(r.symptom_details)
    assert rows[0]["Достоверность"] == CERTAINTY_RU["uncertain"]
    rows[0]["Текст"] = "Начался около 7 дней назад"
    rows.append({"_idx": None, "Раздел": "Заметка", "Текст": "Спросить о курении", "Записи": "", "Достоверность": "со слов пациента"})
    new = rows_to_items(rows, r.symptom_details, {"E1"})
    assert new[0].edited_by_reviewer and new[0].category == "onset" and new[0].certainty == "uncertain"
    assert new[-1].edited_by_reviewer and "no_source" in new[-1].flags
    unchanged = rows_to_items(items_to_rows(r.symptom_details), r.symptom_details, {"E1"})
    assert not any(i.edited_by_reviewer for i in unchanged)


def test_env_example_lists_all_settings():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    for key in ("OPENAI_API_KEY", "APP_MODE", "OPENAI_CHAT_MODEL", "OPENAI_TRANSCRIPTION_MODEL", "TRANSCRIPTION_LANGUAGE",
                "OPENAI_EMBEDDING_MODEL", "CHUNK_SIZE", "CHUNK_OVERLAP", "RETRIEVAL_TOP_K", "CHROMA_PATH",
                "APPOINTMENTS_PATH", "URGENT_RULES_PATH"):
        assert key in text, key
