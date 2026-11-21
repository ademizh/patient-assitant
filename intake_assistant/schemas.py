"""Data models shared across intake, report generation and export.

Patient data models (this file) are deliberately independent from the RAG models in
`intake_assistant.rag`. Nothing from the medical-document collection is ever written
into these structures.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "2.0"

# Categories a patient-reported fact can belong to.
FactCategory = Literal[
    "main_concern",
    "onset",
    "progression",
    "duration",
    "severity",
    "location",
    "character",
    "associated_symptoms",
    "modifying_factors",
    "relevant_events",
    "medications",
    "allergies",
    "medical_history",
    "patient_question",
    "other",
]

# The intake "slots" the assistant tries to cover, in the order it asks about them.
SLOT_ORDER: list[str] = [
    "main_concern",
    "onset",
    "progression",
    "duration",
    "severity",
    "location",
    "associated_symptoms",
    "modifying_factors",
    "relevant_events",
    "medications",
    "allergies",
    "medical_history",
    "patient_question",
]

SLOT_LABELS: dict[str, str] = {
    "main_concern": "Основная жалоба",
    "onset": "Начало",
    "progression": "Изменения со временем",
    "duration": "Длительность / характер течения",
    "severity": "Выраженность",
    "location": "Локализация",
    "character": "Характер ощущений",
    "associated_symptoms": "Сопутствующие симптомы",
    "modifying_factors": "Что облегчает или усиливает",
    "relevant_events": "Предшествующие события со здоровьем",
    "medications": "Лекарства",
    "allergies": "Аллергии",
    "medical_history": "Хронические заболевания и перенесённое",
    "patient_question": "Вопросы к врачу",
    "other": "Другая информация",
}


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class IntakeEntry(BaseModel):
    """One thing the patient submitted. `original_text` is never modified after creation."""

    id: str  # "E1", "E2", ...
    source: Literal["text", "audio"]
    original_text: str  # verbatim submission (for audio: the raw transcription)
    text: str  # current text after patient corrections / review edits
    question_slot: str | None = None  # slot of the follow-up question this answered, if any
    question_text: str | None = None
    audio_filename: str | None = None
    # Set by the intake graph when the patient says this entry corrects an earlier one ("Поправка: …").
    # The earlier entry is kept unchanged; the summary uses the corrected information.
    corrects_entry_id: str | None = None
    created_at: str = Field(default_factory=utcnow_iso)

    @property
    def was_edited(self) -> bool:
        return self.text.strip() != self.original_text.strip()


# ---- Structured extraction (output of the LLM or the demo extractor) -----------------


class ExtractedFact(BaseModel):
    category: FactCategory
    statement: str = Field(description="Concise statement in the patient's language (Russian), close to their own words.")
    source_entry_ids: list[str] = Field(description="IDs of the entries that state this, e.g. ['E2'].")
    certainty: Literal["reported", "uncertain"] = Field(
        description="'uncertain' when the patient hedged, guessed or gave an approximation they flagged as unsure."
    )
    negated: bool = Field(description="True only if the patient explicitly denied it (e.g. 'no fever').")
    time_reference: str | None = Field(
        description="Time expression exactly as the patient gave it (e.g. 'около трёх дней назад'), or null."
    )


class ExtractedContradiction(BaseModel):
    description: str
    source_entry_ids: list[str]


class Extraction(BaseModel):
    facts: list[ExtractedFact]
    contradictions: list[ExtractedContradiction]
    summary: str = Field(
        description="2-4 neutral sentences IN RUSSIAN summarising ONLY what the patient reported. No diagnosis, no advice."
    )


# ---- Doctor report --------------------------------------------------------------------


class ReportItem(BaseModel):
    label: str
    category: str | None = None
    text: str
    source_entry_ids: list[str] = Field(default_factory=list)
    certainty: Literal["reported", "uncertain"] = "reported"
    time_reference: str | None = None
    edited_by_reviewer: bool = False
    flags: list[str] = Field(default_factory=list)


class MissingItem(BaseModel):
    field: str
    label: str
    reason: Literal["not_reported", "skipped_by_patient"]


class ContradictionItem(BaseModel):
    description: str
    source_entry_ids: list[str] = Field(default_factory=list)


class AIGenerated(BaseModel):
    """Everything written by the AI (not by the patient) lives here, clearly separated."""

    summary: str
    generator: str  # e.g. "openai:gpt-6-luna" or "demo-template"
    notes: list[str] = Field(default_factory=list)
    edited_by_reviewer: bool = False


class ReportMeta(BaseModel):
    schema_version: str = SCHEMA_VERSION
    generated_at: str = Field(default_factory=utcnow_iso)
    mode: Literal["live", "demo"]
    models: dict[str, str] = Field(default_factory=dict)
    patient_label: str = "Пациент (личные данные не собираются)"
    fictional_example: bool = False
    intake_completed: bool = False
    disclaimer: str = (
        "Прототип для демонстрации — не является клинически проверенной системой. "
        "Содержит сведения со слов пациента, упорядоченные программой; это не диагноз, "
        "информацию должен проверить врач на приёме."
    )


class DoctorReport(BaseModel):
    meta: ReportMeta
    main_concern: ReportItem | None
    symptom_details: list[ReportItem] = Field(default_factory=list)
    chronology: list[ReportItem] = Field(default_factory=list)
    progression: list[ReportItem] = Field(default_factory=list)
    health_events: list[ReportItem] = Field(default_factory=list)
    medications: list[ReportItem] = Field(default_factory=list)
    allergies: list[ReportItem] = Field(default_factory=list)
    medical_history: list[ReportItem] = Field(default_factory=list)
    other_reported: list[ReportItem] = Field(default_factory=list)
    negative_findings: list[ReportItem] = Field(default_factory=list)
    missing_information: list[MissingItem] = Field(default_factory=list)
    uncertainties: list[ReportItem] = Field(default_factory=list)
    contradictions: list[ContradictionItem] = Field(default_factory=list)
    patient_questions: list[ReportItem] = Field(default_factory=list)
    ai_generated: AIGenerated
    # Optional, separately labelled AI/rule sections (never mixed with patient statements).
    specialist_guidance: dict | None = None
    urgent_check: dict | None = None
    source_entries: list[IntakeEntry] = Field(default_factory=list)
