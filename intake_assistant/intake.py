"""Conversational intake state: entries, skipped questions and follow-up question selection.

Follow-up questions are chosen deterministically from the slots that are still uncovered.
This keeps the assistant predictable (one concise question at a time, never diagnostic)
while the extraction step (LLM or demo heuristics) decides which slots are covered.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .schemas import SLOT_LABELS, SLOT_ORDER, Extraction, IntakeEntry

QUESTIONS: dict[str, str] = {
    "main_concern": "Что вас беспокоит больше всего? Опишите своими словами, что вы хотите обсудить с врачом.",
    "onset": "Когда это началось? Можно указать примерно.",
    "progression": "Как менялись симптомы со временем: стали чаще, сильнее, слабее или остались такими же?",
    "duration": "Это беспокоит постоянно или периодически? Сколько длится один эпизод?",
    "severity": "Насколько это сильно по шкале от 0 (нет совсем) до 10 (сильнее невозможно представить)?",
    "location": "Где именно вы это ощущаете? Отдаёт ли куда-нибудь?",
    "associated_symptoms": "Замечали ли вы вместе с этим другие симптомы?",
    "modifying_factors": "Что облегчает или, наоборот, усиливает это состояние?",
    "relevant_events": "Было ли что-то важное для здоровья незадолго до начала: болезнь, травма, поездка, стресс, смена лекарств?",
    "medications": "Какие лекарства, витамины или добавки вы принимаете, в том числе от этой проблемы?",
    "allergies": "Есть ли у вас аллергия на лекарства, продукты или что-то ещё?",
    "medical_history": "Есть ли у вас хронические заболевания, перенесённые операции или госпитализации, о которых стоит знать врачу?",
    "patient_question": "Какие вопросы вы хотели бы задать врачу на приёме? Можно перечислить несколько.",
}


class IntakeSession(BaseModel):
    entries: list[IntakeEntry] = Field(default_factory=list)
    skipped_slots: list[str] = Field(default_factory=list)
    finished: bool = False
    fictional_example: bool = False
    extraction: Extraction | None = None
    extraction_generator: str | None = None

    # ---- entries ------------------------------------------------------------------

    def _next_id(self) -> str:
        return f"E{len(self.entries) + 1}"

    def add_entry(
        self,
        text: str,
        source: str = "text",
        original_text: str | None = None,
        question_slot: str | None = None,
        audio_filename: str | None = None,
    ) -> IntakeEntry:
        text = (text or "").strip()
        if not text:
            raise ValueError("Текст записи пуст.")
        entry = IntakeEntry(
            id=self._next_id(),
            source="audio" if source == "audio" else "text",
            original_text=(original_text if original_text is not None else text).strip(),
            text=text,
            question_slot=question_slot,
            question_text=QUESTIONS.get(question_slot) if question_slot else None,
            audio_filename=audio_filename,
        )
        self.entries.append(entry)
        self.finished = False
        return entry

    def update_entry_text(self, entry_id: str, new_text: str) -> None:
        for e in self.entries:
            if e.id == entry_id:
                e.text = new_text.strip() or e.original_text
                return
        raise KeyError(entry_id)

    def entry_map(self) -> dict[str, IntakeEntry]:
        return {e.id: e for e in self.entries}

    # ---- coverage & follow-up --------------------------------------------------------

    def covered_slots(self) -> set[str]:
        covered: set[str] = set()
        if self.extraction:
            for f in self.extraction.facts:
                covered.add(f.category)
        # A direct answer to a question covers that slot even if extraction missed it.
        for e in self.entries:
            if e.question_slot:
                covered.add(e.question_slot)
        # "Duration" is often answered together with onset/character; treat character as partial.
        return covered

    def open_slots(self) -> list[str]:
        covered = self.covered_slots()
        return [s for s in SLOT_ORDER if s not in covered and s not in self.skipped_slots]

    def current_slot(self) -> str | None:
        if self.finished:
            return None
        open_ = self.open_slots()
        return open_[0] if open_ else None

    def current_question(self) -> str | None:
        slot = self.current_slot()
        return QUESTIONS[slot] if slot else None

    def skip_current(self) -> str | None:
        slot = self.current_slot()
        if slot and slot not in self.skipped_slots:
            self.skipped_slots.append(slot)
        return slot

    def finish(self) -> None:
        self.finished = True

    def resume(self) -> None:
        self.finished = False

    def progress(self) -> tuple[int, int]:
        covered = self.covered_slots()
        done = sum(1 for s in SLOT_ORDER if s in covered or s in self.skipped_slots)
        return done, len(SLOT_ORDER)

    def slot_status(self) -> list[tuple[str, str]]:
        covered = self.covered_slots()
        out = []
        for s in SLOT_ORDER:
            if s in covered:
                status = "covered"
            elif s in self.skipped_slots:
                status = "skipped"
            else:
                status = "open"
            out.append((SLOT_LABELS[s], status))
        return out
