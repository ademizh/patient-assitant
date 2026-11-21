"""Turn intake entries + extraction into a structured, editable doctor report.

Guardrails applied here (independent of which model produced the extraction):
  * facts must cite existing entries - otherwise they are dropped (never shown as fact);
  * facts with little word overlap with their cited entries are flagged for verification;
  * a "negative finding" is only kept as negative if the cited entry contains an explicit
    negation; otherwise it is moved to uncertainties - missing info never becomes negative;
  * missing slots are computed from coverage, never inferred from silence;
  * diagnostic wording in the AI summary is flagged;
  * nothing from the medical-document (RAG) collection is used - this module does not
    import the RAG package at all (see tests/test_separation.py).
"""

from __future__ import annotations

from .intake import IntakeSession
from .schemas import (
    SLOT_LABELS,
    SLOT_ORDER,
    AIGenerated,
    ContradictionItem,
    DoctorReport,
    ExtractedFact,
    Extraction,
    IntakeEntry,
    MissingItem,
    ReportItem,
    ReportMeta,
)
from .timeline import estimate_days_ago
from .text_utils import has_negation_cue, is_diagnostic, is_hedged, is_unknown, lexical_support

SYMPTOM_DETAIL_CATEGORIES = [
    "onset",
    "duration",
    "severity",
    "location",
    "character",
    "associated_symptoms",
    "modifying_factors",
]

LOW_SUPPORT_THRESHOLD = 0.4


def _guard_facts(facts: list[ExtractedFact], entries: dict[str, IntakeEntry]) -> tuple[list[tuple[ExtractedFact, list[str]]], list[str]]:
    """Validate facts against their source entries. Returns (kept facts with flags, notes)."""
    kept: list[tuple[ExtractedFact, list[str]]] = []
    notes: list[str] = []
    for f in facts:
        valid_ids = [i for i in f.source_entry_ids if i in entries]
        if not valid_ids:
            notes.append(
                f"Удалено утверждение в разделе «{SLOT_LABELS.get(f.category, f.category)}»: "
                "оно не ссылалось ни на одну запись пациента."
            )
            continue
        f = f.model_copy(update={"source_entry_ids": valid_ids})
        flags: list[str] = []
        source_text = " ".join(entries[i].text for i in valid_ids)
        if lexical_support(f.statement, source_text) < LOW_SUPPORT_THRESHOLD:
            flags.append("low_textual_support")
        if f.negated and not has_negation_cue(source_text):
            # The model marked a denial that the patient never explicitly made.
            f = f.model_copy(update={"negated": False, "certainty": "uncertain"})
            flags.append("negation_not_explicit")
        if not f.negated and f.certainty == "reported" and (is_hedged(f.statement) or is_unknown(f.statement)):
            f = f.model_copy(update={"certainty": "uncertain"})
        kept.append((f, flags))
    return kept, notes


def _item(f: ExtractedFact, flags: list[str]) -> ReportItem:
    return ReportItem(
        label=SLOT_LABELS.get(f.category, f.category),
        category=f.category,
        text=f.statement,
        source_entry_ids=f.source_entry_ids,
        certainty=f.certainty,
        time_reference=f.time_reference,
        flags=flags,
    )


def _first_index(item: ReportItem, order: dict[str, int]) -> int:
    return min((order.get(i, 10**6) for i in item.source_entry_ids), default=10**6)


def build_report(
    session: IntakeSession,
    extraction: Extraction,
    *,
    mode: str,
    generator: str,
    models: dict[str, str] | None = None,
) -> DoctorReport:
    entries = session.entry_map()
    order = {e.id: n for n, e in enumerate(session.entries)}
    guarded, notes = _guard_facts(extraction.facts, entries)

    main_concern: ReportItem | None = None
    symptom_details: list[ReportItem] = []
    events: list[ReportItem] = []
    meds: list[ReportItem] = []
    allergies: list[ReportItem] = []
    history: list[ReportItem] = []
    other: list[ReportItem] = []
    negatives: list[ReportItem] = []
    uncertainties: list[ReportItem] = []
    chronology: list[ReportItem] = []
    progression: list[ReportItem] = []
    questions: list[ReportItem] = []
    covered: set[str] = set()

    for f, flags in guarded:
        item = _item(f, flags)
        covered.add(f.category)
        if "negation_not_explicit" in flags:
            item.flags = flags
            uncertainties.append(item.model_copy(update={"label": f"{item.label} (возможно отрицание — уточнить)"}))
            continue
        if f.negated:
            negatives.append(item)
            continue
        if f.certainty == "uncertain":
            uncertainties.append(item)
        if f.time_reference:
            chronology.append(item.model_copy(update={"label": f"{f.time_reference} — {item.label}"}))
        if f.category == "main_concern":
            if main_concern is None:
                main_concern = item
            else:
                other.append(item.model_copy(update={"label": "Дополнительная жалоба"}))
        elif f.category == "progression":
            progression.append(item)
        elif f.category == "patient_question":
            questions.append(item)
        elif f.category in SYMPTOM_DETAIL_CATEGORIES:
            symptom_details.append(item)
        elif f.category == "relevant_events":
            events.append(item)
        elif f.category == "medications":
            meds.append(item)
        elif f.category == "allergies":
            allergies.append(item)
        elif f.category == "medical_history":
            history.append(item)
        else:
            other.append(item)

    # Chronology: oldest first by an APPROXIMATE estimate used only for ordering; the patient's
    # time expression is kept verbatim and never converted to a date. Undatable items follow
    # in order of mention.
    def _chrono_key(it: ReportItem):
        days = estimate_days_ago(it.time_reference)
        return (0, -days, _first_index(it, order)) if days is not None else (1, 0, _first_index(it, order))

    chronology.sort(key=_chrono_key)
    symptom_details.sort(key=lambda it: (SYMPTOM_DETAIL_CATEGORIES.index(it.category or "character"), _first_index(it, order)))

    # Missing information: computed purely from coverage, never turned into negatives.
    missing: list[MissingItem] = []
    for slot in SLOT_ORDER:
        if slot in covered:
            continue
        reason = "skipped_by_patient" if slot in session.skipped_slots else "not_reported"
        missing.append(MissingItem(field=slot, label=SLOT_LABELS[slot], reason=reason))

    contradictions = [
        ContradictionItem(description=c.description, source_entry_ids=[i for i in c.source_entry_ids if i in entries])
        for c in extraction.contradictions
    ]

    summary = extraction.summary.strip() or "Резюме не сформировано."
    if is_diagnostic(summary):
        notes.append(
            "В резюме ИИ есть формулировки, похожие на диагноз или совет. Они оставлены для прозрачности, "
            "но не должны рассматриваться как клиническая интерпретация."
        )
    n_flag = sum(1 for f, fl in guarded if "low_textual_support" in fl)
    if n_flag:
        notes.append(f"Утверждений со слабым совпадением со словами пациента: {n_flag} — сверьте с исходными записями.")
    if not session.finished:
        notes.append("На момент формирования отчёта опрос не был отмечен как завершённый.")

    return DoctorReport(
        meta=ReportMeta(
            mode="demo" if mode == "demo" else "live",
            models=models or {},
            fictional_example=session.fictional_example,
            patient_label="ВЫМЫШЛЕННЫЙ ДЕМО-ПАЦИЕНТ" if session.fictional_example else "Пациент (личные данные не собираются)",
            intake_completed=session.finished,
        ),
        main_concern=main_concern,
        symptom_details=symptom_details,
        chronology=chronology,
        progression=progression,
        health_events=events,
        medications=meds,
        allergies=allergies,
        medical_history=history,
        other_reported=other,
        negative_findings=negatives,
        missing_information=missing,
        uncertainties=uncertainties,
        contradictions=contradictions,
        patient_questions=questions,
        ai_generated=AIGenerated(summary=summary, generator=generator, notes=notes),
        source_entries=[e.model_copy() for e in session.entries],
    )



# ---- Report sections used by the editor and the PDF renderer --------------------------

LIST_SECTIONS: list[tuple[str, str]] = [
    ("symptom_details", "Подробности о симптомах"),
    ("progression", "Изменения симптомов со временем"),
    ("chronology", "Хронология (сроки — со слов пациента)"),
    ("health_events", "Предшествующие события со здоровьем"),
    ("medications", "Лекарства (со слов пациента)"),
    ("allergies", "Аллергии (со слов пациента)"),
    ("medical_history", "Хронические заболевания и перенесённое (со слов пациента)"),
    ("other_reported", "Другая информация от пациента"),
    ("negative_findings", "Явно отрицаемые пациентом симптомы"),
    ("uncertainties", "Неточная или приблизительная информация"),
    ("patient_questions", "Вопросы пациента к врачу"),
]


def report_to_json(report: DoctorReport) -> str:
    return report.model_dump_json(indent=2)
