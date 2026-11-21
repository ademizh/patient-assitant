"""Optional orientation: which TYPE of doctor the patient could contact first.

Grounded only in the knowledge base (with sources shown). It is never a diagnosis. When the
sources are insufficient, the patient is asked to clarify and the default is a general
practitioner. Urgency is handled separately by the pathway rules (urgent.py).
"""

from __future__ import annotations

from .grounding import build_sources, validate_citations
from .rag.qa import retrieve
from .records import AppointmentRecord, SpecialistGuidance, SpecialistOption
from .text_utils import is_diagnostic

DEFAULT_ADVICE = (
    "Если вы не уверены, к кому обратиться, начните с терапевта (врача общей практики): он оценит ситуацию "
    "и при необходимости направит к профильному специалисту."
)
NOT_A_DIAGNOSIS = (
    "Это ориентир по выбору специалиста на основе справочных материалов, а не диагноз. "
    "Окончательное решение принимает врач."
)
CONTEXT_CATEGORIES = ("main_concern", "location", "character", "associated_symptoms", "progression", "duration")


def build_context(record: AppointmentRecord) -> str:
    """The patient's own statements used for the search (no re-asking: taken from the intake)."""
    ex = record.intake.extraction
    parts: list[str] = []
    if ex is not None:
        for cat in CONTEXT_CATEGORIES:
            parts.extend(f.statement for f in ex.facts if f.category == cat and not f.negated)
    if not parts:
        parts = [e.text for e in record.intake.entries[:3]]
    return " ".join(dict.fromkeys(p.strip() for p in parts if p.strip()))


def suggest(record: AppointmentRecord, store, provider, top_k: int, min_relevance: float) -> SpecialistGuidance:
    context = build_context(record)
    sig = record.intake_signature()
    if not context:
        return SpecialistGuidance(
            status="insufficient_evidence", default_advice=DEFAULT_ADVICE,
            uncertainty="Пока нет описания жалоб.",
            clarifying_question="Сначала расскажите о жалобах в разделе «Рассказ о симптомах».",
            generator="—", based_on_signature=sig,
        )
    if store is None or store.count() == 0:
        return SpecialistGuidance(
            status="no_documents", default_advice=DEFAULT_ADVICE, query=context,
            uncertainty="В базе знаний нет документов, на которые можно опереться.",
            generator="—", based_on_signature=sig,
        )
    _, relevant = retrieve(context, store, provider, top_k + 2, min_relevance)
    if not relevant:
        return SpecialistGuidance(
            status="insufficient_evidence", default_advice=DEFAULT_ADVICE, query=context,
            uncertainty="Подходящих фрагментов в базе знаний не найдено.",
            clarifying_question="Опишите, пожалуйста, подробнее, что именно беспокоит, где и как давно.",
            generator=provider.name, based_on_signature=sig,
        )
    advice = provider.suggest_specialist(context, relevant)
    by_id = {c["chunk_id"]: c for c in relevant}
    notes: list[str] = []
    options: list[SpecialistOption] = []
    for opt in advice.options[:3] if advice.sufficient_evidence else []:
        reason, valid, removed = validate_citations(opt.reason, opt.cited_chunk_ids, by_id)
        if not valid:
            notes.append(f"Вариант «{opt.specialist}» убран: нет ссылки на найденный источник.")
            continue
        if is_diagnostic(reason):
            notes.append(f"Вариант «{opt.specialist}» убран: обоснование звучало как диагноз.")
            continue
        sources, src_notes = build_sources(valid, by_id, advice.quotes)
        notes.extend(src_notes)
        options.append(SpecialistOption(specialist=opt.specialist.strip(), reason=reason.strip(), sources=sources))
    status = "suggested" if options else "insufficient_evidence"
    clarifying = advice.clarifying_question
    if not options and not clarifying:
        clarifying = "Опишите, пожалуйста, подробнее, что именно беспокоит, где и как давно."
    return SpecialistGuidance(
        status=status, options=options,
        uncertainty=(advice.uncertainty or "").strip(),
        clarifying_question=clarifying, default_advice=DEFAULT_ADVICE, query=context,
        generator=provider.name, notes=list(dict.fromkeys(notes)), based_on_signature=sig,
    )
