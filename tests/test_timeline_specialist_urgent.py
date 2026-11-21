"""Before the appointment: timeline, symptom changes, patient questions, specialist orientation, urgent rules."""

from __future__ import annotations

import json

from intake_assistant.intake import IntakeSession
from intake_assistant.providers import SpecialistAdvice, SpecialistOptionOut
from intake_assistant.records import AppointmentRecord, sync_questions_from_intake
from intake_assistant.sample import build_fictional_session
from intake_assistant.specialist import DEFAULT_ADVICE, suggest
from intake_assistant.timeline import estimate_days_ago, find_time_reference, symptom_timeline
from intake_assistant.urgent import check_entries, load_pathway


def test_russian_time_expressions_kept_verbatim():
    cases = {
        "Начались примерно две недели назад, может чуть больше.": ("примерно две недели назад", 14),
        "Три недели назад я вышла на новую работу.": ("Три недели назад", 21),
        "В прошлом месяце была простуда.": ("В прошлом месяце", 30),
        "Около 3 дней назад": ("Около 3 дней назад", 3),
        "В детстве была сыпь.": ("В детстве", 365 * 25),
    }
    for sentence, (expr, days) in cases.items():
        assert find_time_reference(sentence) == expr
        assert estimate_days_ago(expr) == days


def test_symptom_timeline_orders_and_keeps_wording(demo_provider):
    s = build_fictional_session(demo_provider)
    rows = symptom_timeline(s.extraction.facts, {e.id: n for n, e in enumerate(s.entries)})
    assert [r.time_reference for r in rows][:3] == ["в детстве", "В прошлом месяце", "Три недели назад"]
    assert all(r.time_reference in r.statement for r in rows)


def test_progression_and_questions_collected(demo_provider):
    rec = AppointmentRecord(intake=build_fictional_session(demo_provider))
    assert any(f.category == "progression" for f in rec.intake.extraction.facts)
    assert sync_questions_from_intake(rec) == 2
    assert sync_questions_from_intake(rec) == 0  # no duplicates
    assert all(q.origin == "patient" and q.source == "intake" for q in rec.patient_questions)


def test_no_questions_answer_adds_nothing(demo_provider):
    s = IntakeSession()
    s.add_entry("Болит голова.")
    s.add_entry("Нет вопросов.", question_slot="patient_question")
    s.extraction = demo_provider.extract(s.entries)
    rec = AppointmentRecord(intake=s)
    assert sync_questions_from_intake(rec) == 0


def test_specialist_grounded_with_sources(demo_provider, indexed_store):
    rec = AppointmentRecord(intake=build_fictional_session(demo_provider))
    g = suggest(rec, indexed_store, demo_provider, 4, 0.12)
    assert g.status == "suggested"
    names = [o.specialist for o in g.options]
    assert "Невролог" in names and "Терапевт / врач общей практики" in names
    for o in g.options:
        assert o.sources and o.sources[0].title.startswith("ДЕМО-материал")
        assert f"[{o.sources[0].chunk_id}]" in o.reason
    assert g.default_advice == DEFAULT_ADVICE


def test_specialist_insufficient_falls_back_to_gp(demo_provider, indexed_store):
    s = IntakeSession()
    s.add_entry("Странное ощущение.")
    s.extraction = demo_provider.extract(s.entries)
    g = suggest(AppointmentRecord(intake=s), indexed_store, demo_provider, 4, 0.12)
    assert g.status == "insufficient_evidence" and not g.options
    assert g.clarifying_question and "терапевт" in g.default_advice


def test_specialist_no_documents_and_no_intake(demo_provider, store):
    assert suggest(AppointmentRecord(), store, demo_provider, 4, 0.12).clarifying_question
    rec = AppointmentRecord(intake=build_fictional_session(demo_provider))
    assert suggest(rec, store, demo_provider, 4, 0.12).status == "no_documents"


class FakeAdvisor:
    name = "fake"

    def __init__(self, base, advice):
        self.base, self.advice = base, advice
        self.embedding_signature = base.embedding_signature

    def embed(self, t):
        return self.base.embed(t)

    def search_queries(self, q):
        return [q]

    def suggest_specialist(self, ctx, chunks):
        self.chunks = chunks
        return self.advice


def test_specialist_drops_uncited_and_diagnostic_options(demo_provider, indexed_store):
    rec = AppointmentRecord(intake=build_fictional_session(demo_provider))
    probe = suggest(rec, indexed_store, demo_provider, 4, 0.12)
    cid = probe.options[0].sources[0].chunk_id
    advice = SpecialistAdvice(sufficient_evidence=True, uncertainty="", clarifying_question=None, quotes=[], options=[
        SpecialistOptionOut(specialist="Невролог", reason=f"Источник связывает повторяющиеся головные боли с неврологом [{cid}].", cited_chunk_ids=[cid]),
        SpecialistOptionOut(specialist="Кардиолог", reason="Так принято.", cited_chunk_ids=["deadbeef0000-p1-c001"]),
        SpecialistOptionOut(specialist="Офтальмолог", reason=f"Скорее всего, у вас мигрень [{cid}].", cited_chunk_ids=[cid]),
    ])
    g = suggest(rec, indexed_store, FakeAdvisor(demo_provider, advice), 4, 0.12)
    assert [o.specialist for o in g.options] == ["Невролог"]
    assert any("Кардиолог" in n for n in g.notes) and any("диагноз" in n for n in g.notes)


def test_urgent_rules_are_labelled_demo_and_handle_negation(demo_settings):
    pw = load_pathway(demo_settings.urgent_rules_path)
    assert pw["validated"] is False and "ДЕМОНСТРАЦИОННЫЕ" in pw["title"]
    s = IntakeSession()
    s.add_entry("Боли в груди нет. Вчера была внезапная сильная головная боль.")
    u = check_entries(s.entries, pw)
    assert {(m.label, m.negated) for m in u.matches} == {
        ("Боль или давление в груди", True), ("Внезапная сильнейшая головная боль", False)}
    assert "ДЕМОНСТРАЦИОННЫМ" in u.message and "103" in u.message
    assert u.pathway_validated is False


def test_validated_pathway_requires_reviewer(tmp_path, demo_settings):
    data = json.loads(demo_settings.urgent_rules_path.read_text(encoding="utf-8"))
    data["validated"] = True  # claims validation but names no reviewer
    p = tmp_path / "rules.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    assert load_pathway(p)["validated"] is False
    data["reviewed_by"] = "Клиническая группа X, протокол v1"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    assert load_pathway(p)["validated"] is True
