"""Missing information is reported as missing - never invented, never turned into negatives (Russian input)."""

from __future__ import annotations

from intake_assistant.intake import IntakeSession
from intake_assistant.report import build_report
from intake_assistant.schemas import Extraction, ExtractedFact


def _report(session, extraction):
    return build_report(session, extraction, mode="demo", generator="test")


def _fact(cat, text, ids, negated=False, certainty="reported", time=None):
    return ExtractedFact(category=cat, statement=text, source_entry_ids=ids, certainty=certainty,
                         negated=negated, time_reference=time)


def test_unmentioned_topics_are_missing_not_negative(demo_provider):
    s = IntakeSession()
    s.add_entry("У меня сыпь на левой руке.")
    s.extraction = demo_provider.extract(s.entries)
    r = _report(s, s.extraction)
    missing = {m.field: m.reason for m in r.missing_information}
    for slot in ("allergies", "medications", "medical_history", "severity", "progression"):
        assert missing.get(slot) == "not_reported"
    assert r.negative_findings == [] and r.allergies == [] and r.medications == []


def test_skipped_question_is_missing_with_reason(demo_provider):
    s = IntakeSession()
    s.add_entry("Болит колено.")
    s.extraction = demo_provider.extract(s.entries)
    assert s.current_slot() == "onset"
    s.skip_current()
    assert s.current_slot() == "progression"  # next question, one at a time
    assert s.current_question().startswith("Как менялись симптомы")
    r = _report(s, s.extraction)
    missing = {m.field: m.reason for m in r.missing_information}
    assert missing["onset"] == "skipped_by_patient"
    assert not any((n.category or "") == "onset" for n in r.negative_findings)


def test_dont_know_is_uncertain_not_negative(demo_provider):
    s = IntakeSession()
    s.add_entry("Болит живот.")
    s.add_entry("Не знаю, может быть, на прошлой неделе.", question_slot="onset")
    s.extraction = demo_provider.extract(s.entries)
    r = _report(s, s.extraction)
    assert r.negative_findings == []
    assert any("Не знаю" in u.text for u in r.uncertainties)
    assert "onset" not in {m.field for m in r.missing_information}


def test_model_negation_without_explicit_denial_is_downgraded():
    s = IntakeSession()
    s.add_entry("У меня кашель.")
    ext = Extraction(
        facts=[_fact("main_concern", "Кашель", ["E1"]),
               _fact("associated_symptoms", "Температуры нет", ["E1"], negated=True)],  # never stated
        contradictions=[], summary="Пациент сообщает о кашле.",
    )
    r = _report(s, ext)
    assert r.negative_findings == []
    assert any("negation_not_explicit" in u.flags for u in r.uncertainties)


def test_explicit_denial_is_kept_as_negative():
    s = IntakeSession()
    s.add_entry("У меня кашель. Температуры нет.")
    ext = Extraction(
        facts=[_fact("main_concern", "Кашель", ["E1"]), _fact("associated_symptoms", "Температуры нет", ["E1"], negated=True)],
        contradictions=[], summary="",
    )
    assert [n.text for n in _report(s, ext).negative_findings] == ["Температуры нет"]


def test_facts_without_valid_source_are_removed():
    s = IntakeSession()
    s.add_entry("Болит спина с понедельника.")
    ext = Extraction(
        facts=[_fact("main_concern", "Боль в спине", ["E1"]),
               _fact("medical_history", "Мочекаменная болезнь в анамнезе", ["E7"]),  # E7 does not exist
               _fact("medications", "Принимает аспирин", [])],
        contradictions=[], summary="",
    )
    r = _report(s, ext)
    assert r.medical_history == [] and r.medications == []
    assert any("не ссылалось ни на одну запись" in n for n in r.ai_generated.notes)


def test_low_support_statement_is_flagged():
    s = IntakeSession()
    s.add_entry("Болит голова.")
    ext = Extraction(facts=[_fact("main_concern", "Кластерная мигрень с аурой", ["E1"])], contradictions=[], summary="")
    assert "low_textual_support" in _report(s, ext).main_concern.flags


def test_approximate_time_kept_verbatim_and_uncertain(demo_provider):
    s = IntakeSession()
    s.add_entry("Головокружение. Началось примерно три недели назад.")
    s.extraction = demo_provider.extract(s.entries)
    r = _report(s, s.extraction)
    onset = [i for i in r.symptom_details if i.category == "onset"]
    assert onset and onset[0].certainty == "uncertain"
    assert onset[0].time_reference == "примерно три недели назад"
    assert any("примерно три недели назад" in c.label for c in r.chronology)


def test_diagnostic_wording_in_summary_is_flagged():
    s = IntakeSession()
    s.add_entry("Болит голова.")
    ext = Extraction(facts=[_fact("main_concern", "Головная боль", ["E1"])], contradictions=[],
                     summary="Скорее всего, у вас мигрень.")
    assert any("похожие на диагноз" in n for n in _report(s, ext).ai_generated.notes)


def test_finish_anytime_and_resume():
    s = IntakeSession()
    s.add_entry("Болит горло.")
    s.finish()
    assert s.current_question() is None
    s.resume()
    assert s.current_question() is not None
