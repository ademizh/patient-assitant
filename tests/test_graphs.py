"""LangGraph workflows: explanation (retry), intake (interrupt + checkpointer), checklist (medications), API."""

from __future__ import annotations

import importlib
import sys

import pytest

from intake_assistant.graphs.checklist_graph import run_checklist, strip_dosing
from intake_assistant.graphs.explain_graph import MAX_ATTEMPTS, ExplainContext, explain_graph
from intake_assistant.graphs.intake_graph import IntakeFlow, find_corrected_entry
from intake_assistant.providers import ExplanationOut, ProviderError
from intake_assistant.records import AppointmentRecord, ConsultationItem, DoctorInstruction

from .conftest import ROOT


def _lab(text="Ферритин 9 нг/мл ↓\nГемоглобин 128 г/л 120–140"):
    return ConsultationItem(id="C1", kind="test_result", source="text", original_text=text, text=text)


def _nodes(graph, inputs, ctx):
    return [n for step in graph.stream(inputs, context=ctx, stream_mode="updates") for n in step]


# ---- explanation graph -------------------------------------------------------------------


def test_explain_graph_runs_all_steps_once_when_valid(indexed_store, demo_provider):
    ctx = ExplainContext(demo_provider, indexed_store, 4, 0.12, True)
    nodes = _nodes(explain_graph, {"item": _lab(), "focus": "Ферритин"}, ctx)
    assert nodes == ["prepare_input", "retrieve", "generate", "validate", "finalize"]


def test_explain_graph_skips_the_model_without_evidence(store, demo_provider):
    ctx = ExplainContext(demo_provider, store, 4, 0.12, True)
    nodes = _nodes(explain_graph, {"item": _lab("Гомоцистеин 11 мкмоль/л")}, ctx)
    assert nodes == ["prepare_input", "retrieve", "finalize"]  # no documents -> no generate


def test_missing_interval_and_unit_are_flagged_before_the_model(indexed_store, demo_provider):
    ctx = ExplainContext(demo_provider, indexed_store, 4, 0.12, True)
    x = explain_graph.invoke({"item": _lab("Ферритин 9\nГемоглобин 128 г/л 120–140")}, context=ctx)["explanation"]
    asks = " ".join(x.ask_doctor)
    assert "«Ферритин» в документе не указан референсный интервал" in asks
    assert "«Ферритин» в документе не распознана единица" in asks
    assert all(r.reference_range in (None, "120–140") for r in x.lab_rows)  # nothing filled in


class CountingBadExplainer:
    """Always cites a passage that was never retrieved -> validation fails every time."""

    name = "bad"

    def __init__(self, base):
        self.base, self.calls, self.feedback = base, 0, []
        self.embedding_signature = base.embedding_signature

    def embed(self, t):
        return self.base.embed(t)

    def search_queries(self, q):
        return [q]

    def explain(self, kind, patient_text, query, chunks, feedback=None):
        self.calls += 1
        self.feedback.append(feedback)
        return ExplanationOut(document_quotes=[], background="Выдуманное утверждение [abcdef-p1-c999].",
                              background_sufficient=True, cited_chunk_ids=["abcdef-p1-c999"], quotes=[], ask_doctor=[])


def test_failed_validation_regenerates_once_with_feedback_then_reports(indexed_store, demo_provider):
    bad = CountingBadExplainer(demo_provider)
    ctx = ExplainContext(bad, indexed_store, 4, 0.12, False)
    nodes = _nodes(explain_graph, {"item": _lab(), "focus": "Ферритин"}, ctx)
    assert nodes.count("generate") == MAX_ATTEMPTS == 2
    assert bad.feedback[0] is None and "not retrieved" in bad.feedback[1]
    x = explain_graph.invoke({"item": _lab(), "focus": "Ферритин"}, context=ctx)["explanation"]
    assert x.background_status == "insufficient_evidence" and not x.sources
    assert any("повторной генерации" in n for n in x.notes)


# ---- intake graph ------------------------------------------------------------------------


def test_intake_pauses_resumes_and_survives_restart(tmp_path, demo_provider):
    db = tmp_path / "ckpt.sqlite"
    rec = AppointmentRecord()
    flow = IntakeFlow(db)
    r = flow.send(rec, {"action": "answer", "text": "Болит голова две недели."}, demo_provider)
    assert r.slot == "onset" and r.trace[:2] == ["ask_patient", "add_entry"] and r.trace[-1] == "ask_patient ⏸"
    flow.send(rec, {"action": "skip"}, demo_provider)
    # "Restart the app": a new object reads the same SQLite file and is still waiting on the same question.
    flow2 = IntakeFlow(db)
    waiting = flow2.current(rec)
    assert waiting["slot"] == "progression"
    r = flow2.send(rec, {"action": "answer", "text": "Стали чаще."}, demo_provider)
    assert [e.question_slot for e in rec.intake.entries] == ["main_concern", "progression"]
    assert rec.intake.skipped_slots == ["onset"] and r.slot == "duration"


def test_intake_correction_is_a_new_entry_linked_to_the_old_one(tmp_path, demo_provider):
    rec = AppointmentRecord()
    flow = IntakeFlow(tmp_path / "c.sqlite")
    flow.send(rec, {"action": "answer", "text": "Голова болит две недели."}, demo_provider)
    flow.send(rec, {"action": "answer", "text": "Поправка: голова болит три недели, а не две."}, demo_provider)
    e1, e2 = rec.intake.entries
    assert e1.text == "Голова болит две недели." and e2.corrects_entry_id == "E1"
    assert e2.question_slot is None  # not counted as an answer to the question that was shown
    assert find_corrected_entry("Болит горло", rec.intake.entries) is None


def test_intake_edits_outside_the_graph_are_synced(tmp_path, demo_provider):
    rec = AppointmentRecord()
    flow = IntakeFlow(tmp_path / "c.sqlite")
    flow.send(rec, {"action": "answer", "text": "Болит ухо."}, demo_provider)
    rec.intake.update_entry_text("E1", "Болит левое ухо.")  # review tab
    flow.send(rec, {"action": "answer", "text": "Три дня назад."}, demo_provider)
    assert [e.text for e in rec.intake.entries] == ["Болит левое ухо.", "Три дня назад."]


def test_intake_provider_error_keeps_the_entry_and_retry_works(tmp_path, demo_provider):
    class Flaky:
        name, fail = "flaky", True

        def extract(self, entries):
            if self.fail:
                raise ProviderError("Ошибка сети.")
            return demo_provider.extract(entries)

    p = Flaky()
    rec = AppointmentRecord()
    flow = IntakeFlow(tmp_path / "c.sqlite")
    r = flow.send(rec, {"action": "answer", "text": "Болит спина."}, p)
    assert r.error == "Ошибка сети." and len(rec.intake.entries) == 1 and rec.intake.extraction is None
    p.fail = False
    r = flow.send(rec, {"action": "retry"}, p)
    assert r.error is None and rec.intake.extraction is not None and len(rec.intake.entries) == 1


def test_finish_resume_and_forget(tmp_path, demo_provider):
    rec = AppointmentRecord()
    flow = IntakeFlow(tmp_path / "c.sqlite")
    flow.send(rec, {"action": "answer", "text": "Кашель."}, demo_provider)
    assert flow.send(rec, {"action": "finish"}, demo_provider).finished
    r = flow.send(rec, {"action": "resume"}, demo_provider)
    assert not r.finished and flow.current(rec) is not None
    flow.forget(rec.id)
    assert flow.graph.get_state(flow.config(rec.id)).values == {}


# ---- checklist graph ---------------------------------------------------------------------

INSTR = ("1. Ибупрофен 400 мг при сильной головной боли, не чаще 2 раз в сутки.\n"
         "2. Магний B6 по 1 таблетке 2 раза в день в течение 1 месяца.\n"
         "3. Магний B6 принимать 1 раз в день утром.\n"
         "4. Сдать анализ крови на ферритин.")


def test_checklist_graph_explains_medications_and_keeps_doctor_wording(indexed_store, demo_provider):
    ins = [DoctorInstruction(id="D1", source="text", original_text=INSTR, text=INSTR)]
    r = run_checklist(ins, demo_provider, indexed_store, top_k=4, min_relevance=0.12, is_demo=True)
    meds = [a for a in r.actions if a.category == "medication"]
    assert [a.medication.dose for a in meds] == ["400 мг", "по 1 таблетке", None]
    ibu = meds[0].explanation
    assert ibu.background_status == "answered" and "нестероидным" in ibu.background
    assert ibu.sources and ibu.ask_doctor[0].startswith("Зачем мне назначен «Ибупрофен»")
    assert all(a.explanation is None for a in r.actions if a.category != "medication")
    assert any(i.kind == "contradiction" for i in r.issues)


def test_checklist_reuses_explanations_and_resolved_flags(indexed_store, demo_provider):
    ins = [DoctorInstruction(id="D1", source="text", original_text=INSTR, text=INSTR)]
    r1 = run_checklist(ins, demo_provider, indexed_store, top_k=4, min_relevance=0.12, is_demo=True)
    for i in r1.issues:
        i.resolved = True
    r2 = run_checklist(ins, demo_provider, indexed_store, previous_actions=r1.actions, previous_issues=r1.issues,
                       top_k=4, min_relevance=0.12, is_demo=True)
    assert r2.actions[0].explanation is r1.actions[0].explanation  # reused, no new model call
    assert all(i.resolved for i in r2.issues)


def test_medication_check_removes_doses_the_doctor_did_not_write():
    text, removed = strip_dosing("Препарат снижает боль. Обычно принимают 200 мг 3 раза в день.", "Ибупрофен 400 мг")
    assert removed and text == "Препарат снижает боль."
    text, removed = strip_dosing("Назначено 400 мг.", "Ибупрофен 400 мг")
    assert not removed


# ---- API ---------------------------------------------------------------------------------


@pytest.fixture()
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_MODE", "demo")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("LIBRARY_PATH", str(tmp_path / "lib"))
    monkeypatch.setenv("APPOINTMENTS_PATH", str(tmp_path / "appointments"))
    sys.path.insert(0, str(ROOT))
    sys.modules.pop("api", None)
    mod = importlib.import_module("api")
    for p in sorted((ROOT / "data" / "sample_documents").iterdir()):
        mod.store.add_file(p.name, p.read_bytes(), 900, 150)
    from fastapi.testclient import TestClient

    return TestClient(mod.app)


def test_api_intake_explain_checklist(api):
    s = api.post("/intake/sessions").json()
    assert s["status"] == "waiting_for_patient" and s["question"].startswith("Что вас беспокоит")
    sid = s["session_id"]
    r = api.post(f"/intake/sessions/{sid}/reply", json={"action": "answer", "text": "Болит голова."}).json()
    assert r["entries"] == 1 and "extract" in r["trace"] and r["question"].startswith("Когда это началось")
    assert api.post(f"/intake/sessions/{sid}/reply", json={"action": "answer"}).status_code == 422
    assert api.post(f"/intake/sessions/{sid}/reply", json={"action": "finish"}).json()["status"] == "finished"
    assert api.get("/intake/sessions/ffffffffffff").status_code == 404
    assert api.delete(f"/intake/sessions/{sid}").json()["status"] == "deleted"

    x = api.post("/explain", json={"kind": "test_result", "text": "Ферритин 9 нг/мл", "focus": "Ферритин"}).json()
    assert x["trace"] == ["prepare_input", "retrieve", "generate", "validate", "finalize"]
    assert x["explanation"]["lab_rows"][0]["value"] == "9"

    c = api.post("/checklist", json={"instructions": [INSTR]}).json()
    assert c["status"] == "needs_clarification" and len(c["actions"]) == 4
