"""Minimal HTTP API for website integration (Week 11). The same LangGraph workflows as the Streamlit app.

Run:   uvicorn api:app --reload        then open http://127.0.0.1:8000/docs (interactive documentation)

Endpoints
  GET  /health                          mode and model
  POST /intake/sessions                 start a symptom intake -> first question
  POST /intake/sessions/{id}/reply      answer / skip / finish / resume / retry -> next question
  GET  /intake/sessions/{id}            current state of the intake
  DELETE /intake/sessions/{id}          delete the record, its files and its saved graph state
  POST /explain                         explain a term, note or lab result (explanation graph)
  POST /checklist                       doctor's instructions -> checklist (+ medication explanations)

Every response has a `status`: waiting_for_patient | finished | answered | needs_clarification | error.
Prototype: no authentication. Do not expose it to the internet.
"""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from intake_assistant.config import load_settings
from intake_assistant.graphs.checklist_graph import run_checklist
from intake_assistant.graphs.explain_graph import ExplainContext, explain_graph
from intake_assistant.graphs.intake_graph import IntakeFlow
from intake_assistant.providers import ProviderError, get_provider
from intake_assistant.rag.store import DocumentStore
from intake_assistant.records import AppointmentRecord, ConsultationItem, DoctorInstruction, RecordStore
from intake_assistant.urgent import load_pathway

log = logging.getLogger("patient_api")  # logs node names and error types only - never patient text

settings = load_settings()
provider = get_provider(settings)
store = DocumentStore(settings.chroma_path, settings.library_path, settings.collection_prefix, provider)
records = RecordStore(settings.appointments_path)
flow = IntakeFlow(settings.appointments_path / "intake_graph_checkpoints.sqlite")
try:
    pathway = load_pathway(settings.urgent_rules_path)
except (OSError, ValueError):
    pathway = None

app = FastAPI(title="Помощник пациента — API (прототип)", version="0.3.0")


# ---------------------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------------------


class Reply(BaseModel):
    action: Literal["answer", "skip", "finish", "resume", "retry"] = "answer"
    text: str | None = Field(None, description="The patient's answer (for action=answer).")


class IntakeResponse(BaseModel):
    session_id: str
    status: Literal["waiting_for_patient", "finished", "error"]
    question: str | None = None
    topics_done: int
    topics_total: int
    summary: str | None = None
    urgent_message: str | None = None
    entries: int
    error: str | None = None
    trace: list[str] = Field(default_factory=list, description="Graph nodes that ran in this request.")


class ExplainRequest(BaseModel):
    kind: Literal["term", "note", "test_result", "question"]
    text: str = Field(..., min_length=1, max_length=20000)
    focus: str | None = None


class ChecklistRequest(BaseModel):
    instructions: list[str] = Field(..., min_length=1)
    explain_medications: bool = True


# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------


def _load(session_id: str) -> AppointmentRecord:
    try:
        if records.exists(session_id):
            return records.load(session_id)
    except ValueError:
        pass
    raise HTTPException(404, "Сессия не найдена.")


def _intake_response(rec: AppointmentRecord, error: str | None = None, trace: list[str] | None = None) -> IntakeResponse:
    done, total = rec.intake.progress()
    u = rec.urgent_check
    urgent = u.message if u and any(not m.negated for m in u.matches) else None
    status = "error" if error else "finished" if rec.intake.finished else "waiting_for_patient"
    return IntakeResponse(
        session_id=rec.id, status=status, question=None if rec.intake.finished else rec.intake.current_question(),
        topics_done=done, topics_total=total,
        summary=rec.intake.extraction.summary if rec.intake.extraction else None,
        urgent_message=urgent, entries=len(rec.intake.entries), error=error, trace=trace or [],
    )


def _provider_error(exc: ProviderError) -> HTTPException:
    log.warning("provider error: %s", type(exc).__name__)
    return HTTPException(502, str(exc))


# ---------------------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "mode": "demo" if settings.is_demo else "live",
            "model": None if settings.is_demo else settings.chat_model, "documents": store.count()}


@app.post("/intake/sessions", response_model=IntakeResponse)
def start_intake() -> IntakeResponse:
    rec = AppointmentRecord(title="Запись из API")
    records.save(rec)
    result = flow.send(rec, {"action": "resume"}, provider, pathway)  # runs to the first question and pauses
    records.save(rec)
    return _intake_response(rec, result.error, result.trace)


@app.post("/intake/sessions/{session_id}/reply", response_model=IntakeResponse)
def reply_intake(session_id: str, body: Reply) -> IntakeResponse:
    rec = _load(session_id)
    if body.action == "answer" and not (body.text or "").strip():
        raise HTTPException(422, "Для action=answer нужен непустой text.")
    result = flow.send(rec, body.model_dump(), provider, pathway)
    records.save(rec)
    log.info("intake turn %s: %s", session_id, " -> ".join(result.trace or []))
    return _intake_response(rec, result.error, result.trace)


@app.get("/intake/sessions/{session_id}", response_model=IntakeResponse)
def get_intake(session_id: str) -> IntakeResponse:
    return _intake_response(_load(session_id))


@app.delete("/intake/sessions/{session_id}")
def delete_intake(session_id: str) -> dict:
    _load(session_id)
    records.delete(session_id)
    flow.forget(session_id)
    return {"status": "deleted", "session_id": session_id}


@app.post("/explain")
def explain(body: ExplainRequest) -> dict:
    item = ConsultationItem(id="C1", kind=body.kind, source="text", original_text=body.text, text=body.text)
    ctx = ExplainContext(provider=provider, store=store, top_k=settings.top_k,
                         min_relevance=settings.min_relevance, is_demo=settings.is_demo)
    trace: list[str] = []
    final: dict = {}
    try:
        for step in explain_graph.stream({"item": item, "focus": body.focus}, context=ctx, stream_mode="updates"):
            for node, update in step.items():
                trace.append(node)
                final.update(update or {})
    except ProviderError as exc:
        raise _provider_error(exc) from None
    x = final["explanation"]
    log.info("explain: %s", " -> ".join(trace))
    return {"status": "answered" if x.background_status == "answered" else "needs_clarification",
            "explanation": x.model_dump(), "trace": trace}


@app.post("/checklist")
def checklist(body: ChecklistRequest) -> dict:
    instructions = [DoctorInstruction(id=f"D{i}", source="text", original_text=t, text=t)
                    for i, t in enumerate(body.instructions, start=1) if t.strip()]
    if not instructions:
        raise HTTPException(422, "Пустые указания врача.")
    try:
        result = run_checklist(instructions, provider, store, explain_medications=body.explain_medications,
                               top_k=settings.top_k, min_relevance=settings.min_relevance, is_demo=settings.is_demo)
    except ProviderError as exc:
        raise _provider_error(exc) from None
    open_issues = [i for i in result.issues if not i.resolved]
    return {"status": "needs_clarification" if open_issues else "answered",
            "actions": [a.model_dump() for a in result.actions],
            "issues": [i.model_dump() for i in result.issues],
            "notes": result.notes, "unrecognised_sentences": result.unmatched}
