"""Build the bundled FICTIONAL appointment record (before -> during -> after) for demos and tests.

Everything here is fictional and labelled as such. The pipeline is the real one: extraction,
specialist orientation, explanations and the checklist all run through the configured provider
(demo heuristics or the live model) and the same guards as user input.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

from .actions import mark_done
from .consult import explain_item
from .intake import IntakeSession
from .labs import extract_patient_document
from .records import (
    AppointmentRecord,
    ConsultationItem,
    DoctorInstruction,
    PatientQuestion,
    RecordStore,
    next_id,
    sync_questions_from_intake,
)
from .schemas import utcnow_iso

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_patient"
SCRIPT_PATH = SAMPLE_DIR / "fictional_patient_script.json"


def load_script() -> dict:
    return json.loads(SCRIPT_PATH.read_text(encoding="utf-8"))


def prewritten_transcript(audio_file: str) -> str:
    sidecar = SAMPLE_DIR / audio_file.replace(".wav", ".transcript.txt")
    return sidecar.read_text(encoding="utf-8").strip()


def build_fictional_session(provider, run_extraction: bool = True) -> IntakeSession:
    """Replay the scripted fictional intake.

    For the audio answer the clip's prepared transcript is used directly, so the example is
    identical in demo and live mode. Real transcription is exercised by recording or uploading
    speech in the UI (the bundled clip can also be uploaded there).
    """
    script = load_script()
    session = IntakeSession(fictional_example=True)
    for step in script["intake_steps"]:
        action = step["action"]
        if action == "entry":
            if step["source"] == "audio":
                text = prewritten_transcript(step["audio_file"])
                session.add_entry(text, source="audio", original_text=text,
                                  question_slot=step.get("slot"), audio_filename=step["audio_file"])
            else:
                session.add_entry(step["text"], source="text", question_slot=step.get("slot"))
        elif action == "skip":
            if step["slot"] not in session.skipped_slots:
                session.skipped_slots.append(step["slot"])
        elif action == "finish":
            session.finish()
    if run_extraction:
        session.extraction = provider.extract(session.entries)
        session.extraction_generator = provider.name
    return session


def build_fictional_record(
    provider,
    store,
    record_store: RecordStore,
    top_k: int,
    min_relevance: float,
    is_demo: bool,
    pathway: dict | None = None,
    today: date | None = None,
) -> AppointmentRecord:
    """Full fictional record. `store` should already contain the demo knowledge documents."""
    from .report import build_report
    from .specialist import suggest
    from .urgent import check_entries

    script = load_script()
    today = today or date.today()
    after = script["after_steps"]
    rec = AppointmentRecord(
        title=script["title"], fictional=True,
        appointment_date=(today - timedelta(days=after["appointment_days_ago"])).isoformat(),
    )

    # ---- Before -------------------------------------------------------------------------
    rec.intake = build_fictional_session(provider)
    sync_questions_from_intake(rec)
    if pathway is not None:
        rec.urgent_check = check_entries(rec.intake.entries, pathway)
    rec.specialist_guidance = suggest(rec, store, provider, top_k, min_relevance)
    rec.report = build_report(rec.intake, rec.intake.extraction, mode="demo" if is_demo else "live",
                              generator=rec.intake.extraction_generator or provider.name)
    attach_guidance_to_report(rec)
    rec.report_signature = rec.intake_signature()
    record_store.save(rec)

    # ---- During -------------------------------------------------------------------------
    for step in script["consultation_steps"]:
        upload_id = None
        audio = None
        if step["source"] == "upload":
            data = (SAMPLE_DIR / step["file"]).read_bytes()
            up = record_store.save_upload(rec, step["file"], data, "test_result")
            doc = extract_patient_document(step["file"], data)
            up.n_pages, up.text_extracted = doc.n_pages, True
            text, upload_id = doc.text, up.id
        elif step["source"] == "audio":
            text, audio = prewritten_transcript(step["audio_file"]), step["audio_file"]
        else:
            text = step["text"]
        item = ConsultationItem(id=next_id("C", rec.consultation_items), kind=step["kind"],
                                source=step["source"], original_text=text, text=text,
                                upload_id=upload_id, audio_filename=audio)
        rec.consultation_items.append(item)
        if step["kind"] == "question":
            rec.patient_questions.append(PatientQuestion(id=next_id("Q", rec.patient_questions), text=text,
                                                         source="consultation"))
        if step.get("explain"):
            ex = explain_item(item, store, provider, top_k, min_relevance, is_demo, focus=step.get("focus"))
            if step.get("save"):
                ex.id = next_id("X", rec.explanations)
                rec.explanations.append(ex)
    for q in script.get("extra_questions", []):
        rec.patient_questions.append(PatientQuestion(id=next_id("Q", rec.patient_questions), text=q, source="manual"))
    for q in rec.patient_questions[:2]:
        q.asked = True

    # ---- After --------------------------------------------------------------------------
    fname = script["instructions_file"]
    data = (SAMPLE_DIR / fname).read_bytes()
    up = record_store.save_upload(rec, fname, data, "instructions")
    doc = extract_patient_document(fname, data)
    up.text_extracted = True
    rec.instructions.append(DoctorInstruction(id="D1", source="upload", original_text=doc.text, text=doc.text,
                                              upload_id=up.id))
    from .graphs.checklist_graph import run_checklist  # LangGraph workflow (Week 10)

    result = run_checklist(rec.instructions, provider, store, explain_medications=store is not None,
                           top_k=top_k, min_relevance=min_relevance, is_demo=is_demo)
    rec.actions, rec.instruction_issues = result.actions, result.issues
    rec.actions_signature = rec.instructions_signature()
    rec.actions_generator = provider.name
    for a in rec.actions:
        if any(a.source_statement.startswith(p) for p in after["done"]):
            mark_done(a, True, utcnow_iso())
        for prefix, days in after["patient_target_days_from_today"].items():
            if a.source_statement.startswith(prefix) and not a.doctor_due_date:
                a.patient_target_date = (today + timedelta(days=days)).isoformat()
    record_store.save(rec)
    return rec


def attach_guidance_to_report(rec: AppointmentRecord) -> None:
    """Copy the (separately labelled) specialist orientation and urgent check into the report."""
    if rec.report is None:
        return
    rec.report.specialist_guidance = rec.specialist_guidance.model_dump() if rec.specialist_guidance else None
    rec.report.urgent_check = rec.urgent_check.model_dump() if rec.urgent_check else None
