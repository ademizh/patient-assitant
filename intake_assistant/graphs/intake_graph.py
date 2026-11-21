"""Symptom intake before the appointment as a LangGraph state graph (Weeks 8-9).

The assistant asks ONE question, waits for the patient, processes the answer and chooses the next
question. Waiting is done with `interrupt()`: the graph stops in the middle of `ask_patient`, its state
is saved by the checkpointer (SQLite file), and it continues later from exactly that point when the
app calls `graph.invoke(Command(resume=reply), config)`. The session can be resumed after closing the
page or restarting the app, because the checkpoint lives on disk under the record's id (thread_id).

    START ─► choose_next ──(finished)──────────────────────────────────────────► END
                  │
                  └─(not finished)─► ask_patient  ⏸ interrupt: waits for the patient's reply
                                         │
            ┌──────────────┬─────────────┼───────────────┬─────────────┐
       (answer)         (skip)        (finish)        (retry)
            ▼              ▼             ▼               │
        add_entry      skip_topic     finish ─► END      │
            ▼              │                             │
     detect_correction     │                             │
            ▼              │                             │
         extract ◄─────────┼─────────────────────────────┘
            ▼              │
      check_urgent         │
            ▼              ▼
        choose_next ◄──────┘   (loop)

State holds plain JSON-like data only (dicts, strings), so every step can be checkpointed.
The AI provider and the urgent-rules file are passed in the runtime context instead.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from ..intake import QUESTIONS, IntakeSession
from ..providers import ProviderError
from ..records import AppointmentRecord, sync_questions_from_intake
from ..text_utils import content_tokens
from ..urgent import check_entries

# ---------------------------------------------------------------------------------------
# State and context
# ---------------------------------------------------------------------------------------


class IntakeState(TypedDict, total=False):
    intake: dict  # IntakeSession.model_dump(): entries, skipped topics, extraction…
    urgent_check: dict | None  # UrgentCheck.model_dump()
    slot: str | None  # topic of the question currently asked (None = free story / all covered)
    question: str | None  # question text currently shown to the patient
    reply: dict | None  # what the patient answered: {"action": "answer"|"skip"|"finish"|"retry", ...}
    last_entry_id: str | None
    error: str | None  # provider error of the last extraction (shown in the UI, never crashes the graph)


@dataclass
class IntakeContext:
    provider: Any
    pathway: dict | None = None


def _session(state: IntakeState) -> IntakeSession:
    return IntakeSession.model_validate(state.get("intake") or {})


# ---------------------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------------------


def choose_next(state: IntakeState) -> dict:
    """Pick the next uncovered topic (deterministic order; never a diagnostic question)."""
    s = _session(state)
    slot = s.current_slot()
    return {"slot": slot, "question": QUESTIONS[slot] if slot else None, "reply": None}


def ask_patient(state: IntakeState) -> dict:
    """Pause here until the patient replies. `interrupt` returns the value passed in Command(resume=…)."""
    reply = interrupt({"question": state.get("question"), "slot": state.get("slot")})
    return {"reply": reply}


def add_entry(state: IntakeState) -> dict:
    s = _session(state)
    r = state["reply"] or {}
    entry = s.add_entry(r.get("text", ""), source=r.get("source", "text"), original_text=r.get("original"),
                        question_slot=state.get("slot"), audio_filename=r.get("filename"))
    return {"intake": s.model_dump(), "last_entry_id": entry.id}


_CORRECTION_CUE = re.compile(
    r"(?:поправк\w*|исправ\w*|ошиб(?:ся|лась|лись|ка)|на\s+самом\s+деле|точнее|неправильно\s+(?:сказал|написал)\w*)",
    re.IGNORECASE,
)


def find_corrected_entry(text: str, earlier: list) -> str | None:
    """If `text` starts like a correction, return the id of the earlier entry it most likely corrects."""
    if not earlier or not _CORRECTION_CUE.search((text or "")[:60]):
        return None
    toks = content_tokens(text)
    scored = [(len(toks & content_tokens(e.text)), i, e.id) for i, e in enumerate(earlier)]
    best = max(scored)
    return best[2] if best[0] > 0 else earlier[-1].id


def detect_correction(state: IntakeState) -> dict:
    """A correction is stored as a NEW entry that points to the earlier one; nothing is overwritten."""
    s = _session(state)
    entry = s.entries[-1]
    target = find_corrected_entry(entry.text, s.entries[:-1])
    if not target:
        return {}
    entry.corrects_entry_id = target
    entry.question_slot = None  # a correction is not an answer to the question currently shown
    entry.question_text = None
    return {"intake": s.model_dump()}


def extract(state: IntakeState, runtime: Runtime[IntakeContext]) -> dict:
    s = _session(state)
    if not s.entries:
        return {"error": None}
    provider = runtime.context.provider
    try:
        s.extraction = provider.extract(s.entries)
        s.extraction_generator = provider.name
    except ProviderError as exc:  # keep the patient's entry; the UI offers "retry"
        return {"intake": s.model_dump(), "error": str(exc)}
    return {"intake": s.model_dump(), "error": None}


def check_urgent(state: IntakeState, runtime: Runtime[IntakeContext]) -> dict:
    pathway = runtime.context.pathway
    s = _session(state)
    if pathway is None or not s.entries:
        return {}
    return {"urgent_check": check_entries(s.entries, pathway).model_dump()}


def skip_topic(state: IntakeState) -> dict:
    s = _session(state)
    s.skip_current()
    return {"intake": s.model_dump()}


def finish(state: IntakeState) -> dict:
    s = _session(state)
    s.finish()
    return {"intake": s.model_dump(), "slot": None, "question": None}


# ---------------------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------------------


def route_after_choose(state: IntakeState) -> Literal["ask_patient", "__end__"]:
    return END if _session(state).finished else "ask_patient"


def route_reply(state: IntakeState) -> Literal["add_entry", "skip_topic", "finish", "extract"]:
    action = (state.get("reply") or {}).get("action")
    if action == "answer" and (state["reply"].get("text") or "").strip():
        return "add_entry"
    if action == "skip":
        return "skip_topic"
    if action == "finish":
        return "finish"
    return "extract"  # "retry" (re-process after an error) or an empty answer


def build_intake_graph(checkpointer=None):
    g = StateGraph(IntakeState, context_schema=IntakeContext)
    for name, fn in [("choose_next", choose_next), ("ask_patient", ask_patient), ("add_entry", add_entry),
                     ("detect_correction", detect_correction), ("extract", extract), ("check_urgent", check_urgent),
                     ("skip_topic", skip_topic), ("finish", finish)]:
        g.add_node(name, fn)
    g.add_edge(START, "choose_next")
    g.add_conditional_edges("choose_next", route_after_choose, ["ask_patient", END])
    g.add_conditional_edges("ask_patient", route_reply, ["add_entry", "skip_topic", "finish", "extract"])
    g.add_edge("add_entry", "detect_correction")
    g.add_edge("detect_correction", "extract")
    g.add_edge("extract", "check_urgent")
    g.add_edge("check_urgent", "choose_next")
    g.add_edge("skip_topic", "choose_next")
    g.add_edge("finish", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


# ---------------------------------------------------------------------------------------
# IntakeFlow: the bridge between the graph and the appointment record used by the UI / API
# ---------------------------------------------------------------------------------------


@dataclass
class TurnResult:
    question: str | None
    slot: str | None
    finished: bool
    error: str | None = None
    questions_added: int = 0
    trace: list[str] | None = None  # names of the nodes that ran in this turn (for debugging / the API)


class IntakeFlow:
    """One compiled graph + one checkpointer. thread_id = appointment record id."""

    def __init__(self, db_path: Path | None = None):
        self._lock = threading.Lock()  # Streamlit runs sessions in threads; one SQLite connection is shared
        if db_path is None:
            self.checkpointer = InMemorySaver()
        else:
            from langgraph.checkpoint.sqlite import SqliteSaver

            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
            self.checkpointer = SqliteSaver(self._conn)
        self.graph = build_intake_graph(self.checkpointer)

    @staticmethod
    def config(record_id: str) -> dict:
        return {"configurable": {"thread_id": record_id}}

    def _sync(self, record: AppointmentRecord, ctx: IntakeContext) -> None:
        """Make the paused graph hold the same intake as the record (it may have been edited elsewhere)."""
        cfg = self.config(record.id)
        snap = self.graph.get_state(cfg)
        dump = record.intake.model_dump()
        urgent = record.urgent_check.model_dump() if record.urgent_check else None
        if not snap.next:  # no checkpoint yet, or the previous run ended (finished)
            self.graph.invoke({"intake": dump, "urgent_check": urgent, "error": None}, cfg, context=ctx)
        elif snap.values.get("intake") != dump:
            # Human-in-the-loop edit of a paused graph: replace the intake, then pause again at ask_patient.
            self.graph.update_state(cfg, {"intake": dump, "urgent_check": urgent}, as_node="check_urgent")
            self.graph.invoke(None, cfg, context=ctx)

    def current(self, record: AppointmentRecord) -> dict | None:
        """The question the paused graph is waiting on, or None."""
        snap = self.graph.get_state(self.config(record.id))
        return snap.interrupts[0].value if snap.interrupts else None

    def send(self, record: AppointmentRecord, reply: dict, provider, pathway: dict | None = None) -> TurnResult:
        """Resume the graph with the patient's reply and copy the new state back into the record."""
        ctx = IntakeContext(provider=provider, pathway=pathway)
        with self._lock:
            if record.intake.finished and reply.get("action") in ("answer", "resume"):
                record.intake.resume()
            self._sync(record, ctx)
            cfg = self.config(record.id)
            trace: list[str] = []
            if reply.get("action") != "resume" and self.graph.get_state(cfg).next:
                # stream(..., "updates") yields {node_name: update} after every step -> a trace of the run
                for step in self.graph.stream(Command(resume=reply), cfg, context=ctx, stream_mode="updates"):
                    trace.extend("ask_patient ⏸" if k == "__interrupt__" else k for k in step)
            values = self.graph.get_state(cfg).values
        record.intake = IntakeSession.model_validate(values["intake"])
        if values.get("urgent_check"):
            from ..records import UrgentCheck

            record.urgent_check = UrgentCheck.model_validate(values["urgent_check"])
        added = sync_questions_from_intake(record) if not values.get("error") else 0
        return TurnResult(question=values.get("question"), slot=values.get("slot"), finished=record.intake.finished,
                          error=values.get("error"), questions_added=added, trace=trace)

    def forget(self, record_id: str) -> None:
        """Delete every checkpoint of this record (called when the patient deletes the record)."""
        with self._lock:
            self.checkpointer.delete_thread(record_id)
