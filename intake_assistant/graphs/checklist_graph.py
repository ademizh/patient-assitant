"""Doctor's instructions -> checklist (+ medication explanations) as a LangGraph state graph (Week 10).

    START ─► extract_items ─► check_against_source ──(problems, 1st try)──► extract_items   (one retry)
                                       │
                                       └─(ok, or retry used)──► flag_issues ──┬─(medications to explain)──► explain_medications ─► END
                                                                              └─(nothing to explain)───────────────────────────► END

* extract_items        - one model call (or demo rules): split the instructions into items with structured
                         fields: action, medication name, dose, frequency, duration, deadline.
* check_against_source - every item must quote the doctor verbatim; every medication field and deadline must
                         be a verbatim fragment of that quote. Anything else is dropped or left empty.
* flag_issues          - ambiguous wording, missing details and contradictions are flagged (never resolved);
                         issues the patient already marked as clarified stay clarified.
* explain_medications  - for each medicine: runs the explanation graph (a graph inside a graph) to say what the
                         medicine is usually used for, from the knowledge base, then a medication-specific check
                         removes any dose or schedule that the doctor did not write.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from ..actions import build_actions
from ..records import ActionItem, ConsultationItem, DoctorInstruction, Explanation, InstructionIssue
from ..text_utils import split_sentences, squash
from .explain_graph import explain_graph, ExplainContext

MAX_ATTEMPTS = 2


class ChecklistState(TypedDict, total=False):
    instructions: list[DoctorInstruction]
    previous_actions: list[ActionItem]
    previous_issues: list[InstructionIssue]
    explain_medications: bool
    attempts: int
    extraction: Any
    unmatched: list[str]
    actions: list[ActionItem]
    issues: list[InstructionIssue]
    notes: list[str]
    problems: list[str]


@dataclass
class ChecklistContext:
    provider: Any
    store: Any = None
    top_k: int = 4
    min_relevance: float = 0.3
    is_demo: bool = False


# ---------------------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------------------


def extract_items(state: ChecklistState, runtime: Runtime[ChecklistContext]) -> dict:
    provider = runtime.context.provider
    feedback = "\n".join(f"- {p}" for p in state.get("problems") or [])
    fn = provider.extract_actions
    if feedback and "feedback" in inspect.signature(fn).parameters:
        extraction, unmatched = fn(state["instructions"], feedback=feedback)
    else:
        extraction, unmatched = fn(state["instructions"])
    return {"extraction": extraction, "unmatched": unmatched, "attempts": state.get("attempts", 0) + 1}


def check_against_source(state: ChecklistState) -> dict:
    actions, issues, notes = build_actions(state["extraction"], state["instructions"],
                                           previous=state.get("previous_actions") or [])
    problems = list(notes)
    for a in actions:
        for f in a.flags:
            problems.append(f"Item {a.id}: {f.replace('_', ' ')} (copy the doctor's wording exactly).")
    return {"actions": actions, "issues": issues, "notes": notes, "problems": problems}


def flag_issues(state: ChecklistState) -> dict:
    resolved = {(i.kind, tuple(i.action_ids)) for i in state.get("previous_issues") or [] if i.resolved}
    issues = state.get("issues") or []
    for i in issues:
        i.resolved = (i.kind, tuple(i.action_ids)) in resolved
    notes = list(state.get("notes") or [])
    if state.get("problems") and state.get("attempts", 0) >= MAX_ATTEMPTS:
        notes.append("Проверка по словам врача не пройдена и после повторного разбора: неподтверждённые поля "
                     "не показаны, их нужно уточнить у врача.")
    return {"issues": issues, "notes": notes}


_DOSING = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:мг|мкг|г\b|мл|ед|ме\b|таб\w*|капс\w*|капел\w*|раз\w*|mg|ml|mcg)", re.IGNORECASE)


def strip_dosing(background: str, statement: str) -> tuple[str, bool]:
    """Medication-specific check: general text may not contain doses or schedules the doctor did not write."""
    kept, removed = [], False
    doctor = squash(statement)
    for line in background.splitlines():
        sents = split_sentences(line) or [line]
        ok = [s for s in sents if not any(squash(m.group(0)) not in doctor for m in _DOSING.finditer(s))]
        removed = removed or len(ok) < len(sents)
        if ok:
            kept.append(" ".join(ok))
    return "\n".join(kept).strip(), removed


def explain_medications(state: ChecklistState, runtime: Runtime[ChecklistContext]) -> dict:
    ctx = runtime.context
    sub_ctx = ExplainContext(provider=ctx.provider, store=ctx.store, top_k=ctx.top_k,
                             min_relevance=ctx.min_relevance, is_demo=ctx.is_demo)
    prev = {(p.id, squash(p.source_statement)): p.explanation for p in state.get("previous_actions") or []}
    actions = state["actions"]
    for a in actions:
        if a.category != "medication" or not a.medication or not a.medication.name:
            continue
        cached = prev.get((a.id, squash(a.source_statement)))
        if cached is not None:  # instruction unchanged -> reuse, no new model call
            a.explanation = cached
            continue
        item = ConsultationItem(id=a.id, kind="term", source="text", original_text=a.source_statement,
                                text=a.source_statement)
        x: Explanation = explain_graph.invoke({"item": item, "focus": a.medication.name}, context=sub_ctx)["explanation"]
        if x.background_status == "answered":
            text, removed = strip_dosing(x.background, a.source_statement)
            if removed:
                x.notes.append("Удалено предложение с дозировкой или схемой приёма, которой нет в словах врача: "
                               "принимайте так, как написал врач.")
            if text:
                x.background = text
            else:
                x.background_status, x.background, x.sources = "insufficient_evidence", "", []
        x.ask_doctor = [f"Зачем мне назначен «{a.medication.name}» и чего ожидать от приёма?"] + [
            q for q in x.ask_doctor if "означает лично в моём случае" not in q]
        a.explanation = x
    return {"actions": actions}


# ---------------------------------------------------------------------------------------
# Routing + build
# ---------------------------------------------------------------------------------------


def route_after_check(state: ChecklistState) -> Literal["extract_items", "flag_issues"]:
    if state.get("problems") and state.get("attempts", 0) < MAX_ATTEMPTS:
        return "extract_items"
    return "flag_issues"


def route_after_flags(state: ChecklistState) -> Literal["explain_medications", "__end__"]:
    if state.get("explain_medications") and any(
        a.category == "medication" and a.medication and a.medication.name for a in state.get("actions") or []
    ):
        return "explain_medications"
    return END


def build_checklist_graph():
    g = StateGraph(ChecklistState, context_schema=ChecklistContext)
    g.add_node("extract_items", extract_items)
    g.add_node("check_against_source", check_against_source)
    g.add_node("flag_issues", flag_issues)
    g.add_node("explain_medications", explain_medications)
    g.add_edge(START, "extract_items")
    g.add_edge("extract_items", "check_against_source")
    g.add_conditional_edges("check_against_source", route_after_check, ["extract_items", "flag_issues"])
    g.add_conditional_edges("flag_issues", route_after_flags, ["explain_medications", END])
    g.add_edge("explain_medications", END)
    return g.compile()


checklist_graph = build_checklist_graph()


@dataclass
class ChecklistResult:
    actions: list[ActionItem]
    issues: list[InstructionIssue]
    notes: list[str]
    unmatched: list[str]


def run_checklist(instructions: list[DoctorInstruction], provider, store=None, *, previous_actions=None,
                  previous_issues=None, explain_medications: bool = True, top_k: int = 4,
                  min_relevance: float = 0.3, is_demo: bool = False) -> ChecklistResult:
    ctx = ChecklistContext(provider=provider, store=store, top_k=top_k, min_relevance=min_relevance, is_demo=is_demo)
    final = checklist_graph.invoke(
        {"instructions": instructions, "previous_actions": previous_actions or [],
         "previous_issues": previous_issues or [], "explain_medications": explain_medications},
        context=ctx,
    )
    return ChecklistResult(final["actions"], final["issues"], final.get("notes", []), final.get("unmatched", []))
