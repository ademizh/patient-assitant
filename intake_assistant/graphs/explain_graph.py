"""Explanation workflow as a LangGraph state graph (Week 7).

    START
      │
      ▼
  prepare_input ──► retrieve ──┬─(no documents / no relevant passages)──► finalize ──► END
                               │
                               └─(evidence found)──► generate ──► validate ──┬─(problems, 1st try)──► generate
                                                                             │
                                                                             └─(ok, or retry used)──► finalize ──► END

* prepare_input - builds the search query and parses lab rows (name, value, unit, interval are exact
                  substrings of the document). Missing intervals and unrecognised units are flagged here,
                  BEFORE any model call, and are never filled in.
* retrieve      - searches the knowledge base (Chroma) for supporting passages.
* generate      - one call to the model (OpenAI SDK, or offline demo heuristics).
* validate      - grounding checks: quotes must be verbatim, citations must point to retrieved passages,
                  no personal verdicts or diagnoses, no numbers that are absent from the sources.
* finalize      - assembles the three-part explanation and the questions for the doctor.

External objects (provider, store) are passed through the *runtime context*, not the state:
the state only holds plain data that changes while the graph runs.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from ..actions import locate_verbatim
from ..grounding import build_sources, language_note, number_note, validate_citations
from ..labs import parse_lab_lines
from ..rag.qa import retrieve as rag_retrieve
from ..records import ConsultationItem, Explanation, LabRow
from ..text_utils import content_tokens, is_diagnostic, is_personal_evaluation, split_sentences

MAX_ATTEMPTS = 2  # the first generation + one regeneration after a failed validation

NO_KB_TEXT = (
    "В базе знаний не нашлось материалов, объясняющих это. Общее объяснение не формировалось — "
    "задайте этот вопрос врачу."
)


# ---------------------------------------------------------------------------------------
# 1. State and context
# ---------------------------------------------------------------------------------------


class ExplainState(TypedDict, total=False):
    # input
    item: ConsultationItem
    focus: str | None
    # prepare_input
    query: str
    lab_rows: list[LabRow]
    input_questions: list[str]  # deterministic questions found BEFORE the model (missing interval, unit…)
    # retrieve
    status: Literal["answered", "insufficient_evidence", "no_documents"]
    relevant: list[dict]
    # generate
    attempts: int
    raw: Any  # ExplanationOut from the provider
    # validate
    problems: list[str]  # what failed in the last validation (empty = passed)
    document_states: list[str]
    background: str
    sources: list
    notes: list[str]
    # finalize
    explanation: Explanation


@dataclass
class ExplainContext:
    provider: Any
    store: Any
    top_k: int = 4
    min_relevance: float = 0.3
    is_demo: bool = False


# ---------------------------------------------------------------------------------------
# 2. Nodes - plain functions: (state, runtime) -> partial state update
# ---------------------------------------------------------------------------------------


def build_query(item: ConsultationItem, focus: str | None = None) -> str:
    if focus and focus.strip():
        return focus.strip()
    if item.kind == "test_result":
        names = [r.name for r in parse_lab_lines(item.text)]
        if names:
            return ", ".join(dict.fromkeys(names))
    return item.text.strip()[:400]


def prepare_input(state: ExplainState) -> dict:
    item, focus = state["item"], state.get("focus")
    rows = parse_lab_lines(item.text) if item.kind == "test_result" else []
    if focus and rows:
        ftoks = content_tokens(focus)
        rows = [r for r in rows if ftoks & content_tokens(r.name)] or rows
    questions: list[str] = []
    for r in rows:
        if not r.reference_range:
            questions.append(f"Для показателя «{r.name}» в документе не указан референсный интервал. "
                             "Какой интервал применим в моём случае?")
        if not r.unit:
            questions.append(f"Для показателя «{r.name}» в документе не распознана единица измерения. "
                             "В каких единицах указан результат?")
        if r.flag:
            questions.append(f"В бланке рядом с показателем «{r.name}» стоит отметка «{r.flag}». Что она означает для меня?")
    if rows:
        questions.append("Как эти результаты связаны с моими жалобами и нужно ли что-то предпринять?")
    return {"query": build_query(item, focus), "lab_rows": rows, "input_questions": questions,
            "attempts": 0, "notes": [], "problems": []}


def retrieve(state: ExplainState, runtime: Runtime[ExplainContext]) -> dict:
    ctx = runtime.context
    if ctx.store is None or ctx.store.count() == 0:
        return {"status": "no_documents", "relevant": []}
    _, relevant = rag_retrieve(state["query"], ctx.store, ctx.provider, ctx.top_k, ctx.min_relevance)
    return {"status": "insufficient_evidence", "relevant": relevant}


def _accepts_feedback(fn) -> bool:
    try:
        return "feedback" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def generate(state: ExplainState, runtime: Runtime[ExplainContext]) -> dict:
    provider = runtime.context.provider
    item = state["item"]
    args = (item.kind, item.text, state["query"], state["relevant"])
    feedback = "\n".join(f"- {p}" for p in state.get("problems") or [])
    if feedback and _accepts_feedback(provider.explain):
        raw = provider.explain(*args, feedback=feedback)
    else:
        raw = provider.explain(*args)
    return {"raw": raw, "attempts": state.get("attempts", 0) + 1}


def _filter_background(text: str) -> tuple[str, list[str]]:
    """Remove sentences that judge the patient's personal result or sound diagnostic."""
    notes: list[str] = []
    kept_lines = []
    for line in (text or "").splitlines():
        sents = split_sentences(line) or [line]
        kept = []
        for s in sents:
            if is_personal_evaluation(s):
                notes.append("Удалено предложение, оценивающее ваш личный результат как норму или отклонение — "
                             "такую оценку может дать только врач.")
                continue
            if is_diagnostic(s):
                notes.append("Удалено предложение, похожее на диагноз или назначение.")
                continue
            kept.append(s)
        if kept:
            prefix = "- " if line.lstrip().startswith("- ") and not kept[0].startswith("- ") else ""
            kept_lines.append(prefix + " ".join(kept))
    return "\n".join(kept_lines).strip(), list(dict.fromkeys(notes))


def validate(state: ExplainState) -> dict:
    """Run every grounding check. Keep only what passed; list what failed in `problems`."""
    item, out, relevant = state["item"], state["raw"], state["relevant"]
    problems: list[str] = []
    notes: list[str] = []

    # a) the patient's document: only verbatim fragments survive
    document_states, dropped = [], 0
    for q in out.document_quotes:
        exact = locate_verbatim(q, item.text)
        if exact:
            document_states.append(exact)
        else:
            dropped += 1
    if dropped:
        notes.append(f"Удалено фрагментов, которых нет дословно в вашем документе: {dropped}.")
        problems.append(f"{dropped} document quote(s) are not verbatim in the patient's document.")
    if not document_states and item.kind in ("note", "question"):
        document_states = [item.text.strip()]
    lab_lines = {r.line for r in state.get("lab_rows", [])}
    document_states = [d for d in dict.fromkeys(document_states) if d not in lab_lines]

    # b) background: citations, personal verdicts, numbers
    by_id = {c["chunk_id"]: c for c in relevant}
    text, valid, removed = validate_citations(out.background, out.cited_chunk_ids, by_id)
    if removed:
        notes.append(f"Удалено ссылок на не найденные источники: {len(removed)}.")
        problems.append(f"Citations to passages that were not retrieved: {', '.join(removed)}.")
    background, sources = "", []
    if out.background_sufficient and not valid:
        problems.append("The background cites none of the retrieved passages.")
    if out.background_sufficient and valid:
        text, filt_notes = _filter_background(text)
        if filt_notes:
            notes.extend(filt_notes)
            problems.append("Sentences judged the patient's own result or sounded like a diagnosis.")
        if text:
            background = text
            sources, src_notes = build_sources(valid, by_id, out.quotes)
            if src_notes:
                notes.extend(src_notes)
                problems.append("Some source quotes are not verbatim in the cited passage.")
            num = number_note(background, *[by_id[c]["text"] for c in valid], item.text, state["query"])
            if num:
                notes.append(num)
                problems.append("The background contains numbers that are absent from the sources.")
    return {"problems": problems, "notes": list(dict.fromkeys(notes)), "document_states": document_states,
            "background": background, "sources": sources}


def finalize(state: ExplainState, runtime: Runtime[ExplainContext]) -> dict:
    ctx = runtime.context
    item, query = state["item"], state["query"]
    background = state.get("background", "")
    status = "answered" if background else state.get("status", "insufficient_evidence")
    notes = list(state.get("notes") or [])
    if state.get("problems") and state.get("attempts", 0) >= MAX_ATTEMPTS:
        notes.append("Проверка источников не пройдена и после повторной генерации: неподтверждённые части "
                     "удалены или отмечены выше.")

    ask = list(state.get("input_questions") or [])
    out = state.get("raw")
    if out is not None:
        ask.extend(q.strip() for q in out.ask_doctor if q and q.strip() and not is_diagnostic(q))
    if status == "answered" and item.kind in ("term", "note", "question"):
        subject = query if len(query) <= 80 else query[:77] + "…"
        ask.append(f"Что «{subject}» означает лично в моём случае и что мне нужно делать?")
    if status != "answered":
        subject = query if len(query) <= 120 else query[:117] + "…"
        ask.append(f"Объясните, пожалуйста, что означает: «{subject}».")

    document_states = state.get("document_states")
    if document_states is None:  # the model was never called
        document_states = [item.text.strip()] if item.kind in ("note", "question") else []

    explanation = Explanation(
        id="",
        item_id=item.id,
        kind=item.kind,
        query=query,
        document_states=document_states,
        lab_rows=state.get("lab_rows", []),
        background=background if status == "answered" else NO_KB_TEXT if status == "insufficient_evidence" else "",
        background_status=status,  # type: ignore[arg-type]
        sources=state.get("sources", []) if status == "answered" else [],
        ask_doctor=list(dict.fromkeys(ask)),
        notes=list(dict.fromkeys(notes)),
        generator=getattr(ctx.provider, "name", ""),
        answer_language_note=language_note(state.get("sources", []), ctx.is_demo) if status == "answered" else None,
    )
    return {"explanation": explanation}


# ---------------------------------------------------------------------------------------
# 3. Routing functions - decide which edge to follow
# ---------------------------------------------------------------------------------------


def route_after_retrieve(state: ExplainState) -> Literal["generate", "finalize"]:
    return "generate" if state.get("relevant") else "finalize"


def route_after_validate(state: ExplainState) -> Literal["generate", "finalize"]:
    if state.get("problems") and state.get("attempts", 0) < MAX_ATTEMPTS:
        return "generate"  # one regeneration, with the problems sent back as feedback
    return "finalize"


# ---------------------------------------------------------------------------------------
# 4. Build and compile the graph (once, at import time)
# ---------------------------------------------------------------------------------------


def build_explain_graph():
    g = StateGraph(ExplainState, context_schema=ExplainContext)
    g.add_node("prepare_input", prepare_input)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("validate", validate)
    g.add_node("finalize", finalize)

    g.add_edge(START, "prepare_input")
    g.add_edge("prepare_input", "retrieve")
    g.add_conditional_edges("retrieve", route_after_retrieve, ["generate", "finalize"])
    g.add_edge("generate", "validate")
    g.add_conditional_edges("validate", route_after_validate, ["generate", "finalize"])
    g.add_edge("finalize", END)
    return g.compile()


explain_graph = build_explain_graph()


def run_explain(item: ConsultationItem, store, provider, top_k: int, min_relevance: float, is_demo: bool,
                focus: str | None = None) -> Explanation:
    ctx = ExplainContext(provider=provider, store=store, top_k=top_k, min_relevance=min_relevance, is_demo=is_demo)
    final = explain_graph.invoke({"item": item, "focus": focus}, context=ctx)
    return final["explanation"]
