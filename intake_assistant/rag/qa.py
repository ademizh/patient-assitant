"""Question answering grounded strictly in retrieved chunks, with citation and quote validation."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..grounding import build_sources, language_note, number_note, validate_citations
from ..records import SourceRef

INSUFFICIENT_MESSAGE = (
    "В подключённых документах недостаточно сведений, чтобы ответить на этот вопрос. "
    "Ответ из общих знаний модели не формировался."
)
NO_DOCUMENTS_MESSAGE = "В базе знаний пока нет документов. Добавьте их в разделе «База знаний»."


@dataclass
class QAResult:
    status: str  # "answered" | "insufficient_evidence" | "no_documents"
    answer: str
    cited: list[dict] = field(default_factory=list)  # chunks actually cited (subset of retrieved)
    retrieved: list[dict] = field(default_factory=list)  # all chunks retrieved
    sources: list[SourceRef] = field(default_factory=list)
    model_note: str | None = None
    removed_citations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    language_note: str | None = None


def retrieve(question: str, store, provider, top_k: int, min_relevance: float) -> tuple[list[dict], list[dict]]:
    """(all retrieved, relevant). Uses the provider's query rewriting for cross-language search."""
    if store.count() == 0:
        return [], []
    queries = provider.search_queries(question)
    retrieved = store.query_multi(queries, top_k) if hasattr(store, "query_multi") else store.query(question, top_k)
    relevant = [c for c in retrieved if c["score"] >= min_relevance]
    return retrieved, relevant


def ask(question: str, store, provider, top_k: int, min_relevance: float, is_demo: bool = False) -> QAResult:
    question = (question or "").strip()
    if not question:
        raise ValueError("Введите вопрос.")
    if store.count() == 0:
        return QAResult("no_documents", NO_DOCUMENTS_MESSAGE)

    retrieved, relevant = retrieve(question, store, provider, top_k, min_relevance)
    if not relevant:
        # Do not even call the model - nothing to ground an answer in.
        return QAResult("insufficient_evidence", INSUFFICIENT_MESSAGE, retrieved=retrieved)

    result = provider.answer(question, relevant)
    by_id = {c["chunk_id"]: c for c in relevant}
    answer_text, valid, removed = validate_citations(result.answer, result.cited_chunk_ids, by_id)

    if not result.sufficient_evidence or not valid:
        return QAResult(
            "insufficient_evidence", INSUFFICIENT_MESSAGE, retrieved=retrieved,
            model_note=answer_text.strip() or None, removed_citations=removed,
        )
    sources, notes = build_sources(valid, by_id, getattr(result, "quotes", []))
    num = number_note(answer_text, *[by_id[c]["text"] for c in valid], question)
    if num:
        notes.append(num)
    return QAResult(
        "answered",
        answer_text.strip(),
        cited=[by_id[c] for c in valid],
        retrieved=retrieved,
        sources=sources,
        removed_citations=removed,
        notes=notes,
        language_note=language_note(sources, is_demo),
    )
