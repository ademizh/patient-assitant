"""Unsupported questions are not answered from model memory; citations, quotes and numbers are checked."""

from __future__ import annotations

from intake_assistant.providers import GroundedAnswer, SourceQuote
from intake_assistant.rag.qa import INSUFFICIENT_MESSAGE, ask

Q = "Что такое референсный интервал?"


class SpyProvider:
    """Uses the demo embedder but returns a scripted answer and counts LLM calls."""

    name = "spy"

    def __init__(self, base, answer: GroundedAnswer | None = None):
        self.base = base
        self.embedding_signature = base.embedding_signature
        self.scripted = answer
        self.calls = 0

    def embed(self, texts):
        return self.base.embed(texts)

    def search_queries(self, q):
        return [q]

    def answer(self, question, chunks):
        self.calls += 1
        self.seen_chunks = chunks
        return self.scripted


def _ga(**kw):
    kw.setdefault("quotes", [])
    return GroundedAnswer(**kw)


def test_no_documents(store, demo_provider):
    assert ask("что-нибудь?", store, demo_provider, 4, 0.1).status == "no_documents"


def test_irrelevant_question_skips_the_model(indexed_store, demo_provider):
    spy = SpyProvider(demo_provider, _ga(sufficient_evidence=True, answer="выдумка", cited_chunk_ids=[]))
    res = ask("Как лечат перелом ключицы у спортсменов?", indexed_store, spy, 4, 0.12)
    assert res.status == "insufficient_evidence" and res.answer == INSUFFICIENT_MESSAGE
    assert spy.calls == 0 and res.cited == []


def test_demo_unsupported_dosing_question(indexed_store, demo_provider):
    res = ask("Какая максимальная суточная доза ибупрофена для детей?", indexed_store, demo_provider, 4, 0.12)
    assert res.status == "insufficient_evidence" and res.cited == []


def test_model_says_insufficient(indexed_store, demo_provider):
    spy = SpyProvider(demo_provider, _ga(sufficient_evidence=False, answer="В фрагментах нет дозировок.", cited_chunk_ids=[]))
    res = ask(Q, indexed_store, spy, 4, 0.12)
    assert spy.calls == 1 and res.status == "insufficient_evidence" and res.model_note == "В фрагментах нет дозировок."


def test_fabricated_citations_and_quotes_are_removed(indexed_store, demo_provider):
    probe = ask(Q, indexed_store, SpyProvider(demo_provider, _ga(sufficient_evidence=True, answer="x", cited_chunk_ids=[])), 4, 0.12)
    real = probe.retrieved[0]
    spy = SpyProvider(demo_provider, _ga(
        sufficient_evidence=True,
        answer=f"Это диапазон для сравнения [{real['chunk_id']}]. См. также [deadbeef0000-p9-c999].",
        cited_chunk_ids=[real["chunk_id"], "deadbeef0000-p9-c999"],
        quotes=[SourceQuote(chunk_id=real["chunk_id"], quote="Цитата, которой нет в источнике", translation_ru=None)],
    ))
    res = ask(Q, indexed_store, spy, 4, 0.12)
    assert res.status == "answered"
    assert [c["chunk_id"] for c in res.cited] == [real["chunk_id"]]
    assert res.removed_citations == ["deadbeef0000-p9-c999"] and "deadbeef0000" not in res.answer
    assert res.sources[0].quote is None
    assert any("не найдена дословно" in n for n in res.notes)


def test_only_fabricated_citations_means_insufficient(indexed_store, demo_provider):
    spy = SpyProvider(demo_provider, _ga(sufficient_evidence=True, answer="Что-то [abcdef123456-p1-c001]",
                                         cited_chunk_ids=["abcdef123456-p1-c001"]))
    res = ask(Q, indexed_store, spy, 4, 0.12)
    assert res.status == "insufficient_evidence" and res.cited == []


def test_numbers_absent_from_sources_are_flagged(indexed_store, demo_provider):
    probe = ask(Q, indexed_store, SpyProvider(demo_provider, _ga(sufficient_evidence=True, answer="x", cited_chunk_ids=[])), 4, 0.12)
    cid = probe.retrieved[0]["chunk_id"]
    spy = SpyProvider(demo_provider, _ga(sufficient_evidence=True, answer=f"Обычно интервал 135–160 г/л [{cid}].",
                                         cited_chunk_ids=[cid]))
    res = ask(Q, indexed_store, spy, 4, 0.12)
    assert any("135" in n and "нет в источниках" in n for n in res.notes)


def test_translation_is_labelled_and_changed_numbers_are_flagged(indexed_store, demo_provider):
    q_en = "What should I include in my medication list?"
    res0 = ask(q_en, indexed_store, SpyProvider(demo_provider, _ga(sufficient_evidence=True, answer="x", cited_chunk_ids=[])), 4, 0.12)
    en = next(c for c in res0.retrieved if c["language"] == "en" and c["score"] >= 0.12)
    quote = "For each one, write the name, the dose, how often you take it, and since when."
    assert quote in " ".join(en["text"].split())

    def run(translation):
        spy = SpyProvider(demo_provider, _ga(
            sufficient_evidence=True, answer=f"Запишите название, дозу и частоту приёма [{en['chunk_id']}].",
            cited_chunk_ids=[en["chunk_id"]],
            quotes=[SourceQuote(chunk_id=en["chunk_id"], quote=quote, translation_ru=translation)],
        ))
        return ask(q_en, indexed_store, spy, 4, 0.12)

    ok = run("Для каждого запишите название, дозу, как часто принимаете и с какого времени.")
    src = ok.sources[0]
    assert src.language == "en" and src.quote == quote and src.translation_numbers_ok is True
    assert "переводы фрагментов выполнены ии" in ok.language_note.lower()
    bad = run("Для каждого запишите название, дозу 400 мг, частоту 2 раза в день.")
    assert bad.sources[0].translation_numbers_ok is False
    assert any("числа не совпадают" in n for n in bad.notes)


def test_model_only_sees_relevant_chunks(indexed_store, demo_provider):
    spy = SpyProvider(demo_provider, _ga(sufficient_evidence=False, answer="", cited_chunk_ids=[]))
    res = ask(Q, indexed_store, spy, 4, 0.12)
    assert all(c["score"] >= 0.12 for c in spy.seen_chunks) and len(spy.seen_chunks) <= len(res.retrieved)
