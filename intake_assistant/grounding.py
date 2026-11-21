"""Shared source-grounding guards for every AI output that relies on the knowledge base.

* citations must refer to chunks that were actually retrieved and passed to the model;
* quotes must occur verbatim in the cited chunk (otherwise they are dropped);
* translations of non-Russian quotes are labelled, and numbers are checked to be unchanged;
* numbers in AI text that appear in none of the supporting texts are reported.
"""

from __future__ import annotations

import re

from .records import SourceRef
from .text_utils import detect_language, is_verbatim, numbers_preserved, unsupported_numbers

CITE_RE = re.compile(r"\[([A-Za-z0-9]{6,}-(?:p\d+|s)-c\d{3})\]")
REMOVED_CITATION = "[ссылка удалена: не найденный источник]"


def validate_citations(text: str, claimed: list[str], by_id: dict[str, dict]) -> tuple[str, list[str], list[str]]:
    ids = list(dict.fromkeys(list(claimed) + CITE_RE.findall(text or "")))
    valid = [c for c in ids if c in by_id]
    removed = [c for c in ids if c not in by_id]
    for bad in removed:
        text = (text or "").replace(f"[{bad}]", REMOVED_CITATION)
    return text or "", valid, removed


def chunk_language(c: dict) -> str:
    return c.get("language") or detect_language(c.get("text", ""))


def to_source(c: dict, quote: str | None = None, translation: str | None = None) -> SourceRef:
    lang = chunk_language(c)
    tr = translation.strip() if (translation and lang != "ru") else None
    return SourceRef(
        chunk_id=c["chunk_id"], title=c.get("title", ""), filename=c.get("filename", ""), page=c.get("page"),
        heading=c.get("heading"), excerpt=c.get("text", ""), quote=quote, language=lang,
        translation_ru=tr, translation_numbers_ok=numbers_preserved(quote or "", tr) if tr else None,
        score=c.get("score"),
    )


def build_sources(valid_ids: list[str], by_id: dict[str, dict], quotes: list) -> tuple[list[SourceRef], list[str]]:
    """One SourceRef per cited chunk, with the first verbatim quote for it (if any)."""
    notes: list[str] = []
    first_quote: dict[str, tuple[str, str | None]] = {}
    for q in quotes or []:
        cid = getattr(q, "chunk_id", None) if not isinstance(q, dict) else q.get("chunk_id")
        text = getattr(q, "quote", None) if not isinstance(q, dict) else q.get("quote")
        tr = getattr(q, "translation_ru", None) if not isinstance(q, dict) else q.get("translation_ru")
        if cid not in by_id or not text:
            continue
        if not is_verbatim(text, by_id[cid]["text"]):
            notes.append("Одна цитата не найдена дословно в источнике и была удалена.")
            continue
        first_quote.setdefault(cid, (text.strip(), tr))
    sources = []
    for cid in valid_ids:
        q, tr = first_quote.get(cid, (None, None))
        src = to_source(by_id[cid], q, tr)
        if src.translation_numbers_ok is False:
            notes.append(f"В переводе фрагмента {cid} числа не совпадают с оригиналом — опирайтесь на оригинал.")
        sources.append(src)
    return sources, list(dict.fromkeys(notes))


def language_note(sources: list[SourceRef], is_demo: bool) -> str | None:
    foreign = [s for s in sources if s.language != "ru"]
    if not foreign:
        return None
    if is_demo:
        return ("Часть источников на английском языке. В демо-режиме перевод недоступен — "
                "фрагменты показаны в оригинале.")
    return ("Ответ на русском составлен по источникам на английском языке. Переводы фрагментов выполнены ИИ и "
            "помечены как перевод — при сомнениях сверяйтесь с оригиналом.")


def number_note(text: str, *supports: str) -> str | None:
    bad = unsupported_numbers(text, *supports)
    if bad:
        return (f"В тексте ИИ есть числа, которых нет в источниках: {', '.join(bad[:5])}. "
                "Не опирайтесь на них — проверьте по оригиналу.")
    return None
