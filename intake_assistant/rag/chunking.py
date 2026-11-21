"""Configurable character-based chunking that prefers paragraph and sentence boundaries."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..text_utils import detect_language
from .loaders import LoadedDocument


@dataclass
class Chunk:
    chunk_id: str
    text: str
    doc_hash: str
    filename: str
    title: str
    page: int | None
    heading: str | None
    chunk_index: int
    language: str = "ru"

    def metadata(self, chunk_size: int, chunk_overlap: int) -> dict:
        # Chroma metadata values must be str/int/float/bool - use -1 / "" for "not available".
        return {
            "doc_hash": self.doc_hash,
            "filename": self.filename,
            "title": self.title,
            "page": self.page if self.page is not None else -1,
            "heading": self.heading or "",
            "chunk_index": self.chunk_index,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "language": self.language,
        }


def _units(text: str, max_len: int) -> list[tuple[str, bool]]:
    """Split text into (piece, starts_paragraph) pairs no longer than max_len.

    Prefers paragraph boundaries, then sentences, then words.
    """
    out: list[tuple[str, bool]] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_len:
            out.append((para, True))
            continue
        first = True
        for sent in re.split(r"(?<=[.!?])\s+", para):
            if len(sent) <= max_len:
                out.append((sent, first))
                first = False
                continue
            cur = ""
            for w in sent.split():
                if len(cur) + len(w) + 1 > max_len and cur:
                    out.append((cur, first))
                    first = False
                    cur = ""
                if len(w) > max_len:  # pathological token
                    for i in range(0, len(w), max_len):
                        out.append((w[i : i + max_len], first))
                        first = False
                    continue
                cur = f"{cur} {w}".strip()
            if cur:
                out.append((cur, first))
                first = False
    return out


def _tail(text: str, n: int) -> str:
    """Last ~n characters of text, starting at a word boundary."""
    if n <= 0 or not text:
        return ""
    if len(text) <= n:
        return text
    tail = text[-n:]
    space = tail.find(" ")
    return tail[space + 1 :] if 0 <= space < len(tail) - 1 else tail


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_size < 100:
        raise ValueError("Размер фрагмента должен быть не меньше 100 символов.")
    if not 0 <= chunk_overlap < chunk_size // 2 + 1:
        raise ValueError("Перекрытие должно быть от 0 до половины размера фрагмента.")
    body_max = chunk_size - chunk_overlap - 1
    chunks: list[str] = []
    current = ""
    for unit, new_para in _units(text, body_max):
        sep = "\n\n" if new_para else " "
        candidate = f"{current}{sep}{unit}".strip() if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        overlap = _tail(current, chunk_overlap)
        current = f"{overlap} {unit}".strip() if overlap else unit
    if current:
        chunks.append(current)
    return chunks


def chunk_document(doc: LoadedDocument, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = 0
    short = doc.doc_hash[:12]
    language = detect_language(" ".join(sec.text for sec in doc.sections))
    for section in doc.sections:
        for piece in split_text(section.text, chunk_size, chunk_overlap):
            page_tag = f"p{section.page}" if section.page else "s"
            chunks.append(
                Chunk(
                    chunk_id=f"{short}-{page_tag}-c{idx:03d}",
                    text=piece,
                    doc_hash=doc.doc_hash,
                    filename=doc.filename,
                    title=doc.title,
                    page=section.page,
                    heading=section.heading,
                    chunk_index=idx,
                    language=language,
                )
            )
            idx += 1
    return chunks
