"""Load PDF, TXT and Markdown files into page/section units with metadata."""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_DOC_EXTENSIONS = {"pdf", "txt", "md", "markdown"}


class DocumentError(ValueError):
    pass


@dataclass
class Section:
    text: str
    page: int | None = None  # 1-based page number (PDF only)
    heading: str | None = None  # Markdown section heading


@dataclass
class LoadedDocument:
    filename: str
    title: str
    doc_hash: str  # SHA-256 of the file bytes - used for de-duplication
    file_type: str
    sections: list[Section] = field(default_factory=list)

    @property
    def n_pages(self) -> int | None:
        pages = {s.page for s in self.sections if s.page}
        return len(pages) or None


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1251", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_document(filename: str, data: bytes) -> LoadedDocument:
    name = Path(filename).name
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in SUPPORTED_DOC_EXTENSIONS:
        raise DocumentError(f"«{name}»: тип файла не поддерживается. Используйте PDF, TXT или Markdown (.md).")
    if not data:
        raise DocumentError(f"«{name}»: файл пуст.")
    doc_hash = hashlib.sha256(data).hexdigest()
    stem = Path(name).stem.replace("_", " ").replace("-", " ").strip()

    if ext == "pdf":
        doc = _load_pdf(name, data, doc_hash, stem)
    elif ext in ("md", "markdown"):
        doc = _load_markdown(name, _decode(data), doc_hash, stem)
    else:
        doc = _load_text(name, _decode(data), doc_hash, stem)

    if not any(s.text.strip() for s in doc.sections):
        raise DocumentError(
            f"«{name}»: не найден извлекаемый текст. Для сканов и PDF из изображений нужно распознавание (OCR), "
            "которого в прототипе нет."
        )
    return doc


def _load_pdf(name: str, data: bytes, doc_hash: str, stem: str) -> LoadedDocument:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:  # noqa: BLE001
                raise DocumentError(f"«{name}» защищён паролем.") from None
        sections = []
        for i, page in enumerate(reader.pages, start=1):
            txt = _clean(page.extract_text() or "")
            if txt:
                sections.append(Section(text=txt, page=i))
        meta_title = None
        try:
            meta_title = (reader.metadata.title or "").strip() if reader.metadata else None
        except Exception:  # noqa: BLE001
            meta_title = None
    except DocumentError:
        raise
    except (PdfReadError, Exception) as exc:  # noqa: BLE001
        raise DocumentError(f"«{name}»: не удалось прочитать PDF ({type(exc).__name__}).") from None
    return LoadedDocument(name, meta_title or stem, doc_hash, "pdf", sections)


def _load_markdown(name: str, text: str, doc_hash: str, stem: str) -> LoadedDocument:
    text = _clean(text)
    title = None
    sections: list[Section] = []
    heading: str | None = None
    buf: list[str] = []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section(text=body, heading=heading))

    for line in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            if title is None and len(m.group(1)) == 1:
                title = m.group(2).strip()
            flush()
            buf = [line]
            heading = m.group(2).strip()
        else:
            buf.append(line)
    flush()
    return LoadedDocument(name, title or stem, doc_hash, "markdown", sections)


def _load_text(name: str, text: str, doc_hash: str, stem: str) -> LoadedDocument:
    text = _clean(text)
    first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    title = first if 0 < len(first) <= 100 else stem
    return LoadedDocument(name, title, doc_hash, "text", [Section(text=text)])
