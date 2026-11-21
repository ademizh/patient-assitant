"""Patient documents (test results, doctor's instructions): text extraction and lab-line parsing.

Patient documents are NEVER indexed into the medical knowledge base. They are stored with the
appointment record only.

Lab parsing is deliberately conservative: every field of a LabRow is an exact substring of
the original line, so values, units and reference ranges are preserved character-for-character.
Nothing is converted, rounded, re-labelled as normal/abnormal, or filled in when missing.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

from .records import LabRow

PATIENT_DOC_EXTENSIONS = {"pdf", "txt", "md"}

# Below this many extractable characters per page a PDF is treated as scanned (image-only).
MIN_CHARS_PER_PAGE = 20


class PatientDocumentError(ValueError):
    """User-presentable error (Russian)."""


class ScannedPDFError(PatientDocumentError):
    pass


@dataclass
class ExtractedDocument:
    text: str
    n_pages: int | None
    file_type: str


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1251", "koi8-r", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def extract_patient_document(filename: str, data: bytes) -> ExtractedDocument:
    name = Path(filename).name
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in PATIENT_DOC_EXTENSIONS:
        raise PatientDocumentError(f"«{name}»: формат не поддерживается. Загрузите PDF с текстом или файл TXT.")
    if not data:
        raise PatientDocumentError(f"«{name}»: файл пуст.")
    if ext in ("txt", "md"):
        text = _decode(data).replace("\r\n", "\n").strip()
        if not text:
            raise PatientDocumentError(f"«{name}»: в файле нет текста.")
        return ExtractedDocument(text, None, ext)

    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:  # noqa: BLE001
                raise PatientDocumentError(f"«{name}»: PDF защищён паролем. Снимите защиту или вставьте текст вручную.") from None
        pages = [(p.extract_text() or "") for p in reader.pages]
    except PatientDocumentError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise PatientDocumentError(f"«{name}»: не удалось прочитать PDF ({type(exc).__name__}).") from None
    n = len(pages)
    total = sum(len(re.sub(r"\s+", "", p)) for p in pages)
    if n == 0 or total < MIN_CHARS_PER_PAGE * n:
        raise ScannedPDFError(
            f"«{name}» похож на скан или фотографию: в PDF нет распознаваемого текста "
            f"(найдено символов: {total}, страниц: {n}). Распознавание изображений (OCR) в прототипе не "
            "поддерживается. Вставьте текст результатов вручную или загрузите PDF, сохранённый из "
            "лаборатории в текстовом виде. Помните: значения и единицы нужно переписать точно как в бланке."
        )
    text = "\n\n".join(f"[стр. {i}]\n{p.strip()}" for i, p in enumerate(pages, start=1) if p.strip())
    return ExtractedDocument(text, n, "pdf")


# ---- Lab-line parsing ------------------------------------------------------------------

_NUMBER = r"\d+(?:[.,]\d+)?"
_VALUE = rf"(?:[<>≤≥]\s*)?{_NUMBER}(?![.,]?\d)"
_UNIT = (
    r"(?:[×x*]\s*10\s*\^?\s*\d+\s*/\s*[^\s\d()]+"  # ×10^9/л, x10*12/L
    r"|%|‰"
    r"|[^\s\d()\[\]<>≤≥;,]+(?:/[^\s\d()\[\]<>≤≥;,]+)*"  # г/л, ммоль/л, нг/мл, мм/ч, fl, mmol/L
    r")"
)
_RANGE = (
    rf"(?:{_NUMBER}\s*[-–—]\s*{_NUMBER}"
    rf"|(?:[<>≤≥]|до|менее|более|не\s+более|не\s+менее|up\s+to)\s*{_NUMBER})"
)
_FLAG = r"(?<![\w])(?:HH|LL|H|L|↑|↓|▲|▼|\*|выше|ниже)(?![\w])"

_LINE = re.compile(
    rf"^(?P<name>[A-Za-zА-Яа-яЁё][^\d\n:]*?(?:\([^)\n]*\))?)\s*[:\-–]?\s+(?P<value>{_VALUE})"
    rf"(?:\s*(?P<unit>{_UNIT}))?(?P<rest>.*)$"
)
_RANGE_RE = re.compile(_RANGE, re.IGNORECASE)
_FLAG_RE = re.compile(_FLAG)
_SKIP_NAMES = re.compile(
    r"^(дата|время|возраст|пол|№|номер|тел|телефон|стр|страница|пациент|фио|заказ|бланк|лаборатор|"
    r"date|age|sex|page|patient|order)\b",
    re.IGNORECASE,
)
_UNIT_WORDS_NOT_UNITS = {"раз", "раза", "года", "лет", "год", "дня", "дней"}


def parse_lab_lines(text: str) -> list[LabRow]:
    rows: list[LabRow] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if len(line) < 4 or len(line) > 200:
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name = m.group("name").strip(" :-–\t")
        if not name or _SKIP_NAMES.match(name) or len(name) > 70:
            continue
        unit = (m.group("unit") or "").strip() or None
        rest = m.group("rest") or ""
        if unit and (unit.lower() in _UNIT_WORDS_NOT_UNITS or _RANGE_RE.fullmatch(unit)):
            rest = f"{unit} {rest}" if unit.lower() not in _UNIT_WORDS_NOT_UNITS else rest
            unit = None if unit.lower() not in _UNIT_WORDS_NOT_UNITS else unit
            if unit in _UNIT_WORDS_NOT_UNITS:
                continue  # "Возраст 34 года" etc. - not a lab value
        rng = _RANGE_RE.search(rest)
        flag = _FLAG_RE.search(rest[: rng.start()] + " " + rest[rng.end():] if rng else rest)
        row = LabRow(
            line=line,
            name=name,
            value=m.group("value").strip(),
            unit=unit,
            reference_range=rng.group(0).strip() if rng else None,
            flag=flag.group(0) if flag else None,
        )
        # Defensive: every field must be a verbatim substring of the line.
        if all(v is None or v in line for v in (row.name, row.value, row.unit, row.reference_range, row.flag)):
            rows.append(row)
    return rows
