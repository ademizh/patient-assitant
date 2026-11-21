"""Chronological views of the patient's information.

Two views are produced:
  * entry timeline  - every submission in the order it was made (with corrections shown);
  * symptom timeline - statements that carry a time expression, ordered by an APPROXIMATE
    "how long ago" estimate. The patient's wording is always shown; the estimate is only used
    for ordering and is never presented as a date.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .schemas import SLOT_LABELS, IntakeEntry
from .text_utils import normalize

_RU_NUM = {
    "один": 1, "одну": 1, "одна": 1, "одного": 1, "два": 2, "две": 2, "двух": 2, "пару": 2, "пара": 2,
    "три": 3, "трёх": 3, "трех": 3, "четыре": 4, "четырёх": 4, "пять": 5, "шесть": 6, "семь": 7,
    "восемь": 8, "девять": 9, "десять": 10, "несколько": 3, "полтора": 1.5, "полторы": 1.5,
}
_EN_NUM = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "few": 3, "couple of": 2, "several": 3,
}
_UNIT_DAYS = [
    (r"(минут\w*|час\w*|minutes?|hours?)", 0.04),
    (r"(день|дня|дней|сутки|суток|days?)", 1),
    (r"(недел\w*|weeks?)", 7),
    (r"(месяц\w*|months?)", 30),
    (r"(год|года|лет|years?)", 365),
]

_RU_NUMWORD = "|".join(sorted(_RU_NUM, key=len, reverse=True))
_EN_NUMWORD = "|".join(sorted(_EN_NUM, key=len, reverse=True))
_UNITS_RE = "|".join(u for u, _ in _UNIT_DAYS)
_HEDGE = r"(?:(?:около|примерно|приблизительно|где-то|почти|больше|более|меньше|менее|уже|about|around|roughly|approximately|maybe|almost|nearly|over|more than|less than)\s+)?"

# Time expressions as the patient wrote them (Russian first, then English). Order matters.
TIME_PATTERNS = [
    rf"{_HEDGE}(?:\d+(?:[.,]\d+)?|{_RU_NUMWORD})?\s*(?:{_UNITS_RE})\s+назад(?:\s+или\s+чуть\s+(?:больше|меньше)|\s+с\s+небольшим)?",
    rf"{_HEDGE}(?:полгода|полгод)\s+назад",
    r"(?:уже\s+)?(?:около|примерно|больше|более)?\s*(?:\d+|" + _RU_NUMWORD + r")?\s*(?:" + _UNITS_RE + r")\s+подряд",
    r"(?:за\s+)?последн\w+\s+(?:\d+\s+|" + _RU_NUMWORD + r"\s+)?(?:" + _UNITS_RE + r")",
    r"(?:в\s+течение|на\s+протяжении)\s+(?:последн\w+\s+)?(?:\d+\s+|" + _RU_NUMWORD + r"\s+)?(?:" + _UNITS_RE + r")",
    r"с\s+(?:прошлой|позапрошлой|этой)\s+недели|с\s+прошлого\s+(?:месяца|года)|с\s+начала\s+\w+|с\s+(?:понедельника|вторника|среды|четверга|пятницы|субботы|воскресенья)",
    r"\b(?:позавчера|вчера|сегодня)\b",
    r"(?:на\s+прошлой\s+неделе|в\s+прошлом\s+(?:месяце|году)|в\s+позапрошлом\s+году)",
    r"(?:в\s+детстве|в\s+юности|в\s+школе|в\s+подростковом\s+возрасте)",
    r"(?:в|с)\s+(?:январе|феврале|марте|апреле|мае|июне|июле|августе|сентябре|октябре|ноябре|декабре|января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)(?:\s+\d{4})?",
    rf"\b{_HEDGE}(?:\d+|{_EN_NUMWORD})\s+(?:minutes?|hours?|days?|weeks?|months?|years?)\s+ago(?:\s+or\s+so)?\b",
    r"\b(?:yesterday|last\s+(?:night|week|month|year))\b",
    r"\bas\s+a\s+(?:child|teenager|kid)\b",
]


def find_time_reference(sentence: str) -> str | None:
    """Return the first time expression exactly as written in the sentence, or None."""
    norm = normalize(sentence)
    low = norm.lower()
    for p in TIME_PATTERNS:
        m = re.search(p, low)
        if m and m.group(0).strip():
            start, end = m.span()
            return sentence[start:end].strip(" ,.;")
    return None


def estimate_days_ago(expr: str | None) -> float | None:
    """Very rough 'how many days ago' for ordering only. None when it cannot be estimated."""
    if not expr:
        return None
    low = normalize(expr).lower()
    fixed = [
        (r"позавчера", 2), (r"вчера|yesterday|last night", 1), (r"сегодня", 0),
        (r"на прошлой неделе|с прошлой недели|last week", 7), (r"позапрошлой недели", 14),
        (r"в прошлом месяце|с прошлого месяца|last month", 30),
        (r"в позапрошлом году", 730), (r"в прошлом году|с прошлого года|last year", 365),
        (r"в детстве|as a (child|kid)|в школе", 365 * 25), (r"в юности|подростков|as a teenager", 365 * 15),
        (r"полгода", 182),
    ]
    for pat, days in fixed:
        if re.search(pat, low):
            return float(days)
    if "назад" not in low and "ago" not in low and "последн" not in low:
        return None  # durations ("в течение недели") say nothing about WHEN something happened
    qty: float | None = None
    m = re.search(r"(\d+(?:[.,]\d+)?)", low)
    if m:
        qty = float(m.group(1).replace(",", "."))
    else:
        for w, v in list(_RU_NUM.items()) + list(_EN_NUM.items()):
            if re.search(rf"(^|\s){re.escape(w)}(\s|$)", low):
                qty = float(v)
                break
    for unit_re, days in _UNIT_DAYS:
        if re.search(unit_re, low):
            return (qty if qty is not None else 1.0) * days
    return None


@dataclass
class TimelineRow:
    time_reference: str
    statement: str
    category_label: str
    entry_ids: list[str]
    approx_days_ago: float | None
    uncertain: bool


def symptom_timeline(facts, entry_order: dict[str, int]) -> list[TimelineRow]:
    """Facts with a time expression, oldest first (approximate); undatable ones at the end."""
    rows: list[tuple[tuple, TimelineRow]] = []
    for f in facts:
        if not f.time_reference or f.negated:
            continue
        days = estimate_days_ago(f.time_reference)
        first = min((entry_order.get(i, 10**6) for i in f.source_entry_ids), default=10**6)
        row = TimelineRow(
            time_reference=f.time_reference,
            statement=f.statement,
            category_label=SLOT_LABELS.get(f.category, f.category),
            entry_ids=list(f.source_entry_ids),
            approx_days_ago=days,
            uncertain=f.certainty == "uncertain",
        )
        key = (0, -days, first) if days is not None else (1, 0, first)
        rows.append((key, row))
    rows.sort(key=lambda kv: kv[0])
    return [r for _, r in rows]


def entry_timeline(entries: list[IntakeEntry]) -> list[dict]:
    return [
        {
            "id": e.id,
            "created_at": e.created_at,
            "source": "голос" if e.source == "audio" else "текст",
            "topic": SLOT_LABELS.get(e.question_slot, "свободный рассказ") if e.question_slot else "свободный рассказ",
            "text": e.text,
            "original_text": e.original_text if e.was_edited else None,
        }
        for e in sorted(entries, key=lambda e: (e.created_at, len(e.id), e.id))
    ]


def describe_days(days: float | None) -> str:
    if days is None:
        return "срок не определён"
    if days < 1:
        return "сегодня"
    if days < 14:
        return f"≈ {int(round(days))} дн. назад"
    if days < 60:
        return f"≈ {int(round(days / 7))} нед. назад"
    if days < 700:
        return f"≈ {int(round(days / 30))} мес. назад"
    return f"≈ {int(round(days / 365))} г. назад"
