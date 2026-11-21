"""Convert report sections to editable rows and back, tracking reviewer edits.

Kept free of Streamlit so it can be unit-tested.
"""

from __future__ import annotations

import re

from .schemas import ContradictionItem, DoctorReport, MissingItem, ReportItem, SLOT_LABELS

CERTAINTY_RU = {"reported": "со слов пациента", "uncertain": "неточно / приблизительно"}
CERTAINTY_FROM_RU = {v: k for k, v in CERTAINTY_RU.items()}
REASON_RU = {"not_reported": "не сообщено", "skipped_by_patient": "пропущено пациентом"}
REASON_FROM_RU = {v: k for k, v in REASON_RU.items()}
FLAG_LABELS = {
    "low_textual_support": "сверить формулировку",
    "negation_not_explicit": "отрицание не явное",
    "no_source": "нет ссылки на запись",
    "unknown_source_removed": "удалена несуществующая ссылка",
}
COLUMNS = ["_idx", "Раздел", "Текст", "Записи", "Достоверность", "Изменено", "Пометки"]


def flag_text(flags: list[str]) -> str:
    return ", ".join(FLAG_LABELS.get(f, f) for f in flags)


def _ids(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip().upper() for v in value if str(v).strip()]
    return [p.strip().upper() for p in re.split(r"[,\s;]+", str(value)) if p.strip()]


def items_to_rows(items: list[ReportItem]) -> list[dict]:
    return [
        {
            "_idx": n,
            "Раздел": it.label,
            "Текст": it.text,
            "Записи": ", ".join(it.source_entry_ids),
            "Достоверность": CERTAINTY_RU[it.certainty],
            "Изменено": it.edited_by_reviewer,
            "Пометки": flag_text(it.flags),
        }
        for n, it in enumerate(items)
    ]


def _row_index(row: dict) -> int | None:
    v = row.get("_idx")
    try:
        if v is None or v != v:  # None or NaN
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def rows_to_items(rows: list[dict], previous: list[ReportItem], valid_ids: set[str]) -> list[ReportItem]:
    """Rebuild items from edited rows. Any changed or new row is marked edited_by_reviewer.

    Source IDs that do not exist are dropped (and the row flagged) so provenance stays valid.
    """
    out: list[ReportItem] = []
    for row in rows:
        text = str(row.get("Текст") or "").strip()
        if not text:
            continue  # empty rows are treated as deleted
        label = str(row.get("Раздел") or "Добавлено при проверке").strip()
        raw_cert = row.get("Достоверность")
        certainty = CERTAINTY_FROM_RU.get(raw_cert, raw_cert if raw_cert in ("reported", "uncertain") else "reported")
        ids = _ids(row.get("Записи"))
        good = [i for i in ids if i in valid_ids]
        idx = _row_index(row)
        prev = previous[idx] if idx is not None and 0 <= idx < len(previous) else None
        changed = (
            prev is None
            or prev.label != label
            or prev.text != text
            or prev.certainty != certainty
            or prev.source_entry_ids != good
        )
        flags = list(prev.flags) if prev else []
        if len(good) != len(ids):
            flags.append("unknown_source_removed")
        if not good and "no_source" not in flags:
            flags.append("no_source")
        out.append(
            ReportItem(
                label=label,
                category=prev.category if prev else None,
                text=text,
                source_entry_ids=good,
                certainty=certainty,
                time_reference=prev.time_reference if prev else None,
                edited_by_reviewer=(prev.edited_by_reviewer if prev else False) or changed,
                flags=sorted(set(flags)),
            )
        )
    return out


def missing_to_rows(items: list[MissingItem]) -> list[dict]:
    return [{"Тема": m.label, "Причина": REASON_RU[m.reason]} for m in items]


def rows_to_missing(rows: list[dict]) -> list[MissingItem]:
    label_to_field = {v: k for k, v in SLOT_LABELS.items()}
    out = []
    for r in rows:
        label = str(r.get("Тема") or "").strip()
        if not label:
            continue
        reason = REASON_FROM_RU.get(r.get("Причина"), r.get("Причина"))
        reason = reason if reason in ("not_reported", "skipped_by_patient") else "not_reported"
        out.append(MissingItem(field=label_to_field.get(label, "reviewer_added"), label=label, reason=reason))
    return out


def contradictions_to_rows(items: list[ContradictionItem]) -> list[dict]:
    return [{"Описание": c.description, "Записи": ", ".join(c.source_entry_ids)} for c in items]


def rows_to_contradictions(rows: list[dict], valid_ids: set[str]) -> list[ContradictionItem]:
    out = []
    for r in rows:
        d = str(r.get("Описание") or "").strip()
        if d:
            out.append(ContradictionItem(description=d, source_entry_ids=[i for i in _ids(r.get("Записи")) if i in valid_ids]))
    return out


def report_to_markdown(report: DoctorReport) -> str:
    """Compact Markdown preview used in the UI."""
    from .report import LIST_SECTIONS

    def fmt(it: ReportItem) -> str:
        tags = []
        if it.source_entry_ids:
            tags.append("`" + ", ".join(it.source_entry_ids) + "`")
        if it.certainty == "uncertain":
            tags.append("**неточно**")
        if it.edited_by_reviewer:
            tags.append("*исправлено*")
        return f"- **{it.label}:** {it.text} " + " ".join(tags)

    lines = ["#### 🧑 Основная жалоба (со слов пациента)"]
    lines.append(fmt(report.main_concern) if report.main_concern else "_Не указана._")
    lines.append("\n#### 🤖 Резюме, составленное ИИ — не слова пациента")
    lines.append(f"> {report.ai_generated.summary}")
    for key, title in LIST_SECTIONS:
        items = getattr(report, key)
        lines.append(f"\n#### {title}")
        lines.extend(fmt(i) for i in items) if items else lines.append("_Нет._")
    lines.append("\n#### Не хватает информации")
    if report.missing_information:
        lines.extend(f"- {m.label} — {REASON_RU[m.reason]}" for m in report.missing_information)
    else:
        lines.append("_Все темы затронуты._")
    lines.append("\n#### Противоречия — уточнить на приёме")
    if report.contradictions:
        lines.extend(f"- {c.description} `{', '.join(c.source_entry_ids)}`" for c in report.contradictions)
    else:
        lines.append("_Не обнаружено._")
    return "\n".join(lines)
