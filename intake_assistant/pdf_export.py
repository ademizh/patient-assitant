"""PDF exports with fpdf2 and the bundled DejaVu fonts (full Cyrillic support).

  report_to_pdf(report)      - the pre-consultation report for the doctor
  record_to_pdf(record)      - the whole appointment record (before / during / after)
  checklist_to_pdf(record)   - the follow-up checklist only

Every block is labelled with its origin (patient / patient's document / doctor / AI / sources / rules).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from .actions import action_state
from .records import ACTION_CATEGORY_LABELS, CONSULT_KIND_LABELS as KIND_LABELS, ORIGIN_LABELS, AppointmentRecord, Explanation, SourceRef
from .report import LIST_SECTIONS
from .report_edit import FLAG_LABELS, REASON_RU
from .schemas import DoctorReport, ReportItem

FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"

INK = (30, 41, 59)
MUTED = (100, 116, 139)
ACCENT = (15, 118, 110)
WARN_BG = (254, 243, 199)
RULE = (203, 213, 225)
ORIGIN_BG = {
    "patient": (224, 242, 254),
    "patient_document": (255, 237, 213),
    "doctor": (220, 252, 231),
    "ai": (238, 242, 255),
    "knowledge_base": (241, 245, 249),
    "rules": (254, 243, 199),
}

FOOTER = "Прототип · не проверен клинически · не является диагнозом · проверяйте информацию с врачом"


class _PDF(FPDF):
    def footer(self):
        self.set_y(-13)
        self.set_font("DejaVu", "", 7.5)
        self.set_text_color(*MUTED)
        self.cell(0, 5, f"{FOOTER}   ·   стр. {self.page_no()}/{{nb}}", align="C")


def _new_pdf(title: str) -> _PDF:
    pdf = _PDF(format="A4")
    pdf.add_font("DejaVu", "", str(FONT_DIR / "DejaVuSans.ttf"))
    pdf.add_font("DejaVu", "B", str(FONT_DIR / "DejaVuSans-Bold.ttf"))
    pdf.add_font("DejaVu", "I", str(FONT_DIR / "DejaVuSans-Oblique.ttf"))
    pdf.set_title(title)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(16, 16, 16)
    pdf.alias_nb_pages()
    pdf.add_page()
    return pdf


def _para(pdf: _PDF, text: str, size=9.5, style="", color=INK, h=5.0, fill=None):
    pdf.set_font("DejaVu", style, size)
    pdf.set_text_color(*color)
    if fill:
        pdf.set_fill_color(*fill)
    pdf.multi_cell(0, h, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=bool(fill), align="L")


def _title(pdf: _PDF, text: str):
    pdf.set_font("DejaVu", "B", 16)
    pdf.set_text_color(*INK)
    pdf.multi_cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")


def _h1(pdf: _PDF, text: str):
    pdf.ln(3)
    pdf.set_font("DejaVu", "B", 13)
    pdf.set_text_color(*INK)
    pdf.set_fill_color(226, 232, 240)
    pdf.multi_cell(0, 8, "  " + text, new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True, align="L")
    pdf.ln(1)


def _heading(pdf: _PDF, text: str):
    pdf.ln(2.5)
    pdf.set_font("DejaVu", "B", 11)
    pdf.set_text_color(*ACCENT)
    pdf.multi_cell(0, 6.5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
    pdf.set_draw_color(*RULE)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(1.5)


def _origin(pdf: _PDF, origin: str, text: str, size=9.3, style=""):
    """A block with a coloured background and an origin tag."""
    _para(pdf, f"[{ORIGIN_LABELS[origin]}]", size=7.2, style="B", color=MUTED, h=4)
    _para(pdf, text, size=size, style=style, fill=ORIGIN_BG[origin], h=5)
    pdf.ln(0.8)


def _legend(pdf: _PDF):
    _para(pdf, "Обозначения происхождения информации:", size=8, style="B", color=MUTED, h=4.4)
    for key in ("patient", "patient_document", "doctor", "ai", "knowledge_base", "rules"):
        _para(pdf, f"   {ORIGIN_LABELS[key]}", size=7.8, fill=ORIGIN_BG[key], h=4.2)
    pdf.ln(1)


def _item_line(item: ReportItem) -> str:
    tags = []
    if item.source_entry_ids:
        tags.append("записи " + ", ".join(item.source_entry_ids))
    if item.certainty == "uncertain":
        tags.append("НЕТОЧНО")
    if item.edited_by_reviewer:
        tags.append("исправлено при проверке")
    tags.extend(FLAG_LABELS[f] for f in item.flags if f in FLAG_LABELS)
    return f"{item.label}: {item.text}" + (f"  [{' | '.join(tags)}]" if tags else "")


def _items(pdf: _PDF, items: list[ReportItem], empty: str):
    if not items:
        _para(pdf, empty, style="I", color=MUTED)
        return
    for it in items:
        _para(pdf, "•  " + _item_line(it), h=5.0)
        pdf.ln(0.5)


def _source_lines(pdf: _PDF, sources: list[SourceRef]):
    for s in sources:
        loc = f"стр. {s.page}" if s.page else (f"раздел «{s.heading}»" if s.heading else "без номера страницы")
        _para(pdf, f"Источник: «{s.title}» — {s.filename}, {loc}, фрагмент {s.chunk_id}", size=7.8, color=MUTED, h=4.2)
        if s.quote:
            lang = "" if s.language == "ru" else f" (оригинал, язык: {s.language})"
            _para(pdf, f"Цитата{lang}: «{s.quote}»", size=7.8, style="I", color=MUTED, h=4.2)
        if s.translation_ru:
            warn = "" if s.translation_numbers_ok is not False else " — ВНИМАНИЕ: числа не совпадают с оригиналом"
            _para(pdf, f"Перевод ИИ{warn}: «{s.translation_ru}»", size=7.8, style="I", color=MUTED, h=4.2)


# ---------------------------------------------------------------------------------------
# Doctor report
# ---------------------------------------------------------------------------------------


def _report_body(pdf: _PDF, report: DoctorReport):
    _heading(pdf, "Основная жалоба (со слов пациента)")
    if report.main_concern:
        _items(pdf, [report.main_concern], "")
    else:
        _para(pdf, "Не указана.", style="I", color=MUTED)

    _heading(pdf, "Резюме, составленное ИИ — не слова пациента, сверяйте с записями")
    _origin(pdf, "ai", report.ai_generated.summary)
    _para(pdf, f"Сформировано: {report.ai_generated.generator}"
               f"{'   ·   исправлено при проверке' if report.ai_generated.edited_by_reviewer else ''}",
          size=7.5, color=MUTED)

    for key, title in LIST_SECTIONS:
        _heading(pdf, title)
        empty = {
            "negative_findings": "Явных отрицаний нет. (Отсутствие информации НЕ означает отсутствие симптома.)",
            "patient_questions": "Вопросов не записано.",
        }.get(key, "Не сообщалось.")
        _items(pdf, getattr(report, key), empty)

    _heading(pdf, "Не хватает информации — уточнить на приёме")
    if report.missing_information:
        for mi in report.missing_information:
            _para(pdf, f"•  {mi.label} — {REASON_RU[mi.reason]}")
    else:
        _para(pdf, "Все темы опроса затронуты.", style="I", color=MUTED)

    _heading(pdf, "Противоречия — уточнить на приёме")
    if report.contradictions:
        for c in report.contradictions:
            _para(pdf, f"•  {c.description}  [записи {', '.join(c.source_entry_ids)}]")
    else:
        _para(pdf, "Не обнаружено.", style="I", color=MUTED)

    g = report.specialist_guidance
    if g:
        _heading(pdf, "Ориентир по выбору специалиста (не диагноз)")
        _guidance(pdf, g)
    u = report.urgent_check
    if u:
        _heading(pdf, "Проверка признаков срочности")
        _urgent(pdf, u)

    if report.ai_generated.notes:
        _heading(pdf, "Служебные заметки (проверки программы, не сведения о пациенте)")
        for n in report.ai_generated.notes:
            _para(pdf, f"•  {n}", size=8.5, color=MUTED)


def _guidance(pdf: _PDF, g: dict):
    if g.get("status") == "suggested":
        for o in g.get("options", []):
            _origin(pdf, "ai", f"{o['specialist']}: {o['reason']}")
            _source_lines(pdf, [SourceRef(**s) for s in o.get("sources", [])])
    else:
        _para(pdf, "Источников недостаточно для ориентира по специалисту.", style="I", color=MUTED)
    if g.get("uncertainty"):
        _para(pdf, "Неопределённость: " + g["uncertainty"], size=8.5, color=MUTED)
    if g.get("clarifying_question"):
        _para(pdf, "Уточняющий вопрос: " + g["clarifying_question"], size=8.5)
    _para(pdf, g.get("default_advice", ""), size=8.5)


def _urgent(pdf: _PDF, u: dict):
    status = ("Маршрут проверен клиницистом: " + (u.get("reviewed_by") or "")) if u.get("pathway_validated") else (
        "ДЕМОНСТРАЦИОННЫЕ ПРАВИЛА — не проверены клиницистом")
    _para(pdf, f"{u.get('pathway_title', '')} · {status}", size=8.5, style="B", fill=WARN_BG, h=4.8)
    found = [m for m in u.get("matches", []) if not m.get("negated")]
    for m in found:
        _origin(pdf, "rules", f"{m['label']}: «{m['matched_text']}» (запись {m['entry_id']})")
    neg = [m for m in u.get("matches", []) if m.get("negated")]
    for m in neg:
        _para(pdf, f"Упомянуто с отрицанием: {m['label']} — «{m['matched_text']}» ({m['entry_id']})", size=8, color=MUTED)
    _para(pdf, u.get("message", ""), size=8.8)


def _report_header(pdf: _PDF, report: DoctorReport, title: str):
    m = report.meta
    _title(pdf, title)
    _para(pdf, f"{m.patient_label}   ·   сформирован {m.generated_at} (UTC)   ·   режим: {m.mode.upper()}"
               f"   ·   опрос {'завершён' if m.intake_completed else 'НЕ завершён'}", size=8.5, color=MUTED)
    pdf.ln(1.5)
    _para(pdf, "⚠  " + m.disclaimer, size=8.5, fill=WARN_BG, h=4.6)
    if m.fictional_example:
        pdf.ln(1)
        _para(pdf, "ВЫМЫШЛЕННЫЙ ПРИМЕР — этого человека не существует. Создано только для демонстрации.",
              size=8.5, style="B", fill=WARN_BG, h=4.6)


def _entries_appendix(pdf: _PDF, entries):
    _heading(pdf, "Приложение — исходные записи пациента (дословно)")
    for e in entries:
        src = "голос" if e.source == "audio" else "текст"
        header = f"[{e.id}] {src}{' (' + e.audio_filename + ')' if e.audio_filename else ''} · {e.created_at}"
        if e.question_text:
            header += f" · ответ на вопрос: «{e.question_text}»"
        _para(pdf, header, size=8, style="B", color=MUTED, h=4.4)
        _para(pdf, e.original_text, size=9, h=4.8, fill=ORIGIN_BG["patient"])
        if e.was_edited:
            _para(pdf, "Исправлено пациентом: " + e.text, size=9, style="I", h=4.8)
        pdf.ln(1)


def report_to_pdf(report: DoctorReport) -> bytes:
    pdf = _new_pdf("Отчёт для врача перед приёмом")
    _report_header(pdf, report, "Сведения пациента перед приёмом")
    _report_body(pdf, report)
    _entries_appendix(pdf, report.source_entries)
    return bytes(pdf.output())


# ---------------------------------------------------------------------------------------
# Whole appointment record
# ---------------------------------------------------------------------------------------


def _explanation(pdf: _PDF, x: Explanation):
    _para(pdf, f"{x.id} · {KIND_LABELS.get(x.kind, x.kind)} · «{x.query}» · {x.created_at}", size=8.5, style="B", h=4.6)
    _para(pdf, "1. Что сказано в вашем документе или заметке", size=8.5, style="B", color=ACCENT, h=4.6)
    if x.lab_rows:
        for r in x.lab_rows:
            parts = [f"{r.name}: {r.value}"]
            if r.unit:
                parts.append(r.unit)
            rng = f"референсный интервал в бланке: {r.reference_range}" if r.reference_range else "референсный интервал в бланке НЕ указан"
            flag = f"; отметка в бланке: {r.flag}" if r.flag else ""
            _origin(pdf, "patient_document", " ".join(parts) + f" ({rng}{flag})")
    for d in x.document_states:
        _origin(pdf, "patient_document" if x.kind == "test_result" else "patient", f"«{d}»")
    if not x.lab_rows and not x.document_states:
        _para(pdf, "—", color=MUTED)
    _para(pdf, "2. Общие сведения из медицинских источников", size=8.5, style="B", color=ACCENT, h=4.6)
    if x.background_status == "answered":
        _origin(pdf, "ai", x.background)
        if x.answer_language_note:
            _para(pdf, x.answer_language_note, size=7.8, style="I", color=MUTED)
        _source_lines(pdf, x.sources)
    else:
        _para(pdf, x.background or "База знаний не содержит ответа.", style="I", color=MUTED)
    _para(pdf, "3. Что уточнить у врача", size=8.5, style="B", color=ACCENT, h=4.6)
    for q in x.ask_doctor or ["—"]:
        _para(pdf, f"•  {q}", size=9)
    for n in x.notes:
        _para(pdf, f"Проверка: {n}", size=7.8, color=MUTED)
    pdf.ln(2)


def _checklist(pdf: _PDF, record: AppointmentRecord, today: date):
    if not record.actions:
        _para(pdf, "Действий пока нет.", style="I", color=MUTED)
        return
    for cat, label in ACTION_CATEGORY_LABELS.items():
        items = [a for a in record.actions if a.category == cat]
        if not items:
            continue
        _heading(pdf, label)
        for a in items:
            state = action_state(a, today)
            box = "☑" if state == "done" else "☐"
            st = {"done": "выполнено", "overdue": "ПРОСРОЧЕНО", "pending": "ожидает"}[state]
            _para(pdf, f"{box}  {a.id}. {a.description}   [{st}]", size=9.5, style="B", h=5.2)
            if a.added_by == "doctor_instruction":
                _origin(pdf, "doctor", f"«{a.source_statement}»", size=8.8)
            else:
                _para(pdf, "Добавлено пациентом.", size=8, color=MUTED)
            if a.medication and any(a.medication.model_dump().values()):
                med = a.medication
                _para(pdf, "Препарат (дословно): " + "; ".join(
                    f"{k}: {v}" for k, v in (("название", med.name), ("доза", med.dose), ("частота", med.frequency),
                                             ("длительность", med.duration)) if v), size=8.5)
            due = []
            if a.doctor_due_text:
                due.append(f"срок от врача: «{a.doctor_due_text}»" + (f" = {a.doctor_due_date}" if a.doctor_due_date else ""))
            else:
                due.append("срок от врача: не указан")
            if a.due_date_note:
                due.append(a.due_date_note)
            if a.patient_target_date:
                due.append(f"ваша целевая дата (не от врача): {a.patient_target_date}")
            if a.completed_at:
                due.append(f"выполнено: {a.completed_at}")
            _para(pdf, " · ".join(due), size=8.2, color=MUTED, h=4.4)
            if a.patient_note:
                _origin(pdf, "patient", "Заметка: " + a.patient_note, size=8.5)
            pdf.ln(1)


def _issues(pdf: _PDF, record: AppointmentRecord):
    open_issues = [i for i in record.instruction_issues if not i.resolved]
    if not open_issues:
        return
    _heading(pdf, "Требует уточнения у врача")
    kinds = {"ambiguous": "Неоднозначно", "contradiction": "Противоречие", "missing_detail": "Не хватает деталей"}
    for i in open_issues:
        src = "правило" if i.origin == "rules" else "ИИ"
        _para(pdf, f"⚠  {kinds[i.kind]} ({src}): {i.description}", size=9, fill=WARN_BG)
        pdf.ln(0.6)


def record_to_pdf(record: AppointmentRecord, today: date | None = None) -> bytes:
    today = today or date.today()
    pdf = _new_pdf("Запись о приёме")
    _title(pdf, record.title)
    _para(pdf, f"Дата приёма: {record.appointment_date or 'не указана'}   ·   запись {record.id}   ·   обновлена "
               f"{record.updated_at} (UTC)", size=8.5, color=MUTED)
    pdf.ln(1)
    _para(pdf, "⚠  Прототип для демонстрации — не является клинически проверенной системой и не ставит диагнозы. "
               "Сведения пациента, указания врача и объяснения ИИ показаны раздельно.", size=8.5, fill=WARN_BG, h=4.6)
    if record.fictional:
        pdf.ln(1)
        _para(pdf, "ВЫМЫШЛЕННЫЙ ПРИМЕР — этого человека и этого приёма не существует.", size=8.5, style="B",
              fill=WARN_BG, h=4.6)
    pdf.ln(1)
    _legend(pdf)

    # ---- Before
    _h1(pdf, "1. До приёма")
    if record.report:
        _report_body(pdf, record.report)
    else:
        _para(pdf, "Отчёт для врача ещё не сформирован.", style="I", color=MUTED)
    _heading(pdf, "Вопросы к врачу (список пациента)")
    if record.patient_questions:
        for q in record.patient_questions:
            mark = "☑" if q.asked else "☐"
            _origin(pdf, "patient", f"{mark} {q.id}. {q.text}" + (f"\nОтвет (запись пациента): {q.answer_note}" if q.answer_note else ""))
    else:
        _para(pdf, "Вопросов нет.", style="I", color=MUTED)

    # ---- During
    _h1(pdf, "2. Во время приёма")
    if record.consultation_items:
        _heading(pdf, "Записи с приёма")
        for c in record.consultation_items:
            src = {"text": "текст", "audio": "голос", "upload": "файл"}[c.source]
            up = record.find_upload(c.upload_id)
            _para(pdf, f"{c.id} · {KIND_LABELS[c.kind]} · {src}{' · ' + up.filename if up else ''}", size=8, style="B", color=MUTED, h=4.4)
            _origin(pdf, c.origin, c.text if len(c.text) < 1500 else c.text[:1500] + " …", size=8.8)
    if record.explanations:
        _heading(pdf, "Сохранённые объяснения")
        for x in record.explanations:
            _explanation(pdf, x)
    if not record.consultation_items and not record.explanations:
        _para(pdf, "Записей нет.", style="I", color=MUTED)

    # ---- After
    _h1(pdf, "3. После приёма")
    _heading(pdf, "Указания врача (дословно, как их передал пациент)")
    if record.instructions:
        for d in record.instructions:
            src = {"text": "текст", "audio": "голос", "upload": "файл"}[d.source]
            _para(pdf, f"{d.id} · {src}", size=8, style="B", color=MUTED, h=4.4)
            _origin(pdf, "doctor", d.original_text, size=8.8)
            if d.was_edited:
                _para(pdf, "Исправлено пациентом: " + d.text, size=8.8, style="I")
    else:
        _para(pdf, "Указаний нет.", style="I", color=MUTED)
    _issues(pdf, record)
    _h1(pdf, "Список дел после приёма")
    _checklist(pdf, record, today)

    if record.intake.entries:
        _entries_appendix(pdf, record.intake.entries)
    return bytes(pdf.output())


def checklist_to_pdf(record: AppointmentRecord, today: date | None = None) -> bytes:
    today = today or date.today()
    pdf = _new_pdf("Список дел после приёма")
    _title(pdf, "Список дел после приёма")
    _para(pdf, f"{record.title} · дата приёма: {record.appointment_date or 'не указана'} · по состоянию на "
               f"{today.strftime('%d.%m.%Y')}", size=8.5, color=MUTED)
    if record.fictional:
        _para(pdf, "ВЫМЫШЛЕННЫЙ ПРИМЕР — только для демонстрации.", size=8.5, style="B", fill=WARN_BG, h=4.6)
    _para(pdf, "Слова врача приведены дословно. Даты, которые вы задали сами, помечены «ваша целевая дата» и не "
               "являются указанием врача. Если срок не указан врачом, он отмечен как «не указан».",
          size=8.3, fill=WARN_BG, h=4.5)
    _issues(pdf, record)
    _checklist(pdf, record, today)
    return bytes(pdf.output())
