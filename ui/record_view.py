"""«Вся запись»: one coherent overview of the appointment record + exports."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from intake_assistant.actions import split_views
from intake_assistant.consult import KIND_LABELS, short
from intake_assistant.pdf_export import checklist_to_pdf, record_to_pdf, report_to_pdf
from intake_assistant.records import ORIGIN_LABELS

from .common import Ctx, fmt_date, origin_tag


def render_record(ctx: Ctx) -> None:
    rec = ctx.record
    st.markdown(f"### 📁 {rec.title}")
    st.caption(f"Дата приёма: {fmt_date(rec.appointment_date)} · запись `{rec.id}` · "
               f"хранится локально: `{ctx.records.root.name}/{rec.id}/`")
    if rec.fictional:
        st.warning("Это **вымышленная** запись для демонстрации.", icon="🧪")
    st.markdown("**Обозначения:** " + " ".join(origin_tag(k) for k in ORIGIN_LABELS), unsafe_allow_html=True)

    views = split_views(rec.actions, ctx.today)
    cols = st.columns(6)
    cols[0].metric("Записей пациента", len(rec.intake.entries))
    cols[1].metric("Вопросов к врачу", len(rec.patient_questions))
    cols[2].metric("Записей с приёма", len(rec.consultation_items))
    cols[3].metric("Объяснений", len(rec.explanations))
    cols[4].metric("Дел выполнено", f"{len(views['done'])}/{len(rec.actions)}")
    cols[5].metric("Просрочено", len(views["overdue"]))

    st.markdown("##### Скачать")
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    pre = "FICTIONAL_" if rec.fictional else ""
    d1, d2, d3, d4 = st.columns(4)
    try:
        d1.download_button("⬇️ Вся запись (PDF)", record_to_pdf(rec, ctx.today), f"{pre}appointment_record_{stamp}.pdf",
                           "application/pdf", type="primary", width="stretch")
    except Exception as exc:  # noqa: BLE001
        d1.error(f"PDF не создан ({type(exc).__name__}).")
    d2.download_button("⬇️ Вся запись (JSON)", rec.model_dump_json(indent=2), f"{pre}appointment_record_{stamp}.json",
                       "application/json", width="stretch")
    if rec.actions:
        d3.download_button("⬇️ Список дел (PDF)", checklist_to_pdf(rec, ctx.today), f"{pre}followup_checklist_{stamp}.pdf",
                           "application/pdf", width="stretch")
    if rec.report:
        d4.download_button("⬇️ Отчёт для врача (PDF)", report_to_pdf(rec.report), f"{pre}doctor_report_{stamp}.pdf",
                           "application/pdf", width="stretch")

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown("#### 🗓️ До приёма")
        mc = rec.report.main_concern.text if rec.report and rec.report.main_concern else (
            rec.intake.entries[0].text if rec.intake.entries else "—")
        st.markdown(f"{origin_tag('patient')} **Жалоба:** {short(mc, 160)}", unsafe_allow_html=True)
        st.markdown(f"Отчёт для врача: {'сформирован' if rec.report else 'нет'}"
                    f"{' (устарел)' if rec.report_is_stale() else ''}")
        g = rec.specialist_guidance
        if g and g.status == "suggested":
            st.markdown(f"{origin_tag('ai')} Ориентир: " + ", ".join(o.specialist for o in g.options), unsafe_allow_html=True)
        st.markdown("**Вопросы:**")
        for q in rec.patient_questions:
            st.markdown(f"- {'☑' if q.asked else '☐'} {q.text}" + (f"  \n  _Ответ:_ {q.answer_note}" if q.answer_note else ""))
    with c2:
        st.markdown("#### 🩺 Во время приёма")
        for c in rec.consultation_items:
            st.markdown(f"{origin_tag(c.origin)} **{c.id}** {KIND_LABELS[c.kind]}: {short(c.text, 70)}", unsafe_allow_html=True)
        for x in rec.explanations:
            st.markdown(f"{origin_tag('ai')} **{x.id}** «{short(x.query, 40)}» · источников: {len(x.sources)}",
                        unsafe_allow_html=True)
        if not rec.consultation_items:
            st.caption("Пока пусто.")
    with c3:
        st.markdown("#### ✅ После приёма")
        for d in rec.instructions:
            st.markdown(f"{origin_tag('doctor')} **{d.id}** {short(d.text, 70)}", unsafe_allow_html=True)
        for a in rec.actions:
            state = "☑" if a.status == "done" else ("🔴" if a in views["overdue"] else "☐")
            st.markdown(f"- {state} {a.description}")
        if not rec.instructions:
            st.caption("Пока пусто.")

    if rec.uploads:
        st.markdown("##### Загруженные файлы этой записи")
        for u in rec.uploads:
            kind = {"test_result": "анализы", "instructions": "указания врача", "other": "другое"}[u.kind]
            st.markdown(f"- `{u.id}` {u.filename} · {kind} · {u.size // 1024 + 1} КБ · sha256 {u.sha256[:10]}…")
        st.caption("Файлы хранятся только вместе с этой записью и удаляются вместе с ней. В базу знаний они не попадают.")
