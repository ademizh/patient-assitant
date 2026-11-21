"""«После приёма»: doctor's instructions -> editable follow-up checklist with in-app reminders."""

from __future__ import annotations

from datetime import date, datetime

import streamlit as st

from intake_assistant.actions import mark_done, reminders, relative_due_hint, split_views
from intake_assistant.graphs.checklist_graph import run_checklist
from intake_assistant.labs import PatientDocumentError, ScannedPDFError, extract_patient_document
from intake_assistant.pdf_export import checklist_to_pdf
from intake_assistant.providers import ProviderError
from intake_assistant.records import ACTION_CATEGORY_LABELS, ActionItem, DoctorInstruction, next_id
from intake_assistant.schemas import utcnow_iso

from .common import Ctx, audio_block, esc, fmt_date, origin_box, render_source

ISSUE_KINDS = {"ambiguous": "Неоднозначно", "contradiction": "Противоречие", "missing_detail": "Не хватает деталей"}
FLAG_TEXT = {
    "description_replaced_with_source": "описание заменено словами врача (в нём были числа, которых нет у врача)",
    "due_text_not_verbatim": "срок не найден дословно в указаниях — не показан",
    "medication_name_not_verbatim": "название препарата не найдено дословно — не показано",
    "medication_dose_not_verbatim": "дозировка не найдена дословно — не показана",
    "medication_frequency_not_verbatim": "частота не найдена дословно — не показана",
    "medication_duration_not_verbatim": "длительность не найдена дословно — не показана",
}


def _add_instruction(ctx: Ctx, text: str, source: str, original: str | None = None,
                     upload_id: str | None = None, audio: str | None = None) -> None:
    rec = ctx.record
    rec.instructions.append(DoctorInstruction(id=next_id("D", rec.instructions), source=source,
                                              original_text=original if original is not None else text, text=text,
                                              upload_id=upload_id, audio_filename=audio))
    ctx.flash("success", "Указания добавлены. Нажмите «Составить список дел».")
    st.session_state.form_nonce += 1  # clears the text box / uploader
    ctx.commit()


def build_checklist(ctx: Ctx, explain_medications: bool = True) -> None:
    """Runs the LangGraph checklist workflow (intake_assistant/graphs/checklist_graph.py)."""
    rec = ctx.record
    ss = st.session_state
    try:
        with st.spinner("Разбираю указания врача и проверяю каждый пункт по их тексту…"):
            result = run_checklist(rec.instructions, ctx.provider, ctx.store, previous_actions=rec.actions,
                                   previous_issues=rec.instruction_issues, explain_medications=explain_medications,
                                   top_k=ss.top_k, min_relevance=ss.min_relevance, is_demo=ctx.settings.is_demo)
    except ProviderError as exc:
        st.error(str(exc))
        return
    actions, issues, notes, unmatched = result.actions, result.issues, result.notes, result.unmatched
    rec.actions, rec.instruction_issues = actions, issues
    rec.actions_signature = rec.instructions_signature()
    rec.actions_generator = ctx.provider.name
    ss.unmatched_sentences = unmatched
    ss.action_notes = notes
    ctx.flash("success", f"Список дел обновлён: {len(actions)} пунктов. Статусы и ваши даты сохранены.")
    ctx.commit()


def reminder_banner(ctx: Ctx, where: str) -> None:
    items = reminders(ctx.record.actions, ctx.today)
    if not items:
        return
    lines = "\n".join(f"- **{a.id}** {esc(a.description)} — {txt}" for a, txt in items)
    if where == "sidebar":
        st.warning(f"⏰ **Напоминания ({len(items)})**\n\n{lines}")
    else:
        st.warning(f"⏰ **Напоминания внутри приложения** — по датам, которые подтверждены врачом или заданы вами:\n\n{lines}")


def render_after(ctx: Ctx) -> None:
    ss = st.session_state
    rec = ctx.record
    st.markdown(
        "<div class='stage-hint'>После приёма: добавьте указания врача (текстом, голосом или файлом). Программа "
        "составит список дел, <b>дословно сохраняя</b> названия лекарств, дозировки, единицы, частоту и длительность, "
        "и отметит неясные места. Сроки, которые врач не назвал, остаются «не указан» — вы можете задать свою целевую дату.</div>",
        unsafe_allow_html=True,
    )
    reminder_banner(ctx, "main")

    with st.expander("➕ Добавить указания врача", expanded=not rec.instructions):
        t_text, t_audio, t_file = st.tabs(["✍️ Вставить текст", "🎙️ Надиктовать", "📎 Файл (PDF/TXT)"],
                                          key="instr_input_tab", on_change="rerun")
        with t_text:
            with st.form(f"instr_text_{ss.form_nonce}", clear_on_submit=True, border=False):
                text = st.text_area("Указания врача", height=150, placeholder="Вставьте текст рекомендаций или выписки…")
                go = st.form_submit_button("Добавить", type="primary")
            if go:
                if text.strip():
                    _add_instruction(ctx, text.strip(), "text")
                else:
                    st.warning("Введите текст.")
        with t_audio:
            audio_block(ctx, "instr", lambda t, o, f: _add_instruction(ctx, t, "audio", original=o, audio=f),
                        label="Надиктуйте указания врача")
        with t_file:
            up = st.file_uploader("PDF с текстом или TXT", type=["pdf", "txt"], key=f"instr_file_{ss.form_nonce}")
            if st.button("Загрузить и добавить", disabled=up is None, key="instr_upload_btn"):
                data = up.getvalue()
                try:
                    doc = extract_patient_document(up.name, data)
                except ScannedPDFError as exc:
                    st.error(str(exc), icon="🖼️")
                except PatientDocumentError as exc:
                    st.error(str(exc))
                else:
                    meta = ctx.records.save_upload(rec, up.name, data, "instructions")
                    meta.n_pages, meta.text_extracted = doc.n_pages, True
                    _add_instruction(ctx, doc.text, "upload", upload_id=meta.id)

    if not rec.instructions:
        st.info("Указаний пока нет. Можно загрузить вымышленный пример в боковой панели.", icon="✅")
        return

    st.markdown("##### Указания врача (дословно)")
    for d in rec.instructions:
        src = {"text": "текст", "audio": "голос", "upload": "файл"}[d.source]
        up = rec.find_upload(d.upload_id)
        with st.expander(f"{d.id} · {src}{' · ' + up.filename if up else ''}{' · исправлено' if d.was_edited else ''}"):
            origin_box("doctor", d.text)
            if d.was_edited:
                st.caption("Исходный текст сохранён:")
                st.text(d.original_text)
            new = st.text_area("Исправить (исходный текст сохранится)", d.text, key=f"edit_d_{d.id}_{ss.form_nonce}", height=140)
            c1, c2 = st.columns(2)
            if c1.button("Сохранить исправление", key=f"save_d_{d.id}") and new.strip() and new.strip() != d.text:
                d.text = new.strip()
                ctx.flash("info", "Текст исправлен. Обновите список дел — статусы и ваши даты сохранятся.")
                ctx.commit()
            if c2.button("Удалить эти указания", key=f"del_d_{d.id}"):
                rec.instructions = [i for i in rec.instructions if i.id != d.id]
                if d.upload_id:
                    ctx.records.delete_upload(rec, d.upload_id)
                ctx.commit()

    stale = rec.actions_are_stale()
    label = "📋 Составить список дел" if not rec.actions else "🔄 Обновить список дел"
    if stale and rec.actions:
        st.warning("Указания изменились после составления списка. Обновите его — выполненные пункты и ваши даты сохранятся.")
    b1, b2 = st.columns([1, 2], vertical_alignment="center")
    explain_meds = b2.toggle("💊 Объяснить лекарства по базе знаний", value=True, key="explain_meds",
                             help="Для каждого лекарства помощник найдёт в источниках, что это за препарат и для чего его "
                                  "обычно применяют. Дозировки берутся только из слов врача.")
    if b1.button(label, type="primary", disabled=ctx.provider is None):
        build_checklist(ctx, explain_medications=explain_meds)

    if ss.get("unmatched_sentences"):
        with st.expander(f"Не распознано как действие ({len(ss.unmatched_sentences)}) — проверьте, ничего ли не потерялось"):
            for s in ss.unmatched_sentences:
                st.markdown(f"- «{esc(s)}»", unsafe_allow_html=True)
            st.caption("Если это действие, добавьте его вручную ниже.")
    for n in ss.get("action_notes", []):
        st.caption(f"Проверка: {n}")

    open_issues = [i for i in rec.instruction_issues if not i.resolved]
    if rec.instruction_issues:
        st.markdown(f"##### ⚠️ Требует уточнения у врача ({len(open_issues)})")
        for n, i in enumerate(rec.instruction_issues):
            c1, c2 = st.columns([0.8, 0.2])
            src = "правило" if i.origin == "rules" else "ИИ"
            text = f"**{ISSUE_KINDS[i.kind]}** ({src}, пункты {', '.join(i.action_ids) or '—'}): {i.description}"
            (c1.success if i.resolved else c1.warning)(text + (" — ✅ уточнено" if i.resolved else ""))
            if c2.button("Уточнено" if not i.resolved else "Вернуть", key=f"res_{n}"):
                i.resolved = not i.resolved
                ctx.commit()

    if rec.actions:
        _checklist(ctx)
    _add_own(ctx)


def _checklist(ctx: Ctx) -> None:
    rec = ctx.record
    views = split_views(rec.actions, ctx.today)
    st.markdown(f"##### Список дел · сформирован: {rec.actions_generator or '—'}")
    tabs = st.tabs([f"⏳ Ожидают ({len(views['pending'])})", f"🔴 Просрочено ({len(views['overdue'])})",
                    f"✅ Выполнено ({len(views['done'])})", f"Все ({len(rec.actions)})"],
                   key="actions_view_tab", on_change="rerun")
    st.caption("«Просрочено» — только для пунктов с подтверждённой датой (от врача или вашей целевой).")
    groups = [views["pending"], views["overdue"], views["done"], rec.actions]
    for tab, items, name in zip(tabs, groups, ["p", "o", "d", "a"]):
        with tab:
            if not items:
                st.caption("Пусто.")
            for a in items:
                _action_card(ctx, a, name)
    c1, c2 = st.columns([1, 3])
    c1.download_button("⬇️ Список дел (PDF)", checklist_to_pdf(rec, ctx.today),
                       f"{'FICTIONAL_' if rec.fictional else ''}followup_checklist_{datetime.now():%Y%m%d}.pdf",
                       "application/pdf", width="stretch")


def _action_card(ctx: Ctx, a: ActionItem, view: str) -> None:
    rec = ctx.record
    ss = st.session_state
    k = f"{view}_{a.id}"
    with st.container(border=True, key=f"card_{k}"):
        c1, c2 = st.columns([0.06, 0.94])
        done = c1.checkbox("Выполнено", value=a.status == "done", key=f"done_{k}_{a.status}", label_visibility="collapsed")
        if done != (a.status == "done"):
            mark_done(a, done, utcnow_iso())
            ctx.commit()
        c2.markdown(f"<span class='entry-id'>{a.id}</span> <b>{esc(a.description)}</b> "
                    f"<span class='small-muted'>· {ACTION_CATEGORY_LABELS[a.category]}"
                    f"{' · исправлено вами' if a.edited_by_patient else ''}</span>", unsafe_allow_html=True)
        with c2:
            if a.added_by == "doctor_instruction":
                origin_box("doctor", f"«{a.source_statement}»", extra_label=a.instruction_id or "")
            else:
                origin_box("patient", "Пункт добавлен вами.")
            if a.medication and any(a.medication.model_dump().values()):
                m = a.medication
                st.markdown(
                    "<span class='small-muted'>Препарат (дословно из слов врача):</span> "
                    + " · ".join(f"<b>{lbl}:</b> {esc(v) if v else '<i>не указано</i>'}" for lbl, v in
                                 (("название", m.name), ("доза", m.dose), ("частота", m.frequency), ("длительность", m.duration))),
                    unsafe_allow_html=True,
                )
            if a.explanation is not None:
                _medication_explanation(a)
            due_html = []
            if a.doctor_due_text:
                due_html.append(f"<span class='doctor-date'>Срок от врача: «{esc(a.doctor_due_text)}»"
                                f"{' = ' + fmt_date(a.doctor_due_date) if a.doctor_due_date else ''}</span>")
            elif a.added_by == "doctor_instruction":
                due_html.append("<span class='no-date'>Срок от врача: не указан</span>")
            if a.patient_target_date:
                due_html.append(f"<span class='target-date'>Ваша целевая дата: {fmt_date(a.patient_target_date)} (не от врача)</span>")
            if a.completed_at:
                due_html.append(f"<span class='small-muted'>выполнено {a.completed_at[:10]}</span>")
            st.markdown(" ".join(due_html), unsafe_allow_html=True)
            if a.due_date_note:
                st.caption(a.due_date_note)
            for f in a.flags:
                st.caption(f"Проверка: {FLAG_TEXT.get(f, f)}")
            with st.expander("Изменить: ваша дата, заметка, описание"):
                hint = relative_due_hint(a.doctor_due_text, rec.appointment_date) if not a.doctor_due_date else None
                if hint:
                    st.caption(f"Подсказка: «{a.doctor_due_text}» от даты приёма ({fmt_date(rec.appointment_date)}) — "
                               f"это около {fmt_date(hint)}. Это ваш расчёт, а не дата от врача.")
                cur = date.fromisoformat(a.patient_target_date) if a.patient_target_date else None
                with st.form(f"edit_{k}_{ss.form_nonce}"):
                    target = st.date_input("Ваша целевая дата", value=cur, format="DD.MM.YYYY")
                    note = st.text_input("Ваша заметка", value=a.patient_note)
                    desc = st.text_input("Короткое описание", value=a.description,
                                         help="Слова врача не меняются — они хранятся отдельно.")
                    ok = st.form_submit_button("Сохранить")
                if ok:
                    a.patient_target_date = target.isoformat() if target else None
                    a.patient_note = note.strip()
                    if desc.strip() and desc.strip() != a.description:
                        a.description = desc.strip()
                        a.edited_by_patient = True
                    ctx.commit()
                if a.patient_target_date and st.button("Убрать вашу дату", key=f"cleardate_{k}"):
                    a.patient_target_date = None
                    ctx.commit()
                if a.added_by == "patient" and st.button("Удалить пункт", key=f"delact_{k}"):
                    rec.actions = [x for x in rec.actions if x.id != a.id]
                    ctx.commit()
            if a.patient_note:
                origin_box("patient", a.patient_note, extra_label="ваша заметка")


def _medication_explanation(a: ActionItem) -> None:
    x = a.explanation
    with st.expander(f"💊 Что это за лекарство — «{a.medication.name if a.medication else x.query}» (по источникам)"):
        if x.background_status == "answered":
            origin_box("ai", x.background, extra_label=x.generator)
            for n, s in enumerate(x.sources):
                render_source(s, key=f"med_{a.id}_{n}")
        else:
            st.info("В базе знаний нет сведений об этом лекарстве. Спросите врача или фармацевта.", icon="📚")
        st.markdown("**Что спросить у врача**")
        for q in x.ask_doctor:
            st.markdown(f"- {esc(q)}", unsafe_allow_html=True)
        st.caption("Как принимать — только по словам врача выше. Общие сведения не заменяют назначение.")
        if x.notes:
            for n in x.notes:
                st.caption(f"Проверка: {n}")


def _add_own(ctx: Ctx) -> None:
    rec = ctx.record
    with st.expander("➕ Добавить свой пункт (не от врача)"):
        with st.form("own_action", clear_on_submit=True):
            desc = st.text_input("Что сделать")
            cat = st.selectbox("Категория", list(ACTION_CATEGORY_LABELS), format_func=ACTION_CATEGORY_LABELS.get, index=4)
            target = st.date_input("Ваша целевая дата (необязательно)", value=None, format="DD.MM.YYYY")
            ok = st.form_submit_button("Добавить")
        if ok and desc.strip():
            rec.actions.append(ActionItem(id=next_id("A", rec.actions), category=cat, description=desc.strip(),
                                          source_statement="", added_by="patient",
                                          patient_target_date=target.isoformat() if target else None))
            ctx.commit()
