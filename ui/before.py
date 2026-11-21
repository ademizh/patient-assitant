"""«До приёма»: symptom intake, review, timeline, questions, specialist orientation, doctor report."""

from __future__ import annotations

import hashlib
from datetime import datetime

import pandas as pd
import streamlit as st

from intake_assistant.intake import IntakeSession
from intake_assistant.pdf_export import report_to_pdf
from intake_assistant.providers import ProviderError
from intake_assistant.records import PatientQuestion, next_id, sync_questions_from_intake
from intake_assistant.report import LIST_SECTIONS, build_report, report_to_json
from intake_assistant.report_edit import (
    CERTAINTY_RU,
    COLUMNS,
    REASON_RU,
    contradictions_to_rows,
    items_to_rows,
    missing_to_rows,
    report_to_markdown,
    rows_to_contradictions,
    rows_to_items,
    rows_to_missing,
)
from intake_assistant.sample import attach_guidance_to_report
from intake_assistant.schemas import SLOT_LABELS, ReportItem
from intake_assistant.specialist import NOT_A_DIAGNOSIS, suggest
from intake_assistant.timeline import describe_days, entry_timeline, symptom_timeline
from intake_assistant.urgent import check_entries

from .common import Ctx, audio_block, esc, fmt_ts, origin_box, render_source

# ---------------------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------------------


def refresh_urgent(ctx: Ctx) -> None:
    if ctx.pathway is not None and ctx.record.intake.entries:
        ctx.record.urgent_check = check_entries(ctx.record.intake.entries, ctx.pathway)


def run_extraction(ctx: Ctx) -> bool:
    ss = st.session_state
    intake: IntakeSession = ctx.record.intake
    refresh_urgent(ctx)
    if ctx.provider is None or not intake.entries:
        return False
    try:
        with st.spinner("Упорядочиваю сведения…"):
            intake.extraction = ctx.provider.extract(intake.entries)
            intake.extraction_generator = ctx.provider.name
        added = sync_questions_from_intake(ctx.record)
        if added:
            ctx.flash("info", f"Добавлено в список вопросов к врачу: {added}.")
        ss.extraction_stale = False
        return True
    except ProviderError as exc:
        ss.extraction_stale = True
        ctx.flash("error", f"{exc} Ваша запись сохранена; нажмите «Повторить обработку».")
        return False


def intake_turn(ctx: Ctx, reply: dict, spinner: str | None = "Упорядочиваю сведения…") -> None:
    """Send one patient action to the LangGraph intake workflow and save the result.

    reply = {"action": "answer", "text": …} | {"action": "skip"} | {"action": "finish"} |
            {"action": "resume"} | {"action": "retry"}
    The graph is paused at `ask_patient`; this resumes it, runs the processing steps and pauses again.
    """
    ss = st.session_state
    if ctx.intake_flow is None or ctx.provider is None:
        st.error("Обработка недоступна: не настроен поставщик ИИ.")
        return
    if spinner:
        with st.spinner(spinner):
            result = ctx.intake_flow.send(ctx.record, reply, ctx.provider, ctx.pathway)
    else:
        result = ctx.intake_flow.send(ctx.record, reply, ctx.provider, ctx.pathway)
    if result.error:
        ss.extraction_stale = True
        ctx.flash("error", f"{result.error} Ваша запись сохранена; нажмите «Повторить обработку».")
    elif reply.get("action") in ("answer", "retry"):
        ss.extraction_stale = False
    if result.questions_added:
        ctx.flash("info", f"Добавлено в список вопросов к врачу: {result.questions_added}.")
    ctx.commit()


def submit_entry(ctx: Ctx, text: str, source: str, original: str | None = None, filename: str | None = None) -> None:
    intake_turn(ctx, {"action": "answer", "text": text, "source": source, "original": original, "filename": filename})


# ---------------------------------------------------------------------------------------
# Urgent banner
# ---------------------------------------------------------------------------------------


def urgent_banner(ctx: Ctx) -> None:
    u = ctx.record.urgent_check
    if u is None:
        return
    positive = [m for m in u.matches if not m.negated]
    label = ("ДЕМОНСТРАЦИОННЫЕ ПРАВИЛА — не проверены клиницистом" if not u.pathway_validated
             else f"маршрут проверен: {u.reviewed_by}")
    if positive:
        found = "; ".join(f"«{m.matched_text}» ({m.entry_id})" for m in positive)
        st.error(f"🚨 **{u.message}**\n\nНайдено: {found}\n\n_{label}_", icon="🚨")


# ---------------------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------------------


def render_before(ctx: Ctx) -> None:
    st.markdown(
        "<div class='stage-hint'>Перед приёмом: расскажите о симптомах своими словами, проверьте, что записано верно, "
        "подготовьте вопросы к врачу и скачайте отчёт. Всё сохраняется в одну запись о приёме.</div>",
        unsafe_allow_html=True,
    )
    urgent_banner(ctx)
    tabs = st.tabs(
        ["💬 Рассказ о симптомах", "📝 Проверка и исправления", "🕒 Хронология", "❓ Вопросы к врачу",
         "🧭 К какому врачу?", "📄 Отчёт для врача"],
        key="before_tab", on_change="rerun",
    )
    with tabs[0]:
        _intake(ctx)
    with tabs[1]:
        _review(ctx)
    with tabs[2]:
        _timeline(ctx)
    with tabs[3]:
        _questions(ctx)
    with tabs[4]:
        _specialist(ctx)
    with tabs[5]:
        _report(ctx)


def _intake(ctx: Ctx) -> None:
    ss = st.session_state
    intake = ctx.record.intake
    col_chat, col_side = st.columns([2.1, 1], gap="large")
    with col_chat:
        with st.chat_message("assistant", avatar="🩺"):
            st.markdown(
                "Здравствуйте! Я помогу подготовиться к приёму: запишу, что вас беспокоит, чтобы врачу было проще. "
                "Я не ставлю диагноз и не советую лечение. Любой вопрос можно **пропустить**, а рассказ — "
                "**завершить** в любой момент."
            )
        for e in intake.entries:
            if e.question_text:
                with st.chat_message("assistant", avatar="🩺"):
                    st.markdown(e.question_text)
            with st.chat_message("user", avatar="🎙️" if e.source == "audio" else "🧑"):
                st.markdown(
                    f"<span class='entry-id'>{e.id}</span><span class='small-muted'>"
                    f"{'голос · ' + esc(e.audio_filename or 'запись') if e.source == 'audio' else 'текст'}"
                    f"{' · исправлено' if e.was_edited else ''}"
                    f"{' · поправка к ' + e.corrects_entry_id if e.corrects_entry_id else ''}</span>",
                    unsafe_allow_html=True,
                )
                st.write(e.text)

        question = intake.current_question()
        if intake.finished:
            st.success("Рассказ завершён. Проверьте записи во вкладке «Проверка и исправления», затем сформируйте «Отчёт для врача».")
        elif question:
            with st.chat_message("assistant", avatar="🩺"):
                st.markdown(f"**{question}**")
        else:
            st.info("Все основные темы затронуты. Добавьте, что считаете важным, или завершите рассказ.")

        if ss.extraction_stale and st.button("🔁 Повторить обработку"):
            intake_turn(ctx, {"action": "retry"})

        st.caption(
            "🧪 Демо-режим: текст остаётся на этом компьютере." if ctx.settings.is_demo else
            "ℹ️ Текст и аудио отправляются в OpenAI для расшифровки и обработки. Не указывайте ФИО, адрес и номера документов."
        )
        t_text, t_audio = st.tabs(["✍️ Написать", "🎙️ Голосом"], key="intake_input_tab", on_change="rerun")
        with t_text:
            with st.form(f"text_entry_{ss.form_nonce}", clear_on_submit=True, border=False):
                text = st.text_area("Ваш ответ", height=110, label_visibility="collapsed",
                                    placeholder="Опишите своими словами…" if not intake.entries else "Напишите ответ…")
                sent = st.form_submit_button("Отправить", type="primary", disabled=ctx.provider is None)
            if sent:
                if not text.strip():
                    st.warning("Сначала напишите ответ или нажмите «Пропустить вопрос».")
                else:
                    submit_entry(ctx, text, "text")
        with t_audio:
            audio_block(ctx, "intake", lambda t, o, f: submit_entry(ctx, t, "audio", original=o, filename=f),
                        label="Запишите ответ голосом")

        b1, b2, _ = st.columns([1, 1, 2])
        if b1.button("⏭️ Пропустить вопрос", disabled=not question or intake.finished):
            intake_turn(ctx, {"action": "skip"}, spinner=None)
        if intake.finished:
            if b2.button("↩️ Продолжить рассказ"):
                intake_turn(ctx, {"action": "resume"}, spinner=None)
        elif b2.button("✅ Завершить рассказ", disabled=not intake.entries):
            ctx.flash("success", "Рассказ завершён.")
            intake_turn(ctx, {"action": "finish"}, spinner=None)

    with col_side:
        done, total = intake.progress()
        st.markdown("**Темы**")
        st.progress(done / total, text=f"Отвечено или пропущено: {done} из {total}")
        icons = {"covered": "✅", "skipped": "⏭️", "open": "◻️"}
        st.markdown("\n".join(f"- {icons[s]} {label}" for label, s in intake.slot_status()))
        st.caption("◻️ не затронуто · ✅ есть ответ · ⏭️ пропущено. Пропущенные темы попадут в отчёт как "
                   "«не хватает информации», а не как отрицание симптома.")


def _review(ctx: Ctx) -> None:
    ss = st.session_state
    intake = ctx.record.intake
    if not intake.entries:
        st.info("Пока ничего не записано. Начните с вкладки «Рассказ о симптомах» или загрузите вымышленный пример в боковой панели.", icon="📝")
        return
    st.markdown("##### Исходные записи")
    st.caption("Исходная формулировка всегда сохраняется. Исправления хранятся рядом с ней и обрабатываются заново.")
    with st.form("review_entries"):
        edits: dict[str, str] = {}
        for e in intake.entries:
            c1, c2 = st.columns([1, 4])
            c1.markdown(
                f"<span class='entry-id'>{e.id}</span><br><span class='small-muted'>"
                f"{'голос' if e.source == 'audio' else 'текст'}<br>"
                f"{SLOT_LABELS.get(e.question_slot, '') if e.question_slot else 'свободный рассказ'}</span>",
                unsafe_allow_html=True,
            )
            text_key = hashlib.sha256(e.text.encode()).hexdigest()[:8]
            edits[e.id] = c2.text_area(f"Текст {e.id}", e.text, key=f"rev_{e.id}_{ss.form_nonce}_{text_key}",
                                       label_visibility="collapsed", height=68)
            if e.was_edited:
                c2.caption(f"Исходно: «{e.original_text}»")
        saved = st.form_submit_button("Сохранить исправления и обработать заново", type="primary")
    if saved:
        changed = False
        for e in intake.entries:
            new = edits[e.id].strip()
            if new and new != e.text:
                intake.update_entry_text(e.id, new)
                changed = True
        if changed:
            run_extraction(ctx)
            ctx.flash("success", "Исправления сохранены. Обновите отчёт для врача, чтобы они в него вошли.")
        else:
            ctx.flash("info", "Изменений нет.")
        ctx.commit()

    st.markdown("##### Что удалось выделить (со ссылками на записи)")
    ex = intake.extraction
    if ex is None or ss.extraction_stale:
        st.warning("Последние записи ещё не обработаны.")
        if st.button("Обработать сейчас"):
            run_extraction(ctx)
            ctx.commit()
    if ex is not None:
        st.caption(f"Сформировано: {intake.extraction_generator}")
        df = pd.DataFrame([
            {
                "Тема": SLOT_LABELS.get(f.category, f.category),
                "Утверждение": f.statement,
                "Записи": ", ".join(f.source_entry_ids),
                "Достоверность": CERTAINTY_RU[f.certainty],
                "Явное отрицание": "да" if f.negated else "",
                "Срок (как сказано)": f.time_reference or "",
            }
            for f in ex.facts
        ])
        st.dataframe(df, width="stretch", hide_index=True)
        if ex.contradictions:
            st.markdown("**Возможные противоречия**")
            for c in ex.contradictions:
                st.markdown(f"- ⚠️ {c.description} `{', '.join(c.source_entry_ids)}`")
        if intake.skipped_slots:
            st.markdown("**Пропущено пациентом:** " + ", ".join(SLOT_LABELS[s] for s in intake.skipped_slots))


def _timeline(ctx: Ctx) -> None:
    intake = ctx.record.intake
    if not intake.entries:
        st.info("Хронология появится после первых записей.", icon="🕒")
        return
    st.markdown("##### Как развивались события (со слов пациента)")
    st.caption("Сроки показаны так, как вы их назвали. Порядок примерный: «≈ 2 нед. назад» — это оценка только для "
               "сортировки, а не дата.")
    order = {e.id: n for n, e in enumerate(intake.entries)}
    rows = symptom_timeline(intake.extraction.facts if intake.extraction else [], order)
    if rows:
        st.dataframe(pd.DataFrame([
            {"Когда (как сказано)": r.time_reference, "Примерно": describe_days(r.approx_days_ago),
             "Что": r.statement, "Тема": r.category_label, "Записи": ", ".join(r.entry_ids),
             "Неточно": "да" if r.uncertain else ""}
            for r in rows
        ]), width="stretch", hide_index=True)
    else:
        st.caption("В записях пока нет указаний на сроки.")
    prog = [f for f in (intake.extraction.facts if intake.extraction else []) if f.category == "progression"]
    if prog:
        st.markdown("**Изменения со временем**")
        for f in prog:
            origin_box("patient", f.statement, extra_label=", ".join(f.source_entry_ids))
    st.markdown("##### Все записи в порядке поступления")
    st.dataframe(pd.DataFrame([
        {"Запись": r["id"], "Время": fmt_ts(r["created_at"]), "Способ": r["source"], "Тема": r["topic"],
         "Текст": r["text"], "Исходный текст (если исправлен)": r["original_text"] or ""}
        for r in entry_timeline(intake.entries)
    ]), width="stretch", hide_index=True)


def render_questions_list(ctx: Ctx, key_prefix: str, compact: bool = False) -> None:
    rec = ctx.record
    if not rec.patient_questions:
        st.caption("Вопросов пока нет.")
    changed = False
    for q in rec.patient_questions:
        c1, c2 = st.columns([0.08, 0.92])
        asked = c1.checkbox("Задан", value=q.asked, key=f"{key_prefix}_asked_{q.id}_{q.asked}", label_visibility="collapsed")
        if asked != q.asked:
            q.asked = asked
            changed = True
        src = {"intake": "из рассказа", "manual": "добавлен вручную", "consultation": "во время приёма"}[q.source]
        c2.markdown(f"<span class='entry-id'>{q.id}</span>{esc(q.text)} <span class='small-muted'>· {src}"
                    f"{' · ' + ', '.join(q.entry_ids) if q.entry_ids else ''}</span>", unsafe_allow_html=True)
        if not compact or q.asked:
            note = c2.text_input("Что ответил врач (ваша запись)", value=q.answer_note, key=f"{key_prefix}_note_{q.id}_{hashlib.sha256(q.answer_note.encode()).hexdigest()[:8]}",
                                 placeholder="Запишите ответ врача своими словами…")
            if note != q.answer_note:
                q.answer_note = note
                changed = True
    if changed:
        ctx.save()


def _questions(ctx: Ctx) -> None:
    rec = ctx.record
    st.caption("Вопросы, которые вы назвали в рассказе, добавляются сюда автоматически. Список доступен и во время приёма.")
    render_questions_list(ctx, "qb")
    with st.form("add_question", clear_on_submit=True):
        new_q = st.text_input("Новый вопрос к врачу")
        add = st.form_submit_button("➕ Добавить вопрос")
    if add and new_q.strip():
        rec.patient_questions.append(PatientQuestion(id=next_id("Q", rec.patient_questions), text=new_q.strip(), source="manual"))
        ctx.commit()
    if rec.patient_questions:
        with st.popover("🗑️ Удалить вопрос"):
            qid = st.selectbox("Вопрос", [q.id for q in rec.patient_questions],
                               format_func=lambda i: next(f"{q.id}: {q.text[:60]}" for q in rec.patient_questions if q.id == i))
            if st.button("Удалить", key="del_q"):
                rec.patient_questions = [q for q in rec.patient_questions if q.id != qid]
                ctx.commit()


def _specialist(ctx: Ctx) -> None:
    rec = ctx.record
    st.warning(NOT_A_DIAGNOSIS, icon="🧭")
    u = rec.urgent_check
    if u is not None:
        label = ("**ДЕМОНСТРАЦИОННЫЕ ПРАВИЛА** — не проверены клиницистом; не используйте их для реальных решений."
                 if not u.pathway_validated else f"Маршрут проверен: {u.reviewed_by}.")
        with st.expander("🚨 Проверка признаков срочности", expanded=any(not m.negated for m in u.matches)):
            st.markdown(f"**{u.pathway_title}**  \n{label}")
            st.caption(u.pathway_source)
            pos = [m for m in u.matches if not m.negated]
            for m in pos:
                origin_box("rules", f"{m.label}: «{m.matched_text}» (запись {m.entry_id})")
            for m in [m for m in u.matches if m.negated]:
                st.caption(f"Упомянуто с отрицанием: {m.label} — «{m.matched_text}» ({m.entry_id})")
            (st.error if pos else st.info)(u.message)
    if not rec.intake.entries:
        st.info("Сначала расскажите о симптомах — подсказка строится на ваших словах, повторно ничего спрашивать не нужно.")
        return
    g = rec.specialist_guidance
    stale = g is not None and g.based_on_signature != rec.intake_signature()
    if st.button("🧭 Подсказать, к какому врачу обратиться" if g is None else "🔄 Обновить подсказку",
                 type="primary", disabled=ctx.provider is None):
        try:
            with st.spinner("Ищу в справочных материалах…"):
                rec.specialist_guidance = suggest(rec, ctx.store, ctx.provider, st.session_state.top_k,
                                                  st.session_state.min_relevance)
            attach_guidance_to_report(rec)
            ctx.commit()
        except ProviderError as exc:
            st.error(str(exc))
    if g is None:
        st.caption("Подсказка необязательна. Она опирается только на документы из «Базы знаний» и показывает источники.")
        return
    if stale:
        st.warning("Рассказ изменился после этой подсказки — обновите её.")
    st.caption(f"Поиск по вашим словам: «{g.query[:200]}» · сформировано: {g.generator}")
    if g.status == "suggested":
        for o in g.options:
            st.markdown(f"#### 👩‍⚕️ {o.specialist}")
            origin_box("ai", o.reason, extra_label=g.generator)
            for n, s in enumerate(o.sources):
                render_source(s, key=f"sp_{o.specialist}_{n}")
    elif g.status == "no_documents":
        st.warning("В базе знаний нет документов. Загрузите демонстрационные материалы во вкладке «База знаний».")
    else:
        st.warning("В источниках недостаточно сведений для подсказки по вашей ситуации.", icon="🚫")
    if g.uncertainty:
        st.markdown(f"**Что неясно:** {g.uncertainty}")
    if g.clarifying_question:
        st.info(f"**Уточните, пожалуйста:** {g.clarifying_question}  \nДобавьте ответ в «Рассказ о симптомах» и обновите подсказку.")
    st.success(g.default_advice, icon="🩺")
    for n in g.notes:
        st.caption(f"Проверка: {n}")


def generate_report(ctx: Ctx) -> None:
    rec = ctx.record
    ss = st.session_state
    if ss.extraction_stale or rec.intake.extraction is None:
        if not run_extraction(ctx):
            return
    with st.spinner("Формирую отчёт…"):
        models = ({"обработка": "демо-эвристики"} if ctx.settings.is_demo else
                  {"обработка": ctx.settings.chat_model, "расшифровка": ctx.settings.transcription_model})
        rec.report = build_report(rec.intake, rec.intake.extraction, mode=ctx.settings.mode,
                                  generator=rec.intake.extraction_generator or ctx.provider.name, models=models)
        attach_guidance_to_report(rec)
        rec.report_signature = rec.intake_signature()


def _report(ctx: Ctx) -> None:
    ss = st.session_state
    rec = ctx.record
    if not rec.intake.entries:
        st.info("Данных пока нет. Сначала заполните «Рассказ о симптомах» (или загрузите вымышленный пример).", icon="📄")
        return
    c1, c2 = st.columns([1, 3])
    if c1.button("⚙️ Сформировать отчёт" if rec.report is None else "🔄 Обновить отчёт", type="primary",
                 disabled=ctx.provider is None):
        generate_report(ctx)
        ss.form_nonce += 1
        ctx.commit()
    if not rec.intake.finished:
        c2.caption("Рассказ не отмечен как завершённый — отчёт можно сформировать, в нём это будет указано.")
    report = rec.report
    if report is None:
        return
    if rec.report_is_stale():
        st.warning("Рассказ изменился после формирования отчёта. Обновите отчёт, чтобы учесть изменения "
                   "(ваши правки в отчёте будут заменены).")
    st.caption("Сведения пациента показаны со ссылками на записи (E1, E2…). Резюме ИИ отделено. Можно править ячейки, "
               "добавлять и удалять строки, затем нажать «Сохранить правки». Исправленные пункты помечаются для врача.")
    valid_ids = {e.id for e in report.source_entries}
    with st.form(f"report_edit_{ss.form_nonce}"):
        mc = report.main_concern
        mc_text = st.text_input("Основная жалоба (со слов пациента)", mc.text if mc else "")
        st.markdown("🤖 **Резюме, составленное ИИ** — это не слова пациента, сверяйте с записями")
        summary = st.text_area("Резюме ИИ", report.ai_generated.summary, height=100, label_visibility="collapsed")
        col_cfg = {
            "_idx": None,
            "Раздел": st.column_config.TextColumn("Раздел", width="small"),
            "Текст": st.column_config.TextColumn("Со слов пациента", width="large"),
            "Записи": st.column_config.TextColumn("Записи", help="Номера записей, например E2, E5", width="small"),
            "Достоверность": st.column_config.SelectboxColumn("Достоверность", options=list(CERTAINTY_RU.values()), width="small"),
            "Изменено": st.column_config.CheckboxColumn("Изменено", disabled=True, width="small"),
            "Пометки": st.column_config.TextColumn("Пометки", disabled=True, width="small"),
        }
        edited = {}
        for key, title in LIST_SECTIONS:
            st.markdown(f"**{title}**")
            df = pd.DataFrame(items_to_rows(getattr(report, key)), columns=COLUMNS)
            edited[key] = st.data_editor(df, column_config=col_cfg, num_rows="dynamic", hide_index=True,
                                         width="stretch", key=f"ed_{key}_{ss.form_nonce}")
        st.markdown("**Не хватает информации — уточнить на приёме**")
        missing_df = st.data_editor(
            pd.DataFrame(missing_to_rows(report.missing_information), columns=["Тема", "Причина"]),
            column_config={"Причина": st.column_config.SelectboxColumn(options=list(REASON_RU.values()))},
            num_rows="dynamic", hide_index=True, width="stretch", key=f"ed_missing_{ss.form_nonce}",
        )
        st.markdown("**Противоречия — уточнить на приёме**")
        contra_df = st.data_editor(
            pd.DataFrame(contradictions_to_rows(report.contradictions), columns=["Описание", "Записи"]),
            num_rows="dynamic", hide_index=True, width="stretch", key=f"ed_contra_{ss.form_nonce}",
        )
        save = st.form_submit_button("💾 Сохранить правки", type="primary")
    if save:
        upd = report.model_copy(deep=True)
        if mc_text.strip():
            if mc is None:
                upd.main_concern = ReportItem(label="Основная жалоба", category="main_concern", text=mc_text.strip(),
                                              edited_by_reviewer=True, flags=["no_source"])
            elif mc_text.strip() != mc.text:
                upd.main_concern = mc.model_copy(update={"text": mc_text.strip(), "edited_by_reviewer": True})
        else:
            upd.main_concern = None
        if summary.strip() != report.ai_generated.summary:
            upd.ai_generated.summary = summary.strip()
            upd.ai_generated.edited_by_reviewer = True
        for key, _ in LIST_SECTIONS:
            setattr(upd, key, rows_to_items(edited[key].to_dict("records"), getattr(report, key), valid_ids))
        upd.missing_information = rows_to_missing(missing_df.to_dict("records"))
        upd.contradictions = rows_to_contradictions(contra_df.to_dict("records"), valid_ids)
        rec.report = upd
        ss.form_nonce += 1
        ctx.flash("success", "Правки отчёта сохранены.")
        ctx.commit()

    st.markdown("##### Скачать")
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    prefix = "FICTIONAL_" if report.meta.fictional_example else ""
    d1, d2, _ = st.columns([1, 1, 2])
    try:
        d1.download_button("⬇️ PDF для врача", report_to_pdf(report), f"{prefix}doctor_report_{stamp}.pdf",
                           "application/pdf", width="stretch", type="primary")
    except Exception as exc:  # noqa: BLE001
        d1.error(f"Не удалось создать PDF ({type(exc).__name__}). JSON доступен.")
    d2.download_button("⬇️ JSON", report_to_json(report), f"{prefix}doctor_report_{stamp}.json",
                       "application/json", width="stretch")
    with st.expander("Предпросмотр", expanded=False):
        st.markdown(report_to_markdown(report))
    if report.ai_generated.notes:
        with st.expander("Служебные заметки (проверки программы)"):
            for n in report.ai_generated.notes:
                st.markdown(f"- {n}")
