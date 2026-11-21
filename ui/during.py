"""«Во время приёма»: terms, notes, test results and questions -> grounded plain-language explanations."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from intake_assistant.consult import KIND_LABELS, explain_item, lab_names, short
from intake_assistant.labs import PatientDocumentError, ScannedPDFError, extract_patient_document
from intake_assistant.providers import ProviderError
from intake_assistant.records import ConsultationItem, Explanation, PatientQuestion, next_id

from .before import render_questions_list
from .common import Ctx, audio_block, esc, origin_box, render_source

KIND_HELP = {
    "term": "Слово или выражение, которое сказал врач или которое встретилось в документе (например, «МРТ»).",
    "note": "Что вы записали во время приёма — своими словами.",
    "test_result": "Текст из бланка анализов. Значения и единицы сохраняются точно как в документе.",
    "question": "Что осталось непонятным в объяснении врача.",
}


def _add_item(ctx: Ctx, kind: str, text: str, source: str, original: str | None = None,
              upload_id: str | None = None, audio: str | None = None) -> None:
    rec = ctx.record
    item = ConsultationItem(id=next_id("C", rec.consultation_items), kind=kind, source=source,
                            original_text=(original if original is not None else text), text=text,
                            upload_id=upload_id, audio_filename=audio)
    rec.consultation_items.append(item)
    if kind == "question":
        rec.patient_questions.append(PatientQuestion(id=next_id("Q", rec.patient_questions), text=text, source="consultation"))
    st.session_state.open_item = item.id
    st.session_state.form_nonce += 1  # clears the text box / uploader
    ctx.flash("success", f"Добавлено: {item.id} · {KIND_LABELS[kind]}.")
    ctx.commit()


def render_during(ctx: Ctx) -> None:
    ss = st.session_state
    rec = ctx.record
    st.markdown(
        "<div class='stage-hint'>Во время приёма: запишите термины, заметки, результаты анализов или вопросы. "
        "Объяснение всегда разделено на три части: <b>что сказано в вашем документе</b>, <b>общие сведения из "
        "источников</b> и <b>что уточнить у врача</b>. Это не замена объяснению врача.</div>",
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.6, 1], gap="large")
    with left:
        st.markdown("##### Добавить")
        kind = st.radio("Что добавить", list(KIND_LABELS), format_func=KIND_LABELS.get, horizontal=True,
                        key="consult_kind")
        st.caption(KIND_HELP[kind])
        t_text, t_audio, t_file = st.tabs(["✍️ Текст", "🎙️ Голос", "📎 Файл (PDF/TXT)"], key="consult_input_tab",
                                          on_change="rerun")
        with t_text:
            with st.form(f"consult_text_{ss.form_nonce}", clear_on_submit=True, border=False):
                text = st.text_area("Текст", height=130 if kind == "test_result" else 80, label_visibility="collapsed",
                                    placeholder={"term": "Например: ферритин", "note": "Что сказал врач…",
                                                 "test_result": "Вставьте строки из бланка анализов…",
                                                 "question": "Что осталось непонятным?"}[kind])
                go = st.form_submit_button("Добавить", type="primary")
            if go:
                if text.strip():
                    _add_item(ctx, kind, text.strip(), "text")
                else:
                    st.warning("Введите текст.")
        with t_audio:
            audio_block(ctx, "consult", lambda t, o, f: _add_item(ctx, kind, t, "audio", original=o, audio=f),
                        label="Запишите голосом")
        with t_file:
            up = st.file_uploader("PDF с текстом или TXT", type=["pdf", "txt"], key=f"consult_file_{ss.form_nonce}",
                                  help="Сканы и фотографии без текстового слоя не распознаются (OCR нет).")
            if st.button("Загрузить и добавить", disabled=up is None, key="consult_upload_btn"):
                data = up.getvalue()
                try:
                    doc = extract_patient_document(up.name, data)
                except ScannedPDFError as exc:
                    st.error(str(exc), icon="🖼️")
                except PatientDocumentError as exc:
                    st.error(str(exc))
                else:
                    meta = ctx.records.save_upload(rec, up.name, data, "test_result" if kind == "test_result" else "other")
                    meta.n_pages, meta.text_extracted = doc.n_pages, True
                    _add_item(ctx, kind, doc.text, "upload", upload_id=meta.id)
    with right:
        st.markdown("##### ❓ Ваши вопросы к врачу")
        st.caption("Отмечайте заданные вопросы и записывайте ответы. Список общий с этапом «До приёма».")
        render_questions_list(ctx, "qd", compact=True)

    st.divider()
    st.markdown(f"##### Записи с приёма ({len(rec.consultation_items)})")
    if not rec.consultation_items:
        st.info("Пока ничего не добавлено.", icon="🩺")
    for item in reversed(rec.consultation_items):
        _render_item(ctx, item)

    st.divider()
    st.markdown(f"##### 💾 Сохранённые объяснения ({len(rec.explanations)})")
    if not rec.explanations:
        st.caption("Сохраните объяснение кнопкой «Сохранить в запись о приёме».")
    for x in reversed(rec.explanations):
        with st.expander(f"{x.id} · {KIND_LABELS.get(x.kind, x.kind)} · {short(x.query, 60)}"):
            render_explanation(x)
            if st.button("Удалить объяснение", key=f"del_x_{x.id}"):
                rec.explanations = [e for e in rec.explanations if e.id != x.id]
                ctx.commit()


def _render_item(ctx: Ctx, item: ConsultationItem) -> None:
    ss = st.session_state
    rec = ctx.record
    src = {"text": "текст", "audio": "голос", "upload": "файл"}[item.source]
    up = rec.find_upload(item.upload_id)
    title = f"{item.id} · {KIND_LABELS[item.kind]} · {src}{' · ' + up.filename if up else ''} · {short(item.text, 50)}"
    with st.expander(title, expanded=ss.get("open_item") == item.id):
        origin_box(item.origin, item.text if len(item.text) < 4000 else item.text[:4000] + " …",
                   extra_label="исправлено" if item.was_edited else "")
        if item.was_edited:
            st.caption(f"Исходный текст: «{short(item.original_text, 300)}»")
        with st.popover("✏️ Исправить текст"):
            st.caption("Исходный текст сохраняется. Для анализов переписывайте значения и единицы точно как в бланке.")
            new = st.text_area("Текст", item.text, key=f"edit_c_{item.id}_{ss.form_nonce}", height=160)
            if st.button("Сохранить", key=f"save_c_{item.id}"):
                if new.strip() and new.strip() != item.text:
                    item.text = new.strip()
                    ss.pending_expl.pop(item.id, None)
                    ctx.commit()

        focus = None
        if item.kind == "test_result":
            names = lab_names(item.text)
            if names:
                st.markdown("**Показатели, найденные в документе (значения дословно):**")
                from intake_assistant.labs import parse_lab_lines

                st.dataframe(pd.DataFrame([
                    {"Показатель": r.name, "Значение": r.value, "Ед. изм.": r.unit or "—",
                     "Референсный интервал (из бланка)": r.reference_range or "не указан в документе",
                     "Отметка в бланке": r.flag or ""}
                    for r in parse_lab_lines(item.text)
                ]), hide_index=True, width="stretch")
                chosen = st.multiselect("Что объяснить", names, default=names[:1], key=f"focus_{item.id}")
                focus = ", ".join(chosen) if chosen else None
            else:
                st.caption("Строки со значениями не распознаны автоматически — объяснение будет по тексту целиком.")
        elif item.kind == "note":
            focus = st.text_input("Что именно объяснить (необязательно)", key=f"focus_{item.id}",
                                  placeholder="например: мигрень") or None

        c1, c2 = st.columns([1, 1])
        if c1.button("💡 Объяснить", key=f"explain_{item.id}", type="primary", disabled=ctx.provider is None):
            try:
                with st.spinner("Ищу в источниках и готовлю объяснение…"):
                    ss.pending_expl[item.id] = explain_item(item, ctx.store, ctx.provider, ss.top_k, ss.min_relevance,
                                                            ctx.settings.is_demo, focus=focus)
                ss.open_item = item.id
            except ProviderError as exc:
                st.error(str(exc))
        if c2.button("🗑️ Удалить запись", key=f"del_c_{item.id}"):
            rec.consultation_items = [c for c in rec.consultation_items if c.id != item.id]
            if item.upload_id and not any(c.upload_id == item.upload_id for c in rec.consultation_items):
                ctx.records.delete_upload(rec, item.upload_id)
            ss.pending_expl.pop(item.id, None)
            ctx.commit()

        x: Explanation | None = ss.pending_expl.get(item.id)
        if x is not None:
            st.markdown("---")
            render_explanation(x)
            s1, s2 = st.columns(2)
            if s1.button("💾 Сохранить в запись о приёме", key=f"save_x_{item.id}", type="primary"):
                x.id = next_id("X", rec.explanations)
                rec.explanations.append(x)
                ss.pending_expl.pop(item.id, None)
                ctx.flash("success", f"Объяснение сохранено как {x.id}.")
                ctx.commit()
            if x.ask_doctor and s2.button("➕ Добавить вопросы в список к врачу", key=f"addq_{item.id}"):
                known = {q.text.strip().lower() for q in rec.patient_questions}
                n = 0
                for q in x.ask_doctor:
                    if q.strip().lower() not in known:
                        rec.patient_questions.append(PatientQuestion(id=next_id("Q", rec.patient_questions), text=q,
                                                                     source="consultation"))
                        n += 1
                ctx.flash("success", f"Добавлено вопросов: {n}.")
                ctx.commit()


def render_explanation(x: Explanation) -> None:
    st.markdown("**1. Что сказано в вашем документе или заметке**")
    if x.lab_rows:
        st.dataframe(pd.DataFrame([
            {"Показатель": r.name, "Значение": r.value, "Ед. изм.": r.unit or "—",
             "Референсный интервал (из бланка)": r.reference_range or "не указан в документе",
             "Отметка в бланке": r.flag or "", "Строка документа": r.line}
            for r in x.lab_rows
        ]), hide_index=True, width="stretch")
        st.caption("Значения, единицы и интервалы скопированы из документа без изменений. Программа не оценивает, "
                   "в норме ли результат, и не дописывает отсутствующие интервалы.")
    for d in x.document_states:
        origin_box("patient_document" if x.kind == "test_result" else "patient", f"«{d}»")
    if not x.lab_rows and not x.document_states:
        st.caption("—")

    st.markdown("**2. Общие сведения из медицинских источников**")
    if x.background_status == "answered":
        origin_box("ai", x.background, extra_label=x.generator)
        if x.answer_language_note:
            st.info(x.answer_language_note, icon="🌐")
        st.markdown(f"<span class='small-muted'>Источники ({len(x.sources)}):</span>", unsafe_allow_html=True)
        for n, s in enumerate(x.sources):
            render_source(s, key=f"{x.id}_{n}")
    elif x.background_status == "no_documents":
        st.warning("В базе знаний нет документов — общее объяснение не формировалось. Загрузите материалы во вкладке «База знаний».", icon="📚")
    else:
        st.warning(x.background, icon="🚫")

    st.markdown("**3. Что уточнить у врача**")
    for q in x.ask_doctor:
        st.markdown(f"- {esc(q)}", unsafe_allow_html=True)
    if x.notes:
        with st.expander("Проверки программы"):
            for n in x.notes:
                st.markdown(f"- {n}")
