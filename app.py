"""Streamlit UI — Помощник пациента до, во время и после приёма (прототип, не проверен клинически).

Запуск:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import date
from pathlib import Path

import streamlit as st

from intake_assistant.config import Settings, load_settings
from intake_assistant.graphs.intake_graph import IntakeFlow
from intake_assistant.providers import ProviderError, get_provider
from intake_assistant.rag.store import DocumentStore
from intake_assistant.records import AppointmentRecord, RecordStore
from intake_assistant.sample import build_fictional_record
from intake_assistant.urgent import load_pathway
from ui.after import render_after, reminder_banner
from ui.before import render_before
from ui.common import CSS, Ctx, fmt_date, footer, hero
from ui.during import render_during
from ui.knowledge import load_demo_documents, render_knowledge
from ui.record_view import render_record

# ---- Privacy: keep third-party loggers quiet; this app never logs patient content or keys.
logging.basicConfig(level=logging.WARNING)
for _name in ("httpx", "httpcore", "openai", "chromadb", "urllib3"):
    logging.getLogger(_name).setLevel(logging.WARNING)

st.set_page_config(page_title="Помощник пациента: до, во время и после приёма", page_icon="🩺", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def _provider(_settings: Settings, cache_key: tuple):
    return get_provider(_settings)


@st.cache_resource(show_spinner=False)
def _store(_provider_obj, chroma_path: str, library_path: str, prefix: str, signature: str):
    return DocumentStore(Path(chroma_path), Path(library_path), prefix, _provider_obj)


@st.cache_resource(show_spinner=False)
def _intake_flow(db_path: str) -> IntakeFlow:
    """One compiled LangGraph intake graph with a SQLite checkpointer, shared by all browser sessions."""
    return IntakeFlow(Path(db_path))


settings = load_settings()
provider_error: str | None = None
try:
    provider = _provider(settings, (settings.mode, settings.chat_model, settings.embedding_model,
                                    settings.transcription_model, settings.transcription_language))
except ProviderError as exc:
    provider = None
    provider_error = str(exc)

store = None
store_error: str | None = None
if provider is not None:
    try:
        store = _store(provider, str(settings.chroma_path), str(settings.library_path),
                       settings.collection_prefix, provider.embedding_signature)
    except Exception as exc:  # noqa: BLE001
        store_error = f"Не удалось открыть хранилище документов {settings.chroma_path} ({type(exc).__name__})."

records = RecordStore(settings.appointments_path)
# Checkpoints of the intake graph contain patient text, so they live next to the records (and are deleted with them).
intake_flow = _intake_flow(str(settings.appointments_path / "intake_graph_checkpoints.sqlite"))
try:
    pathway = load_pathway(settings.urgent_rules_path)
except (OSError, ValueError) as exc:
    pathway = None
    store_error = (store_error or "") + f" Файл правил срочности не загружен ({type(exc).__name__})."

try:
    TODAY = date.fromisoformat(os.environ["APP_TODAY"]) if os.getenv("APP_TODAY") else date.today()
except ValueError:
    TODAY = date.today()

# ---------------------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------------------

ss = st.session_state
ss.setdefault("flash", [])
ss.setdefault("form_nonce", 0)
ss.setdefault("qa_history", [])
ss.setdefault("chunk_size", settings.chunk_size)
ss.setdefault("chunk_overlap", settings.chunk_overlap)
ss.setdefault("top_k", settings.top_k)
ss.setdefault("min_relevance", float(settings.min_relevance))

PER_RECORD_STATE = ("pending_expl", "unmatched_sentences", "action_notes", "extraction_stale", "open_item",
                    "pending_audio_intake", "pending_audio_consult", "pending_audio_instr")


def reset_record_state() -> None:
    for k in PER_RECORD_STATE:
        ss.pop(k, None)
    ss.pending_expl = {}
    ss.extraction_stale = False
    ss.form_nonce += 1


if "record" not in ss:
    last = records.most_recent()
    ss.record = records.load(last.id) if last else AppointmentRecord()
    reset_record_state()
ss.setdefault("pending_expl", {})
ss.setdefault("extraction_stale", False)


def flash(kind: str, msg: str) -> None:
    ss.flash.append((kind, msg))


def save() -> None:
    rec: AppointmentRecord = ss.record
    if not rec.is_empty() or records.exists(rec.id):
        records.save(rec)


def commit() -> None:
    save()
    st.rerun()


ctx = Ctx(settings=settings, provider=provider, store=store, records=records, pathway=pathway, today=TODAY,
          flash=flash, commit=commit, save=save, intake_flow=intake_flow)
record: AppointmentRecord = ss.record

# ---------------------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------------------

mode_badge = (
    '<span class="badge badge-demo">Демо-режим · без ИИ-модели</span>'
    if settings.is_demo
    else f'<span class="badge badge-live">Рабочий режим · OpenAI {settings.chat_model}</span>'
)
hero(mode_badge + '<span class="badge badge-proto">LangGraph · RAG · проверка по источникам</span>'
     + ('<span class="badge badge-demo">Вымышленная запись</span>' if record.fictional else ''))
if settings.is_demo:
    st.markdown(
        "<div class='notice'>🧪 <b>Демо-режим</b>: ключ <code>OPENAI_API_KEY</code> не задан, поэтому работают простые "
        "офлайн-правила. Данные не покидают компьютер.</div>",
        unsafe_allow_html=True,
    )
if provider_error:
    st.error(provider_error)
if store_error:
    st.error(store_error)

# ---------------------------------------------------------------------------------------
# Sidebar: the appointment record
# ---------------------------------------------------------------------------------------

with st.sidebar:
    st.markdown("<div class='side-brand'>Medical<b>.kz</b> · помощник пациента</div>", unsafe_allow_html=True)
    st.subheader("📁 Запись о приёме")
    saved = records.list_records()
    options = [r.id for r in saved]
    # Labels are unique (short id suffix): two records with the same title must never be confused.
    labels = {r.id: f"{'🧪 ' if r.fictional else ''}{r.title} · {fmt_date(r.appointment_date)} · #{r.id[:4]}" for r in saved}
    if record.id not in options:
        options.insert(0, record.id)
        labels[record.id] = f"{record.title} (не сохранена — пока пустая)"
    # The widget key depends on the current record and the option list, so a re-ordered or extended list can
    # never silently switch the patient to another record (regression test: test_record_selector_is_stable).
    sel_key = "record_select_" + hashlib.sha256(("|".join(options) + record.id).encode()).hexdigest()[:12]
    chosen = st.selectbox("Текущая запись", options, index=options.index(record.id), format_func=lambda i: labels[i],
                          key=sel_key)
    if chosen != record.id:
        ss.record = records.load(chosen)
        reset_record_state()
        st.rerun()

    c1, c2 = st.columns(2)
    if c1.button("➕ Новая", width="stretch", help="Создать новую пустую запись о приёме."):
        save()
        ss.record = AppointmentRecord()
        reset_record_state()
        st.rerun()
    if c2.button("👤 Пример", width="stretch", disabled=provider is None,
                 help="Создать ВЫМЫШЛЕННУЮ запись со всеми этапами: рассказ → отчёт → объяснение → список дел."):
        try:
            with st.spinner("Создаю вымышленный пример (до, во время и после приёма)…"):
                save()
                load_demo_documents(ctx, quiet=True)
                ss.record = build_fictional_record(provider, store, records, ss.top_k, ss.min_relevance,
                                                   settings.is_demo, pathway, today=TODAY)
                reset_record_state()
            flash("success", "Вымышленный пример создан. Пройдите вкладки «До приёма» → «Во время приёма» → «После приёма».")
            st.rerun()
        except ProviderError as exc:
            st.error(str(exc))

    with st.form(f"record_meta_{ss.form_nonce}"):
        title = st.text_input("Название записи", value=record.title)
        appt = st.date_input("Дата приёма", value=date.fromisoformat(record.appointment_date) if record.appointment_date else None,
                             format="DD.MM.YYYY")
        if st.form_submit_button("Сохранить"):
            record.title = title.strip() or record.title
            record.appointment_date = appt.isoformat() if appt else None
            records.save(record)
            flash("success", "Сохранено.")
            st.rerun()

    reminder_banner(ctx, "sidebar")

    with st.popover("🗑️ Удалить запись", width="stretch"):
        st.write("Будут безвозвратно удалены эта запись о приёме и **все загруженные в неё файлы**. "
                 "Документы базы знаний не затрагиваются.")
        if st.button("Да, удалить запись и файлы", type="primary"):
            records.delete(record.id) if records.exists(record.id) else None
            intake_flow.forget(record.id)  # also delete the saved LangGraph session of this record
            ss.record = AppointmentRecord()
            ss.qa_history = []
            reset_record_state()
            flash("success", "Запись и связанные файлы удалены.")
            st.rerun()

    st.divider()
    st.subheader("Настройки")
    st.markdown(
        f"<div class='small-muted'>Режим: <b>{'демо' if settings.is_demo else 'рабочий'}</b><br>"
        f"Модель: <code>{settings.chat_model if not settings.is_demo else '—'}</code><br>"
        f"Расшифровка: <code>{settings.transcription_model if not settings.is_demo else '—'}</code>"
        f" (язык: {settings.transcription_language or 'авто'})<br>"
        f"Эмбеддинги: <code>{provider.embedding_signature if provider else '—'}</code></div>",
        unsafe_allow_html=True,
    )
    with st.expander("Параметры поиска"):
        ss.chunk_size = st.number_input("Размер фрагмента (символов)", 200, 4000, ss.chunk_size, 50)
        ss.chunk_overlap = st.number_input("Перекрытие фрагментов (символов)", 0, ss.chunk_size // 2,
                                           min(ss.chunk_overlap, ss.chunk_size // 2), 25)
        ss.top_k = st.slider("Сколько фрагментов искать", 1, 10, ss.top_k)
        ss.min_relevance = st.slider("Минимальная релевантность (косинус)", 0.0, 0.9, ss.min_relevance, 0.01,
                                     help="Фрагменты ниже порога не считаются подтверждением.")
        st.caption("Размер и перекрытие применяются к новым документам; для старых — «Переиндексировать всё».")
    st.divider()
    st.markdown(
        "<div class='small-muted'>🔒 <b>Конфиденциальность</b>: записи о приёме хранятся локально в папке "
        f"<code>{settings.appointments_path.name}/</code> на этом компьютере, отдельно от базы знаний, и удаляются "
        "кнопкой «Удалить запись». В рабочем режиме текст и аудио отправляются в OpenAI для обработки. "
        "Приложение не записывает в журнал сведения пациента и ключи API.</div>",
        unsafe_allow_html=True,
    )

for kind, msg in ss.flash:
    getattr(st, kind)(msg)
ss.flash = []

# ---------------------------------------------------------------------------------------
# Stage cards: where am I, what is done, one click to open a stage
# ---------------------------------------------------------------------------------------

MAIN_TABS = ["🗓️ 1 · До приёма", "🩺 2 · Во время приёма", "✅ 3 · После приёма", "📁 Вся запись", "📚 База знаний"]


def _open_tab(label: str) -> None:
    ss.main_tab = label


done_topics, total_topics = record.intake.progress()
n_done = sum(1 for a in record.actions if a.status == "done")
n_open_issues = sum(1 for i in record.instruction_issues if not i.resolved)
stages = [
    ("1", "До приёма", "Расскажите о симптомах, подготовьте вопросы и отчёт для врача.",
     f"Записей: {len(record.intake.entries)} · темы {done_topics}/{total_topics}"
     + (" · отчёт готов" if record.report else ""), bool(record.report), MAIN_TABS[0]),
    ("2", "Во время приёма", "Запишите термины и анализы — получите объяснение по источникам.",
     f"Записей: {len(record.consultation_items)} · объяснений: {len(record.explanations)}",
     bool(record.explanations), MAIN_TABS[1]),
    ("3", "После приёма", "Добавьте указания врача — получите список дел и объяснение лекарств.",
     f"Пунктов: {len(record.actions)} · выполнено: {n_done}"
     + (f" · уточнить: {n_open_issues}" if n_open_issues else ""), bool(record.actions) and not n_open_issues,
     MAIN_TABS[2]),
]
for col, (num, title, desc, status, ok, tab_label) in zip(st.columns(3, gap="medium"), stages):
    with col, st.container(key=f"stage_card_{num}"):
        st.markdown(
            f"<div><span class='stage-num'>{num}</span><span class='stage-title'>{title}</span></div>"
            f"<div class='stage-desc'>{desc}</div><span class='stage-status{' done' if ok else ''}'>{status}</span>",
            unsafe_allow_html=True,
        )
        st.button("Открыть →", key=f"open_stage_{num}", on_click=_open_tab, args=(tab_label,))

tab_before, tab_during, tab_after, tab_record, tab_kb = st.tabs(MAIN_TABS, key="main_tab", on_change="rerun")
with tab_before:
    render_before(ctx)
with tab_during:
    render_during(ctx)
with tab_after:
    render_after(ctx)
with tab_record:
    render_record(ctx)
with tab_kb:
    render_knowledge(ctx)

footer()
