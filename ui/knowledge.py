"""«База знаний»: questions over the medical documents + document library (separate from patient data)."""

from __future__ import annotations

import streamlit as st

from intake_assistant.config import PROJECT_ROOT
from intake_assistant.providers import ProviderError
from intake_assistant.rag.loaders import DocumentError
from intake_assistant.rag.qa import ask

from .common import Ctx, esc, render_source

SAMPLE_DOCS = PROJECT_ROOT / "data" / "sample_documents"


def load_demo_documents(ctx: Ctx, quiet: bool = False) -> None:
    if ctx.store is None:
        return
    ss = st.session_state
    with st.spinner("Индексирую демонстрационные документы…"):
        for path in sorted(SAMPLE_DOCS.iterdir()):
            if path.suffix.lower() not in (".pdf", ".md", ".txt"):
                continue
            try:
                r = ctx.store.add_file(path.name, path.read_bytes(), ss.chunk_size, ss.chunk_overlap)
                if not quiet:
                    ctx.flash("success" if r.status == "indexed" else "info",
                              f"{r.filename}: " + (f"проиндексировано фрагментов: {r.n_chunks}." if r.status == "indexed"
                                                   else "уже есть в базе — пропущено."))
            except (DocumentError, ProviderError) as exc:
                ctx.flash("error", str(exc))


def render_knowledge(ctx: Ctx) -> None:
    t_ask, t_docs = st.tabs(["🔎 Вопрос по документам", "📚 Документы"], key="kb_tab", on_change="rerun")
    with t_ask:
        _ask(ctx)
    with t_docs:
        _library(ctx)


def _ask(ctx: Ctx) -> None:
    ss = st.session_state
    st.caption("Ответы строятся **только** по документам базы знаний, которые хранятся отдельно от сведений пациента. "
               "Текст документов считается данными, а не инструкциями. Это не замена консультации врача.")
    n = ctx.store.count() if ctx.store else 0
    if n == 0:
        st.info("В базе знаний пока нет документов.", icon="📚")
        if st.button("Загрузить демонстрационные документы", key="ask_load_demo", disabled=ctx.store is None):
            load_demo_documents(ctx)
            ctx.commit()
        return
    examples = [
        "Что такое референсный интервал?",
        "Что нужно указать в списке лекарств?",
        "Какая максимальная суточная доза ибупрофена для детей?",
    ]
    st.markdown("<span class='small-muted'>Примеры:</span>", unsafe_allow_html=True)
    chosen = None
    for col, q in zip(st.columns(len(examples)), examples):
        if col.button(q, key=f"ex_{q}", width="stretch"):
            chosen = q
    with st.form("ask_form", clear_on_submit=True):
        q_text = st.text_input("Вопрос", placeholder="Задайте вопрос по документам базы знаний…")
        asked = st.form_submit_button("Спросить", type="primary")
    question = chosen or (q_text.strip() if asked else None)
    if asked and not q_text.strip():
        st.warning("Введите вопрос.")
    if question:
        try:
            with st.spinner("Ищу подходящие фрагменты и готовлю ответ…"):
                res = ask(question, ctx.store, ctx.provider, ss.top_k, ss.min_relevance, is_demo=ctx.settings.is_demo)
            ss.qa_history.insert(0, (question, res))
        except ProviderError as exc:
            st.error(str(exc))
    for i, (q, res) in enumerate(ss.qa_history):
        st.markdown(f"#### ❓ {esc(q)}", unsafe_allow_html=True)
        if res.status == "answered":
            st.markdown(res.answer)
            if res.language_note:
                st.info(res.language_note, icon="🌐")
            for note in res.notes:
                st.caption(f"Проверка: {note}")
            if res.removed_citations:
                st.caption(f"⚠️ Удалено ссылок на не найденные источники: {len(res.removed_citations)}.")
            st.markdown(f"**Источники ({len(res.sources)})**")
            for n_, s in enumerate(res.sources):
                render_source(s, expanded=False, key=f"qa_{i}_{n_}")
        else:
            st.warning(res.answer, icon="🚫")
            if res.model_note:
                st.caption(f"Пояснение модели: {res.model_note}")
        others = [c for c in res.retrieved if c not in res.cited]
        if others:
            with st.expander(f"Другие найденные фрагменты, не использованные в ответе ({len(others)})"):
                for c in others:
                    render_source(c)
        st.divider()


def _library(ctx: Ctx) -> None:
    ss = st.session_state
    store = ctx.store
    if store is None:
        st.error("Хранилище документов недоступно.")
        return
    st.caption(f"Коллекция `{store.collection_name}` · эмбеддинги `{ctx.provider.embedding_signature}` · размер фрагмента "
               f"{ss.chunk_size}, перекрытие {ss.chunk_overlap}. Здесь только справочные медицинские документы — "
               "**не загружайте сюда документы пациента** (для них есть разделы «Во время приёма» и «После приёма»).")
    up_col, demo_col = st.columns([3, 1])
    with up_col:
        files = st.file_uploader("Загрузите документы PDF, TXT или Markdown", type=["pdf", "txt", "md", "markdown"],
                                 accept_multiple_files=True, key=f"docs_{ss.form_nonce}")
        if st.button("Добавить в базу знаний", type="primary", disabled=not files):
            prog = st.progress(0.0, text="Индексирование…")
            for n, f in enumerate(files, start=1):
                try:
                    r = store.add_file(f.name, f.getvalue(), ss.chunk_size, ss.chunk_overlap)
                    ctx.flash("success" if r.status == "indexed" else "info",
                              f"{f.name}: " + (f"проиндексировано фрагментов: {r.n_chunks}." if r.status == "indexed"
                                               else "такой документ уже есть — пропущен (без дубликатов)."))
                except (DocumentError, ProviderError, ValueError) as exc:
                    ctx.flash("error", str(exc))
                prog.progress(n / len(files), text=f"Готово {n}/{len(files)}")
            ss.form_nonce += 1
            ctx.commit()
    with demo_col:
        st.markdown("**Демонстрационные материалы**")
        st.caption("Тексты автора прототипа — не авторитетный медицинский источник. Есть документы на русском и один на английском.")
        if st.button("Загрузить демо-документы", width="stretch"):
            load_demo_documents(ctx)
            ctx.commit()

    docs = store.list_documents()
    st.markdown(f"##### Документы в базе ({len(docs)})")
    if not docs:
        st.info("Документов пока нет. Загрузите свои или демонстрационные.", icon="📚")
    for d in docs:
        c1, c2, c3 = st.columns([5, 2, 1])
        demo_tag = (" <span class='badge badge-demo'>ДЕМО</span>"
                    if d.title.upper().startswith(("DEMO", "ДЕМО")) else "")
        c1.markdown(f"**{esc(d.title)}**{demo_tag}<br><span class='small-muted'>{esc(d.filename)} · "
                    f"страниц: {d.pages or '–'} · sha256 {d.doc_hash[:12]}…</span>", unsafe_allow_html=True)
        c2.markdown(f"<span class='small-muted'>фрагментов: {d.n_chunks}<br>размер {d.chunk_size} / перекрытие {d.chunk_overlap}</span>",
                    unsafe_allow_html=True)
        if c3.button("Удалить", key=f"del_{d.doc_hash}"):
            store.delete_document(d.doc_hash)
            ctx.flash("success", f"Удалён: {d.filename}.")
            ctx.commit()
    pending = store.unindexed_files()
    r1, r2 = st.columns(2)
    if pending:
        r1.warning(f"Файлов, не проиндексированных для текущей модели эмбеддингов: {len(pending)} "
                   "(например, после переключения между демо- и рабочим режимом).")
    if store.stored_files() and r2.button("Переиндексировать всё с текущими настройками"):
        with st.spinner("Переиндексирование…"):
            try:
                res = store.reindex_all(ss.chunk_size, ss.chunk_overlap)
                ctx.flash("success", f"Переиндексировано документов: {len(res)}.")
            except (DocumentError, ProviderError) as exc:
                ctx.flash("error", str(exc))
        ctx.commit()
