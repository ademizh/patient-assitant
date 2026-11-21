"""Shared UI helpers: context object, origin badges, audio input, source cards."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

import streamlit as st

from intake_assistant.config import Settings
from intake_assistant.providers import SUPPORTED_AUDIO_EXTENSIONS, ProviderError
from intake_assistant.records import ORIGIN_LABELS, AppointmentRecord, RecordStore, SourceRef

CSS = """
<style>
/* ======================================================================================
   sky-blue accents · very light blue page · white rounded cards · serif headings · pill buttons
   ====================================================================================== */
:root {
  --sky: #5BBFF5;          /* decorative blue (blob, underline accents) */
  --blue: #1E86D4;         /* interactive blue (buttons, links)          */
  --blue-dark: #11578F;
  --blue-soft: #EAF4FD;
  --page: #F6F9FE;
  --card: #FFFFFF;
  --ink: #1F2A37;
  --muted: #6B7A8C;
  --line: #E3ECF7;
  --shadow: 0 6px 24px rgba(30, 80, 140, 0.07), 0 1px 3px rgba(30, 80, 140, 0.05);
}
[data-testid="stMainBlockContainer"], .block-container {padding-top: 1.2rem; max-width: 1240px;}
[data-testid="stHeader"] {background: transparent;}

/* ---- Hero ---------------------------------------------------------------------------- */
.hero {position: relative; overflow: hidden; background: var(--card); border-radius: 22px;
       box-shadow: var(--shadow); padding: 14px 34px 20px 34px; margin-bottom: 14px;}
.hero-blob {position: absolute; right: -30px; top: -70px; width: 430px; height: 310px; z-index: 0;}
.hero-nav {position: relative; z-index: 1; display: flex; justify-content: space-between; align-items: center;
           font-size: 0.92rem; margin-bottom: 10px;}
.hero-brand {font-weight: 700; color: var(--ink); letter-spacing: 0.01em;}
.hero-brand b {color: var(--blue);}
.hero-links span {color: #FFFFFF; margin-left: 26px; font-weight: 600; text-shadow: 0 1px 2px rgba(17,87,143,.25);}
.hero-body {position: relative; z-index: 1; max-width: 560px;}
.eyebrow {color: var(--blue); font-weight: 700; font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase;
          margin-bottom: 6px;}
.hero-title {font-family: "Playfair Display", serif; font-weight: 700; font-size: 2rem; line-height: 1.15;
             color: var(--ink); margin: 0 0 8px 0;}
.hero-sub {color: var(--muted); font-size: .95rem; line-height: 1.55; margin: 0 0 10px 0;}
.hero-chips {position: relative; z-index: 1;}
.hero-icon {position: absolute; right: 110px; top: 58px; z-index: 1; width: 104px; height: 104px; border-radius: 50%;
            background: rgba(255,255,255,0.92); box-shadow: 0 10px 30px rgba(17,87,143,0.18);
            display: flex; align-items: center; justify-content: center;}
@media (max-width: 1100px) {.hero-icon, .hero-links {display: none;} .hero-blob {opacity: .35;}}

/* ---- Chips / badges ------------------------------------------------------------------- */
.badge {display:inline-block;padding:3px 12px;border-radius:999px;font-size:0.76rem;font-weight:700;
        margin:0 6px 6px 0;border:1px solid transparent}
.badge-demo {background:#FFF7E6;color:#8A5300;border-color:#FBD99A}
.badge-live {background:#E8F8EE;color:#166534;border-color:#A7E3BC}
.badge-proto {background:var(--blue-soft);color:var(--blue-dark);border-color:#C9E2F8}

/* ---- Stage cards (1 · 2 · 3) ------------------------------------------------------------ */
[class*="st-key-stage_card_"] {background: var(--card); border-radius: 18px; box-shadow: var(--shadow);
        padding: 14px 18px 12px 18px; border: 1px solid transparent; height: 100%;}
[class*="st-key-stage_card_"].active, .stage-active {border-color: var(--sky);}
.stage-num {display:inline-flex;align-items:center;justify-content:center;width:30px;height:30px;border-radius:50%;
            background:var(--blue-soft);color:var(--blue);font-weight:700;margin-right:10px;font-size:.95rem}
.stage-title {font-family:"Playfair Display",serif;font-weight:700;font-size:1.25rem;color:var(--ink)}
.stage-desc {color:var(--muted);font-size:.86rem;line-height:1.45;margin:6px 0 8px 0;min-height:2.9em}
.stage-status {font-size:.82rem;color:var(--blue-dark);background:var(--blue-soft);border-radius:999px;
               padding:3px 12px;display:inline-block;margin-bottom:6px}
.stage-status.done {background:#E8F8EE;color:#166534}

/* ---- Section headings: serif + short blue underline (like "Projects" in the mockup) ------- */
[data-testid="stMainBlockContainer"] h5 {font-family: "Playfair Display", serif; font-weight: 700; font-size: 1.3rem;
       margin-top: .6rem;}
[data-testid="stMainBlockContainer"] h5::after {content: ""; display: block; width: 46px; height: 3px;
       background: var(--sky); border-radius: 3px; margin-top: 7px;}

/* ---- Tabs as pills ---------------------------------------------------------------------- */
.stTabs [role="tablist"] {gap: 6px; flex-wrap: wrap; border-bottom: none !important;}
.stTabs [data-testid="stTab"] {border-radius: 999px; padding: 6px 13px !important; background: transparent;
       border: 1px solid transparent; transition: background .15s;}
.stTabs [data-testid="stTab"]:hover {background: var(--blue-soft);}
.stTabs [data-testid="stTab"][aria-selected="true"] {background: var(--blue-soft); border-color: #C9E2F8;}
.stTabs [data-testid="stTab"][aria-selected="true"] p {color: var(--blue-dark) !important; font-weight: 700;}
.stTabs [role="tablist"]::after, .stTabs [role="tablist"]::before, .stTabs [data-testid="stTab"]::after,
.stTabs .react-aria-SelectionIndicator {display: none !important; content: none !important;}
.st-key-main_tab > div > [role="tablist"] {background: var(--card); padding: 7px; border-radius: 999px;
       box-shadow: var(--shadow); display: inline-flex; margin-bottom: 8px;}
.st-key-main_tab > div > [role="tablist"] [data-testid="stTab"] {padding: 9px 20px !important;}
.st-key-main_tab > div > [role="tablist"] [data-testid="stTab"][aria-selected="true"] {background: var(--blue); border-color: var(--blue);}
.st-key-main_tab > div > [role="tablist"] [data-testid="stTab"][aria-selected="true"] p {color: #FFFFFF !important;}

/* ---- Cards: expanders, bordered containers, forms, chat ---------------------------------- */
[data-testid="stExpander"] details {background: var(--card); border: 1px solid var(--line) !important;
       border-radius: 16px !important; box-shadow: 0 2px 10px rgba(30,80,140,.04);}
[data-testid="stExpander"] summary:hover {color: var(--blue);}
[class*="st-key-card_"] {background: var(--card); border: 1px solid var(--line); border-radius: 16px;
       box-shadow: 0 2px 10px rgba(30,80,140,.04); padding: 12px 14px;}
[data-testid="stForm"] {background: var(--card); border-color: var(--line) !important; border-radius: 16px;}
[data-testid="stChatMessage"] {background: var(--card); border-radius: 16px; border: 1px solid var(--line);
       box-shadow: 0 2px 8px rgba(30,80,140,.03); padding: 12px 16px;}
[data-testid="stAlert"] > div {border-radius: 14px;}

/* ---- Buttons: pills. Primary = filled blue; secondary = outlined (like "LinkedIn") -------- */
[data-testid^="stBaseButton-primary"] {box-shadow: 0 4px 12px rgba(30,134,212,.25); font-weight: 700;
       padding-left: 1.1rem; padding-right: 1.1rem;}
[data-testid^="stBaseButton-secondary"], [data-testid="stPopoverButton"] {background: var(--card);
       border: 1.5px solid #C9D6E4; color: var(--ink); font-weight: 600;}
[data-testid^="stBaseButton-secondary"]:hover, [data-testid="stPopoverButton"]:hover {border-color: var(--blue); color: var(--blue);}

/* ---- Sidebar ----------------------------------------------------------------------------- */
[data-testid="stSidebar"] {border-right: 1px solid var(--line);}
[data-testid="stSidebar"] h3 {font-family: "Playfair Display", serif; font-size: 1.15rem;}
.side-brand {font-weight: 700; font-size: 1.05rem; margin: -6px 0 12px 0; color: var(--ink);}
.side-brand b {color: var(--blue);}

/* ---- Small building blocks -------------------------------------------------------------- */
.entry-id {font-family:ui-monospace,monospace;font-size:0.75rem;background:var(--blue-soft);color:var(--blue-dark);
           padding:1px 7px;border-radius:7px;margin-right:6px;font-weight:600}
.small-muted {font-size:0.82rem;color:var(--muted)}
.excerpt {font-size:0.9rem;line-height:1.55;color:#3A4A5C;border-left:3px solid #C9E2F8;padding-left:12px}
.src-card {border:1px solid var(--line);border-radius:12px;padding:8px 12px;margin:4px 0 6px 0;background:#F9FBFE}
.src-card .meta {font-size:0.78rem;color:var(--muted)}
.stage-hint {position:relative;font-size:0.93rem;line-height:1.55;color:#2D3B4B;background:var(--card);
             border-radius:16px;padding:12px 16px 12px 46px;margin-bottom:12px;box-shadow:0 2px 10px rgba(30,80,140,.04);
             border-left:4px solid var(--sky)}
.stage-hint::before {content:"💡";position:absolute;left:16px;top:11px;font-size:1.05rem}
.notice {font-size:.86rem;color:#6B4A00;background:#FFF9EC;border:1px solid #FBE3B3;border-radius:999px;
         padding:6px 16px;margin:-4px 0 14px 0;display:inline-block}

/* ---- Origin colours: who said it (patient, document, doctor, AI, source, rules) --------- */
.o-tag {display:inline-block;font-size:0.72rem;font-weight:700;padding:2px 9px;border-radius:999px;margin-bottom:4px}
.o-box {border-radius:12px;padding:9px 14px;margin:2px 0 8px 0;line-height:1.55}
.o-patient .o-tag, .o-patient.o-box {background:#E8F4FD;color:#0C4A6E}
.o-patient_document .o-tag, .o-patient_document.o-box {background:#FFF1E6;color:#7C2D12}
.o-doctor .o-tag, .o-doctor.o-box {background:#E8F8EE;color:#14532D}
.o-ai .o-tag, .o-ai.o-box {background:#F0F1FE;color:#1E1B4B}
.o-knowledge_base .o-tag, .o-knowledge_base.o-box {background:#F2F5F9;color:#1E293B}
.o-rules .o-tag, .o-rules.o-box {background:#FFF7E6;color:#78350F}
.o-box.o-patient {border-left:4px solid #5BBFF5}
.o-box.o-patient_document {border-left:4px solid #F59E5B}
.o-box.o-doctor {border-left:4px solid #34C27A}
.o-box.o-ai {border-left:4px solid #818CF8}
.o-box.o-knowledge_base {border-left:4px solid #A8B4C4}
.o-box.o-rules {border-left:4px solid #E0A33A}
.target-date {background:#F5F3FF;color:#5B21B6;border:1px dashed #A78BFA;border-radius:999px;padding:2px 10px;font-size:0.82rem}
.doctor-date {background:#E8F8EE;color:#14532D;border:1px solid #A7E3BC;border-radius:999px;padding:2px 10px;font-size:0.82rem}
.no-date {background:#F2F5F9;color:#475569;border:1px solid #D5DEE8;border-radius:999px;padding:2px 10px;font-size:0.82rem}

/* ---- Footer ------------------------------------------------------------------------------ */
.footer {text-align:center;color:var(--muted);font-size:.82rem;margin:34px 0 10px 0;line-height:1.7}
.footer .brand {font-weight:700;color:var(--ink)}
.footer .brand b {color:var(--blue)}

/* Russian labels for Streamlit's built-in file-uploader texts (Streamlit has no UI locale setting). */
[data-testid="stFileUploaderDropzone"] {border-radius:16px;background:#F9FBFE}
[data-testid="stFileUploaderDropzone"] button [data-testid="stMarkdownContainer"] p {font-size:0}
[data-testid="stFileUploaderDropzone"] button [data-testid="stMarkdownContainer"] p::after {content:"Выбрать файл";font-size:0.875rem}
[data-testid="stFileUploaderDropzoneInstructions"] span {font-size:0}
[data-testid="stFileUploaderDropzoneInstructions"] span::after {content:"Перетащите файл сюда · не более 50 МБ";font-size:0.8rem}
</style>
"""


@dataclass
class Ctx:
    settings: Settings
    provider: object | None
    store: object | None
    records: RecordStore
    pathway: dict | None
    today: date
    flash: Callable[[str, str], None]
    commit: Callable[[], None]
    save: Callable[[], None]
    intake_flow: object | None = None  # LangGraph intake workflow (intake_assistant/graphs/intake_graph.py)
    notes: list[str] = field(default_factory=list)

    @property
    def record(self) -> AppointmentRecord:
        return st.session_state.record


BLOB_SVG = (
    '<svg class="hero-blob" viewBox="0 0 470 340" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<path fill="#5BBFF5" d="M118,-4 C178,-6 470,-8 486,-6 L486,250 C470,300 420,338 352,334 C276,330 238,286 176,284 '
    'C104,282 52,300 22,262 C-8,224 34,176 64,140 C96,100 72,58 86,26 C94,8 104,-2 118,-4 Z"/>'
    '<path fill="#8FD4FA" opacity="0.55" d="M300,40 C350,30 420,60 430,110 C440,160 400,190 350,184 C300,178 270,150 '
    '268,110 C266,72 270,46 300,40 Z"/></svg>'
)
PULSE_SVG = (
    '<svg width="62" height="62" viewBox="0 0 24 24" fill="none" stroke="#1E86D4" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M19.5 12.6 12 20l-7.5-7.4A4.6 4.6 0 0 1 12 6.6a4.6 4.6 0 0 1 7.5 6z"/>'
    '<path d="M3.5 12h4l1.5-3 3 6 1.5-3h7" stroke="#5BBFF5"/></svg>'
)


def hero(badges_html: str) -> None:
    """Top banner in the style of the Medical.kz mockup: brand line, serif title, soft blue blob."""
    st.markdown(
        f"""<div class="hero">{BLOB_SVG}<div class="hero-icon">{PULSE_SVG}</div>
<div class="hero-nav"><span class="hero-brand">Medical<b>.kz</b></span>
<span class="hero-links"><span>До приёма</span><span>Во время</span><span>После</span></span></div>
<div class="hero-body"><div class="eyebrow">Помощник пациента · прототип</div>
<div class="hero-title">Подготовьтесь к приёму и ничего не упустите</div>
<p class="hero-sub">Расскажите о симптомах, разберитесь в терминах и анализах по источникам, получите понятный
список дел после приёма. <b>Помощник не ставит диагноз и не назначает лечение.</b></p></div>
<div class="hero-chips">{badges_html}</div></div>""",
        unsafe_allow_html=True,
    )


def footer() -> None:
    st.markdown(
        "<div class='footer'><span class='brand'>Medical<b>.kz</b></span> · помощник пациента<br>"
        "Прототип для демонстрации · не медицинское изделие · не проверен клинически · "
        "всегда сверяйте информацию с врачом</div>",
        unsafe_allow_html=True,
    )


def esc(text: str) -> str:
    return html.escape(text or "").replace("\n", "<br>")


def origin_box(origin: str, text: str, extra_label: str = "", as_html: bool = False) -> None:
    body = text if as_html else esc(text)
    label = ORIGIN_LABELS[origin] + (f" · {extra_label}" if extra_label else "")
    st.markdown(
        f"<div class='o-{origin}'><span class='o-tag'>{html.escape(label)}</span></div>"
        f"<div class='o-box o-{origin}'>{body}</div>",
        unsafe_allow_html=True,
    )


def origin_tag(origin: str) -> str:
    return f"<span class='o-{origin}'><span class='o-tag'>{html.escape(ORIGIN_LABELS[origin])}</span></span>"


def reflow(text: str) -> str:
    """Display-only: join hard-wrapped lines, keep paragraph breaks, escape HTML."""
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    return "<br><br>".join(html.escape(p) for p in paras)


def render_source(s: SourceRef | dict, expanded: bool = False, key: str = "") -> None:
    if isinstance(s, dict):  # retrieved-but-not-cited chunk
        s = SourceRef(chunk_id=s["chunk_id"], title=s["title"], filename=s["filename"], page=s.get("page"),
                      heading=s.get("heading"), excerpt=s["text"], language=s.get("language", "ru"), score=s.get("score"))
    loc = f"стр. {s.page}" if s.page else (f"раздел «{html.escape(s.heading)}»" if s.heading else "без номера страницы")
    lang = "" if s.language == "ru" else f" · язык оригинала: <b>{html.escape(s.language)}</b>"
    score = f" · релевантность {s.score:.2f}" if s.score is not None else ""
    st.markdown(
        f"<div class='src-card'>{origin_tag('knowledge_base')} <b>{html.escape(s.title)}</b><div class='meta'>"
        f"{html.escape(s.filename)} · {loc} · фрагмент <code>{s.chunk_id}</code>{score}{lang}</div></div>",
        unsafe_allow_html=True,
    )
    if s.quote:
        st.markdown(f"<div class='excerpt'>«{esc(s.quote)}»</div>", unsafe_allow_html=True)
        if s.language != "ru":
            if s.translation_ru:
                warn = "" if s.translation_numbers_ok is not False else " ⚠️ числа в переводе не совпадают с оригиналом — опирайтесь на оригинал"
                st.markdown(
                    f"<div class='excerpt'><span class='o-ai'><span class='o-tag'>Перевод ИИ</span></span> "
                    f"«{esc(s.translation_ru)}»{warn}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption("Перевод недоступен (демо-режим) — показан оригинал.")
    with st.expander("Полный фрагмент источника", expanded=expanded):
        st.markdown(f"<div class='excerpt'>{reflow(s.excerpt)}</div>", unsafe_allow_html=True)


def audio_block(ctx: Ctx, prefix: str, on_submit: Callable[[str, str, str], None], label: str = "Запишите ответ") -> None:
    """Record or upload audio -> transcription shown for correction -> on_submit(text, original, filename)."""
    ss = st.session_state
    pend_key = f"pending_audio_{prefix}"
    pending = ss.get(pend_key)
    nonce = ss.form_nonce
    if pending is None:
        rec = st.audio_input(label, key=f"rec_{prefix}_{nonce}")
        up = st.file_uploader(
            "…или загрузите аудиофайл", type=sorted(SUPPORTED_AUDIO_EXTENSIONS), key=f"aup_{prefix}_{nonce}",
            help=f"До {ctx.settings.max_audio_mb} МБ. Форматы: {', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}. "
                 "Речь распознаётся на русском языке.",
        )
        audio = rec or up
        if st.button("Расшифровать", key=f"tr_{prefix}", disabled=audio is None or ctx.provider is None):
            data = audio.getvalue()
            name = getattr(audio, "name", None) or "recording.wav"
            if "." not in name:
                name += ".wav"
            try:
                with st.spinner("Расшифровываю…"):
                    result = ctx.provider.transcribe(data, name)
                ss[pend_key] = {"raw": result.text, "filename": name, "note": result.note,
                                "placeholder": result.is_placeholder}
                st.rerun()
            except ProviderError as exc:
                st.error(str(exc))
    else:
        st.markdown(f"**Расшифровка** `{pending['filename']}` — проверьте и исправьте текст перед отправкой.")
        if pending.get("note"):
            (st.warning if pending["placeholder"] else st.info)(pending["note"])
        corrected = st.text_area("Расшифровка (можно исправить)", value=pending["raw"], height=110,
                                 key=f"pending_text_{prefix}_{nonce}")
        c1, c2 = st.columns(2)
        if c1.button("Отправить расшифровку", type="primary", key=f"sub_{prefix}"):
            if pending["placeholder"] and corrected.strip() == pending["raw"].strip():
                st.warning("Замените текст-заглушку демо-режима тем, что вы сказали.")
            elif not corrected.strip():
                st.warning("Расшифровка пуста.")
            else:
                ss[pend_key] = None
                ss.form_nonce += 1
                on_submit(corrected.strip(), pending["raw"], pending["filename"])
        if c2.button("Отменить", key=f"disc_{prefix}"):
            ss[pend_key] = None
            ss.form_nonce += 1
            st.rerun()


def fmt_date(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        return date.fromisoformat(iso).strftime("%d.%m.%Y")
    except ValueError:
        return iso


def fmt_ts(iso: str | None) -> str:
    if not iso:
        return "—"
    return iso.replace("T", " ").replace("+00:00", " UTC")
