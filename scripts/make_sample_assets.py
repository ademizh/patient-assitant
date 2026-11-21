"""Regenerate the bundled demonstration assets (deterministic output).

    python scripts/make_sample_assets.py

Creates:
  data/sample_documents/DEMO_preparing_for_a_consultation.pdf   English demo PDF (shows cross-language citations)
  data/sample_documents/DEMO_ru_tests_glossary.pdf               Russian demo glossary PDF (2 pages, page citations)
  data/sample_patient/FICTIONAL_lab_results.pdf                  fictional test results, text-based PDF
  data/sample_patient/FICTIONAL_lab_results_scan.pdf             the same as an image-only "scan" (no text layer)
  data/sample_patient/*.wav + *.transcript.txt                   Russian speech clips (synthesised with espeak-ng)

The speech clips need the optional package `espeakng-loader` (pip install espeakng-loader), which bundles
the espeak-ng engine with Russian voice data. The voice is robotic but it is real Russian speech, so it can
be sent to the live transcription API. In demo mode the clip's transcript is returned (matched by SHA-256).
All people, laboratories and documents in these assets are FICTIONAL.
"""

from __future__ import annotations

import ctypes
import io
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fpdf import FPDF  # noqa: E402
from fpdf.enums import XPos, YPos  # noqa: E402

FONTS = ROOT / "assets" / "fonts"
DOCS = ROOT / "data" / "sample_documents"
PATIENT = ROOT / "data" / "sample_patient"
FIXED_DATE = datetime(2026, 1, 1, tzinfo=timezone.utc)

EN_PAGES = [
    (
        "DEMO material: Preparing for a consultation",
        [
            "DEMONSTRATION MATERIAL ONLY. Written for the prototype demo by the project author. "
            "Not an authoritative medical reference, not clinically reviewed, and it cites no publications. "
            "Replace it with reviewed medical documents before any real use.",
            "Before the appointment",
            "Write down the main problem you want to discuss and when it started. If you are unsure of an "
            "exact date, an approximate time such as 'about two weeks ago' is still useful. Note whether the "
            "problem has changed over time, for example becoming more frequent or more intense.",
            "Medication list",
            "Prepare a list of every medicine you take, including prescription medicines, over-the-counter "
            "painkillers, vitamins, supplements and herbal products. For each one, write the name, the dose, "
            "how often you take it, and since when. Mention any medicine you recently stopped.",
        ],
    ),
    (
        "DEMO material: During the consultation",
        [
            "Allergies and past reactions",
            "Tell the clinician about any allergy or past reaction to a medicine, food or other substance, "
            "and describe what happened, for example a rash or swelling. Mention it even if you are not sure "
            "it was a true allergy, so that it can be clarified.",
            "Questions to ask",
            "It can help to prepare two or three questions in advance, for example what the next steps are, "
            "what to watch for, and when to come back. Writing down the answers during the visit makes them "
            "easier to remember.",
            "Bringing someone with you",
            "Some people find it helpful to bring a relative or friend to help remember information, if they "
            "are comfortable sharing their health information with that person.",
        ],
    ),
]

RU_GLOSSARY_PAGES = [
    (
        "ДЕМО-материал: словарь терминов и анализов (часть 1)",
        [
            "ТОЛЬКО ДЛЯ ДЕМОНСТРАЦИИ. Текст написан автором прототипа для показа поиска и ссылок на источники. "
            "Это не авторитетный медицинский источник, он не проверен клиницистом и не ссылается на публикации. "
            "Перед реальным использованием замените его проверенными документами.",
            "Референсный интервал",
            "Референсный интервал — это диапазон значений, который лаборатория указывает в бланке для сравнения. "
            "Он зависит от лаборатории, метода исследования, возраста и пола, поэтому в разных бланках может "
            "отличаться. Значение вне интервала само по себе не означает болезнь, а значение внутри интервала "
            "не исключает её. Оценку результата в вашей ситуации даёт врач.",
            "Общий анализ крови",
            "Общий анализ крови — лабораторное исследование, в котором определяют количество эритроцитов, "
            "лейкоцитов и тромбоцитов, уровень гемоглобина и другие показатели. Результаты врач оценивает вместе "
            "с жалобами и осмотром.",
            "Гемоглобин",
            "Гемоглобин — белок в эритроцитах, который переносит кислород. В бланках его обычно указывают в граммах "
            "на литр (г/л).",
            "Ферритин",
            "Ферритин — белок, в котором организм запасает железо. Анализ на ферритин используют, чтобы оценить "
            "запасы железа в организме. На уровень ферритина может влиять воспаление, поэтому результат "
            "оценивают вместе с другими показателями.",
        ],
    ),
    (
        "ДЕМО-материал: словарь терминов и анализов (часть 2)",
        [
            "СОЭ",
            "СОЭ — скорость оседания эритроцитов. Это неспецифический показатель: он может меняться при разных "
            "состояниях, поэтому сам по себе не указывает на конкретную причину. Обычно его указывают в миллиметрах "
            "в час (мм/ч).",
            "С-реактивный белок",
            "С-реактивный белок (СРБ) — белок, уровень которого в крови может меняться при воспалении. Это "
            "неспецифический показатель, его оценивают вместе с другими данными.",
            "МРТ",
            "МРТ (магнитно-резонансная томография) — метод исследования, при котором изображения органов получают "
            "с помощью сильного магнитного поля и радиоволн, без рентгеновского излучения. Перед МРТ спрашивают о "
            "металлических имплантах и кардиостимуляторе.",
            "Мигрень",
            "Мигрень — название одного из видов головной боли. Приступы часто бывают односторонними и пульсирующими, "
            "могут сопровождаться тошнотой и чувствительностью к свету и звуку. Диагноз мигрени ставит врач после "
            "расспроса и осмотра.",
            "Дневник головной боли",
            "Дневник головной боли — записи о каждом приступе: дата, время начала, длительность, сила боли по шкале "
            "от 0 до 10, возможные провоцирующие факторы и принятые лекарства с дозой. Такие записи помогают врачу "
            "увидеть, как меняются приступы.",
        ],
    ),
]

LAB_LINES = [
    "ВЫМЫШЛЕННЫЙ ПРИМЕР — бланк для демонстрации прототипа.",
    "Пациент и лаборатория не существуют.",
    "",
    "Демо-лаборатория «Пример»",
    "Дата взятия: 03.10.2026",
    "",
    "Показатель / Результат / Ед. изм. / Референсный интервал",
    "Гемоглобин (HGB) 128 г/л 120–140",
    "Лейкоциты (WBC) 6,4 ×10^9/л 4,0–9,0",
    "Тромбоциты (PLT) 250 ×10^9/л 180–320",
    "СОЭ 12 мм/ч 2–15",
    "Ферритин 9 нг/мл ↓",
    "С-реактивный белок <1 мг/л до 5",
    "",
    "Комментарий лаборатории: интерпретацию результатов проводит лечащий врач.",
]

AUDIO_CLIPS = {
    "fictional_severity_answer_ru": "Примерно шесть из десяти. Вечером иногда сильнее.",
    "fictional_consult_question_ru": "Что такое ферритин и зачем его проверяют?",
}


def _font_pdf() -> FPDF:
    pdf = FPDF(format="A4")
    pdf.add_font("DejaVu", "", str(FONTS / "DejaVuSans.ttf"))
    pdf.add_font("DejaVu", "B", str(FONTS / "DejaVuSans-Bold.ttf"))
    pdf.set_creation_date(FIXED_DATE)
    pdf.set_margins(20, 20, 20)
    return pdf


def make_doc_pdf(path: Path, title: str, author: str, pages, banner_prefix: str) -> None:
    pdf = _font_pdf()
    pdf.set_title(title)
    pdf.set_author(author)
    for page_title, blocks in pages:
        pdf.add_page()
        pdf.set_font("DejaVu", "B", 15)
        pdf.multi_cell(0, 8, page_title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(3)
        for b in blocks:
            is_heading = len(b) < 40
            pdf.set_font("DejaVu", "B" if is_heading else "", 12 if is_heading else 10.5)
            if b.startswith(banner_prefix):
                pdf.set_fill_color(254, 243, 199)
                pdf.multi_cell(0, 5.5, b, new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
            else:
                pdf.multi_cell(0, 6 if is_heading else 5.5, b, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(2)
    path.write_bytes(bytes(pdf.output()))


def make_lab_pdf(path: Path) -> None:
    pdf = _font_pdf()
    pdf.set_title("ВЫМЫШЛЕННЫЙ бланк анализов (демо)")
    pdf.set_author("Прототип (вымышленные данные)")
    pdf.add_page()
    for line in LAB_LINES:
        if not line:
            pdf.ln(3)
            continue
        bold = line.startswith(("ВЫМЫШЛЕННЫЙ", "Демо-лаборатория", "Показатель"))
        pdf.set_font("DejaVu", "B" if bold else "", 11)
        pdf.cell(0, 7, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    path.write_bytes(bytes(pdf.output()))


def make_scanned_lab_pdf(path: Path) -> None:
    """The same form rendered as a picture - a PDF without a text layer (like a phone scan)."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (1240, 1754), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(str(FONTS / "DejaVuSans.ttf"), 30)
    y = 120
    for line in LAB_LINES:
        draw.text((110, y), line, fill=(20, 20, 20), font=font)
        y += 52
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    pdf = FPDF(format="A4")
    pdf.set_creation_date(FIXED_DATE)
    pdf.add_page()
    pdf.image(buf, x=0, y=0, w=210, h=297)
    path.write_bytes(bytes(pdf.output()))


def synthesize_ru(text: str, path: Path) -> bool:
    """Russian speech via the espeak-ng engine bundled in `espeakng-loader`. Returns False if unavailable."""
    try:
        import espeakng_loader as loader
    except ImportError:
        return False
    lib = ctypes.cdll.LoadLibrary(loader.get_library_path())
    rate = lib.espeak_Initialize(2, 0, loader.get_data_path().encode(), 0)  # AUDIO_OUTPUT_SYNCHRONOUS
    buf = bytearray()
    callback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_void_p)

    def _cb(wav, n, _events):
        if wav and n > 0:
            buf.extend(ctypes.string_at(wav, n * 2))
        return 0

    cb = callback_type(_cb)
    lib.espeak_SetSynthCallback(cb)
    if lib.espeak_SetVoiceByName(b"ru") != 0:
        return False
    lib.espeak_SetParameter(1, 145, 0)  # words per minute
    data = text.encode("utf-8")
    lib.espeak_Synth(data, len(data) + 1, 0, 0, 0, 1, None, None)  # espeakCHARS_UTF8
    lib.espeak_Synchronize()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(buf))
    return len(buf) > 0


if __name__ == "__main__":
    make_doc_pdf(DOCS / "DEMO_preparing_for_a_consultation.pdf", "DEMO material: Preparing for a consultation",
                 "Prototype demo (not an authoritative source)", EN_PAGES, "DEMONSTRATION")
    make_doc_pdf(DOCS / "DEMO_ru_tests_glossary.pdf", "ДЕМО-материал: словарь терминов и анализов",
                 "Демонстрация прототипа (не авторитетный источник)", RU_GLOSSARY_PAGES, "ТОЛЬКО ДЛЯ")
    make_lab_pdf(PATIENT / "FICTIONAL_lab_results.pdf")
    make_scanned_lab_pdf(PATIENT / "FICTIONAL_lab_results_scan.pdf")
    for stem, text in AUDIO_CLIPS.items():
        wav = PATIENT / f"{stem}.wav"
        if synthesize_ru(text, wav):
            (PATIENT / f"{stem}.transcript.txt").write_text(text + "\n", encoding="utf-8")
            print(f"wrote {wav.relative_to(ROOT)}")
        else:
            print(f"skipped {stem}.wav (pip install espeakng-loader to synthesise Russian speech)")
    print("done")
