"""Offline demo-mode implementations (no network, no API key).

These are deterministic heuristics so the full workflow can be shown without an API key.
They are clearly labelled in the UI and in every generated report as DEMO output.
They are NOT a substitute for the live model and are not medically meaningful.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

from .timeline import find_time_reference
from .schemas import Extraction, ExtractedContradiction, ExtractedFact, IntakeEntry, SLOT_LABELS
from .text_utils import (
    STOPWORDS,
    content_tokens,
    is_denial,
    is_hedged,
    is_unknown,
    normalize,
    split_sentences,
    tokens,
    _stem,
)

DEMO_GENERATOR = "демо-эвристики (офлайн, без ИИ-модели)"

# Order matters: the first matching category wins for a free-text sentence.
CATEGORY_PATTERNS: list[tuple[str, list[str]]] = [
    ("allergies", [r"allerg", r"аллерг"]),
    ("medications", [
        r"\b\d+\s*(mg|мг|мкг|мл)\b",
        r"\b(ibuprofen|paracetamol|acetaminophen|aspirin|pill\w*|tablet\w*|medication\w*|medicine\w*|supplement\w*|vitamin\w*)\b",
        r"(ибупрофен|парацетамол|аспирин|нурофен|цитрамон|но-шп|анальгин|таблет\w*|лекарств\w*|препарат\w*|"
        r"витамин\w*|добавк\w*|капл\w*|ингалятор\w*|антибиотик\w*|обезболива\w*)",
        r"(принимаю|пью\s+(таблет|лекарств)|выпила|выпил|принимала|принимал)",
    ]),
    ("medical_history", [
        r"\bdiagnosed\b", r"\bhistory of\b", r"\b(surgery|operation|hospitali[sz]ed|chronic|asthma|diabetes)\b",
        r"(операци\w*|госпитализ\w*|лежал\w*\s+в\s+больниц\w*|хроническ\w*|астм\w*|диабет\w*|гипертони\w*|"
        r"щитовидн\w*|ставили\s+диагноз|на\s+учёте|на\s+учете)",
    ]),
    ("progression", [
        r"(стал[аио]?\s+(чаще|реже|сильнее|слабее|хуже|лучше|дольше|короче))", r"(усилил\w*|ослаб\w*|участил\w*|нарастает|прогрессир\w*)",
        r"(не\s+меня\w*|без\s+изменений|так\s+же,?\s+как)", r"\b(getting\s+(worse|better)|more\s+often|more\s+frequent)\b",
    ]),
    ("severity", [r"\b\d{1,2}\s*(/|out of|из)\s*10\b", r"\b(mild|moderate|severe|unbearable|worst)\b",
                  r"(слаб\w*\s+бол|умеренн\w*|сильн\w*\s+бол|невыносим\w*|терпим\w*)",
                  r"(один|два|три|четыре|пять|шесть|семь|восемь|девять|десять)\s+из\s+десяти"]),
    ("associated_symptoms", [
        r"\b(nause\w*|vomit\w*|fever\w*|dizz\w*|cough\w*|rash\w*|fatigue\w*|vision|light|sound|numb\w*)\b",
        r"(тошн\w*|рвот\w*|температур\w*|головокруж\w*|кашл\w*|сыпь|слабост\w*|устал\w*|зрени\w*|"
        r"свет\w*|звук\w*|онемен\w*|покалыван\w*|потлив\w*|озноб\w*|одышк\w*|сердцебиен\w*|мутит)",
    ]),
    ("modifying_factors", [r"\b(worse|better|helps?|relie\w*|eases?|trigger\w*)\b",
                           r"(хуже|лучше|помогает|облегча\w*|усилива\w*|провоцир\w*|проходит\s+после|снимает)"]),
    ("onset", [r"\b(started|began|since|ago|yesterday)\b", r"(начал\w*|появил\w*|назад|вчера|с\s+тех\s+пор)"]),
    ("relevant_events", [r"\b(accident|fell|injur\w*|travel\w*|stress\w*|flu|infection|new job)\b",
                         r"(травм\w*|упал\w*|поездк\w*|путешеств\w*|стресс\w*|грипп\w*|простуд\w*|инфекци\w*|"
                         r"нов\w+\s+работ\w*|переезд\w*|ОРВИ)"]),
    ("duration", [r"\b(constant\w*|comes and goes|on and off|lasts?|all day)\b",
                  r"(постоянн\w*|периодическ\w*|то\s+появля\w*|приступ\w*|длится|длятся|продолжа\w*|весь\s+день|каждый\s+(день|вечер))"]),
    ("location", [r"\b(forehead|temple\w*|behind (my|the) eyes?|chest|abdomen|stomach|neck|lower back)\b",
                  r"(лоб|лбу|висок|виске|висках|затыл\w*|за\s+глаз\w*|груд\w*|живот\w*|желуд\w*|ше[яеи]\b|поясниц\w*|"
                  r"справа|слева|с\s+обеих\s+сторон|правой\s+стороне|левой\s+стороне)"]),
]

STRONG_CATEGORIES = {"allergies", "medications", "medical_history"}
NEGATION_WORDS = {"no", "none", "not", "never", "nothing", "don't", "dont", "haven't", "didn't",
                  "allergy", "allergi", "allerg", "нет", "не", "никаких", "ничего", "аллерг", "аллерги"}


def _match_category(sentence: str) -> str | None:
    low = normalize(sentence).lower()
    for cat, pats in CATEGORY_PATTERNS:
        if any(re.search(p, low) for p in pats):
            return cat
    return None


def _time_ref(sentence: str) -> str | None:
    return find_time_reference(sentence)


def _fact(category: str, sentence: str, entry: IntakeEntry) -> ExtractedFact:
    return ExtractedFact(
        category=category,  # type: ignore[arg-type]
        statement=sentence.strip(),
        source_entry_ids=[entry.id],
        certainty="uncertain" if is_hedged(sentence) else "reported",
        negated=is_denial(sentence),
        time_reference=_time_ref(sentence),
    )


def demo_extract(entries: list[IntakeEntry]) -> Extraction:
    facts: list[ExtractedFact] = []
    have_main = False

    for entry in entries:
        sentences = split_sentences(entry.text) or [entry.text]
        if entry.question_slot == "patient_question":
            # Every sentence of the answer is one of the patient's own questions ("Нет вопросов" is skipped).
            for s in sentences:
                if not is_denial(s):
                    facts.append(_fact("patient_question", s, entry))
            continue
        if entry.question_slot and entry.question_slot != "main_concern":
            # First sentence answers the question that was asked; extra sentences are classified.
            facts.append(_fact(entry.question_slot, sentences[0], entry))
            have_main = have_main or entry.question_slot == "main_concern"
            for s in sentences[1:]:
                cat = _match_category(s)
                # Extra sentences stay with the question asked unless they clearly belong elsewhere.
                if cat not in STRONG_CATEGORIES:
                    cat = entry.question_slot
                facts.append(_fact(cat, s, entry))
            continue
        prev_cat: str | None = None
        for i, s in enumerate(sentences):
            if not have_main and i == 0:
                facts.append(_fact("main_concern", s, entry))
                have_main = True
                # The opening sentence may also carry a time reference -> record onset too.
                if _time_ref(s):
                    facts.append(_fact("onset", s, entry))
                prev_cat = "main_concern"
                continue
            cat = _match_category(s) or "other"
            if s.rstrip().endswith("?"):
                cat = "patient_question"
            # A follow-on sentence ("I got a rash from it") stays with a preceding allergy/medication/history.
            if prev_cat in STRONG_CATEGORIES and cat not in STRONG_CATEGORIES:
                cat = prev_cat
            facts.append(_fact(cat, s, entry))
            prev_cat = cat

    # Corrections ("Поправка: …"): facts of the corrected entry are replaced by the correcting entry's facts
    # of the same topic. The corrected entry itself stays in the record unchanged.
    corrected_by = {e.corrects_entry_id: e.id for e in entries if e.corrects_entry_id}
    if corrected_by:
        cats_of: dict[str, set[str]] = {}
        for f in facts:
            for eid in f.source_entry_ids:
                cats_of.setdefault(eid, set()).add(f.category)
        facts = [
            f for f in facts
            if not any(eid in corrected_by and f.category in cats_of.get(corrected_by[eid], set())
                       for eid in f.source_entry_ids)
        ]

    contradictions = _find_contradictions(facts)
    return Extraction(facts=facts, contradictions=contradictions, summary=_summary(facts))


def _sorted_ids(ids: list[str]) -> list[str]:
    return sorted(set(ids), key=lambda i: (len(i), i))


def _find_contradictions(facts: list[ExtractedFact]) -> list[ExtractedContradiction]:
    out: list[ExtractedContradiction] = []
    seen: set[tuple] = set()
    by_cat: dict[str, list[ExtractedFact]] = {}
    for f in facts:
        by_cat.setdefault(f.category, []).append(f)
    for cat, items in by_cat.items():
        if cat in ("main_concern", "other", "patient_question"):
            continue
        denials = [f for f in items if f.negated]
        positives = [f for f in items if not f.negated and not is_unknown(f.statement)]
        for d in denials:
            for p in positives:
                if set(d.source_entry_ids) == set(p.source_entry_ids):
                    continue
                # A blanket denial ("No allergies", "None") conflicts with any positive statement in a
                # list-type category; otherwise the two statements must share a topic word.
                d_topic = content_tokens(d.statement) - NEGATION_WORDS - {_stem(cat)}
                blanket = cat in STRONG_CATEGORIES and len(d_topic) <= 1
                key = (cat, tuple(_sorted_ids(d.source_entry_ids + p.source_entry_ids)))
                if key in seen:
                    continue
                if blanket or (d_topic & content_tokens(p.statement)):
                    seen.add(key)
                    out.append(
                        ExtractedContradiction(
                            description=(
                                f"{SLOT_LABELS[cat]}: «{d.statement}» ({', '.join(d.source_entry_ids)}) "
                                f"противоречит «{p.statement}» ({', '.join(p.source_entry_ids)})."
                            ),
                            source_entry_ids=_sorted_ids(d.source_entry_ids + p.source_entry_ids),
                        )
                    )
    return out


def _summary(facts: list[ExtractedFact]) -> str:
    def first(cat: str) -> ExtractedFact | None:
        return next((f for f in facts if f.category == cat and not f.negated), None)

    parts: list[str] = []
    mc = first("main_concern")
    if mc:
        parts.append(f"Основная жалоба со слов пациента: «{mc.statement.rstrip('.!')}».")
    for cat in ("onset", "progression", "severity", "location"):
        f = first(cat)
        if f:
            parts.append(f"{SLOT_LABELS[cat]}: «{f.statement.rstrip('.!')}».")
    n_neg = sum(1 for f in facts if f.negated)
    if n_neg:
        parts.append(f"Явных отрицаний: {n_neg}.")
    if not parts:
        return "Информация пока не собрана."
    return "[Демо-шаблон — собран из слов пациента, ИИ-модель не использовалась] " + " ".join(parts)


# ---- Demo embeddings: feature hashing over stemmed content words ---------------------

DEMO_EMBED_DIM = 512


def demo_embed(texts: list[str]) -> list[list[float]]:
    vectors = []
    for t in texts:
        vec = [0.0] * DEMO_EMBED_DIM
        feats = [_stem(w) for w in tokens(t) if w not in STOPWORDS and len(w) > 1]
        # unigrams + bigrams
        grams = feats + [f"{a}_{b}" for a, b in zip(feats, feats[1:])]
        for g in grams:
            h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
            idx = h % DEMO_EMBED_DIM
            sign = 1.0 if (h >> 9) & 1 else -1.0
            vec[idx] += sign * (1.0 if "_" not in g else 0.5)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vectors.append([v / norm for v in vec])
    return vectors


# ---- Demo transcription --------------------------------------------------------------

SAMPLE_AUDIO_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_patient"


def demo_transcribe(audio_bytes: bytes, filename: str | None) -> tuple[str, bool]:
    """Return (text, matched_sample).

    Only bundled sample clips have a pre-written transcript (matched by SHA-256).
    Any other audio returns an explicit placeholder that the patient must replace.
    """
    digest = hashlib.sha256(audio_bytes).hexdigest()
    for sidecar in SAMPLE_AUDIO_DIR.glob("*.transcript.txt"):
        audio = sidecar.with_name(sidecar.name.replace(".transcript.txt", ".wav"))
        if audio.exists() and hashlib.sha256(audio.read_bytes()).hexdigest() == digest:
            return sidecar.read_text(encoding="utf-8").strip(), True
    return (
        "[ДЕМО-РЕЖИМ — аудио НЕ расшифровано. Чтобы включить настоящую расшифровку, задайте OPENAI_API_KEY. "
        "Замените этот текст тем, что вы сказали.]",
        False,
    )


# ---- Demo extractive helpers ---------------------------------------------------------

# Share of the question's content words a sentence must contain to be quoted as evidence.
DEMO_MIN_SENTENCE_OVERLAP = 0.3


def _unwrap(text: str) -> str:
    """Join hard-wrapped lines into paragraphs; keep short unpunctuated lines (headings) separate."""
    paras: list[str] = []
    cur: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        is_heading = line.startswith("#") or (0 < len(line) < 60 and not line.endswith((".", "!", "?", ",", ";", ":")))
        if not line or is_heading:
            if cur:
                paras.append(" ".join(cur))
                cur = []
            if is_heading:
                paras.append(line)
            continue
        cur.append(line)
    if cur:
        paras.append(" ".join(cur))
    return "\n\n".join(paras)


def _evidence_sentences(chunk: dict) -> list[str]:
    out = []
    for s in split_sentences(_unwrap(chunk["text"])):
        # Only full sentences are quoted as evidence (no headings or fragments).
        if s.startswith("#") or s.startswith(">") or len(s) < 25 or not s.rstrip().endswith((".", "!", "?")):
            continue
        out.append(s)
    return out


def _best_sentences(query: str, chunks: list[dict], limit: int = 3, min_overlap: float = DEMO_MIN_SENTENCE_OVERLAP):
    q = content_tokens(query)
    scored: list[tuple[float, str, dict]] = []
    for ch in chunks:
        for s in _evidence_sentences(ch):
            overlap = len(q & content_tokens(s))
            if overlap:
                scored.append((overlap / (len(q) or 1), s, ch))
    scored.sort(key=lambda x: -x[0])
    seen: set[str] = set()
    best = []
    for sc, s, ch in scored:
        if sc < min_overlap or s in seen:
            continue
        seen.add(s)
        best.append((sc, s, ch))
        if len(best) >= limit:
            break
    return best


def _quote(sentence: str, ch: dict) -> dict:
    # Demo mode cannot translate: non-Russian quotes are shown in the original with a label.
    return {"chunk_id": ch["chunk_id"], "quote": sentence, "translation_ru": None}


DEMO_LABEL = "[Демо-режим — ответ составлен из предложений найденных фрагментов, ИИ-модель не использовалась]"


def demo_answer(question: str, chunks: list[dict]) -> dict:
    """Pick the retrieved sentences that best overlap the question. Never adds outside knowledge."""
    best = _best_sentences(question, chunks)
    if not best:
        return {"sufficient_evidence": False, "answer": "", "cited_chunk_ids": [], "quotes": []}
    lines = [f"- {s} [{ch['chunk_id']}]" for _, s, ch in best]
    cited = list(dict.fromkeys(ch["chunk_id"] for _, _, ch in best))
    return {
        "sufficient_evidence": True,
        "answer": DEMO_LABEL + "\n" + "\n".join(lines),
        "cited_chunk_ids": cited,
        "quotes": [_quote(s, ch) for _, s, ch in best],
    }


# ---- Demo specialist orientation -----------------------------------------------------

SPECIALISTS: list[tuple[str, str]] = [
    (r"терапевт\w*|врач\w*\s+общей\s+практики|general\s+practitioner|\bgp\b", "Терапевт / врач общей практики"),
    (r"невролог\w*|neurologist", "Невролог"),
    (r"кардиолог\w*|cardiologist", "Кардиолог"),
    (r"гастроэнтеролог\w*", "Гастроэнтеролог"),
    (r"дерматолог\w*", "Дерматолог"),
    (r"оториноларинголог\w*|\bлор\w*", "Оториноларинголог (ЛОР)"),
    (r"офтальмолог\w*|окулист\w*", "Офтальмолог"),
    (r"эндокринолог\w*", "Эндокринолог"),
    (r"уролог\w*", "Уролог"),
    (r"гинеколог\w*", "Гинеколог"),
    (r"травматолог\w*|ортопед\w*", "Травматолог-ортопед"),
    (r"пульмонолог\w*", "Пульмонолог"),
    (r"аллерголог\w*", "Аллерголог-иммунолог"),
    (r"ревматолог\w*", "Ревматолог"),
]
_SPECIALIST_WORDS = {_stem(w) for pat, _ in SPECIALISTS for w in re.findall(r"[а-яa-z]{4,}", pat)}


def _specialists_in(sentence: str) -> list[str]:
    low = normalize(sentence).lower()
    return [name for pat, name in SPECIALISTS if re.search(pat, low)]


def demo_specialist(context: str, chunks: list[dict]) -> dict:
    ctx = content_tokens(context)
    scored: dict[str, tuple[int, str, dict]] = {}
    for ch in chunks:
        for s in _evidence_sentences(ch):
            names = _specialists_in(s)
            if not names:
                continue
            overlap = len((content_tokens(s) - _SPECIALIST_WORDS) & ctx)
            if overlap < 2:
                continue
            for n in names:
                if n not in scored or overlap > scored[n][0]:
                    scored[n] = (overlap, s, ch)
    ranked = sorted(scored.items(), key=lambda kv: -kv[1][0])
    if ranked:
        top = ranked[0][1][0]
        ranked = [kv for kv in ranked if kv[1][0] >= max(2, top * 0.75)][:3]
    if not ranked:
        return {
            "sufficient_evidence": False, "options": [],
            "uncertainty": "В базе знаний не найдено фрагментов, которые явно связывают описанные жалобы с определённым специалистом.",
            "clarifying_question": "Опишите, пожалуйста, основную жалобу подробнее: что именно беспокоит, где и как давно?",
            "quotes": [],
        }
    options = [
        {"specialist": name, "reason": f"В источнике сказано: «{s}» [{ch['chunk_id']}]", "cited_chunk_ids": [ch["chunk_id"]]}
        for name, (_, s, ch) in ranked
    ]
    return {
        "sufficient_evidence": True,
        "options": options,
        "uncertainty": (
            "Демо-режим: совпадения найдены по словам, а не по смыслу. Выбор специалиста зависит от осмотра; "
            "если сомневаетесь, начните с терапевта (врача общей практики)."
        ),
        "clarifying_question": None,
        "quotes": [_quote(s, ch) for _, (_, s, ch) in ranked],
    }


# ---- Demo explanation -----------------------------------------------------------------


def demo_explain(kind: str, patient_text: str, query: str, chunks: list[dict]) -> dict:
    q = content_tokens(query)
    doc_quotes: list[str] = []
    for line in (patient_text or "").splitlines():
        for s in split_sentences(line) or [line]:
            if q & content_tokens(s):
                doc_quotes.append(s.strip())
    best = _best_sentences(query, chunks, limit=3)
    if not best:
        return {"document_quotes": doc_quotes[:6], "background": "", "background_sufficient": False,
                "cited_chunk_ids": [], "quotes": [], "ask_doctor": []}
    background = DEMO_LABEL + "\n" + "\n".join(f"- {s} [{ch['chunk_id']}]" for _, s, ch in best)
    return {
        "document_quotes": doc_quotes[:6],
        "background": background,
        "background_sufficient": True,
        "cited_chunk_ids": list(dict.fromkeys(ch["chunk_id"] for _, _, ch in best)),
        "quotes": [_quote(s, ch) for _, s, ch in best],
        "ask_doctor": [],
    }
