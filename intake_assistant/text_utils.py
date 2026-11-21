"""Small, dependency-free text helpers used by guardrails and demo-mode logic.

Russian is the primary language; English patterns are kept so English input still works.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[\w']+", re.UNICODE)
_CYR = re.compile(r"[а-яё]", re.IGNORECASE)
_LAT = re.compile(r"[a-z]", re.IGNORECASE)

STOPWORDS = {
    # English
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at", "for", "with",
    "is", "are", "was", "were", "be", "been", "it", "its", "this", "that", "these", "those",
    "i", "i'm", "im", "my", "me", "we", "you", "your", "he", "she", "they", "them", "his", "her",
    "do", "does", "did", "have", "has", "had", "so", "as", "by", "from", "about", "what",
    "which", "when", "how", "can", "could", "should", "would", "will", "there", "then", "than",
    "also", "just", "some", "any", "very", "really", "into", "out", "up", "down", "over",
    # Russian
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все", "она", "так",
    "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по", "только", "ее", "её", "мне",
    "было", "вот", "от", "меня", "еще", "ещё", "нет", "о", "из", "ему", "теперь", "когда", "даже",
    "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь",
    "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей", "может", "они", "тут",
    "где", "есть", "надо", "ней", "для", "мы", "тебя", "их", "чем", "была", "сам", "чтоб", "без",
    "будто", "чего", "раз", "тоже", "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того",
    "потому", "этого", "какой", "совсем", "ним", "здесь", "этом", "один", "почти", "мой", "тем",
    "чтобы", "нее", "сейчас", "были", "куда", "зачем", "всех", "никогда", "можно", "при",
    "наконец", "два", "об", "другой", "хоть", "после", "над", "больше", "тот", "через", "эти",
    "нас", "про", "всего", "них", "какая", "много", "разве", "три", "эту", "моя", "впрочем",
    "хорошо", "свою", "этой", "перед", "иногда", "лучше", "чуть", "том", "нельзя", "такой", "им",
    "более", "всегда", "конечно", "всю", "между", "это", "мои", "мою", "моё", "моего", "у меня",
    "очень", "такое", "какие", "каких", "ли", "ваш", "ваша", "ваши", "вашего", "означает",
    "значит", "что-то", "такое?", "который", "которые", "которая",
}

# Phrases that signal "I don't know" rather than a denial. Checked BEFORE negation cues.
UNKNOWN_PATTERNS = [
    r"\bi\s*(do\s*not|don'?t)\s*(know|remember|recall)\b",
    r"\bnot\s+sure\b",
    r"\bunsure\b",
    r"\bcan'?t\s+(remember|recall|say)\b",
    r"\bno\s+idea\b",
    r"\bunknown\b",
    r"не\s+знаю",
    r"не\s+помню",
    r"не\s+уверен\w*",
    r"затрудняюсь",
    r"понятия\s+не\s+имею",
    r"сложно\s+сказать",
    r"трудно\s+сказать",
    r"не\s+могу\s+(сказать|вспомнить)",
    r"не\s+замеча\w*\s+точно",
]

HEDGE_PATTERNS = UNKNOWN_PATTERNS + [
    r"\bmaybe\b", r"\bperhaps\b", r"\bprobably\b", r"\bi\s+think\b", r"\bi\s+guess\b",
    r"\b(about|around)\s+(\d|a|an|one|two|three|four|five|six|seven|eight|nine|ten|half|last|mid|early|late)\b",
    r"\broughly\b", r"\bapproximately\b", r"\bor\s+so\b", r"\bsomething\s+like\b", r"\bmight\b",
    r"\bpossibly\b", r"\bkind\s+of\b", r"\bsort\s+of\b",
    r"примерно", r"около\b", r"наверное", r"кажется", r"может\s+быть", r"возможно", r"вроде\b",
    r"где-то", r"приблизительно", r"\bдумаю\b", r"пожалуй", r"по-моему", r"если\s+не\s+ошибаюсь",
    r"или\s+чуть\s+(больше|меньше)", r"чуть\s+больше", r"с\s+небольшим", r"плюс-минус",
]

NEGATION_PATTERNS = [
    r"^\s*(no|none|nope|never|nothing)\b",
    r"\bno\s+\w+", r"\bnot\b", r"\bnever\b", r"\bnone\b", r"\bwithout\b", r"\bdenies\b",
    r"\b(don'?t|do\s+not|didn'?t|did\s+not|haven'?t|have\s+not|hasn'?t|isn'?t|wasn'?t|aren'?t)\b",
    r"\bneither\b", r"\bnor\b",
    r"(^|[^\w])нет([^\w]|$)", r"(^|[^\w])не([^\w]|$)", r"(^|[^\w])ни([^\w]|$)", r"никак\w*", r"ничего",
    r"(^|[^\w])без([^\w]|$)", r"никогда", r"отрица\w*", r"отсутств\w*",
]


def normalize(text: str) -> str:
    return (text or "").replace("ё", "е").replace("Ё", "Е")


def tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(normalize(text))]


# Longest Russian endings first. A light stemmer so "головные"~"головная", "болит"~"боли".
_RU_ENDINGS = sorted(
    [
        "иями", "ями", "ами", "ого", "его", "ому", "ему", "ыми", "ими", "ешь", "ете", "ишь", "ите",
        "ают", "яют", "уют", "ется", "ются", "ась", "ось", "лась", "лись", "ться", "ется",
        "ах", "ях", "ов", "ев", "ей", "ий", "ый", "ой", "ая", "яя", "ое", "ее", "ые", "ие", "ую", "юю",
        "ом", "ем", "ам", "ям", "ть", "ет", "ют", "ит", "ат", "ят", "ла", "ли", "ло",
        "а", "я", "о", "е", "ы", "и", "у", "ю", "ь", "й",
    ],
    key=len,
    reverse=True,
)


def _stem(word: str) -> str:
    word = normalize(word).lower()
    if _CYR.search(word):
        for suf in _RU_ENDINGS:
            if word.endswith(suf) and len(word) - len(suf) >= 3:
                return word[: -len(suf)]
        return word
    for suffix in ("ingly", "edly", "ing", "ies", "ed", "es", "ly", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def content_tokens(text: str) -> set[str]:
    return {_stem(t) for t in tokens(text) if t not in STOPWORDS and len(t) > 1}


def matches_any(text: str, patterns: list[str]) -> bool:
    low = normalize(text).lower()
    return any(re.search(p, low) for p in patterns)


def is_unknown(text: str) -> bool:
    return matches_any(text, UNKNOWN_PATTERNS)


def is_hedged(text: str) -> bool:
    return matches_any(text, HEDGE_PATTERNS)


def has_negation_cue(text: str) -> bool:
    """True if the text contains an explicit negation (and is not merely 'I don't know')."""
    low = normalize(text).lower()
    if is_unknown(low):
        for p in UNKNOWN_PATTERNS:
            low = re.sub(p, " ", low)
    return matches_any(low, NEGATION_PATTERNS)


def is_denial(text: str) -> bool:
    """A statement that is primarily a denial: 'Нет.', 'Аллергии нет', 'Никаких лекарств не принимаю'."""
    low = normalize(text).strip().lower()
    if not low or is_unknown(low):
        return False
    if re.match(r"^(no|none|nope|never|nothing|not really|n/a)\b", low):
        return True
    if re.search(r"\b(don'?t|do not|haven'?t|have not|didn'?t|did not)\s+(take|have|had|use|notice)\b", low):
        return True
    if re.match(r"^(нет|никаких|ничего|никогда|не\s+было|не\s+принимаю|отрицаю)([^\w]|$)", low):
        return True
    # "<something> нет" / "не было" / "не замечал(а)" as the whole short clause, e.g. "Рвоты нет".
    if len(low) <= 60 and re.search(r"(^|[^\w])(нет|не\s+было|не\s+замеча\w*|не\s+принима\w*|отсутству\w*)[.!]?$", low):
        return True
    return False


def lexical_support(statement: str, source_text: str) -> float:
    """Share of the statement's content words that appear in the source text (0..1)."""
    st = content_tokens(statement)
    if not st:
        return 1.0
    src = content_tokens(source_text)
    return len(st & src) / len(st)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?;])\s+|\n+", (text or "").strip())
    return [p.strip() for p in parts if p and p.strip()]


def squash(text: str) -> str:
    """Whitespace-normalised, ё-normalised, lower-cased text used for verbatim checks."""
    return re.sub(r"\s+", " ", normalize(text or "")).strip().lower()


def is_verbatim(fragment: str, source: str) -> bool:
    """True if `fragment` occurs in `source` ignoring only whitespace, case and ё/е differences."""
    f = squash(fragment).strip(" .;,")
    return bool(f) and f in squash(source)


# Numbers as written: 400, 0,5, 1.25, 4,5–5,5 -> individual numbers with ',' normalised to '.'
_NUM = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)?")


def numbers_in(text: str) -> list[str]:
    return [n.replace(",", ".") for n in _NUM.findall(text or "")]


def numbers_preserved(source: str, target: str) -> bool:
    """Translation check: the same numbers appear in source and target (none lost, none added)."""
    return set(numbers_in(source)) == set(numbers_in(target))


def unsupported_numbers(text: str, *supports: str) -> list[str]:
    """Numbers in `text` that do not appear in any of the supporting texts."""
    allowed: set[str] = set()
    for s in supports:
        allowed.update(numbers_in(s))
    # Citation IDs contain digits; strip [..] markers before checking.
    clean = re.sub(r"\[[^\]]*\]", " ", text or "")
    return [n for n in dict.fromkeys(numbers_in(clean)) if n not in allowed]


def detect_language(text: str) -> str:
    """'ru' if the text is mostly Cyrillic letters, 'en' if mostly Latin, else 'other'."""
    cyr = len(_CYR.findall(text or ""))
    lat = len(_LAT.findall(text or ""))
    if cyr == 0 and lat == 0:
        return "other"
    return "ru" if cyr >= lat else "en"


# ---- Diagnostic / evaluative wording guards ------------------------------------------

DIAGNOSTIC_PATTERNS = [
    r"\bdiagnos\w*\b",
    r"\b(likely|probably|probable|possible|suspected)\s+(has|have|a|an|case|diagnosis|condition)\b",
    r"\bconsistent with\b",
    r"\byou (have|should|must)\b",
    r"\b(prescribe|treatment plan)\b",
    r"диагноз",
    r"(вероятно|скорее\s+всего|похоже|возможно),?\s+(у\s+вас|это)\b",
    r"у\s+вас\s+([а-я]+\s+)?(мигрень|инфекция|анемия|заболевание|болезнь|синдром|воспаление|гипертония|диабет)",
    r"(указывает|свидетельствует|говорит)\s+о\s+(наличии|заболевании|болезни|диагнозе)",
    r"(вам\s+)?(следует|нужно|необходимо)\s+(принимать|начать\s+приём|пить\s+препарат|лечиться)",
    r"назнача(ю|ем)",
    r"рекоменду(ю|ем)\s+(принимать|препарат|лечение)",
    r"это\s+(точно\s+)?(мигрень|инфекция|анемия)",
]

# Words that judge a result as normal/abnormal. Used together with a personal reference.
EVALUATIVE_PATTERNS = [
    r"в\s+норме", r"\bнорм(а|ы|альн\w*)\b", r"отклонени\w*", r"повышен\w*", r"понижен\w*",
    r"патолог\w*", r"(хорош|плох)\w*\s+(результат|анализ)", r"\b(ab)?normal\b", r"\belevated\b",
    r"\b(low|high)\s+result\b", r"выше\s+нормы", r"ниже\s+нормы",
]
PERSONAL_PATTERNS = [r"\bваш\w*", r"\bу\s+вас\b", r"\byour\b", r"\byou\b", r"\bвы\b", r"\bвам\b"]


def is_diagnostic(text: str) -> bool:
    return matches_any(text, DIAGNOSTIC_PATTERNS)


def is_personal_evaluation(sentence: str) -> bool:
    """A sentence that judges THE PATIENT'S result (personal reference + evaluative word)."""
    return matches_any(sentence, EVALUATIVE_PATTERNS) and matches_any(sentence, PERSONAL_PATTERNS)
