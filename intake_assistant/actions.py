"""Doctor's instructions -> editable follow-up checklist.

Guardrails (applied to BOTH the live model output and the demo heuristics):
  * every action must quote a statement that occurs verbatim in the doctor's instructions,
    otherwise it is dropped (and reported) - nothing is invented;
  * medication name / dose / frequency / duration are kept only if they are verbatim fragments
    of that statement - they are never normalised, converted or completed;
  * a due date is kept only if its wording appears in the statement; it becomes a calendar date
    only when the doctor wrote an explicit date with a year. Otherwise the deadline stays
    "не указан" (unspecified) and the patient may add their OWN target date, stored separately;
  * ambiguous wording, missing medication details and contradictions are flagged for
    clarification - never resolved automatically;
  * an action's description may not contain numbers that are absent from the doctor's words.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from pydantic import BaseModel, Field

from .records import (
    ACTION_CATEGORY_LABELS,
    ActionItem,
    DoctorInstruction,
    InstructionIssue,
    MedicationDetails,
    next_id,
)
from .text_utils import normalize, split_sentences, squash, unsupported_numbers

# ---------------------------------------------------------------------------------------
# Structured output schema (live model) - also produced by the demo heuristics
# ---------------------------------------------------------------------------------------


class ExtractedMedication(BaseModel):
    name: str | None = Field(description="Medication name exactly as written, or null.")
    dose: str | None = Field(description="Dose WITH unit exactly as written (e.g. '400 мг'), or null.")
    frequency: str | None = Field(description="Frequency exactly as written (e.g. '2 раза в день'), or null.")
    duration: str | None = Field(description="Duration exactly as written (e.g. 'в течение 10 дней'), or null.")


class ExtractedAction(BaseModel):
    category: str = Field(description="One of: follow_up, test, medication, diet, other.")
    description: str = Field(description="Short task in Russian. Must not add numbers, doses or dates.")
    source_statement: str = Field(description="The doctor's sentence, copied EXACTLY (verbatim).")
    due_text: str | None = Field(description="Deadline wording exactly as written in the statement, or null.")
    medication: ExtractedMedication | None = Field(description="Only for medication instructions, else null.")


class ExtractedIssue(BaseModel):
    kind: str = Field(description="One of: ambiguous, contradiction, missing_detail.")
    description: str = Field(description="In Russian: what is unclear and should be asked.")
    source_statements: list[str] = Field(description="The doctor's sentences involved, copied verbatim.")


class ActionExtraction(BaseModel):
    actions: list[ExtractedAction]
    issues: list[ExtractedIssue]


# ---------------------------------------------------------------------------------------
# Verbatim helpers
# ---------------------------------------------------------------------------------------


def locate_verbatim(fragment: str | None, source: str) -> str | None:
    """Return the exact substring of `source` that matches `fragment` up to whitespace/case/ё.

    None if the fragment does not occur. The returned text is the SOURCE wording.
    """
    if not fragment or not fragment.strip():
        return None
    frag = fragment.strip().strip(" .;,")
    if not frag:
        return None
    if frag in source:
        return frag
    words = [re.escape(normalize(w)) for w in frag.split()]
    if not words:
        return None
    pattern = r"\s+".join(words).replace("е", "[её]").replace("Е", "[ЕЁ]")
    m = re.search(pattern, source, flags=re.IGNORECASE)
    return source[m.start():m.end()] if m else None


# ---------------------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------------------

_MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6, "июля": 7,
    "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}
_MONTH_RE = "|".join(_MONTHS)

DUE_PATTERNS = [
    rf"(?:до|к|не\s+позднее|не\s+позже)\s+\d{{1,2}}[./]\d{{1,2}}(?:[./]\d{{2,4}})?",
    rf"(?:до|к|не\s+позднее|не\s+позже)\s+\d{{1,2}}\s+(?:{_MONTH_RE})(?:\s+\d{{4}})?(?:\s+г(?:ода|\.)?)?",
    rf"\d{{1,2}}[./]\d{{1,2}}[./]\d{{2,4}}",
    rf"\d{{1,2}}\s+(?:{_MONTH_RE})(?:\s+\d{{4}})?(?:\s+г(?:ода|\.)?)?",
    r"\d{4}-\d{2}-\d{2}",
    r"через\s+(?:\d+|одну|один|две|два|три|четыре|пять|шесть|несколько|пару)?\s*(?:дн\w*|день|недел\w*|месяц\w*|год\w*)",
    r"(?:в\s+течение|в\s+ближайш\w+)\s+(?:\d+\s+)?(?:дн\w*|недел\w*|месяц\w*)",
    r"(?:на\s+)?(?:следующей|этой)\s+неделе|в\s+(?:следующем|этом)\s+месяце",
    r"(?:завтра|послезавтра)",
]


def find_due_text(statement: str, category: str) -> str | None:
    low = normalize(statement).lower()
    for p in DUE_PATTERNS:
        if category == "medication" and p.startswith(r"(?:в\s+течение"):
            continue  # for medicines "в течение 10 дней" is a duration, not a deadline
        m = re.search(p, low)
        if m:
            return statement[m.start():m.end()].strip()
    return None


def parse_explicit_date(text: str | None) -> tuple[str | None, str | None]:
    """(ISO date, note). A calendar date is produced only for explicit dates WITH a year."""
    if not text:
        return None, None
    low = normalize(text).lower()
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", low)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    else:
        m = re.search(r"(\d{1,2})[./](\d{1,2})[./](\d{2,4})", low)
        if m:
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if y < 100:
                y += 2000
        else:
            m = re.search(rf"(\d{{1,2}})\s+({_MONTH_RE})\s+(\d{{4}})", low)
            if m:
                d, mo, y = int(m.group(1)), _MONTHS[m.group(2)], int(m.group(3))
            else:
                if re.search(rf"\d{{1,2}}[./]\d{{1,2}}|\d{{1,2}}\s+(?:{_MONTH_RE})", low):
                    return None, "Год не указан — подтвердите дату у врача или задайте свою."
                if re.search(r"через|в\s+течение|ближайш|неделе|месяце|завтра", low):
                    return None, "Относительный срок — задайте свою целевую дату."
                return None, None
    try:
        return date(y, mo, d).isoformat(), None
    except ValueError:
        return None, "Дата в тексте некорректна — уточните у врача."


def relative_due_hint(due_text: str | None, appointment_date: str | None) -> str | None:
    """Helper for the UI only: 'через 2 недели' counted from the appointment date. Never stored as the doctor's date."""
    if not due_text or not appointment_date:
        return None
    try:
        base = date.fromisoformat(appointment_date)
    except ValueError:
        return None
    low = normalize(due_text).lower()
    m = re.search(r"через\s+(\d+|одну|один|две|два|три|четыре|пять|шесть)?\s*(дн\w*|день|недел\w*|месяц\w*)", low)
    if not m:
        return None
    words = {"одну": 1, "один": 1, "две": 2, "два": 2, "три": 3, "четыре": 4, "пять": 5, "шесть": 6}
    n = int(m.group(1)) if (m.group(1) or "").isdigit() else words.get(m.group(1) or "", 1)
    unit = m.group(2)
    days = n * (7 if unit.startswith("недел") else 30 if unit.startswith("месяц") else 1)
    return (base + timedelta(days=days)).isoformat()


# ---------------------------------------------------------------------------------------
# Demo heuristics (offline)
# ---------------------------------------------------------------------------------------

_CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("medication", [r"\d+(?:[.,]\d+)?\s*(?:мг|мкг|мл|ед|ме|mg|ml)\b", r"таблет\w*|капсул\w*|капл[ия]\b|препарат\w*|\bприним\w*|\bпить\s+[А-ЯA-Z]",
                    r"отмени\w*\s+\w+|прекрати\w*\s+при[её]м"]),
    ("follow_up", [r"повторн\w+\s+при[её]м|прийти\s+на\s+при[её]м|придите|запис\w+\s+(?:на|к)|контрольн\w+\s+(?:при[её]м|визит|осмотр)|явка|консультаци\w+\s+\w+|обратиться\s+к"]),
    ("test", [r"\bсдать\b|\bсдайте\b|анализ\w*|\bмрт\b|\bкт\b|\bузи\b|\bэкг\b|рентген\w*|обследован\w*|пройти\s+\w+"]),
    ("diet", [r"питани\w*|диет\w*|кофе|ча[йя]\b|алкогол\w*|сол[иь]\b|сахар\w*|\bводы?\b|жидкост\w*|\bешьте\b|\bесть\b|продукт\w*|завтрак\w*"]),
    ("other", [r"\bспать\b|\bсон\b|сна\b|перерыв\w*|нагрузк\w*|прогулк\w*|дневник\w*|избегать|ограничить|соблюдать|измерять|вести\b|делать\b|отдых\w*|экран\w*"]),
]
_ACTION_CUE = re.compile(
    r"принимать|принимайте|пить|выпивать|сдать|сдайте|сделать|сделайте|пройти|пройдите|прийти|придите|"
    r"записаться|запишитесь|явиться|ограничить|ограничьте|исключить|исключите|избегать|избегайте|"
    r"соблюдать|делать|делайте|вести|ведите|измерять|обратиться|обратитесь|отменить|отменяется|прекратить|"
    r"продолжить|продолжайте|добавить|увеличить|уменьшить|снизить|контроль|повторн\w+\s+при[её]м|"
    r"рекомендовано|рекомендуется|назначен\w*|\d+\s*(?:мг|мкг|мл)\b|по\s+\d+\s+(?:таблетк|капсул|капл)\w*",
    re.IGNORECASE,
)

_DOSE = r"(?:по\s+)?\d+(?:[.,]\d+)?\s*(?:мг|мкг|г|мл|ЕД|МЕ|mg|mcg|ml)(?![\w])|по\s+(?:\d+|одной|одну|половине|1/2)\s+(?:таблетк\w*|капсул\w*|капл\w*)"
_FREQ = [
    r"не\s+(?:чаще|более)\s+\d+\s+раз\w*\s+в\s+(?:день|сутки|неделю)",
    r"(?:\d+|один|одну|два|две|три|четыре)\s+раз[а]?\s+в\s+(?:день|сутки|неделю)",
    r"каждые\s+\d+\s+час\w*",
    r"(?:ежедневно|утром\s+и\s+вечером|утром|вечером|на\s+ночь|перед\s+сном)",
    r"(?:по\s+необходимости|при\s+необходимости|при\s+(?:сильной\s+)?(?:головной\s+)?бол\w+)",
]
_DURATION = (
    r"(?:в\s+течение|курсом|курс)\s+\d+\s*(?:дн\w*|день|недел\w*|месяц\w*)"
    r"|(?:в\s+течение|на)\s+(?:одного\s+|двух\s+|трёх\s+|трех\s+)?(?:месяц\w*|недел\w*)"
    r"|не\s+более\s+\d+\s+(?:дн\w*|день)\s+подряд"
    r"|до\s+(?:следующего|повторного)\s+при[её]ма|длительно|постоянно"
)
_NAME_BEFORE_VERB = re.compile(
    r"(?:принимать|принимайте|пить|продолжить\s+при[её]м|назначен\w*|отменить|отменяется|прекратить\s+при[её]м)\s*:?\s+"
    r"([A-Za-zА-Яа-яЁё][\w\-]*(?:\s+[A-Za-zА-Яа-яЁё0-9][\w\-]*)?)",
    re.IGNORECASE,
)
_GENERIC_NAME_WORDS = {"по", "таблетку", "таблетки", "капсулу", "лекарство", "препарат", "препараты", "при", "не"}


def _find(pattern: str, text: str) -> str | None:
    m = re.search(pattern, normalize(text), flags=re.IGNORECASE)
    return text[m.start():m.end()].strip() if m else None


_MED_VERB = re.compile(
    r"(?:принимать|принимайте|пить|продолжить\s+при[её]м|назначен\w*|отменить|отменяется|прекратить\s+при[её]м)",
    re.IGNORECASE,
)


def demo_medication(statement: str) -> ExtractedMedication:
    dose = _find(_DOSE, statement)
    freq = next((f for f in (_find(p, statement) for p in _FREQ) if f), None)
    duration = _find(_DURATION, statement)
    name = None
    verb = _MED_VERB.search(statement)
    first_cut = min(
        [i for i in (verb.start() if verb else None, statement.find(dose) if dose else None) if i is not None and i >= 0],
        default=None,
    )
    if first_cut:
        before = statement[:first_cut].strip(" :,-")
        # "Ибупрофен 400 мг ...", "Магний B6 принимать ..." - the name opens the sentence.
        if before and re.match(r"^[A-ZА-ЯЁ]", before) and len(before.split()) <= 3:
            name = before
    if name is None and verb:
        m = _NAME_BEFORE_VERB.search(statement)
        if m:
            words = [w for w in m.group(1).split() if w.lower() not in _GENERIC_NAME_WORDS and not re.match(r"^\d", w)]
            if words and re.match(r"^[A-ZА-ЯЁ]", words[0]):
                name = locate_verbatim(" ".join(words), statement) or words[0]
    return ExtractedMedication(name=name, dose=dose, frequency=freq, duration=duration)


def _category(sentence: str) -> str | None:
    low = normalize(sentence).lower()
    for cat, pats in _CATEGORY_RULES:
        if any(re.search(p, low) for p in pats):
            return cat
    return None


def _short_description(category: str, statement: str, med: ExtractedMedication | None) -> str:
    if category == "medication" and med and med.name:
        return f"Лекарство: {med.name}"
    text = statement.strip().rstrip(".")
    return text if len(text) <= 90 else text[:87].rsplit(" ", 1)[0] + "…"


def _split_instruction_sentences(text: str) -> list[str]:
    out = []
    for line in re.split(r"\n+", text or ""):
        line = re.sub(r"^\s*(?:[-•*]|\d+[.)])\s+", "", line).strip()
        if not line:
            continue
        out.extend(split_sentences(line))
    return out


def demo_extract_actions(instructions: list[DoctorInstruction]) -> tuple[ActionExtraction, list[str]]:
    """Offline extraction. Returns (extraction, sentences that were not recognised as actions)."""
    actions: list[ExtractedAction] = []
    unmatched: list[str] = []
    for ins in instructions:
        for sent in _split_instruction_sentences(ins.text):
            cat = _category(sent)
            if cat is None or not _ACTION_CUE.search(sent):
                unmatched.append(sent)
                continue
            med = demo_medication(sent) if cat == "medication" else None
            actions.append(
                ExtractedAction(
                    category=cat,
                    description=_short_description(cat, sent, med),
                    source_statement=sent,
                    due_text=find_due_text(sent, cat),
                    medication=med,
                )
            )
    return ActionExtraction(actions=actions, issues=[]), unmatched


# ---------------------------------------------------------------------------------------
# Rule-based issue detection (runs in every mode)
# ---------------------------------------------------------------------------------------

AMBIGUOUS_PATTERNS = [
    (r"по\s+необходимости|при\s+необходимости|если\s+нужно|при\s+боли|при\s+сильной", "не указано, в каких случаях и сколько максимум",
     r"не\s+(?:чаще|более)"),
    (r"иногда|пару\s+раз|несколько\s+раз|время\s+от\s+времени", "частота описана неточно", None),
    (r"как\s+обычно|как\s+раньше|по\s+старой\s+схеме", "ссылка на прежнюю схему без деталей", None),
    (r"примерно|около\b|где-то", "приблизительная формулировка", None),
    (r"на\s+ваше\s+усмотрение|по\s+самочувствию|можно\s+попробовать", "решение оставлено на усмотрение пациента", None),
    (r"какое-то\s+время|некоторое\s+время|в\s+ближайшее\s+время|позже|скоро", "срок не определён", None),
]
_STOP_WORDS = re.compile(r"(?:не\s+принимать|отменить|отменяется|прекратить|исключить)", re.IGNORECASE)


def _med_key(name: str | None) -> str | None:
    if not name:
        return None
    first = squash(name).split()[0]
    return first[:5] if len(first) >= 5 else first


def detect_issues(actions: list[ActionItem]) -> list[InstructionIssue]:
    issues: list[InstructionIssue] = []
    for a in actions:
        if a.added_by != "doctor_instruction":
            continue
        low = normalize(a.source_statement).lower()
        for pat, why, unless in AMBIGUOUS_PATTERNS:
            m = re.search(pat, low)
            if m and not (unless and re.search(unless, low)):
                issues.append(InstructionIssue(
                    kind="ambiguous",
                    description=f"{why[0].upper() + why[1:]}: «{a.source_statement[m.start():m.end()]}». Уточните у врача.",
                    source_statements=[a.source_statement], action_ids=[a.id]))
                break
        if a.category == "medication" and not _STOP_WORDS.search(low):
            med = a.medication or MedicationDetails()
            missing = [lbl for lbl, v in (("название", med.name), ("дозировка", med.dose), ("частота приёма", med.frequency)) if not v]
            if missing:
                issues.append(InstructionIssue(
                    kind="missing_detail",
                    description="В указании о лекарстве не найдено: " + ", ".join(missing) + ". Не додумывайте — уточните у врача.",
                    source_statements=[a.source_statement], action_ids=[a.id]))
    # Contradictions between medication instructions for the same medicine.
    meds = [a for a in actions if a.category == "medication" and a.added_by == "doctor_instruction"]
    for i, a in enumerate(meds):
        for b in meds[i + 1:]:
            ka = _med_key(a.medication.name if a.medication else None)
            kb = _med_key(b.medication.name if b.medication else None)
            if not ka or ka != kb:
                continue
            stop_a, stop_b = bool(_STOP_WORDS.search(a.source_statement)), bool(_STOP_WORDS.search(b.source_statement))
            diffs = []
            if stop_a != stop_b:
                diffs.append("в одном месте препарат назначен, в другом — отменён")
            for field, label in (("dose", "дозировка"), ("frequency", "частота"), ("duration", "длительность")):
                va = getattr(a.medication, field) if a.medication else None
                vb = getattr(b.medication, field) if b.medication else None
                if va and vb and squash(va) != squash(vb):
                    diffs.append(f"{label}: «{va}» и «{vb}»")
            if diffs:
                issues.append(InstructionIssue(
                    kind="contradiction",
                    description=f"«{a.medication.name}» — " + "; ".join(diffs) + ". Уточните у врача, какой вариант верный.",
                    source_statements=[a.source_statement, b.source_statement], action_ids=[a.id, b.id]))
    # Two different deadlines for follow-up visits.
    fus = [a for a in actions if a.category == "follow_up" and a.doctor_due_text and a.added_by == "doctor_instruction"]
    if len({squash(a.doctor_due_text or "") for a in fus}) > 1:
        issues.append(InstructionIssue(
            kind="contradiction",
            description="Указаны разные сроки повторного приёма: " + ", ".join(f"«{a.doctor_due_text}»" for a in fus) + ".",
            source_statements=[a.source_statement for a in fus], action_ids=[a.id for a in fus]))
    return issues


# ---------------------------------------------------------------------------------------
# Guard + build
# ---------------------------------------------------------------------------------------


def build_actions(
    extraction: ActionExtraction,
    instructions: list[DoctorInstruction],
    previous: list[ActionItem] | None = None,
) -> tuple[list[ActionItem], list[InstructionIssue], list[str]]:
    """Validate extracted actions against the doctor's text and build checklist items.

    Returns (actions, issues, notes). Status, completion time, patient target dates and notes
    of unchanged items are carried over from `previous`; patient-added items are kept.
    """
    previous = previous or []
    notes: list[str] = []
    out: list[ActionItem] = []
    used_ids: list[ActionItem] = list(previous)

    for ex in extraction.actions:
        cat = ex.category if ex.category in ACTION_CATEGORY_LABELS else "other"
        statement, ins_id = None, None
        for ins in instructions:
            statement = locate_verbatim(ex.source_statement, ins.text)
            if statement:
                ins_id = ins.id
                break
        if not statement:
            notes.append(
                f"Пропущено действие «{ex.description[:60]}»: его формулировка не найдена дословно в указаниях врача."
            )
            continue
        flags: list[str] = []
        med = None
        if cat == "medication":
            src = ex.medication or ExtractedMedication(name=None, dose=None, frequency=None, duration=None)
            fields = {}
            for field in ("name", "dose", "frequency", "duration"):
                raw = getattr(src, field)
                exact = locate_verbatim(raw, statement)
                if raw and not exact:
                    flags.append(f"medication_{field}_not_verbatim")
                fields[field] = exact
            med = MedicationDetails(**fields)
        due = locate_verbatim(ex.due_text, statement) if ex.due_text else None
        if ex.due_text and not due:
            flags.append("due_text_not_verbatim")
        iso, due_note = parse_explicit_date(due)
        description = ex.description.strip() or statement
        if unsupported_numbers(description, statement):
            description = statement
            flags.append("description_replaced_with_source")

        prev = next(
            (p for p in previous if p.added_by == "doctor_instruction" and squash(p.source_statement) == squash(statement)
             and p.category == cat),
            None,
        )
        item = ActionItem(
            id=prev.id if prev else next_id("A", used_ids + out),
            category=cat,  # type: ignore[arg-type]
            description=prev.description if (prev and prev.edited_by_patient) else description,
            source_statement=statement,
            instruction_id=ins_id,
            added_by="doctor_instruction",
            doctor_due_text=due,
            doctor_due_date=iso,
            due_date_note=due_note,
            patient_target_date=prev.patient_target_date if prev else None,
            medication=med,
            status=prev.status if prev else "pending",
            completed_at=prev.completed_at if prev else None,
            flags=flags,
            patient_note=prev.patient_note if prev else "",
            edited_by_patient=prev.edited_by_patient if prev else False,
        )
        if any(squash(a.source_statement) == squash(item.source_statement) and a.category == item.category for a in out):
            continue  # duplicate
        out.append(item)

    # Keep the patient's own items.
    out.extend(p for p in previous if p.added_by == "patient")

    issues = detect_issues(out)
    for ex_issue in extraction.issues:
        kind = ex_issue.kind if ex_issue.kind in ("ambiguous", "contradiction", "missing_detail") else "ambiguous"
        stmts = [s for s in (locate_verbatim(x, " ".join(i.text for i in instructions)) for x in ex_issue.source_statements) if s]
        ids = [a.id for a in out if any(squash(s) in squash(a.source_statement) or squash(a.source_statement) in squash(s) for s in stmts)]
        dup = any(set(ids) and set(ids) == set(i.action_ids) and i.kind == kind for i in issues)
        if not dup and ex_issue.description.strip():
            issues.append(InstructionIssue(kind=kind, description=ex_issue.description.strip(),  # type: ignore[arg-type]
                                           source_statements=stmts, action_ids=ids, origin="ai"))
    return out, issues, notes


# ---------------------------------------------------------------------------------------
# Views and reminders (in-app only)
# ---------------------------------------------------------------------------------------


def action_state(a: ActionItem, today: date) -> str:
    """'done' | 'overdue' | 'pending'. Overdue only for items with a CONFIRMED date."""
    if a.status == "done":
        return "done"
    d, _ = a.effective_date()
    if d is not None and d < today:
        return "overdue"
    return "pending"


def split_views(actions: list[ActionItem], today: date) -> dict[str, list[ActionItem]]:
    views: dict[str, list[ActionItem]] = {"pending": [], "done": [], "overdue": []}
    for a in actions:
        views[action_state(a, today)].append(a)

    def key(a: ActionItem):
        d, _ = a.effective_date()
        return (d is None, d or date.max, a.id)

    for v in views.values():
        v.sort(key=key)
    return views


def reminders(actions: list[ActionItem], today: date, window_days: int = 3) -> list[tuple[ActionItem, str]]:
    out = []
    for a in actions:
        if a.status == "done":
            continue
        d, who = a.effective_date()
        if d is None:
            continue
        delta = (d - today).days
        who_txt = "срок врача" if who == "doctor" else "ваша цель"
        if delta < 0:
            out.append((a, f"просрочено на {-delta} дн. ({who_txt}: {d.strftime('%d.%m.%Y')})"))
        elif delta <= window_days:
            when = "сегодня" if delta == 0 else f"через {delta} дн."
            out.append((a, f"{when} ({who_txt}: {d.strftime('%d.%m.%Y')})"))
    return out


def mark_done(a: ActionItem, done: bool, when: str) -> None:
    a.status = "done" if done else "pending"
    a.completed_at = when if done else None
