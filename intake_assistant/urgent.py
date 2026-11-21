"""Urgent-care check driven by a pathway file.

The bundled pathway (data/pathways/DEMO_urgent_rules_ru.json) is a DEMONSTRATION rule set written
by the prototype author. It is NOT clinician-reviewed; every output says so. A documented,
clinician-reviewed pathway can be dropped in via URGENT_RULES_PATH (validated=true, reviewed_by,
source) without code changes. The check never diagnoses; it only points to emergency services.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .records import UrgentCheck, UrgentMatch
from .schemas import IntakeEntry
from .text_utils import normalize

_NEG_AFTER = re.compile(r"^[^.;!?,]{0,12}?(?:нет|не\s+было|не\s+беспоко\w*|не\s+замеча\w*|отсутству\w*)(?:[^\w]|$)")
_NEG_BEFORE = re.compile(r"(?:^|[^\w])(?:нет|не|без|ни|никак\w*|не\s+было|отрица\w*|no|not|never|without)(?=[^\w]|$)[^.;!?]{0,25}$")


def load_pathway(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data.setdefault("validated", False)
    data.setdefault("reviewed_by", None)
    data.setdefault("rules", [])
    # A pathway only counts as validated if it names its reviewer and source.
    data["validated"] = bool(data["validated"] and data.get("reviewed_by") and data.get("source"))
    return data


def check_entries(entries: list[IntakeEntry], pathway: dict) -> UrgentCheck:
    matches: list[UrgentMatch] = []
    extra_messages: list[str] = []
    numbers = pathway.get("emergency_numbers", "112")
    for e in entries:
        text = normalize(e.text)
        low = text.lower()
        for rule in pathway["rules"]:
            for pat in rule["patterns"]:
                m = re.search(pat, low)
                if not m:
                    continue
                negated = bool(_NEG_BEFORE.search(low[max(0, m.start() - 40):m.start()])
                               or _NEG_AFTER.search(low[m.end():m.end() + 30]))
                matches.append(UrgentMatch(rule_id=rule["id"], label=rule["label"],
                                           matched_text=e.text[m.start():m.end()], entry_id=e.id, negated=negated))
                if not negated and rule.get("message"):
                    extra_messages.append(rule["message"].format(numbers=numbers))
                break
    positive = [m for m in matches if not m.negated]
    template = pathway.get("message_if_matched" if positive else "message_if_not_matched", "")
    message = template.format(numbers=numbers)
    if extra_messages:
        message = " ".join(dict.fromkeys(extra_messages)) + " " + message
    return UrgentCheck(
        pathway_title=pathway.get("title", "Правила срочности"),
        pathway_validated=pathway["validated"],
        pathway_source=pathway.get("source", ""),
        reviewed_by=pathway.get("reviewed_by"),
        matches=matches,
        message=message,
        emergency_numbers=numbers,
    )
