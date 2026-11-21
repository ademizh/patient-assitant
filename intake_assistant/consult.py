"""During the appointment: plain-language explanations of terms, notes and test results.

Each explanation keeps three parts apart:
  1. document_states - what the patient's document / note ACTUALLY says (verbatim fragments and
     parsed lab rows whose values, units and ranges are exact substrings of the document);
  2. background      - general information from the knowledge base (AI text, cited, validated);
  3. ask_doctor      - what must be clarified with the doctor.

Guards: the AI may not judge the patient's own result as normal/abnormal, may not infer a
diagnosis, may not invent a missing reference range, and may not introduce numbers absent
from the sources. Sentences that do so are removed and the removal is reported.
"""

from __future__ import annotations

import re

# The explanation itself is produced by a LangGraph workflow - see graphs/explain_graph.py.
from .graphs.explain_graph import NO_KB_TEXT, _filter_background, build_query, run_explain  # noqa: F401
from .labs import parse_lab_lines
from .records import CONSULT_KIND_LABELS, ConsultationItem, Explanation

KIND_LABELS = CONSULT_KIND_LABELS


def explain_item(
    item: ConsultationItem,
    store,
    provider,
    top_k: int,
    min_relevance: float,
    is_demo: bool,
    focus: str | None = None,
) -> Explanation:
    """Public entry point used by the UI, the API and the fictional example."""
    return run_explain(item, store, provider, top_k, min_relevance, is_demo, focus=focus)


def lab_names(text: str) -> list[str]:
    return list(dict.fromkeys(r.name for r in parse_lab_lines(text)))


def short(text: str, n: int = 80) -> str:
    t = re.sub(r"\s+", " ", text or "").strip()
    return t if len(t) <= n else t[: n - 1] + "…"
