"""Generate the example outputs for the FICTIONAL demo appointment (before -> during -> after).

    python scripts/generate_examples.py              # uses APP_MODE / OPENAI_API_KEY from .env
    APP_MODE=demo python scripts/generate_examples.py

Writes to docs/example_appointment/:
    FICTIONAL_doctor_report.pdf / .json         report for the doctor (before the appointment)
    FICTIONAL_appointment_record.pdf / .json    the whole appointment record
    FICTIONAL_followup_checklist.pdf            the follow-up checklist
The demo knowledge documents are indexed into a temporary store (the real store is not touched).
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import replace
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from intake_assistant.config import load_settings  # noqa: E402
from intake_assistant.pdf_export import checklist_to_pdf, record_to_pdf, report_to_pdf  # noqa: E402
from intake_assistant.providers import get_provider  # noqa: E402
from intake_assistant.rag.store import DocumentStore  # noqa: E402
from intake_assistant.records import RecordStore  # noqa: E402
from intake_assistant.report import report_to_json  # noqa: E402
from intake_assistant.sample import build_fictional_record  # noqa: E402
from intake_assistant.urgent import load_pathway  # noqa: E402

TODAY = date(2026, 10, 11)  # fixed so the example is reproducible


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    settings = replace(load_settings(), chroma_path=tmp / "chroma", library_path=tmp / "lib")
    provider = get_provider(settings)
    store = DocumentStore(settings.chroma_path, settings.library_path, "examples", provider)
    for p in sorted((ROOT / "data" / "sample_documents").iterdir()):
        store.add_file(p.name, p.read_bytes(), settings.chunk_size, settings.chunk_overlap)
    rec = build_fictional_record(provider, store, RecordStore(tmp / "appointments"), settings.top_k,
                                 settings.min_relevance, settings.is_demo, load_pathway(settings.urgent_rules_path),
                                 today=TODAY)
    out = ROOT / "docs" / "example_appointment"
    out.mkdir(parents=True, exist_ok=True)
    (out / "FICTIONAL_doctor_report.pdf").write_bytes(report_to_pdf(rec.report))
    (out / "FICTIONAL_doctor_report.json").write_text(report_to_json(rec.report), encoding="utf-8")
    (out / "FICTIONAL_appointment_record.pdf").write_bytes(record_to_pdf(rec, TODAY))
    (out / "FICTIONAL_appointment_record.json").write_text(rec.model_dump_json(indent=2), encoding="utf-8")
    (out / "FICTIONAL_followup_checklist.pdf").write_bytes(checklist_to_pdf(rec, TODAY))
    print(f"mode={settings.mode}; wrote {len(list(out.iterdir()))} files to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
