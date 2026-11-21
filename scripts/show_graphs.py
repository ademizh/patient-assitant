"""Learning helper: draw the three LangGraph graphs and run each one with a step-by-step trace.

    python scripts/show_graphs.py            # demo mode, no API key needed, nothing is saved

Prints, for every graph: its Mermaid diagram (paste it into https://mermaid.live to see the picture)
and the nodes that run, in order, with the most important values they change.
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from intake_assistant.config import load_settings  # noqa: E402
from intake_assistant.graphs.checklist_graph import ChecklistContext, checklist_graph  # noqa: E402
from intake_assistant.graphs.explain_graph import ExplainContext, explain_graph  # noqa: E402
from intake_assistant.graphs.intake_graph import IntakeFlow  # noqa: E402
from intake_assistant.providers import DemoProvider  # noqa: E402
from intake_assistant.rag.store import DocumentStore  # noqa: E402
from intake_assistant.records import AppointmentRecord, ConsultationItem, DoctorInstruction  # noqa: E402
from intake_assistant.urgent import load_pathway  # noqa: E402


def title(text: str) -> None:
    print("\n" + "=" * 90 + f"\n{text}\n" + "=" * 90)


def short(v) -> str:
    s = repr(v)
    return s if len(s) <= 110 else s[:107] + "..."


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    settings = replace(load_settings(), app_mode="demo", openai_api_key=None,
                       chroma_path=tmp / "chroma", library_path=tmp / "lib", appointments_path=tmp / "appointments")
    provider = DemoProvider(settings)
    store = DocumentStore(settings.chroma_path, settings.library_path, "demo", provider)
    for p in sorted((ROOT / "data" / "sample_documents").iterdir()):
        store.add_file(p.name, p.read_bytes(), 900, 150)

    # 1) Explanation graph -----------------------------------------------------------------
    title("1. EXPLANATION GRAPH (Week 7) - diagram")
    print(explain_graph.get_graph().draw_mermaid())
    title("1. EXPLANATION GRAPH - run on a fictional lab result")
    item = ConsultationItem(id="C1", kind="test_result", source="text", original_text="",
                            text="Ферритин 9 нг/мл ↓\nГемоглобин 128 г/л 120–140")
    ctx = ExplainContext(provider=provider, store=store, top_k=4, min_relevance=0.12, is_demo=True)
    for step in explain_graph.stream({"item": item, "focus": "Ферритин"}, context=ctx, stream_mode="updates"):
        for node, update in step.items():
            update = update or {}
            keys = {k: update[k] for k in ("query", "attempts", "problems") if k in update}
            if "relevant" in update:  # retrieve: how many passages passed the relevance threshold
                keys["relevant_passages"] = len(update["relevant"])
            print(f"  ▶ {node:<14} {short(keys) if keys else ''}")
    x = explain_graph.invoke({"item": item, "focus": "Ферритин"}, context=ctx)["explanation"]
    print(f"\n  status: {x.background_status}\n  ask the doctor: {x.ask_doctor[0]}")

    # 2) Intake graph ----------------------------------------------------------------------
    title("2. INTAKE GRAPH (Weeks 8-9) - diagram")
    flow = IntakeFlow(tmp / "checkpoints.sqlite")
    print(flow.graph.get_graph().draw_mermaid())
    title("2. INTAKE GRAPH - a short conversation (the graph pauses after every question)")
    rec = AppointmentRecord()
    pathway = load_pathway(settings.urgent_rules_path)
    for reply in [{"action": "answer", "text": "Болит голова уже две недели."},
                  {"action": "skip"},
                  {"action": "answer", "text": "Поправка: голова болит три недели, а не две."},
                  {"action": "finish"}]:
        r = flow.send(rec, reply, provider, pathway)
        print(f"  patient: {short(reply)}\n    nodes: {' -> '.join(r.trace or [])}\n    next question: {r.question}")
    print("\n  saved entries:", [(e.id, e.text, e.corrects_entry_id) for e in rec.intake.entries])
    print("  checkpoints in SQLite for this record:",
          len(list(flow.checkpointer.list(flow.config(rec.id)))))

    # 3) Checklist graph -------------------------------------------------------------------
    title("3. CHECKLIST GRAPH (Week 10) - diagram")
    print(checklist_graph.get_graph().draw_mermaid())
    title("3. CHECKLIST GRAPH - fictional doctor's instructions")
    text = (ROOT / "data" / "sample_patient" / "FICTIONAL_doctor_instructions.txt").read_text(encoding="utf-8")
    ins = [DoctorInstruction(id="D1", source="text", original_text=text, text=text)]
    cctx = ChecklistContext(provider=provider, store=store, top_k=4, min_relevance=0.12, is_demo=True)
    final = {}
    for step in checklist_graph.stream({"instructions": ins, "explain_medications": True}, context=cctx,
                                       stream_mode="updates"):
        for node, update in step.items():
            final.update(update or {})
            print(f"  ▶ {node}")
    for a in final["actions"]:
        med = f"  [{a.medication.name} | {a.medication.dose} | {a.medication.frequency}]" if a.medication else ""
        expl = f"  -> explanation: {a.explanation.background_status}" if a.explanation else ""
        print(f"  {a.id} {a.category:<10} {a.description[:55]}{med}{expl}")
    print("\n  issues to clarify with the doctor:")
    for i in final["issues"]:
        print(f"   - {i.kind}: {i.description[:100]}")


if __name__ == "__main__":
    main()
