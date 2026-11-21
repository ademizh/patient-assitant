"""LangGraph workflows of the patient assistant.

Three graphs, each built from small, separately testable nodes:

* explain_graph   - explains a term, note or laboratory result (Week 7);
* intake_graph    - symptom intake before the appointment: one question at a time, pauses for the
                    patient's answer (interrupt) and is saved after every step (checkpointer) (Weeks 8-9);
* checklist_graph - doctor's instructions -> checklist, with medication explanations (Week 10).

The OpenAI SDK, Chroma and the grounding checks are used directly inside the nodes; LangGraph only
controls the order of the steps, the branching, the retries and the pauses.
See docs/LANGGRAPH_GUIDE.md for a step-by-step explanation.
"""
