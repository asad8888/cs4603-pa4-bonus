"""Synthesizer node (Task 1.6).

`make_synthesizer(llm)` combines the collected `step_results` into one coherent,
cited answer. It writes the answer to BOTH `final_answer` (internal scratch) AND
the `messages` channel as an `AIMessage`. The `messages` append is what makes the
deployed OpenAI-compatible endpoint return a non-empty completion — the serving
container reads the response off the last message, not off `final_answer`.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.planner import _latest_user_query
from agent.prompts import SYNTHESIZER_PROMPT
from agent.state import AnalystState


def _format_results(plan: list[str], results: list[str]) -> str:
    lines = []
    for i, result in enumerate(results):
        step = plan[i] if i < len(plan) else f"step {i + 1}"
        lines.append(f"Step {i + 1} ({step}): {result}")
    return "\n".join(lines) if lines else "(no step results were produced)"


def make_synthesizer(llm):
    def synthesizer(state: AnalystState) -> dict:
        question = _latest_user_query(state.get("messages", []))
        results = state.get("step_results", [])
        plan = state.get("plan", [])

        context = _format_results(plan, results)
        response = llm.invoke(
            [
                SystemMessage(content=SYNTHESIZER_PROMPT),
                HumanMessage(
                    content=f"ORIGINAL question: {question}\n\nRESULTS:\n{context}"
                ),
            ]
        )
        answer = (getattr(response, "content", "") or "").strip()

        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
        }

    return synthesizer
