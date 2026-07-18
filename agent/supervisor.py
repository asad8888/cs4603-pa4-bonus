"""Supervisor node + routing edge (Task 1.3).

The supervisor is called once per step. If every planned step has been executed
it routes to the synthesizer; otherwise it classifies the *current* step as a
retrieval step (`rag_agent`) or a calculation step (`mcp_tools`). Classification
is done by the LLM but parsed defensively so an unexpected phrasing still yields
a valid route.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import SUPERVISOR_PROMPT
from agent.state import AnalystState

RAG = "rag_agent"
MCP = "mcp_tools"
SYNTH = "synthesizer"

# Words that signal a calculation step when we have to fall back to keyword parsing.
_CALC_HINTS = (
    "mcp_tools", "mcp", "calc", "comput", "growth", "percent", "%", "cagr",
    "multiply", "divide", "sum", "difference", "ratio", "convert", "rate",
    "increase", "decrease", "compare", "project",
)


def _classify(text: str) -> str:
    """Map a raw LLM classification to RAG or MCP.

    The model is asked for a single word, but if it adds explanation we pick
    whichever label appears FIRST (so "route to rag_agent, not mcp_tools" is not
    misread as MCP). Keyword hints are a last resort, defaulting to retrieval.
    """
    lowered = (text or "").lower()
    i_mcp = lowered.find(MCP)
    i_rag = lowered.find(RAG)
    if i_mcp != -1 and (i_rag == -1 or i_mcp < i_rag):
        return MCP
    if i_rag != -1:
        return RAG
    if any(hint in lowered for hint in _CALC_HINTS):
        return MCP
    # Default: treat it as a document lookup.
    return RAG


def make_supervisor(llm):
    def supervisor(state: AnalystState) -> dict:
        plan = state.get("plan", [])
        idx = state.get("current_step_index", 0)

        # All steps done -> synthesize the final answer.
        if idx >= len(plan):
            return {"next_agent": SYNTH}

        step = plan[idx]
        response = llm.invoke(
            [
                SystemMessage(content=SUPERVISOR_PROMPT),
                HumanMessage(content=f"CURRENT step: {step}"),
            ]
        )
        decision = _classify(getattr(response, "content", "") or "")
        return {"next_agent": decision}

    return supervisor


def route_from_supervisor(state: AnalystState) -> str:
    return state["next_agent"]
