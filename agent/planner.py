"""Planner node (Task 1.2).

`make_planner(llm)` returns a graph node that turns the user's question into an
ordered list of 2-5 atomic steps. The LLM is asked for a JSON array; parsing is
deliberately forgiving (models sometimes wrap JSON in prose or code fences), and
any failure falls back to a single-step plan so the graph can still run.
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import PLANNER_PROMPT
from agent.state import AnalystState


def _latest_user_query(messages: list) -> str:
    """Return the text of the most recent human turn (empty string if none)."""
    for msg in reversed(messages or []):
        # Support both LangChain message objects and plain {"role","content"} dicts.
        role = getattr(msg, "type", None) or (msg.get("role") if isinstance(msg, dict) else None)
        if role in ("human", "user"):
            content = getattr(msg, "content", None)
            if content is None and isinstance(msg, dict):
                content = msg.get("content", "")
            return content if isinstance(content, str) else str(content)
    # Fall back to the last message of any kind.
    if messages:
        last = messages[-1]
        content = getattr(last, "content", None)
        if content is None and isinstance(last, dict):
            content = last.get("content", "")
        return content if isinstance(content, str) else str(content)
    return ""


def _parse_plan(raw: str, fallback: str) -> list[str]:
    """Parse the LLM output into a list of step strings, tolerating stray prose."""
    text = (raw or "").strip()
    # Strip a ```json ... ``` (or plain ```) code fence if present.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    candidates = [text]
    # Also try the first bracketed array anywhere in the text.
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, list):
            steps = [str(s).strip() for s in parsed if str(s).strip()]
            if steps:
                return steps[:5]

    # Could not parse anything usable -> single-step plan on the raw question.
    return [fallback]


def make_planner(llm):
    def planner(state: AnalystState) -> dict:
        query = _latest_user_query(state.get("messages", []))
        response = llm.invoke(
            [SystemMessage(content=PLANNER_PROMPT), HumanMessage(content=query)]
        )
        plan = _parse_plan(getattr(response, "content", "") or "", fallback=query)
        return {"plan": plan, "current_step_index": 0, "step_results": []}

    return planner
