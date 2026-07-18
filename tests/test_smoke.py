"""Offline smoke test for the Document Analyst graph (Bonus A test target).

This is what the Bonus A CI pipeline runs to prove the graph wires up before any
deploy. It builds the real graph with **fake** LLM / retriever / tool objects, so
it needs no Databricks, no network, and no MCP subprocess. It exercises a combined
retrieval + calculation query and asserts that both specialists ran and the final
answer surfaced on `messages[-1]`.

Run:  uv run pytest -q
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402


def test_graph_module_imports():
    """Minimal collection guard: the graph module must import cleanly."""
    from agent.graph import build_graph  # noqa: F401


# ─── Fakes (no Databricks, no network, no subprocess) ────────────────────────

@tool
def calculate(expression: str) -> str:
    """Fake calculator tool: pretend 1000 grown by 10% is 1100."""
    return f"{expression} = 1100"


class _ToolBoundFakeLLM:
    """What FakeLLM.bind_tools returns: always requests the `calculate` tool."""

    def invoke(self, messages):
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "calculate",
                    "args": {"expression": "1000 * 1.10"},
                    "id": "call_1",
                }
            ],
        )


class FakeLLM:
    """Routes by the node's system prompt so one object serves every node."""

    def bind_tools(self, tools):
        return _ToolBoundFakeLLM()

    def invoke(self, messages):
        system = messages[0].content
        human = messages[-1].content
        if "PLANNER" in system:
            return AIMessage(
                content='["Find the net revenue for 2023", "Compute revenue * 1.10"]'
            )
        if "SUPERVISOR" in system:
            calc = any(w in human.lower() for w in ("comput", "calc", "*", "growth"))
            return AIMessage(content="mcp_tools" if calc else "rag_agent")
        if "extract a single fact" in system:
            return AIMessage(content="Net revenue in 2023 was 1000 [source: annual_report.pdf, p.4]")
        if "SYNTHESIZER" in system:
            return AIMessage(content="Revenue was 1000 [source: annual_report.pdf, p.4]; after 10% growth it is 1100.")
        return AIMessage(content="")


class FakeRetriever:
    """Returns a single fixed document for any query."""

    def invoke(self, query):
        return [
            Document(
                page_content="Meridian reported net revenue of 1000 in fiscal year 2023.",
                metadata={"source": "annual_report.pdf", "page": 4},
            )
        ]


def test_combined_query_runs_end_to_end():
    from agent.graph import build_graph

    graph = build_graph(llm=FakeLLM(), retriever=FakeRetriever(), tools=[calculate])
    result = graph.invoke(
        {
            "messages": [
                {"role": "user", "content": "What was 2023 revenue, and its value after 10% growth?"}
            ]
        }
    )

    # A plan of >= 2 steps was produced.
    assert len(result["plan"]) >= 2

    # Both specialists ran: one retrieval fact (has a citation) and one calc result.
    joined = " ".join(result["step_results"])
    assert "source:" in joined  # rag_agent contributed
    assert "= 1100" in joined  # mcp_tools contributed

    # The final answer surfaced on messages[-1] (the serving contract) AND final_answer.
    assert result["final_answer"]
    last = result["messages"][-1]
    assert isinstance(last, AIMessage)
    assert last.content == result["final_answer"]
    assert last.content.strip()
