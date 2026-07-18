"""Full Document Analyst graph (Tasks 1.5 + 1.7).

Wiring:

    START -> planner -> supervisor -> {rag_agent | mcp_tools} -> supervisor -> ...
                                    -> synthesizer -> END

The MCP tools are loaded ONCE at graph-build time (see the async caveat in
DEPLOYMENT_GUIDE.md) and invoked synchronously, which keeps the stdio subprocess
stable inside the serving container. `build_graph` takes injectable `llm`,
`retriever`, and `tools` so the graph can be unit-tested fully offline with fakes.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agent.planner import make_planner
from agent.prompts import MCP_STEP_PROMPT
from agent.rag_agent import make_rag_agent
from agent.state import AnalystState
from agent.supervisor import MCP, RAG, SYNTH, make_supervisor, route_from_supervisor
from agent.synthesizer import make_synthesizer


def _default_mcp_server() -> str:
    """Locate the GIVEN stdio MCP server file.

    We resolve it through the importable ``tools`` package rather than by
    ``__file__`` arithmetic. That matters in the MLflow serving container: the
    model file (``agent_model.py``) and the ``tools`` code_path package land in
    different directories, so ``dirname(dirname(...))`` from another module does
    not point at ``tools``. ``tools.__file__`` always does.
    """
    import tools

    return os.path.join(os.path.dirname(os.path.abspath(tools.__file__)), "mcp_server.py")


def _run_async(coro):
    """Run a coroutine to completion from synchronous code.

    MLflow's serving path is synchronous, but MCP tools are async. `asyncio.run`
    raises "cannot be called from a running event loop" if a loop is already
    active, so when that happens we run the coroutine on a dedicated thread with
    its own event loop.
    """
    try:
        asyncio.get_running_loop()
        running = True
    except RuntimeError:
        running = False

    if not running:
        return asyncio.run(coro)

    box: dict = {}

    def _runner():
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller thread
            box["error"] = exc

    thread = threading.Thread(target=_runner)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


def _ensure_std_fileno() -> None:
    """Make sure ``sys.stdout``/``sys.stderr`` expose a real ``fileno()``.

    Inside the MLflow serving container these are replaced by a ``StreamToLogger``
    wrapper that has no ``fileno()``. The MCP stdio client hands the child process
    ``sys.stderr`` as its error log, and subprocess creation calls ``.fileno()`` on
    it — which raises ``AttributeError`` and fails model load. We point the wrapper
    at the real OS file descriptors so the subprocess can be spawned. Locally the
    streams already have a ``fileno`` and this is a no-op.
    """
    for stream, fallback_fd in ((sys.stdout, 1), (sys.stderr, 2)):
        try:
            stream.fileno()
        except (AttributeError, OSError, ValueError):
            try:
                stream.fileno = lambda fd=fallback_fd: fd  # type: ignore[method-assign]
            except (AttributeError, TypeError):
                pass


def load_mcp_tools(server_path: str | None = None):
    """Connect to the MCP server and return its tools as LangChain tools.

    Two transports are supported:

    - **Remote HTTP (Bonus C):** if `MCP_SERVER_URL` is set, connect to the
      standalone Databricks App over streamable HTTP with a bearer token.
    - **Local stdio (Parts 1/2):** otherwise spawn `tools/mcp_server.py` as a
      stdio subprocess (the code ships inside the serving container).
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    mcp_url = os.environ.get("MCP_SERVER_URL", "").strip()
    if mcp_url:
        token = os.environ.get("DATABRICKS_TOKEN", "")
        connections = {
            "analyst": {
                "url": f"{mcp_url.rstrip('/')}/mcp",
                "transport": "streamable_http",
                "headers": {"Authorization": f"Bearer {token}"} if token else {},
            }
        }
    else:
        _ensure_std_fileno()
        connections = {
            "analyst": {
                "command": sys.executable,
                "args": [server_path or _default_mcp_server()],
                "transport": "stdio",
            }
        }

    client = MultiServerMCPClient(connections)
    return _run_async(client.get_tools())


def _find_tool(tools, name: str):
    for tool in tools:
        if getattr(tool, "name", None) == name:
            return tool
    return None


def _tool_output_to_text(output) -> str:
    """Normalise an MCP tool result into plain text.

    langchain-mcp-adapters may return the tool output as a list of content blocks
    (e.g. ``[{"type": "text", "text": "..."}]``) rather than a bare string; extract
    the text so ``step_results`` stays human-readable.
    """
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        parts = []
        for item in output:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(output)


def make_mcp_node(tools, llm):
    """Node that executes ONE calculation step by calling exactly one MCP tool."""
    llm_with_tools = llm.bind_tools(tools)

    def mcp_tools(state: AnalystState) -> dict:
        idx = state.get("current_step_index", 0)
        plan = state.get("plan", [])
        step = plan[idx] if idx < len(plan) else ""
        prior = state.get("step_results", [])

        prior_context = "\n".join(prior) if prior else "(none)"
        response = llm_with_tools.invoke(
            [
                SystemMessage(content=MCP_STEP_PROMPT),
                HumanMessage(
                    content=f"CURRENT step: {step}\n\nPrevious results:\n{prior_context}"
                ),
            ]
        )

        tool_calls = getattr(response, "tool_calls", None) or []
        if tool_calls:
            call = tool_calls[0]
            tool = _find_tool(tools, call["name"])
            if tool is not None:
                output = _run_async(tool.ainvoke(call["args"]))
                result = _tool_output_to_text(output)
            else:
                result = f"Requested unknown tool '{call['name']}' for step: {step}"
        else:
            # Model answered without a tool call — keep whatever it produced so the
            # step is not silently dropped, but this is the non-ideal path.
            content = (getattr(response, "content", "") or "").strip()
            result = content or f"No calculation tool was invoked for step: {step}"

        return {
            "step_results": prior + [result],
            "current_step_index": idx + 1,
        }

    return mcp_tools


def build_graph(llm=None, retriever=None, tools=None):
    """Assemble and compile the full Document Analyst graph.

    Dependencies are injected for testability. Real dependencies (LLM, Vector
    Search retriever, MCP tools) are only constructed when not supplied, so the
    offline smoke test can pass fakes without touching Databricks or a subprocess.
    """
    if llm is None:
        from config import get_chat_llm

        llm = get_chat_llm()
    if retriever is None:
        from rag.store import get_retriever

        retriever = get_retriever()
    if tools is None:
        tools = load_mcp_tools()

    builder = StateGraph(AnalystState)
    builder.add_node("planner", make_planner(llm))
    builder.add_node("supervisor", make_supervisor(llm))
    builder.add_node("rag_agent", make_rag_agent(retriever, llm))
    builder.add_node("mcp_tools", make_mcp_node(tools, llm))
    builder.add_node("synthesizer", make_synthesizer(llm))

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {RAG: "rag_agent", MCP: "mcp_tools", SYNTH: "synthesizer"},
    )
    builder.add_edge("rag_agent", "supervisor")
    builder.add_edge("mcp_tools", "supervisor")
    builder.add_edge("synthesizer", END)

    return builder.compile()
