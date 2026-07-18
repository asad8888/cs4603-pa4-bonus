"""MLflow models-from-code definition (Task 2.1).

This is the *deployable* file. MLflow serialises it by re-executing it inside the
serving container, so it must be self-contained: it validates its environment,
rebuilds the graph with production clients, and ends with `set_model` so MLflow
knows what to serve.

Must import cleanly (with a populated environment):

    python -c "import deployment.agent_model"

The heavy lifting lives in the packaged modules shipped via `code_paths`
(`agent`, `rag`, `tools`, `config.py`); this file only assembles them.
"""

from __future__ import annotations

import os

import mlflow

from agent.graph import build_graph, load_mcp_tools
from config import get_chat_llm
from rag.store import get_retriever

# ─── Fail fast with a clear message if the container is misconfigured ─────────
# Without this, a missing secret surfaces as a cryptic DEPLOYMENT_FAILED far from
# the root cause. Validate the credentials the LLM + retriever need at import.
_REQUIRED = ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_MODEL")
_missing = [name for name in _REQUIRED if not os.environ.get(name)]
if _missing:
    raise OSError(
        "Missing required environment variables for the serving container: "
        f"{', '.join(_missing)}. Wire them via the endpoint secret scope "
        "(see deployment/deploy.py)."
    )

# Build the graph once with production dependencies. `load_mcp_tools()` locates
# the bundled stdio MCP server via the packaged `tools` package (robust in the
# serving container); when MCP_SERVER_URL is set (Bonus C) it connects remotely.
graph = build_graph(
    llm=get_chat_llm(),
    retriever=get_retriever(),
    tools=load_mcp_tools(),
)

# Tell MLflow this compiled graph is the model to serve.
mlflow.models.set_model(graph)
