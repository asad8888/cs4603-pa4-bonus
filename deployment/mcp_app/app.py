"""Bonus C — standalone MCP server as a Databricks App.

Reuses the GIVEN tool definitions from `tools/mcp_server.py` but serves them over
the **streamable-http** transport instead of stdio, so the agent connects to a
long-lived HTTP service (decoupled from the model) rather than spawning a
subprocess inside the serving container.

The agent side is already wired: when `MCP_SERVER_URL` is set, `load_mcp_tools`
in `agent/graph.py` connects here over streamable HTTP with a bearer token.
"""

from __future__ import annotations

from tools.mcp_server import mcp

if __name__ == "__main__":
    # Databricks Apps injects the port via $DATABRICKS_APP_PORT (FastMCP/uvicorn
    # honour it); streamable-http exposes the tools at the /mcp path.
    mcp.run(transport="streamable-http")
