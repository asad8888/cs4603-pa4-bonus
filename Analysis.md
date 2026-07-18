# CS4603 PA4 — Document Analyst

> This `README.md` is a **graded deliverable**:
>
> - Document how to set up, run, and deploy your Document Analyst so a TA can reproduce your results.
> - **Answer every ANALYSIS QUESTION** from the assignment in the sections below.
> - Code that runs but is not explained will not receive full marks.
> - Replace every `TODO` before submitting.
> - Keep it self-contained: a reader should be able to follow this file top-to-bottom —
>   setup → ingest → run → deploy → results — without opening the assignment PDF.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in your values
```

## Running locally

Everything runs against Databricks (managed LLM, embeddings, and Vector Search), so a
`.env` with valid `DATABRICKS_HOST` / `DATABRICKS_TOKEN` is required even for "local" runs.

1. **Verify the MCP tool server** (Task 0.2):
   ```bash
   uv run python tools/mcp_server.py     # should start and wait on stdio; Ctrl-C to exit
   ```

2. **Ingest the corpus into Vector Search** (Task 0.3 — run once, from a Databricks
   notebook that has `spark`):
   ```python
   from rag.ingest import build_chunks_table, create_index
   # PDF must already be uploaded to a UC volume, e.g. /Volumes/main/default/pa4/
   build_chunks_table(
       spark,
       volume_path="/Volumes/main/default/pa4/annual_report.pdf",
       chunks_table="main.default.<you>_analyst_chunks",   # also set SOURCE_TABLE in .env
   )
   create_index()   # STANDARD endpoint + TRIGGERED Delta Sync index; waits until READY
   ```
   `build_chunks_table` parses the PDF with `ai_parse_document`, chunks it with
   `ai_prep_search` into a CDF-enabled Delta table (`chunk_id`, `chunk_to_retrieve`,
   `chunk_to_embed`, `source`, `page`), and `create_index` builds the managed-embedding
   index named in `VECTOR_SEARCH_INDEX`.

3. **Build and run the graph** in `pa4.ipynb`:
   ```python
   from agent.graph import build_graph
   graph = build_graph()          # uses config.py + rag/store.py + the bundled MCP server
   result = graph.invoke({"messages": [{"role": "user",
             "content": "What was the net revenue in 2023?"}]})
   print(result["messages"][-1].content)
   ```
   `build_graph()` wires `planner → supervisor → {rag_agent | mcp_tools} → synthesizer`.
   Dependencies (`llm`, `retriever`, `tools`) are injected and default to the real
   Databricks clients when omitted, so the offline smoke test can pass fakes instead.

4. **Test queries** (retrieval-only, computation-only, combined). Run these in `pa4.ipynb`
   and record the actual answers your endpoint returns:

   | Query | Answer produced |
   |-------|-----------------|
   | "What was the net income in 2023?" | TODO (fill after running) |
   | "What is 15% of 2.4 billion?" | TODO (fill after running) |
   | "What was 2023 revenue, and its value after 10% growth?" | TODO (fill after running) |

## Deployment

The deployment pipeline is `deployment/agent_model.py` (models-from-code definition) +
`deployment/deploy.py` (log → register → serve).

1. **Store credentials in a secret scope** (the serving container has no `.env`):
   ```bash
   databricks secrets create-scope cs4603-deploy
   databricks secrets put-secret cs4603-deploy DATABRICKS_TOKEN --string-value "dapi..."
   databricks secrets put-secret cs4603-deploy DATABRICKS_HOST  --string-value "https://<workspace>.databricks.com"
   databricks secrets put-secret cs4603-deploy DATABRICKS_MODEL --string-value "databricks-meta-llama-3-3-70b-instruct"
   ```

2. **Log, register, and serve** (reads names from `.env`: `UC_CATALOG/UC_SCHEMA/MODEL_NAME`,
   `SERVING_ENDPOINT_NAME`, `SECRET_SCOPE`, `VECTOR_SEARCH_*`, `EMBEDDINGS_ENDPOINT`):
   ```bash
   uv run python deployment/deploy.py
   ```
   This calls `mlflow.langchain.log_model(...)` with `code_paths=[agent, rag, tools,
   config.py]` and explicit `pip_requirements`, registers the version in Unity Catalog,
   and creates/updates a `workload_size="Small"`, `scale_to_zero_enabled=True` endpoint
   whose secrets are injected as `{{secrets/cs4603-deploy/...}}` references and whose
   Vector Search env vars are passed in plaintext.

- **Registered model:** `main.default.document_analyst` (from `UC_CATALOG.UC_SCHEMA.MODEL_NAME`)
- **Endpoint name:** TODO (`SERVING_ENDPOINT_NAME`, e.g. `<you>-document-analyst`)
- **Endpoint URL:** `{DATABRICKS_HOST}/serving-endpoints/<endpoint>/invocations`
- **Bonus B** (`deployment/deploy_agents.py`): same log+register, then a single
  `agents.deploy(...)` that also provisions a Review App.
- **Bonus C** (`deployment/mcp_app/`): the MCP server can run as a standalone Databricks
  App; set `MCP_SERVER_URL` and the agent connects to it over streamable HTTP instead of
  a stdio subprocess.

## Design decisions

- **Plan-then-execute with a supervisor loop.** The planner emits an explicit, auditable
  JSON list of atomic steps; the supervisor routes each step to exactly one specialist and
  loops back until the plan is exhausted, then routes to the synthesizer. This cleanly
  handles mixed retrieval + computation queries that a single ReAct agent struggles with.
- **RAG separated from math tools.** Retrieval (`rag_agent` over Databricks Vector Search)
  and deterministic math (`mcp_tools` over the MCP server) are distinct nodes, so each has a
  focused prompt and can be tuned independently; numbers are computed by real Python, never
  hallucinated.
- **One retriever path, local and deployed.** `rag/store.py::get_retriever()` queries a
  managed Vector Search index reachable with `DATABRICKS_HOST`/`DATABRICKS_TOKEN`, so the
  identical code runs locally and inside the serving container — no separate embedding path.
- **messages-in → messages-out.** The synthesizer writes the final answer to both
  `final_answer` and the `messages` channel as an `AIMessage`, satisfying the OpenAI-
  compatible serving contract (the endpoint reads the last message).
- **Injected dependencies.** `build_graph(llm, retriever, tools)` accepts fakes, enabling a
  fully offline smoke test (`tests/test_smoke.py`) that the CI pipeline runs before deploy.

---

## Analysis Questions

### Task 1.2 — Planner
1. TODO
2. TODO

### Task 1.3 — Supervisor
1. TODO
2. TODO

### Task 1.4 — RAG Agent
1. TODO
2. TODO

### Task 2.1 — Model Definition
1. TODO
2. TODO

### Task 2.3 — Serving Endpoint
1. TODO
2. TODO

### Task 3.2 — Client
1. TODO
2. TODO
3. TODO

### Bonus A / B / C (if attempted)
TODO
