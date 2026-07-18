"""Log, register, and serve the Document Analyst (Tasks 2.2 + 2.3).

Run:  uv run python deployment/deploy.py

Pipeline (identical in shape to wk5/15):

    log_model (models-from-code) -> register_model (Unity Catalog)
                                 -> create/update Model Serving endpoint -> READY

Names are read from the environment so the same values flow through `.env`, the
secret scope, and the endpoint config:

    UC_CATALOG, UC_SCHEMA, MODEL_NAME       -> registered model
    SERVING_ENDPOINT_NAME                   -> endpoint
    SECRET_SCOPE                            -> where HOST/TOKEN/MODEL live
    VECTOR_SEARCH_ENDPOINT/INDEX, EMBEDDINGS_ENDPOINT, MCP_SERVER_URL -> retriever/tools
"""

from __future__ import annotations

import os

import mlflow
from dotenv import load_dotenv

# Load .env so `python deployment/deploy.py` sees UC_CATALOG, host/token, VS vars,
# etc. (the serving container gets these via secret scope + environment_vars).
load_dotenv()

# Repo root (one level above this file) so code_paths point at the real packages.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Everything the serving container must install (inference can miss the extras).
PIP_REQUIREMENTS = [
    "mlflow",
    "langgraph",
    "langchain",
    "langchain-core",
    "langchain-openai",
    "databricks-langchain",
    "databricks-vectorsearch",
    "langchain-mcp-adapters",
    "mcp",
    "openai",
    "python-dotenv",
]


def _uc_name() -> str:
    catalog = os.environ.get("UC_CATALOG", "main")
    schema = os.environ.get("UC_SCHEMA", "default")
    model = os.environ.get("MODEL_NAME", "document_analyst")
    return f"{catalog}.{schema}.{model}"


def log_and_register():
    """Log the models-from-code graph and register it in Unity Catalog.

    Returns (uc_name, version) for the endpoint step.
    """
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    experiment = os.environ.get("MLFLOW_EXPERIMENT", "/Shared/cs4603-pa4-document-analyst")
    try:
        mlflow.set_experiment(experiment)
    except Exception as exc:  # noqa: BLE001 - experiment path may need creating by hand
        print(f"[warn] could not set experiment '{experiment}': {exc}")

    uc_name = _uc_name()

    with mlflow.start_run(run_name="document-analyst"):
        model_info = mlflow.langchain.log_model(
            lc_model=os.path.join(_ROOT, "deployment", "agent_model.py"),
            name="agent",  # newer MLflow uses name= (older used artifact_path=)
            code_paths=[
                os.path.join(_ROOT, "agent"),
                os.path.join(_ROOT, "rag"),
                os.path.join(_ROOT, "tools"),
                os.path.join(_ROOT, "config.py"),
            ],
            pip_requirements=PIP_REQUIREMENTS,
            input_example={"messages": [{"role": "user", "content": "What was the revenue?"}]},
        )

    registered = mlflow.register_model(model_info.model_uri, uc_name)
    print(f"Registered {uc_name} version {registered.version}")
    return uc_name, registered.version


def create_or_update_endpoint(uc_name: str, version: str) -> str:
    """Create or update the Model Serving endpoint and wait for READY."""
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import (
        EndpointCoreConfigInput,
        ServedEntityInput,
    )

    endpoint_name = os.environ.get("SERVING_ENDPOINT_NAME", "document-analyst")
    scope = os.environ.get("SECRET_SCOPE", "cs4603-deploy")

    # Credentials come from the secret scope; the non-secret retriever config is
    # passed as plaintext so rag/store.py can reach the Vector Search index.
    environment_vars = {
        "DATABRICKS_HOST": f"{{{{secrets/{scope}/DATABRICKS_HOST}}}}",
        "DATABRICKS_TOKEN": f"{{{{secrets/{scope}/DATABRICKS_TOKEN}}}}",
        "DATABRICKS_MODEL": f"{{{{secrets/{scope}/DATABRICKS_MODEL}}}}",
        "VECTOR_SEARCH_ENDPOINT": os.environ.get("VECTOR_SEARCH_ENDPOINT", ""),
        "VECTOR_SEARCH_INDEX": os.environ.get("VECTOR_SEARCH_INDEX", ""),
        "EMBEDDINGS_ENDPOINT": os.environ.get("EMBEDDINGS_ENDPOINT", "databricks-gte-large-en"),
    }
    # Bonus C: point the deployed agent at a remote MCP server when configured.
    if os.environ.get("MCP_SERVER_URL"):
        environment_vars["MCP_SERVER_URL"] = os.environ["MCP_SERVER_URL"]

    served_entity = ServedEntityInput(
        entity_name=uc_name,
        entity_version=version,
        workload_size="Small",
        scale_to_zero_enabled=True,
        environment_vars=environment_vars,
    )
    config = EndpointCoreConfigInput(name=endpoint_name, served_entities=[served_entity])

    w = WorkspaceClient()
    existing = {e.name for e in w.serving_endpoints.list()}
    if endpoint_name in existing:
        print(f"Updating existing endpoint '{endpoint_name}'...")
        w.serving_endpoints.update_config_and_wait(
            name=endpoint_name, served_entities=[served_entity]
        )
    else:
        print(f"Creating endpoint '{endpoint_name}'...")
        w.serving_endpoints.create_and_wait(name=endpoint_name, config=config)

    endpoint = w.serving_endpoints.get(endpoint_name)
    state = getattr(endpoint.state, "ready", endpoint.state)
    host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    url = f"{host}/serving-endpoints/{endpoint_name}/invocations"
    print(f"Endpoint '{endpoint_name}' state: {state}")
    print(f"Endpoint URL: {url}")
    return url


if __name__ == "__main__":
    name, ver = log_and_register()
    create_or_update_endpoint(name, ver)
