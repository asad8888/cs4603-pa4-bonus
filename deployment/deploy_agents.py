"""Bonus B — deploy via the databricks-agents SDK (deployment/deploy_agents.py).

This reuses the *entire* manual pipeline from `deploy.py` unchanged — the same
models-from-code definition, the same `code_paths`, the same Unity Catalog
registration — and only swaps the final `WorkspaceClient` endpoint step for a
single `agents.deploy(...)` call. That one call provisions the serving endpoint
AND a Review App, and handles auth automatically (no secret scope needed).

Run:  uv run python deployment/deploy_agents.py
"""

from __future__ import annotations

from deployment.deploy import log_and_register


def main() -> None:
    from databricks import agents

    # Identical log + register step as the manual path.
    uc_name, version = log_and_register()

    # One call replaces create/update endpoint + secret wiring.
    deployment = agents.deploy(
        model_name=uc_name,
        model_version=version,
        scale_to_zero=True,
    )

    print(f"Deployed agent: {uc_name} v{version}")
    print(f"Serving endpoint: {deployment.endpoint_name}")
    print(f"Review App URL:  {deployment.review_app_url}")


if __name__ == "__main__":
    main()
