"""Corpus ingestion into Databricks Vector Search (Task 0.3 / rag/ingest.py).

Run inside a Databricks notebook (needs Spark + ai_parse_document / ai_prep_search).
This mirrors the PA2 Part 1 pipeline:

    PDF in a UC volume  --ai_parse_document-->  parsed text
                        --ai_prep_search----->  Delta table of chunks (CDF on)
                        --Delta Sync index --->  Vector Search (managed embeddings)

Typical driver (from a Databricks notebook cell):

    from rag.ingest import build_chunks_table, create_index
    build_chunks_table(spark,
        volume_path="/Volumes/main/default/pa4/annual_report.pdf",
        chunks_table="main.default.<you>_analyst_chunks")
    create_index()   # reads endpoint/index/source-table names from the environment
"""

from __future__ import annotations

import os
import time

from config import get_settings


def build_chunks_table(spark, volume_path: str, chunks_table: str) -> None:
    """Parse the PDF and chunk it into a Delta table with Change Data Feed on.

    Produces columns: chunk_id, chunk_to_retrieve, chunk_to_embed, source, page.
    A Delta Sync Vector Search index requires Change Data Feed on its source table.

    `ai_parse_document` returns a VARIANT `{document: {elements: [...]}}`; feeding
    that straight into `ai_prep_search` yields `{document: {contents: [...]}}` where
    each chunk already carries `chunk_id`, `chunk_to_retrieve`, `chunk_to_embed`, and
    a `pages` array — so we explode `contents` and project those fields directly.
    """
    # Create the table once, then INSERT OVERWRITE. We deliberately avoid
    # CREATE OR REPLACE TABLE: that assigns a NEW Delta table id on every run,
    # which breaks the Vector Search Delta Sync index (its streaming checkpoint
    # tracks the source table by id and fails with
    # DIFFERENT_DELTA_TABLE_READ_BY_STREAMING_SOURCE). INSERT OVERWRITE keeps the
    # same table identity, so re-ingestion refreshes the data without breaking sync.
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {chunks_table} (
            chunk_id          STRING,
            chunk_to_retrieve STRING,
            chunk_to_embed    STRING,
            source            STRING,
            page              INT
        ) TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """
    )
    spark.sql(
        f"""
        INSERT OVERWRITE TABLE {chunks_table}
        WITH parsed AS (
            SELECT
                element_at(split(path, '/'), -1)               AS source,
                ai_prep_search(ai_parse_document(content))     AS prepped
            FROM READ_FILES('{volume_path}', format => 'binaryFile')
        )
        SELECT
            cast(chunk:chunk_id AS STRING)          AS chunk_id,
            cast(chunk:chunk_to_retrieve AS STRING) AS chunk_to_retrieve,
            cast(chunk:chunk_to_embed AS STRING)    AS chunk_to_embed,
            source,
            cast(element_at(cast(chunk:pages AS ARRAY<VARIANT>), 1):page_id AS INT) + 1 AS page
        FROM parsed
        LATERAL VIEW explode(cast(prepped:document.contents AS ARRAY<VARIANT>)) AS chunk
        WHERE chunk:chunk_to_retrieve IS NOT NULL
          AND length(trim(cast(chunk:chunk_to_retrieve AS STRING))) > 0
        """
    )


def create_index() -> None:
    """Create a STANDARD Vector Search endpoint and a TRIGGERED Delta Sync index.

    Endpoint/index/source-table names come from the environment
    (VECTOR_SEARCH_ENDPOINT, VECTOR_SEARCH_INDEX, SOURCE_TABLE) so the same names
    are used by the retriever (`rag/store.py`) and the deployment env vars.
    """
    from databricks.vector_search.client import VectorSearchClient

    s = get_settings()
    endpoint = s["vs_endpoint"]
    index = s["vs_index"]
    embeddings = s["embeddings"]
    source_table = os.environ.get("SOURCE_TABLE")

    if not endpoint or not index:
        raise OSError("VECTOR_SEARCH_ENDPOINT and VECTOR_SEARCH_INDEX must be set.")
    if not source_table:
        raise OSError("SOURCE_TABLE (the chunks Delta table) must be set in the environment.")

    client = VectorSearchClient()  # authenticates via DATABRICKS_HOST / DATABRICKS_TOKEN

    # 1) STANDARD endpoint (idempotent — ignore "already exists").
    existing = {e["name"] for e in client.list_endpoints().get("endpoints", [])}
    if endpoint not in existing:
        client.create_endpoint(name=endpoint, endpoint_type="STANDARD")
        _wait_endpoint_online(client, endpoint)

    # 2) TRIGGERED Delta Sync index with managed embeddings.
    existing_indexes = {
        i["name"] for i in client.list_indexes(endpoint).get("vector_indexes", [])
    }
    if index not in existing_indexes:
        client.create_delta_sync_index(
            endpoint_name=endpoint,
            index_name=index,
            source_table_name=source_table,
            pipeline_type="TRIGGERED",
            primary_key="chunk_id",
            embedding_source_column="chunk_to_retrieve",
            embedding_model_endpoint_name=embeddings,
        )

    # 3) Kick a sync and wait until the index is READY.
    idx = client.get_index(endpoint_name=endpoint, index_name=index)
    try:
        idx.sync()
    except Exception:  # noqa: BLE001 - a freshly created index may still be provisioning
        pass
    idx.wait_until_ready(verbose=True)
    print(f"Vector Search index READY: {index} (endpoint: {endpoint})")


def _wait_endpoint_online(client, endpoint: str, timeout_s: int = 900) -> None:
    """Poll until the Vector Search endpoint reports ONLINE."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        state = client.get_endpoint(endpoint).get("endpoint_status", {}).get("state", "")
        if state == "ONLINE":
            return
        time.sleep(15)
    raise TimeoutError(f"Vector Search endpoint '{endpoint}' did not come online in time")
