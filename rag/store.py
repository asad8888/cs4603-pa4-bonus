"""Vector Search retriever factory (Task 1.4 support / rag/store.py).

`get_retriever()` returns a LangChain retriever over the Databricks Vector Search
index built by `ingest.py`. Because the index is a managed Databricks service,
this *exact* retriever runs unchanged both locally and inside the deployed serving
container — there is no separate embedding path for deployment.
"""

from __future__ import annotations

from config import get_settings

TEXT_COLUMN = "chunk_to_retrieve"
CITATION_COLUMNS = ["chunk_id", "source", "page"]


def _require_vs_settings() -> dict:
    s = get_settings()
    if not s["vs_endpoint"] or not s["vs_index"]:
        raise OSError(
            "Vector Search is not configured. Set VECTOR_SEARCH_ENDPOINT and "
            "VECTOR_SEARCH_INDEX in your .env (local) or the endpoint "
            "environment_vars (deployed)."
        )
    return s


def get_vector_store():
    """Return a `DatabricksVectorSearch` handle over the managed index.

    The index uses managed embeddings (Databricks computes them from the source
    column `chunk_to_retrieve`). For such indexes the client auto-detects the text
    column from the index config, so we must NOT pass `text_column` — passing it
    raises. We only request the citation `columns` to be returned with each hit.
    """
    from databricks_langchain import DatabricksVectorSearch

    s = _require_vs_settings()
    return DatabricksVectorSearch(
        endpoint=s["vs_endpoint"],
        index_name=s["vs_index"],
        columns=CITATION_COLUMNS + [TEXT_COLUMN],
    )


def get_retriever(k: int = 4):
    """Return a top-k retriever over the Vector Search index."""
    return get_vector_store().as_retriever(search_kwargs={"k": k})
