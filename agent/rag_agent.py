"""RAG agent node (Task 1.4) — retrieves from Databricks Vector Search.

`make_rag_agent(retriever, llm)` retrieves the top-k chunks for the *current*
step, formats them with source citations, and asks the LLM to extract a single
cited fact (or "not found in documents" when retrieval is empty / unhelpful).
The same `retriever` object is used locally and inside the deployed endpoint, so
this code path never changes between environments.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import RAG_EXTRACT_PROMPT
from agent.state import AnalystState

NOT_FOUND = "not found in documents"


def _citation(metadata: dict) -> str:
    """Build a [source: file, p.N] tag from a document's metadata."""
    source = metadata.get("source") or metadata.get("chunk_id") or "unknown"
    page = metadata.get("page")
    if page in (None, ""):
        return f"[source: {source}]"
    # Vector Search returns numeric columns as floats; render whole pages as ints.
    if isinstance(page, float) and page.is_integer():
        page = int(page)
    return f"[source: {source}, p.{page}]"


def format_docs(docs) -> str:
    """Render retrieved documents as citation-tagged context blocks."""
    blocks = []
    for doc in docs:
        content = getattr(doc, "page_content", None)
        metadata = getattr(doc, "metadata", None)
        if content is None and isinstance(doc, dict):
            content = doc.get("page_content", "")
            metadata = doc.get("metadata", {})
        content = (content or "").strip()
        blocks.append(f"{content} {_citation(metadata or {})}")
    return "\n\n".join(blocks)


def make_rag_agent(retriever, llm):
    def rag_agent(state: AnalystState) -> dict:
        idx = state.get("current_step_index", 0)
        plan = state.get("plan", [])
        step = plan[idx] if idx < len(plan) else ""

        docs = retriever.invoke(step)
        if not docs:
            fact = NOT_FOUND
        else:
            context = format_docs(docs)
            response = llm.invoke(
                [
                    SystemMessage(content=RAG_EXTRACT_PROMPT),
                    HumanMessage(content=f"CURRENT step: {step}\n\nCONTEXT:\n{context}"),
                ]
            )
            fact = (getattr(response, "content", "") or "").strip() or NOT_FOUND

        return {
            "step_results": state.get("step_results", []) + [fact],
            "current_step_index": idx + 1,
        }

    return rag_agent
