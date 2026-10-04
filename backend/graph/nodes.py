from langchain_core.messages import HumanMessage, SystemMessage

from backend.config.settings import settings
from backend.graph.state import RAGState
from backend.ingestion.cleaner import clean_text
from backend.retrieval.indexes import index_manager
from backend.retrieval.retriever import hybrid_search
from backend.services.llm import get_llm

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered context passages provided. "
    "If the passages do not contain the answer, say you could not find it in the documents. "
    "Be concise. Cite the passages you used like [1] or [2]."
)

REWRITE_SYSTEM_PROMPT = (
    "You are a search query optimizer. "
    "Given a user question that failed to find relevant documents, "
    "rewrite it into a better search query. "
    "Rules:\n"
    "- Make it more specific and keyword-rich\n"
    "- Use synonyms or alternative phrasings\n"
    "- Keep it concise (under 20 words)\n"
    "- Output ONLY the rewritten query, nothing else"
)


def understand_query(state: RAGState) -> dict:
    return {
        "cleaned_question": clean_text(state["question"]),
        "retry_count": 0,
    }


def rewrite_query(state: RAGState) -> dict:
    """LLM rewrites the current query to improve retrieval on retry."""
    current_q = state.get("rewritten_question") or state.get("cleaned_question", "")
    retry_count = state.get("retry_count", 0)

    prompt = (
        f"Attempt #{retry_count + 1}. Original question: {state['question']}\n"
        f"Previous search query: {current_q}\n"
        f"The search returned no relevant results. Rewrite the query."
    )
    response = get_llm().invoke(
        [SystemMessage(content=REWRITE_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    new_query = _to_text(response.content).strip()
    return {
        "rewritten_question": new_query,
        "cleaned_question": new_query,   # retriever uses cleaned_question
        "retry_count": retry_count + 1,
    }


def retrieve(state: RAGState) -> dict:
    return {"chunks": hybrid_search(state["cleaned_question"])}


def validate_context(state: RAGState) -> dict:
    chunks = state.get("chunks", [])
    ok = False
    if chunks and chunks[0].rerank_score is not None:
        ok = chunks[0].rerank_score >= settings.min_rerank_score
    return {"context_ok": ok}


def route_after_validation(state: RAGState) -> str:
    """
    Conditional edge:
      - good context  → generate
      - bad context, retries left → rewrite_query
      - bad context, retries exhausted → no_context
    """
    if state.get("context_ok"):
        return "generate"
    if state.get("retry_count", 0) < 3:
        return "rewrite_query"
    return "no_context"


def no_context(state: RAGState) -> dict:
    if index_manager.is_empty:
        answer = "No documents have been uploaded yet. Please upload a document first."
    else:
        answer = (
            "I couldn't find relevant information about that in your documents "
            "after rewriting the query 3 times."
        )
    return {"answer": answer, "sources": []}


def _format_context(chunks) -> str:
    blocks = []
    for n, c in enumerate(chunks, start=1):
        page = f", page {c.page_number}" if c.page_number else ""
        blocks.append(f"[{n}] (source: {c.filename}{page})\n{c.text}")
    return "\n\n".join(blocks)

def _to_text(content) -> str:
    """Normalize LLM output: some providers return a list of blocks instead of a string."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)

def generate(state: RAGState) -> dict:
    chunks = state["chunks"]
    user_prompt = f"Context:\n{_format_context(chunks)}\n\nQuestion: {state['cleaned_question']}"
    response = get_llm().invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)]
    )
    sources = [
        {
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "filename": c.filename,
            "page_number": c.page_number,
            "snippet": c.text[:200],
            "score": c.rerank_score,
        }
        for c in chunks
    ]
    return {"answer": _to_text(response.content), "sources": sources}