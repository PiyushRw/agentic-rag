from typing import TypedDict

from backend.retrieval.types import RetrievedChunk


class RAGState(TypedDict, total=False):
    question: str              # raw input
    cleaned_question: str      # after "understand_query" / "rewrite_query"
    chunks: list[RetrievedChunk]
    context_ok: bool           # decision made by "validate_context"
    retry_count: int           # how many rewrite+retrieve cycles have run (max 3)
    rewritten_question: str    # LLM-rewritten query for the current retry
    answer: str
    sources: list[dict]