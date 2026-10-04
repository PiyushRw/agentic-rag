from langsmith import traceable
from sqlalchemy import select

from backend.config.settings import settings
from backend.database.connection import get_session
from backend.database.models import Chunk, Document
from backend.ingestion.cleaner import clean_text
from backend.ingestion.embedder import embed_query
from backend.retrieval.indexes import index_manager
from backend.retrieval.reranker import rerank
from backend.retrieval.types import RetrievedChunk


def reciprocal_rank_fusion(rankings: list[list[int]], k: int) -> dict[int, float]:
    """Each list votes 1/(k+rank) for every ID it contains. Agreement between lists wins."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


def fetch_chunks(chunk_ids: list[int]) -> dict[int, RetrievedChunk]:
    """Look up IDs in SQLite. IDs that no longer exist are silently dropped (stale-index protection)."""
    if not chunk_ids:
        return {}
    with get_session() as session:
        rows = session.execute(
            select(Chunk, Document.filename)
            .join(Document, Chunk.document_id == Document.id)
            .where(Chunk.id.in_(chunk_ids))
        ).all()
    return {
        chunk.id: RetrievedChunk(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            filename=filename,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
        )
        for chunk, filename in rows
    }


@traceable(name="faiss_search", run_type="retriever")
def semantic_search(query: str, k: int) -> list[tuple[int, float]]:
    return index_manager.semantic_search(embed_query(query), k)


@traceable(name="bm25_search", run_type="retriever")
def keyword_search(query: str, k: int) -> list[tuple[int, float]]:
    return index_manager.keyword_search(query, k)


@traceable(name="hybrid_search", run_type="retriever")
def hybrid_search(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    query = clean_text(query)
    if not query or index_manager.is_empty:
        return []

    faiss_hits = semantic_search(query, settings.faiss_top_k)
    bm25_hits = keyword_search(query, settings.bm25_top_k)

    fused = reciprocal_rank_fusion(
        [[i for i, _ in faiss_hits], [i for i, _ in bm25_hits]], settings.rrf_k
    )
    top_ids = sorted(fused, key=fused.get, reverse=True)[: settings.candidate_limit]

    faiss_scores, bm25_scores = dict(faiss_hits), dict(bm25_hits)
    by_id = fetch_chunks(top_ids)

    candidates = []
    for chunk_id in top_ids:
        chunk = by_id.get(chunk_id)
        if chunk is None:
            continue
        chunk.faiss_score = faiss_scores.get(chunk_id)
        chunk.bm25_score = bm25_scores.get(chunk_id)
        chunk.rrf_score = fused[chunk_id]
        candidates.append(chunk)

    return rerank(query, candidates, top_k or settings.rerank_top_k)