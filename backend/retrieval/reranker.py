from functools import lru_cache

from langsmith import traceable
from sentence_transformers import CrossEncoder

from backend.config.settings import settings
from backend.retrieval.types import RetrievedChunk


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(settings.reranker_model_name)


@traceable(name="rerank", run_type="chain")
def rerank(query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    if not candidates:
        return []
    scores = get_reranker().predict([(query, c.text) for c in candidates])
    for chunk, score in zip(candidates, scores):
        chunk.rerank_score = float(score)
    candidates.sort(key=lambda c: c.rerank_score, reverse=True)
    return candidates[:top_k]
    