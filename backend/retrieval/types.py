from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    filename: str
    page_number: int | None
    chunk_index: int
    text: str
    faiss_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None