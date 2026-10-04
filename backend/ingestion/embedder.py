from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.config.settings import settings


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    # Loaded once, then reused. The first call is slow (loading ~90 MB into memory).
    return SentenceTransformer(settings.embedding_model_name)


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    model = get_embedding_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,  # unit length, so inner product == cosine similarity
        convert_to_numpy=True,
    )
    return vectors.astype("float32")  # FAISS requires float32


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])  # shape (1, 384)