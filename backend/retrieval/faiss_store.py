from pathlib import Path

import faiss
import numpy as np


class FaissStore:
    """Wraps one FAISS index. Vectors are stored under our SQLite chunk IDs."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.index = None  # created lazily on first add, so no empty "final index" exists

    def load(self) -> bool:
        if not self.path.exists():
            return False
        self.index = faiss.read_index(str(self.path))
        return True

    def save(self) -> None:
        if self.index is None:
            self.path.unlink(missing_ok=True)
            return
        faiss.write_index(self.index, str(self.path))

    def reset(self) -> None:
        self.index = None
        self.path.unlink(missing_ok=True)

    @property
    def count(self) -> int:
        return 0 if self.index is None else self.index.ntotal

    def ids(self) -> set[int]:
        if self.index is None:
            return set()
        id_map = faiss.downcast_index(self.index).id_map
        return set(faiss.vector_to_array(id_map).tolist())

    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        if self.index is None:
            # Inner product on normalized vectors = cosine similarity; IDMap lets us use our own IDs
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(vectors.shape[1]))
        self.index.add_with_ids(vectors, np.asarray(ids, dtype="int64"))

    def remove(self, ids: list[int]) -> None:
        if self.index is None:
            return
        self.index.remove_ids(np.asarray(ids, dtype="int64"))
        if self.index.ntotal == 0:
            self.reset()

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        k = min(k, self.index.ntotal)
        scores, ids = self.index.search(query_vector, k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]