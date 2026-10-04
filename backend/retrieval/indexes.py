import logging
import threading

import numpy as np
from sqlalchemy import select

from backend.config.settings import settings
from backend.database.connection import get_session
from backend.database.models import Chunk
from backend.ingestion.embedder import embed_texts
from backend.retrieval.bm25_store import BM25Store
from backend.retrieval.faiss_store import FaissStore

logger = logging.getLogger(__name__)


class IndexManager:
    def __init__(self) -> None:
        self.faiss = FaissStore(settings.indexes_dir / "faiss.index")
        self.bm25 = BM25Store(settings.indexes_dir / "bm25.pkl")
        self._lock = threading.RLock()

    # ---- SQLite (source of truth) ----
    @staticmethod
    def _db_chunks() -> list[tuple[int, str]]:
        with get_session() as session:
            rows = session.execute(select(Chunk.id, Chunk.text).order_by(Chunk.id)).all()
        return [(row.id, row.text) for row in rows]

    # ---- lifecycle ----
    def startup(self) -> str:
        """Load persisted indexes; if they disagree with SQLite, rebuild them from SQLite."""
        with self._lock:
            self.faiss.load()
            self.bm25.load()
            db_ids = {chunk_id for chunk_id, _ in self._db_chunks()}
            if self.faiss.ids() == db_ids and set(self.bm25.ids) == db_ids:
                return "loaded"
            logger.warning("Indexes out of sync with SQLite; rebuilding.")
            self.rebuild_from_db()
            return "rebuilt"

    def rebuild_from_db(self) -> None:
        with self._lock:
            rows = self._db_chunks()
            self.faiss.reset()
            if rows:
                vectors = embed_texts([text for _, text in rows])
                self.faiss.add([chunk_id for chunk_id, _ in rows], vectors)
                self.faiss.save()
            self.bm25.rebuild(rows)

    # ---- changes ----
    def add_chunks(self, ids: list[int], vectors: np.ndarray) -> None:
        """Call AFTER the chunks are committed to SQLite."""
        with self._lock:
            self.faiss.add(ids, vectors)
            self.faiss.save()
            self.bm25.rebuild(self._db_chunks())

    def remove_chunks(self, ids: list[int]) -> None:
        """Call AFTER the chunks are deleted from SQLite."""
        with self._lock:
            self.faiss.remove(ids)
            self.faiss.save()
            self.bm25.rebuild(self._db_chunks())

    # ---- queries ----
    @property
    def is_empty(self) -> bool:
        return self.faiss.count == 0

    def semantic_search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        with self._lock:
            return self.faiss.search(query_vector, k)

    def keyword_search(self, query: str, k: int) -> list[tuple[int, float]]:
        with self._lock:
            return self.bm25.search(query, k)


index_manager = IndexManager()