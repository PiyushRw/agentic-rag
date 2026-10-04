import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Store:
    """Keyword index. rank-bm25 cannot add documents, so we rebuild it (fast) on every change."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.ids: list[int] = []
        self.tokens: list[list[str]] = []
        self.bm25: BM25Okapi | None = None

    def _build(self) -> None:
        self.bm25 = BM25Okapi(self.tokens) if self.tokens else None

    def rebuild(self, chunks: list[tuple[int, str]]) -> None:
        self.ids = [chunk_id for chunk_id, _ in chunks]
        self.tokens = [tokenize(text) for _, text in chunks]
        self._build()
        self.save()

    def save(self) -> None:
        if not self.ids:
            self.path.unlink(missing_ok=True)
            return
        with self.path.open("wb") as f:
            pickle.dump({"ids": self.ids, "tokens": self.tokens}, f)

    def load(self) -> bool:
        if not self.path.exists():
            return False
        with self.path.open("rb") as f:  # only ever our own file
            data = pickle.load(f)
        self.ids, self.tokens = data["ids"], data["tokens"]
        self._build()
        return True

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        if self.bm25 is None:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        order = np.argsort(scores)[::-1][:k]
        return [(self.ids[i], float(scores[i])) for i in order if scores[i] > 0]