from backend.ingestion.chunker import chunk_pages
from backend.ingestion.cleaner import clean_text
from backend.ingestion.loader import PageText
from backend.retrieval.bm25_store import BM25Store
from backend.retrieval.retriever import reciprocal_rank_fusion


def test_cleaner_fixes_pdf_damage():
    assert clean_text("inter-\nnational  trade\nagreement") == "international trade agreement"


def test_chunker_keeps_page_numbers():
    pages = [PageText(1, "a. " * 600), PageText(2, "b. " * 600)]
    chunks = chunk_pages(pages)
    assert {c.page_number for c in chunks} == {1, 2}
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_rrf_rewards_agreement():
    scores = reciprocal_rank_fusion([[1, 2, 3], [3, 4, 1]], k=60)
    assert max(scores, key=scores.get) in {1, 3}


def test_bm25_finds_exact_keyword(tmp_path):
    store = BM25Store(tmp_path / "bm25.pkl")
    store.rebuild([
        (10, "refund policy for annual plans"),
        (11, "error ERR_4021 gateway timeout"),
        (12, "office opening hours"),
    ])
    assert store.search("ERR_4021", 3)[0][0] == 11