from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]

# Exports every .env entry (including LANGSMITH_*) to os.environ so LangChain/LangGraph can see them.
load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- Paths ----------
    documents_dir: Path = BASE_DIR / "data" / "documents"
    indexes_dir: Path = BASE_DIR / "data" / "indexes"
    sqlite_path: Path = BASE_DIR / "data" / "rag.db"

    # ---------- Local embedding + reranker models ----------
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ---------- LLM provider ("gemini" or "ollama") ----------
    llm_provider: str = "gemini"

    # Gemini (Google AI Studio free tier)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"

    # Ollama (local fallback)
    ollama_model: str = "llama3.2"
    ollama_base_url: str = "http://localhost:11434"

    # ---------- Chunking (characters) ----------
    chunk_size: int = 800
    chunk_overlap: int = 100

    # ---------- Retrieval ----------
    faiss_top_k: int = 20
    bm25_top_k: int = 20
    candidate_limit: int = 20           # fused candidates sent to the reranker
    rerank_top_k: int = 4               # chunks that reach the LLM
    rrf_k: int = 60                     # standard constant for reciprocal rank fusion
    min_rerank_score: float = -2.0      # below this, context is treated as "not relevant"

    # ---------- Uploads ----------
    max_upload_mb: int = 25

    def ensure_dirs(self) -> None:
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.indexes_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()