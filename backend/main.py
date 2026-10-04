import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.chat import router as chat_router
from backend.api.chat_stream import router as chat_stream_router
from backend.api.documents import router as documents_router
from backend.database.connection import init_db
from backend.ingestion.embedder import get_embedding_model
from backend.retrieval.indexes import index_manager
from backend.retrieval.reranker import get_reranker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()                                # create tables if missing
    status = index_manager.startup()         # load persisted indexes, or repair from SQLite
    logger.info("Indexes %s", status)
    get_embedding_model()                    # warm up so the first request isn't slow
    get_reranker()
    yield


app = FastAPI(title="Agentic RAG API", lifespan=lifespan)

# Allow the Reflex frontend (port 3000) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://0.0.0.0:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)
app.include_router(chat_router)
app.include_router(chat_stream_router)


@app.get("/health")
def health():
    return {"status": "ok"}