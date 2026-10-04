import logging

from fastapi import APIRouter, HTTPException
from langsmith import traceable

from backend.graph.builder import rag_graph
from backend.schemas.api import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


@traceable(name="chat_request", run_type="chain")  # parent trace: graph + retrieval + LLM nest inside
def run_chat(question: str) -> dict:
    result = rag_graph.invoke({"question": question})
    return {"answer": result["answer"], "sources": result.get("sources", [])}


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        return run_chat(request.question)
    except Exception:
        logger.exception("Chat failed")
        raise HTTPException(
            status_code=503,
            detail="The answer could not be generated. Is the LLM service (Ollama or xAI) running?",    
        )