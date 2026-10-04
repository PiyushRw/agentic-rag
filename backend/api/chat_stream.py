"""
Streaming chat endpoint – emits one Server-Sent Event per LangGraph node so
the frontend can show a live pipeline trace panel.

Event schema (JSON):
  { "type": "node_start",    "node": "<name>",  "ts": <epoch_ms> }
  { "type": "node_done",     "node": "<name>",  "ts": <epoch_ms>,
    "duration_ms": <int>,    "data": { ... node-specific payload ... } }
  { "type": "final",         "answer": "...",   "sources": [...] }
  { "type": "error",         "message": "..." }
"""

import json
import logging
import time
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from backend.graph.builder import rag_graph
from backend.schemas.api import ChatRequest

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])

# Human-readable labels & icons sent to the frontend
NODE_META = {
    "understand_query": {"label": "Understand Query",  "icon": "🔍"},
    "retrieve":          {"label": "Hybrid Retrieval", "icon": "📚"},
    "validate_context":  {"label": "Validate Context", "icon": "✅"},
    "rewrite_query":     {"label": "Rewrite Query",    "icon": "✏️"},
    "generate":          {"label": "Generate Answer",  "icon": "🤖"},
    "no_context":        {"label": "No Context",       "icon": "⚠️"},
}


def _node_payload(node_name: str, state_update: dict) -> dict:
    """Extract a small, serialisable summary of what the node produced."""
    payload: dict = {}

    if node_name == "understand_query":
        payload["cleaned_question"] = state_update.get("cleaned_question", "")

    elif node_name == "rewrite_query":
        payload["rewritten_question"] = state_update.get("rewritten_question", "")
        payload["retry_attempt"] = state_update.get("retry_count", 1)

    elif node_name == "retrieve":
        chunks = state_update.get("chunks", [])
        payload["num_chunks"] = len(chunks)
        payload["top_chunks"] = [
            {
                "filename": c.filename,
                "page": c.page_number,
                "score": round(c.rerank_score, 4) if c.rerank_score is not None else None,
                "snippet": c.text[:120] + ("…" if len(c.text) > 120 else ""),
            }
            for c in chunks[:4]
        ]

    elif node_name == "validate_context":
        payload["context_ok"] = state_update.get("context_ok", False)

    elif node_name == "generate":
        answer = state_update.get("answer", "")
        payload["answer_preview"] = answer[:200] + ("…" if len(answer) > 200 else "")
        sources = state_update.get("sources", [])
        payload["num_sources"] = len(sources)

    elif node_name == "no_context":
        payload["answer"] = state_update.get("answer", "")

    return payload


async def _stream_pipeline(question: str) -> AsyncGenerator[str, None]:
    def _sse(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"

    start_times: dict[str, float] = {}
    final_state: dict = {}

    try:
        # stream_mode="updates" yields {node_name: state_update} dicts (one per step)
        for chunk in rag_graph.stream(
            {"question": question},
            stream_mode="updates",
        ):
            for node_name, state_update in chunk.items():
                # ── node_start ──────────────────────────────────────────────
                start_times[node_name] = time.monotonic()
                meta = NODE_META.get(node_name, {"label": node_name, "icon": "⚙️"})
                yield _sse(
                    {
                        "type": "node_start",
                        "node": node_name,
                        "label": meta["label"],
                        "icon": meta["icon"],
                        "ts": int(time.time() * 1000),
                    }
                )

                # ── node_done ───────────────────────────────────────────────
                duration_ms = int((time.monotonic() - start_times[node_name]) * 1000)
                payload = _node_payload(node_name, state_update)
                yield _sse(
                    {
                        "type": "node_done",
                        "node": node_name,
                        "label": meta["label"],
                        "icon": meta["icon"],
                        "ts": int(time.time() * 1000),
                        "duration_ms": duration_ms,
                        "data": payload,
                    }
                )

                # Accumulate state so we can emit the final event
                final_state.update(state_update)


        # ── final answer ─────────────────────────────────────────────────
        yield _sse(
            {
                "type": "final",
                "answer": final_state.get("answer", ""),
                "sources": final_state.get("sources", []),
            }
        )

    except Exception as exc:
        logger.exception("Stream pipeline failed")
        yield _sse({"type": "error", "message": str(exc)})


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(
        _stream_pipeline(request.question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
