from langgraph.graph import END, START, StateGraph

from backend.graph.nodes import (
    generate,
    no_context,
    retrieve,
    rewrite_query,
    route_after_validation,
    understand_query,
    validate_context,
)
from backend.graph.state import RAGState


def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("understand_query", understand_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("validate_context", validate_context)
    graph.add_node("rewrite_query", rewrite_query)   # ← NEW
    graph.add_node("generate", generate)
    graph.add_node("no_context", no_context)

    # Happy path + retry loop
    graph.add_edge(START, "understand_query")
    graph.add_edge("understand_query", "retrieve")
    graph.add_edge("retrieve", "validate_context")
    graph.add_conditional_edges(
        "validate_context",
        route_after_validation,
        {
            "generate":      "generate",
            "rewrite_query": "rewrite_query",   # retry branch
            "no_context":    "no_context",
        },
    )
    graph.add_edge("rewrite_query", "retrieve")     # loop back
    graph.add_edge("generate", END)
    graph.add_edge("no_context", END)

    return graph.compile()


rag_graph = build_graph()