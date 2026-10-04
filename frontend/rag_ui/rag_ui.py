"""
RAG UI with live pipeline observability panel.

The right-hand "Pipeline Trace" panel connects to /chat/stream (SSE) and
renders each LangGraph node in real-time as an append-only event log,
supporting the rewrite-query retry loop (nodes may appear multiple times).
"""

import json
import os

import httpx
import reflex as rx

API_URL = os.getenv("RAG_API_URL", "http://localhost:8001")


# ── State ────────────────────────────────────────────────────────────────────
class State(rx.State):
    # Documents
    documents: list[dict[str, str]] = []

    # Chat
    messages: list[dict[str, str]] = []
    question: str = ""
    is_uploading: bool = False
    is_chatting: bool = False
    error: str = ""

    # Pipeline trace — append-only event log
    # Each entry: {node, label, icon, status, duration_ms, data_json, seq}
    # status: "running" | "done"
    trace_events: list[dict] = []
    trace_visible: bool = False
    _trace_seq: int = 0        # internal counter for unique keys

    # ── helpers ──────────────────────────────────────────────────────────────
    def set_question(self, value: str):
        self.question = value

    def _reset_trace(self):
        self.trace_events = []
        self.trace_visible = True
        self._trace_seq = 0

    def _upsert_event(self, node: str, label: str, icon: str,
                      status: str, duration_ms: int = 0, data: dict | None = None):
        """Add a new card or update the last card for this node (running→done)."""
        seq = self._trace_seq
        # If last event for this node is "running", update it in-place
        for i in range(len(self.trace_events) - 1, -1, -1):
            ev = self.trace_events[i]
            if ev["node"] == node and ev["status"] == "running":
                updated = {**ev, "status": status, "duration_ms": duration_ms,
                           "data_json": json.dumps(data, indent=2) if data else ev["data_json"]}
                self.trace_events = (
                    self.trace_events[:i] + [updated] + self.trace_events[i + 1:]
                )
                return
        # Otherwise append a new card
        self._trace_seq += 1
        self.trace_events = self.trace_events + [{
            "node": node,
            "label": label,
            "icon": icon,
            "status": status,
            "duration_ms": duration_ms,
            "data_json": json.dumps(data, indent=2) if data else "",
            "seq": self._trace_seq,
        }]

    # ── document actions ─────────────────────────────────────────────────────
    async def load_documents(self):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(f"{API_URL}/documents")
                response.raise_for_status()
            self.documents = [
                {
                    "id": str(d["id"]),
                    "filename": d["filename"],
                    "chunks": str(d["num_chunks"]),
                }
                for d in response.json()["documents"]
            ]
        except httpx.HTTPError as exc:
            self.error = f"Cannot reach the backend: {exc}"

    async def handle_upload(self, files: list[rx.UploadFile]):
        if not files:
            self.error = "Choose a file first."
            return
        self.is_uploading = True
        self.error = ""
        yield
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                for file in files:
                    content = await file.read()
                    response = await client.post(
                        f"{API_URL}/documents/upload",
                        files={"file": (file.filename, content)},
                    )
                    if response.status_code != 201:
                        self.error = response.json().get("detail", response.text)
                        break
        except httpx.HTTPError as exc:
            self.error = f"Upload failed: {exc}"
        self.is_uploading = False
        yield rx.clear_selected_files("upload1")
        yield State.load_documents

    async def delete_document(self, doc_id: str):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.delete(f"{API_URL}/documents/{doc_id}")
                response.raise_for_status()
        except httpx.HTTPError as exc:
            self.error = f"Delete failed: {exc}"
        yield State.load_documents

    # ── streaming chat with live pipeline trace ───────────────────────────────
    async def send_question(self, form_data: dict):
        question = self.question.strip()
        if not question or self.is_chatting:
            return

        self.messages.append({"role": "You", "content": question, "sources": ""})
        self.question = ""
        self.is_chatting = True
        self.error = ""
        self._reset_trace()
        yield

        try:
            async with httpx.AsyncClient(timeout=300) as client:
                async with client.stream(
                    "POST",
                    f"{API_URL}/chat/stream",
                    json={"question": question},
                    headers={"Accept": "text/event-stream"},
                ) as resp:
                    resp.raise_for_status()

                    async for raw_line in resp.aiter_lines():
                        if not raw_line.startswith("data: "):
                            continue
                        event = json.loads(raw_line[6:])
                        etype = event.get("type")

                        if etype == "node_start":
                            self._upsert_event(
                                node=event["node"],
                                label=event["label"],
                                icon=event["icon"],
                                status="running",
                            )
                            yield

                        elif etype == "node_done":
                            self._upsert_event(
                                node=event["node"],
                                label=event["label"],
                                icon=event["icon"],
                                status="done",
                                duration_ms=event.get("duration_ms", 0),
                                data=event.get("data"),
                            )
                            yield

                        elif etype == "final":
                            sources = event.get("sources", [])
                            lines = []
                            for s in sources:
                                page = f" (p.{s['page_number']})" if s.get("page_number") else ""
                                score = f" [{s['score']:.3f}]" if s.get("score") is not None else ""
                                lines.append(f"• {s['filename']}{page}{score}")
                            self.messages.append(
                                {
                                    "role": "Assistant",
                                    "content": event.get("answer", ""),
                                    "sources": "\n".join(lines),
                                }
                            )
                            yield

                        elif etype == "error":
                            self.error = event.get("message", "Unknown error")
                            yield

        except httpx.HTTPError as exc:
            self.error = f"Chat failed: {exc}"

        self.is_chatting = False
        yield


# ── UI components ─────────────────────────────────────────────────────────────

def document_row(doc: dict) -> rx.Component:
    return rx.hstack(
        rx.text(doc["filename"], weight="medium", size="2"),
        rx.badge(doc["chunks"] + " chunks", variant="soft", color_scheme="blue"),
        rx.spacer(),
        rx.button(
            "Delete",
            on_click=State.delete_document(doc["id"]),
            size="1",
            color_scheme="red",
            variant="soft",
        ),
        width="100%",
        align="center",
        padding="0.4em 0",
    )


def message_bubble(msg: dict) -> rx.Component:
    is_user = msg["role"] == "You"
    return rx.box(
        rx.text(
            msg["role"],
            weight="bold",
            size="1",
            color=rx.cond(is_user, "var(--accent-9)", "var(--gray-11)"),
        ),
        rx.text(msg["content"], white_space="pre-wrap", size="2"),
        rx.cond(
            msg["sources"] != "",
            rx.box(
                rx.text("Sources", size="1", weight="bold", color="var(--gray-10)", margin_bottom="2px"),
                rx.text(
                    msg["sources"],
                    size="1",
                    color="var(--gray-10)",
                    white_space="pre-wrap",
                    font_family="monospace",
                ),
                padding="0.4em 0.6em",
                background="var(--gray-2)",
                border_radius="6px",
                margin_top="4px",
            ),
        ),
        padding="0.75em 1em",
        border_radius="10px",
        background=rx.cond(is_user, "var(--accent-3)", "var(--gray-3)"),
        border_left=rx.cond(is_user, "3px solid var(--accent-9)", "3px solid var(--gray-7)"),
        width="100%",
    )


def status_dot(status: str) -> rx.Component:
    color = rx.match(
        status,
        ("running", "var(--amber-9)"),
        ("done",    "var(--green-9)"),
        "var(--gray-4)",
    )
    return rx.box(
        width="10px",
        height="10px",
        border_radius="50%",
        background=color,
        flex_shrink="0",
        animation=rx.cond(status == "running", "pulse 1s infinite", "none"),
    )


def trace_event_card(ev: dict) -> rx.Component:
    status = ev["status"]
    is_done = status == "done"
    is_running = status == "running"

    # Color scheme based on node type
    node_color = rx.match(
        ev["node"],
        ("rewrite_query",    "orange"),
        ("validate_context", "green"),
        ("no_context",       "red"),
        ("generate",         "purple"),
        "blue",
    )

    border_color = rx.match(
        status,
        ("running", "var(--amber-7)"),
        ("done",    "var(--green-6)"),
        "var(--gray-3)",
    )
    bg = rx.match(
        status,
        ("running", "var(--amber-2)"),
        ("done",    "var(--gray-1)"),
        "var(--gray-1)",
    )

    return rx.box(
        rx.hstack(
            status_dot(status),
            rx.box(
                rx.text(ev["icon"], size="3"),
                width="28px",
                text_align="center",
            ),
            rx.vstack(
                rx.hstack(
                    rx.text(ev["label"], weight="bold", size="2"),
                    rx.cond(
                        is_done,
                        rx.badge(
                            ev["duration_ms"].to_string() + " ms",
                            color_scheme="green",
                            variant="soft",
                            size="1",
                        ),
                    ),
                    rx.cond(
                        is_running,
                        rx.badge("running…", color_scheme="amber", variant="soft", size="1"),
                    ),
                    align="center",
                    spacing="2",
                    flex_wrap="wrap",
                ),
                rx.cond(
                    is_done & (ev["data_json"] != ""),
                    rx.scroll_area(
                        rx.code_block(
                            ev["data_json"],
                            language="json",
                            font_size="10px",
                            background="transparent",
                            padding="0",
                        ),
                        max_height="150px",
                        width="100%",
                    ),
                ),
                spacing="1",
                width="100%",
            ),
            align="start",
            spacing="2",
            width="100%",
        ),
        padding="0.6em 0.8em",
        border_radius="10px",
        border=f"1px solid {border_color}",
        background=bg,
        width="100%",
        transition="all 0.2s ease",
    )


def pipeline_panel() -> rx.Component:
    return rx.cond(
        State.trace_visible,
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.heading("🔬 Pipeline Trace", size="3"),
                    rx.spacer(),
                    rx.vstack(
                        rx.text("Live LangGraph execution", size="1", color="var(--gray-10)"),
                        rx.text(
                            "Nodes repeat on query rewrites",
                            size="1",
                            color="var(--orange-10)",
                        ),
                        spacing="0",
                        align="end",
                    ),
                    width="100%",
                    align="center",
                ),
                rx.divider(),
                rx.scroll_area(
                    rx.vstack(
                        rx.foreach(State.trace_events, trace_event_card),
                        spacing="2",
                        width="100%",
                    ),
                    max_height="600px",
                    width="100%",
                ),
                spacing="3",
                width="100%",
            ),
            padding="1.2em",
            border_radius="14px",
            border="1px solid var(--gray-4)",
            background="var(--gray-1)",
            width="100%",
        ),
    )


def index() -> rx.Component:
    return rx.container(
        rx.html(
            "<style>"
            "@keyframes pulse {"
            "  0%,100% { opacity:1; transform:scale(1); }"
            "  50%      { opacity:0.4; transform:scale(1.5); }"
            "}"
            "</style>"
        ),
        rx.vstack(
            # ── header ──
            rx.heading("🗂 Local RAG Assistant", size="7"),
            rx.text(
                "Chat with your documents. The Pipeline Trace shows each LangGraph node live — "
                "including query rewrites and retries.",
                size="2",
                color="var(--gray-10)",
            ),
            rx.cond(State.error != "", rx.callout(State.error, color_scheme="red", width="100%")),

            # ── documents ──
            rx.box(
                rx.vstack(
                    rx.heading("📄 Documents", size="4"),
                    rx.upload(
                        rx.text("Drop files here or click to select (PDF, DOCX, TXT, MD)", size="2"),
                        id="upload1",
                        accept={
                            "application/pdf": [".pdf"],
                            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
                            "text/plain": [".txt"],
                            "text/markdown": [".md"],
                        },
                        max_files=5,
                        border="2px dashed var(--gray-5)",
                        padding="1.5em",
                        width="100%",
                        border_radius="10px",
                    ),
                    rx.foreach(rx.selected_files("upload1"), rx.text),
                    rx.button(
                        "⬆ Upload",
                        on_click=State.handle_upload(rx.upload_files(upload_id="upload1")),
                        loading=State.is_uploading,
                        color_scheme="blue",
                    ),
                    rx.cond(
                        State.documents.length() == 0,
                        rx.text("No documents yet.", color="var(--gray-10)", size="2"),
                        rx.vstack(rx.foreach(State.documents, document_row), width="100%", spacing="1"),
                    ),
                    spacing="3",
                    width="100%",
                ),
                padding="1.2em",
                border_radius="14px",
                border="1px solid var(--gray-4)",
                background="var(--gray-1)",
                width="100%",
            ),

            # ── chat + trace ──
            rx.flex(
                # Chat column
                rx.box(
                    rx.vstack(
                        rx.heading("💬 Chat", size="4"),
                        rx.scroll_area(
                            rx.vstack(
                                rx.foreach(State.messages, message_bubble),
                                spacing="3",
                                width="100%",
                            ),
                            max_height="500px",
                            width="100%",
                        ),
                        rx.cond(
                            State.is_chatting,
                            rx.hstack(rx.spinner(), rx.text("Thinking…", size="2"), align="center"),
                        ),
                        rx.form(
                            rx.hstack(
                                rx.input(
                                    name="question",
                                    value=State.question,
                                    on_change=State.set_question,
                                    placeholder="Ask about your documents…",
                                    width="100%",
                                ),
                                rx.button("Send →", type="submit", loading=State.is_chatting, color_scheme="blue"),
                                width="100%",
                            ),
                            on_submit=State.send_question,
                            reset_on_submit=False,
                            width="100%",
                        ),
                        spacing="4",
                        width="100%",
                    ),
                    padding="1.2em",
                    border_radius="14px",
                    border="1px solid var(--gray-4)",
                    background="var(--gray-1)",
                    flex="1",
                    min_width="300px",
                ),

                # Pipeline trace column
                rx.box(
                    pipeline_panel(),
                    width="400px",
                    flex_shrink="0",
                ),

                gap="1.2em",
                flex_wrap="wrap",
                width="100%",
                align="start",
            ),

            spacing="5",
            width="100%",
            padding_y="2em",
        ),
        max_width="1300px",
    )


app = rx.App()
app.add_page(index, route="/", on_load=State.load_documents)