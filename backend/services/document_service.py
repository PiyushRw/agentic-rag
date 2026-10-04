import uuid
from pathlib import Path

from sqlalchemy import select

from backend.config.settings import settings
from backend.database.connection import get_session
from backend.database.models import Chunk, Document
from backend.ingestion.chunker import chunk_pages
from backend.ingestion.cleaner import clean_pages
from backend.ingestion.embedder import embed_texts
from backend.ingestion.loader import (
    SUPPORTED_EXTENSIONS,
    DocumentLoadError,
    compute_file_hash,
    load_document,
)
from backend.retrieval.indexes import index_manager


class DuplicateDocumentError(Exception):
    pass


class DocumentNotFoundError(Exception):
    pass


def upload_document(filename: str, content: bytes) -> Document:
    safe_name = Path(filename).name  # strips any folder parts (path-traversal protection)
    extension = Path(safe_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise DocumentLoadError(f"Unsupported file type '{extension}'. Supported: {supported}")

    settings.ensure_dirs()
    temp_path = settings.documents_dir / f"_upload_{uuid.uuid4().hex}{extension}"
    temp_path.write_bytes(content)

    # 1. Do all expensive/risky work BEFORE touching the database
    try:
        content_hash = compute_file_hash(temp_path)
        with get_session() as session:
            existing = session.scalar(select(Document).where(Document.content_hash == content_hash))
            if existing:
                raise DuplicateDocumentError(f"'{existing.filename}' has already been uploaded.")

        chunks = chunk_pages(clean_pages(load_document(temp_path)))
        if not chunks:
            raise DocumentLoadError("No usable text found after cleaning.")
        vectors = embed_texts([c.text for c in chunks])
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    final_path = settings.documents_dir / f"{content_hash[:12]}_{safe_name}"
    temp_path.replace(final_path)

    # 2. Commit to SQLite (source of truth); chunk IDs are assigned here
    try:
        with get_session() as session:
            doc = Document(
                filename=safe_name,
                stored_path=str(final_path),
                content_hash=content_hash,
                num_chunks=len(chunks),
            )
            doc.chunks = [
                Chunk(chunk_index=c.chunk_index, page_number=c.page_number, text=c.text)
                for c in chunks
            ]
            session.add(doc)
            session.flush()
            chunk_ids = [c.id for c in doc.chunks]  # same order as `vectors`
    except Exception:
        final_path.unlink(missing_ok=True)
        raise

    # 3. Update the indexes; if that fails, undo the SQLite write
    try:
        index_manager.add_chunks(chunk_ids, vectors)
    except Exception:
        with get_session() as session:
            stored = session.get(Document, doc.id)
            if stored:
                session.delete(stored)
        final_path.unlink(missing_ok=True)
        raise  # startup consistency check repairs any half-updated index

    return doc


def list_documents() -> list[Document]:
    with get_session() as session:
        return list(session.scalars(select(Document).order_by(Document.created_at.desc())))


def delete_document(doc_id: int) -> None:
    with get_session() as session:
        doc = session.get(Document, doc_id)
        if doc is None:
            raise DocumentNotFoundError(f"Document {doc_id} not found.")
        chunk_ids = list(session.scalars(select(Chunk.id).where(Chunk.document_id == doc_id)))
        stored_path = Path(doc.stored_path)
        session.delete(doc)  # cascades to its chunks

    index_manager.remove_chunks(chunk_ids)
    stored_path.unlink(missing_ok=True)