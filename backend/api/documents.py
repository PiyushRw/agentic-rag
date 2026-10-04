from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.config.settings import settings
from backend.ingestion.loader import DocumentLoadError
from backend.schemas.api import DocumentListResponse, DocumentOut
from backend.services.document_service import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    delete_document,
    list_documents,
    upload_document,
)

router = APIRouter(prefix="/documents", tags=["documents"])


# Plain `def` (not `async def`): embedding is CPU-heavy, so FastAPI runs it in a worker
# thread instead of blocking the event loop.
@router.post("/upload", response_model=DocumentOut, status_code=201)
def upload(file: UploadFile = File(...)):
    content = file.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb} MB.")
    try:
        return upload_document(file.filename or "unnamed", content)
    except DuplicateDocumentError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except DocumentLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("", response_model=DocumentListResponse)
def list_all():
    return {"documents": list_documents()}


@router.delete("/{doc_id}")
def remove(doc_id: int):
    try:
        delete_document(doc_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"deleted": doc_id}