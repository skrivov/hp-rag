from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from src.server.documents.service import DocumentService
from src.server.settings import Settings, get_settings


router = APIRouter(prefix="/api/documents", tags=["documents"])


def _resolve_service(settings: Settings) -> DocumentService:
    global _DOCUMENT_SERVICE
    if _DOCUMENT_SERVICE is None:
        _DOCUMENT_SERVICE = DocumentService(settings)
    return _DOCUMENT_SERVICE


def get_document_service(settings: Settings = Depends(get_settings)) -> DocumentService:
    return _resolve_service(settings)


def get_document_service_instance(settings: Settings) -> DocumentService:
    return _resolve_service(settings)


_DOCUMENT_SERVICE: DocumentService | None = None


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    title: str | None = None,
    service: DocumentService = Depends(get_document_service),
):
    metadata = await service.upload(file, title=title)
    return {
        "document_id": metadata.id,
        "status": metadata.status,
        "plugins": [state.__dict__ for state in metadata.plugin_states],
    }


@router.get("")
async def list_documents(
    tenant_id: str | None = None,
    service: DocumentService = Depends(get_document_service),
):
    docs = service.list_documents(tenant_id=tenant_id)
    return {
        "documents": [
            {
                "id": doc.id,
                "name": doc.name,
                "status": doc.status,
                "section_count": doc.section_count,
                "chunk_count": doc.chunk_count,
                "plugins": [state.__dict__ for state in doc.plugin_states],
                "tenants": [tenant.__dict__ for tenant in doc.tenants],
            }
            for doc in docs
        ]
    }


@router.get("/{document_id}")
async def get_document(document_id: str, service: DocumentService = Depends(get_document_service)):
    doc = service.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "id": doc.id,
        "name": doc.name,
        "status": doc.status,
        "section_count": doc.section_count,
        "chunk_count": doc.chunk_count,
        "plugins": [state.__dict__ for state in doc.plugin_states],
        "tenants": [tenant.__dict__ for tenant in doc.tenants],
    }


@router.get("/{document_id}/body")
async def get_document_body(document_id: str, service: DocumentService = Depends(get_document_service)):
    body = service.get_document_body(document_id)
    if body is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"document_id": document_id, "body": body}


@router.delete("")
async def delete_all_documents(
    force: bool = False,
    service: DocumentService = Depends(get_document_service),
):
    if not force:
        raise HTTPException(status_code=400, detail="Set force=true to delete all documents")
    await service.delete_all()
    return {"status": "deleted"}


__all__ = ["router", "get_document_service"]
