from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from src.hp_rag.storage import SQLiteHyperlinkStore
from src.server.plugins.base import PluginContext, RetrievalPlugin


class HPRagPlugin(RetrievalPlugin):
    name = "hp-rag"
    display_name = "HP-RAG"
    capabilities = {"toc": True, "hyperlinks": True, "chunks": True}

    def __init__(self, store: SQLiteHyperlinkStore) -> None:
        self.store = store
        self._router = APIRouter(prefix="/api/plugins/hp-rag", tags=["hp-rag"])
        self._register_routes()

    async def ingest(self, context: PluginContext) -> Dict[str, Any]:
        sections = len(self.store.iter_sections_by_document(context.document_id))
        chunks = len(self.store.iter_chunks(context.document_id))
        return {"sections": sections, "chunks": chunks}

    def router(self) -> APIRouter:
        return self._router

    async def cleanup_all(self) -> None:
        # HP-RAG data lives in sections/chunks tables cleared by DocumentService.
        return None

    def _register_routes(self) -> None:
        @self._router.get("/documents/{document_id}/toc")
        async def get_toc(document_id: str) -> Dict[str, Any]:
            sections = self.store.iter_sections_by_document(document_id)
            if not sections:
                raise HTTPException(status_code=404, detail="Document not found")
            nodes = [
                {
                    "path": row["path"],
                    "title": row["title"],
                    "level": row["level"],
                    "parent_path": row["parent_path"],
                    "order_index": row["order_index"],
                }
                for row in sections
            ]
            return {"document_id": document_id, "nodes": nodes}

        @self._router.get("/documents/{document_id}/hyperlinks")
        async def get_hyperlinks(document_id: str) -> Dict[str, Any]:
            sections = self.store.iter_sections_by_document(document_id)
            if not sections:
                raise HTTPException(status_code=404, detail="Document not found")
            ordered = sorted(sections, key=lambda row: (row["level"], row["order_index"]))
            payload: List[Dict[str, Any]] = []
            for idx, row in enumerate(ordered):
                payload.append(
                    {
                        "path": row["path"],
                        "title": row["title"],
                        "level": row["level"],
                        "parent_path": row["parent_path"],
                        "body": row["body"],
                        "prev_path": ordered[idx - 1]["path"] if idx > 0 else None,
                        "next_path": ordered[idx + 1]["path"] if idx + 1 < len(ordered) else None,
                    }
                )
            return {"document_id": document_id, "sections": payload}

        @self._router.get("/documents/{document_id}/chunks")
        async def get_chunks(document_id: str) -> Dict[str, Any]:
            sections = self.store.iter_sections_by_document(document_id)
            if not sections:
                raise HTTPException(status_code=404, detail="Document not found")
            chunks = [
                {
                    "section_path": row["section_path"],
                    "order": row["chunk_order"],
                    "text": row["text"],
                    "section_title": row["section_title"],
                }
                for row in self.store.iter_chunks(document_id)
            ]
            return {"document_id": document_id, "chunks": chunks}


__all__ = ["HPRagPlugin"]
