from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query

from src.hp_rag.storage import SQLiteHyperlinkStore
from src.ingest.chunker import SectionChunk
from src.rag.storage import FaissVectorConfig, FaissVectorStore
from src.server.plugins.base import PluginContext, RetrievalPlugin


class RagPlugin(RetrievalPlugin):
    name = "rag"
    display_name = "Vector RAG"
    capabilities = {"chunks": True, "search": True}

    def __init__(self, store: SQLiteHyperlinkStore, faiss_config: FaissVectorConfig) -> None:
        self.store = store
        self.faiss_config = faiss_config
        self._router = APIRouter(prefix="/api/plugins/rag", tags=["rag"])
        self._register_routes()

    async def ingest(self, context: PluginContext) -> Dict[str, Any]:
        chunks = self._load_all_chunks()
        if not chunks:
            return {"chunks": 0}
        vector_store = FaissVectorStore(self.faiss_config)
        vector_store.config.overwrite = True
        vector_store.build_from_chunks(chunks)
        return {"chunks": len(chunks)}

    def router(self) -> APIRouter:
        return self._router

    async def cleanup_all(self) -> None:
        persist_dir = Path(self.faiss_config.index_path)
        if persist_dir.exists():
            shutil.rmtree(persist_dir)

    # ------------------------------------------------------------------ helpers
    def _load_all_chunks(self) -> List[SectionChunk]:
        rows = self.store.iter_chunks()
        chunks: List[SectionChunk] = []
        for row in rows:
            chunks.append(
                SectionChunk(
                    path=row["section_path"],
                    text=row["text"],
                    order=row["chunk_order"],
                    parent_title=row["section_title"],
                )
            )
        return chunks

    def _register_routes(self) -> None:
        @self._router.get("/documents/{document_id}/chunks")
        async def list_chunks(document_id: str):
            rows = self.store.iter_chunks(document_id)
            if not rows:
                raise HTTPException(status_code=404, detail="Document not found")
            return {
                "document_id": document_id,
                "chunks": [
                    {
                        "section_path": row["section_path"],
                        "order": row["chunk_order"],
                        "text": row["text"],
                        "section_title": row["section_title"],
                    }
                    for row in rows
                ],
            }

        @self._router.get("/documents/{document_id}/search")
        async def search_chunks(document_id: str, q: str = Query(..., min_length=2)):
            rows = self.store.search(q, limit=20)
            filtered = [row for row in rows if row["document_id"] == document_id]
            return {
                "document_id": document_id,
                "query": q,
                "results": [
                    {
                        "path": row["path"],
                        "title": row["title"],
                        "snippet": row["body"][:400],
                        "level": row["level"],
                    }
                    for row in filtered
                ],
            }


__all__ = ["RagPlugin"]
