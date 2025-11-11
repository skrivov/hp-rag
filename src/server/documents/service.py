from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from fastapi import UploadFile

from src.hp_rag.storage import SQLiteHyperlinkConfig, SQLiteHyperlinkStore
from src.ingest.markdown import MarkdownTOCBuilder
from src.ingest.pdf import PyMuPDFTOCBuilder
from src.ingest.simple_chunker import ParagraphChunker
from src.ingest.pipeline import IngestionPipeline
from src.ingest.tenant_extractor import TenantExtractor
from src.ingest.toc_builder import TOCBuilder
from src.models.document import DocumentMetadata, DocumentStatus, PluginStatus
from src.rag.storage import FaissVectorConfig
from src.server.documents.store import DocumentStore
from src.server.plugins.base import PluginContext, PluginRegistry
from src.server.plugins.hp_rag import HPRagPlugin
from src.server.plugins.rag import RagPlugin
from src.server.settings import Settings


@dataclass(slots=True)
class _IngestJob:
    document_id: str
    file_path: Path
    mime_type: Optional[str]
    size_bytes: int


class DocumentService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.uploads_dir = Path(settings.uploads_dir)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.document_store = DocumentStore(settings.sqlite_db_path)
        self.sqlite_store = SQLiteHyperlinkStore(SQLiteHyperlinkConfig(db_path=settings.sqlite_db_path))
        self.sqlite_store.initialize()
        self.registry = PluginRegistry()
        self._register_builtin_plugins()
        self.queue: asyncio.Queue[_IngestJob] = asyncio.Queue(maxsize=max(settings.document_queue_size, 1))
        self._worker_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------ public API
    async def upload(self, file: UploadFile, title: str | None = None) -> DocumentMetadata:
        document_id = str(uuid.uuid4())
        suffix = Path(file.filename or "").suffix or ".txt"
        stored_path = self.uploads_dir / f"{document_id}{suffix}"
        size_bytes = await self._persist_upload(file, stored_path)

        display_name = title or (file.filename or stored_path.name)
        self.document_store.upsert_document(
            document_id=document_id,
            name=display_name,
            source_path=stored_path,
            mime_type=file.content_type,
            size_bytes=size_bytes,
            status=DocumentStatus.QUEUED,
        )

        for plugin_name in self.registry.plugins:
            self.document_store.set_plugin_status(document_id, plugin_name, PluginStatus.QUEUED)

        job = _IngestJob(
            document_id=document_id,
            file_path=stored_path,
            mime_type=file.content_type,
            size_bytes=size_bytes,
        )
        await self.queue.put(job)
        self._ensure_worker()
        return self.document_store.get_document(document_id)  # type: ignore[return-value]

    def list_documents(self, tenant_id: str | None = None) -> List[DocumentMetadata]:
        return self.document_store.list_documents(tenant_id=tenant_id)

    def get_document(self, document_id: str) -> DocumentMetadata | None:
        return self.document_store.get_document(document_id)

    async def delete_all(self) -> None:
        await self._stop_worker()
        self.queue = asyncio.Queue(maxsize=max(self.settings.document_queue_size, 1))
        self.document_store.delete_all()
        self.sqlite_store.clear_all()
        for plugin in self.registry.plugins.values():
            await plugin.cleanup_all()
        if self.settings.faiss_index_path.exists():
            # cleaned via plugin but double check
            pass
        for file in self.uploads_dir.glob("*"):
            if file.is_file():
                file.unlink()

    def get_document_body(self, document_id: str) -> str | None:
        sections = self.sqlite_store.iter_sections_by_document(document_id)
        if not sections:
            return None
        parts: List[str] = []
        for row in sections:
            indent = "#" * max(1, row["level"] or 1)
            parts.append(f"{indent} {row['title']}\n{row['body']}\n")
        return "\n".join(parts)

    # ------------------------------------------------------------------ workers
    def _ensure_worker(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker())

    async def _stop_worker(self) -> None:
        if self._worker_task is None:
            return
        self._worker_task.cancel()
        try:
            await self._worker_task
        except asyncio.CancelledError:
            pass
        self._worker_task = None

    async def _worker(self) -> None:
        while True:
            job = await self.queue.get()
            try:
                await self._process(job)
            except Exception as exc:  # pragma: no cover
                self.document_store.update_status(job.document_id, DocumentStatus.FAILED, error=str(exc))
            finally:
                self.queue.task_done()

    async def _process(self, job: _IngestJob) -> None:
        self.document_store.update_status(job.document_id, DocumentStatus.INGESTING)
        builder = self._resolve_builder(job.file_path)
        root = builder.build(job.file_path)
        chunker = ParagraphChunker()
        pipeline = IngestionPipeline(
            toc_builder=None,
            chunker=chunker,
            sqlite_store=self.sqlite_store,
            vector_store=None,
            tenant_extractor=TenantExtractor(),
        )
        pipeline.ingest_sections([root])

        context = PluginContext(
            document_id=job.document_id,
            sqlite_store=self.sqlite_store,
            faiss_config=FaissVectorConfig(
                index_path=self.settings.faiss_index_path,
                embed_model_name=self.settings.embedding_model,
                overwrite=True,
            ),
        )

        failures = []
        for plugin in self.registry.plugins.values():
            try:
                stats = await plugin.ingest(context)
                self.document_store.set_plugin_status(job.document_id, plugin.name, PluginStatus.READY, stats=stats)
            except Exception as exc:  # pragma: no cover
                self.document_store.set_plugin_status(job.document_id, plugin.name, PluginStatus.FAILED, error=str(exc))
                failures.append(str(exc))

        final_status = DocumentStatus.FAILED if failures else DocumentStatus.READY
        self.document_store.update_status(job.document_id, final_status, error="; ".join(failures) if failures else None)

    # ------------------------------------------------------------------ helpers
    async def _persist_upload(self, file: UploadFile, destination: Path) -> int:
        size = 0
        with destination.open("wb") as sink:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                sink.write(chunk)
                size += len(chunk)
        return size

    def _resolve_builder(self, file_path: Path) -> TOCBuilder:
        if file_path.suffix.lower() == ".pdf":
            return PyMuPDFTOCBuilder()
        if file_path.suffix.lower() in {".md", ".markdown"}:
            return MarkdownTOCBuilder()
        return MarkdownTOCBuilder()

    def _register_builtin_plugins(self) -> None:
        hp_plugin = HPRagPlugin(self.sqlite_store)
        rag_plugin = RagPlugin(
            self.sqlite_store,
            FaissVectorConfig(
                index_path=self.settings.faiss_index_path,
                embed_model_name=self.settings.embedding_model,
                overwrite=True,
            ),
        )
        self.registry.register(hp_plugin)
        self.registry.register(rag_plugin)


__all__ = ["DocumentService"]
