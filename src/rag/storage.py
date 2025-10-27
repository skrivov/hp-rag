from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional

import faiss  # type: ignore
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore as LlamaFaissStore

if TYPE_CHECKING:
    from src.ingest.chunker import SectionChunk


@dataclass(slots=True)
class FaissVectorConfig:
    """Configuration for FAISS-backed llama-index storage."""

    index_path: Path
    embed_model_name: Optional[str] = None
    embed_model: Optional[Any] = None
    overwrite: bool = False


class FaissVectorStore:
    """Wraps llama-index/FAISS persistence and retrieval helpers."""

    def __init__(self, config: FaissVectorConfig) -> None:
        self.config = config
        self._index = None

    def build_from_chunks(self, chunks: Iterable["SectionChunk"]) -> None:
        """Create or refresh stored FAISS index with provided chunks."""

        persist_dir = self.config.index_path
        persist_dir.mkdir(parents=True, exist_ok=True)

        if self.config.overwrite and any(persist_dir.iterdir()):
            for artifact in persist_dir.iterdir():
                if artifact.is_file():
                    artifact.unlink()

        if self.config.embed_model is not None:
            Settings.embed_model = self.config.embed_model
        elif self.config.embed_model_name:
            Settings.embed_model = OpenAIEmbedding(model=self.config.embed_model_name)
        else:
            raise ValueError(
                "FaissVectorStore requires `embed_model` or `embed_model_name` (e.g., text-embedding-3-small)."
            )

        documents = [
            Document(
                text=chunk.text,
                metadata={
                    "path": chunk.path,
                    "chunk_order": chunk.order,
                    "parent_title": chunk.parent_title,
                },
            )
            for chunk in chunks
        ]

        dim = getattr(Settings.embed_model, "dimension", None)
        if not isinstance(dim, int):
            dim = 1536
        vector_store = LlamaFaissStore(faiss_index=faiss.IndexFlatL2(dim))
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        index.storage_context.persist(persist_dir=str(persist_dir))
        self._index = index

    def _load_index(self) -> None:
        if self._index is not None:
            return
        persist_dir = self.config.index_path
        if not persist_dir.exists():
            raise FileNotFoundError(f"FAISS index directory not found: {persist_dir}")

        if self.config.embed_model is not None:
            Settings.embed_model = self.config.embed_model
        elif self.config.embed_model_name:
            Settings.embed_model = OpenAIEmbedding(model=self.config.embed_model_name)

        vector_store = LlamaFaissStore.from_persist_dir(str(persist_dir))
        docstore = SimpleDocumentStore.from_persist_dir(str(persist_dir))
        index_store = SimpleIndexStore.from_persist_dir(str(persist_dir))
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, docstore=docstore, index_store=index_store
        )
        self._index = load_index_from_storage(storage_context)

    def as_retriever(self, *, similarity_top_k: int = 5):
        """Expose llama-index retriever for querying."""

        self._load_index()
        return self._index.as_retriever(similarity_top_k=similarity_top_k)

    def dump_metadata(self, output_path: Path) -> None:
        """Persist docstore metadata for debugging."""

        self._load_index()
        nodes = self._index.storage_context.docstore.docs
        payload = [node.to_dict() for node in nodes.values()]
        output_path.write_text(json.dumps(payload, indent=2))


__all__ = ["FaissVectorStore", "FaissVectorConfig"]
