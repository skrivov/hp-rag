from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, TYPE_CHECKING

from src.ingest.chunker import Chunker, SectionChunk
from src.ingest.toc_builder import TOCBuilder
from src.models.section import SectionNode
from src.hp_rag.storage import SQLiteHyperlinkStore

if TYPE_CHECKING:
    # Avoid importing FAISS code-paths at runtime when only building TOC
    from src.rag.storage import FaissVectorStore


@dataclass(slots=True)
class IngestionResult:
    """Summarizes artifacts produced by an ingestion run."""

    documents_processed: int
    sections_written: int
    chunks_written: int


class IngestionPipeline:
    """Coordinates TOC parsing, chunking, and persistence to storage backends."""

    def __init__(
        self,
        toc_builder: Optional[TOCBuilder],
        chunker: Chunker,
        *,
        sqlite_store: Optional[SQLiteHyperlinkStore] = None,
        vector_store: Optional["FaissVectorStore"] = None,
    ) -> None:
        self.toc_builder = toc_builder
        self.chunker = chunker
        self.sqlite_store = sqlite_store
        self.vector_store = vector_store

    def ingest(self, document_paths: Iterable[Path]) -> IngestionResult:
        if self.toc_builder is None:
            raise RuntimeError("toc_builder must be provided for path-based ingestion")

        roots = [self.toc_builder.build(path) for path in document_paths]
        return self.ingest_sections(roots)

    def ingest_sections(self, section_roots: Iterable[SectionNode]) -> IngestionResult:
        sections: List[SectionNode] = []
        chunks: List[SectionChunk] = []
        documents_processed = 0

        for root in section_roots:
            documents_processed += 1
            sections.extend(self._flatten_sections(root))
            for section in self._walk_sections(root):
                if section.level == 0:
                    continue
                chunks.extend(self.chunker.chunk(section))

        if self.sqlite_store:
            self.sqlite_store.initialize()
            if sections:
                self.sqlite_store.upsert_sections(sections)
            if chunks:
                self.sqlite_store.add_chunks(chunks)

        if self.vector_store and chunks:
            self.vector_store.build_from_chunks(chunks)

        return IngestionResult(
            documents_processed=documents_processed,
            sections_written=len(sections),
            chunks_written=len(chunks),
        )

    def _flatten_sections(self, root: SectionNode) -> List[SectionNode]:
        return list(self._walk_sections(root))

    def _walk_sections(self, node: SectionNode) -> Iterator[SectionNode]:
        yield node
        for child in node.children:
            yield from self._walk_sections(child)


__all__ = ["IngestionPipeline", "IngestionResult"]
