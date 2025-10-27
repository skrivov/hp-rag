"""Ingestion utilities for building TOC-aware corpora."""

from .toc_builder import TOCBuilder, TOCBuilderConfig
from .chunker import Chunker, ChunkConfig, SectionChunk
from .simple_chunker import ParagraphChunker
from .markdown import MarkdownTOCBuilder, MarkdownTOCBuilderConfig
from .pipeline import IngestionPipeline, IngestionResult

__all__ = [
    "TOCBuilder",
    "TOCBuilderConfig",
    "Chunker",
    "ChunkConfig",
    "SectionChunk",
    "IngestionPipeline",
    "IngestionResult",
    "ParagraphChunker",
    "MarkdownTOCBuilder",
    "MarkdownTOCBuilderConfig",
]
