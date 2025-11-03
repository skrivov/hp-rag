"""Ingestion utilities for building TOC-aware corpora."""

from .toc_builder import TOCBuilder, TOCBuilderConfig
from .chunker import Chunker, ChunkConfig, SectionChunk
from .simple_chunker import ParagraphChunker
from .markdown import MarkdownTOCBuilder, MarkdownTOCBuilderConfig
from .pipeline import IngestionPipeline, IngestionResult

try:
    from .pdf import PyMuPDFTOCBuilder, PyMuPDFTOCBuilderConfig

    __all_pdf__ = ["PyMuPDFTOCBuilder", "PyMuPDFTOCBuilderConfig"]
except ModuleNotFoundError:
    PyMuPDFTOCBuilder = PyMuPDFTOCBuilderConfig = None  # type: ignore[assignment]
    __all_pdf__ = []

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
] + __all_pdf__
