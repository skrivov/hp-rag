from pathlib import Path

import fitz

from src.ingest import (
    IngestionPipeline,
    MarkdownTOCBuilder,
    MarkdownTOCBuilderConfig,
    ParagraphChunker,
    PyMuPDFTOCBuilder,
    PyMuPDFTOCBuilderConfig,
)
from src.hp_rag.storage import SQLiteHyperlinkConfig, SQLiteHyperlinkStore


class StubSQLiteStore:
    def __init__(self) -> None:
        self.initialized = False
        self.sections = []
        self.chunks = []

    def initialize(self) -> None:
        self.initialized = True

    def upsert_sections(self, sections) -> None:
        self.sections.extend(list(sections))

    def add_chunks(self, chunks) -> None:
        self.chunks.extend(list(chunks))


class StubVectorStore:
    def __init__(self) -> None:
        self.chunks = None

    def build_from_chunks(self, chunks) -> None:
        self.chunks = list(chunks)


def test_ingestion_pipeline_processes_documents(tmp_path):
    doc_path = tmp_path / "doc.md"
    doc_path.write_text(
        "# Title\n\nIntro paragraph.\n\n## Details\n\nMore info here.\n",
        encoding="utf-8",
    )

    sqlite_store = StubSQLiteStore()
    vector_store = StubVectorStore()

    pipeline = IngestionPipeline(
        toc_builder=MarkdownTOCBuilder(MarkdownTOCBuilderConfig()),
        chunker=ParagraphChunker(),
        sqlite_store=sqlite_store,
        vector_store=vector_store,
    )

    result = pipeline.ingest([Path(doc_path)])

    assert result.documents_processed == 1
    assert result.sections_written == len(sqlite_store.sections)
    assert result.chunks_written == len(sqlite_store.chunks)
    assert sqlite_store.initialized is True
    assert len(sqlite_store.sections) == 3  # root + two headings
    assert len(sqlite_store.chunks) == 2
    assert vector_store.chunks is not None and len(vector_store.chunks) == 2


def _create_pdf_contract(path: Path) -> None:
    document = fitz.open()
    page1 = document.new_page()
    page1.insert_text(
        (72, 72),
        "Section 1 Overview\nDetailed scope of work described here.",
    )
    page2 = document.new_page()
    page2.insert_text(
        (72, 72),
        "Section 2 Payment Terms\nCompensation details and invoicing.",
    )
    document.set_toc(
        [
            [1, "Overview", 1],
            [1, "Payment Terms", 2],
        ]
    )
    document.save(path)
    document.close()


def test_ingestion_pipeline_handles_pdf_sections(tmp_path):
    pdf_path = tmp_path / "contract.pdf"
    _create_pdf_contract(pdf_path)

    sqlite_path = tmp_path / "hyperlink.db"
    sqlite_store = SQLiteHyperlinkStore(SQLiteHyperlinkConfig(db_path=sqlite_path))

    pipeline = IngestionPipeline(
        toc_builder=None,
        chunker=ParagraphChunker(),
        sqlite_store=sqlite_store,
        vector_store=None,
    )

    builder = PyMuPDFTOCBuilder(PyMuPDFTOCBuilderConfig())
    root = builder.build(pdf_path)

    result = pipeline.ingest_sections([root])

    assert result.documents_processed == 1
    assert result.sections_written >= 3  # root + two outline entries
    assert result.chunks_written >= 2

    stored = sqlite_store.fetch_by_path("contract/overview")
    assert stored is not None
    assert "scope" in stored["body"].lower()

    sqlite_store.close()
