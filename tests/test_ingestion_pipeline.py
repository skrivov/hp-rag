from pathlib import Path

from src.ingest import IngestionPipeline, MarkdownTOCBuilder, MarkdownTOCBuilderConfig, ParagraphChunker


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
