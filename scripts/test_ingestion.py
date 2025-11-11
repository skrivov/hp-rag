from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ingest.markdown import MarkdownTOCBuilder, MarkdownTOCBuilderConfig
from src.ingest.pipeline import IngestionPipeline, IngestionResult
from src.ingest.simple_chunker import ParagraphChunker
from src.ingest.tenant_extractor import TenantExtractor
from src.hp_rag.storage import SQLiteHyperlinkConfig, SQLiteHyperlinkStore

try:
    from src.ingest.pdf import PyMuPDFTOCBuilder, PyMuPDFTOCBuilderConfig

    HAS_PYMUPDF = True
except ModuleNotFoundError:
    PyMuPDFTOCBuilder = PyMuPDFTOCBuilderConfig = None  # type: ignore[assignment]
    HAS_PYMUPDF = False


@dataclass(slots=True)
class IngestionSummary:
    documents: int = 0
    sections: int = 0
    chunks: int = 0

    def add(self, result: IngestionResult) -> None:
        self.documents += result.documents_processed
        self.sections += result.sections_written
        self.chunks += result.chunks_written


def discover_documents(corpus: Path) -> Tuple[List[Path], List[Path]]:
    markdown = sorted(corpus.rglob("*.md"))
    pdfs = sorted(corpus.rglob("*.pdf"))
    return markdown, pdfs


def ingest_markdown(pipeline: IngestionPipeline, paths: Iterable[Path]) -> IngestionResult | None:
    docs = list(paths)
    if not docs:
        return None
    pipeline.toc_builder = MarkdownTOCBuilder(MarkdownTOCBuilderConfig())
    return pipeline.ingest(docs)


def ingest_pdfs(pipeline: IngestionPipeline, paths: Iterable[Path]) -> IngestionResult | None:
    if not HAS_PYMUPDF:
        return None
    pdfs = list(paths)
    if not pdfs:
        return None
    builder = PyMuPDFTOCBuilder(PyMuPDFTOCBuilderConfig())
    roots = []
    for path in pdfs:
        try:
            roots.append(builder.build(path))
        except ValueError as exc:
            print(f"[warn] Skipping PDF without outline: {path} ({exc})")
    if not roots:
        return None
    return pipeline.ingest_sections(roots)


def dump_toc(sqlite_path: Path, destination: Path, limit: int | None = None) -> None:
    store = SQLiteHyperlinkStore(SQLiteHyperlinkConfig(db_path=sqlite_path))
    rows = store.iter_sections()
    lines = []
    for idx, row in enumerate(rows, start=1):
        lines.append(f"- {row['path']}: {row['title']}")
        if limit and idx >= limit:
            break
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    store.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Utility script to ingest a document corpus and dump the TOC used by the LLM selector.",
    )
    parser.add_argument(
        "corpus",
        type=Path,
        nargs="?",
        default=Path("data/contracts"),
        help="Directory containing markdown/PDF documents (default: data/contracts)",
    )
    parser.add_argument(
        "--sqlite",
        type=Path,
        default=Path("artifacts/test_ingestion/contracts_hyperlink.db"),
        help="Path to the SQLite hyperlink store (default: artifacts/test_ingestion/contracts_hyperlink.db)",
    )
    parser.add_argument(
        "--toc-output",
        type=Path,
        default=Path("artifacts/test_ingestion/toc_dump.txt"),
        help="File to write the TOC bullet list shown to the selector LLM (default: artifacts/test_ingestion/toc_dump.txt)",
    )
    parser.add_argument(
        "--only-markdown",
        action="store_true",
        help="Ingest only markdown files (skip PDFs even if available)",
    )
    parser.add_argument(
        "--only-pdf",
        action="store_true",
        help="Ingest only PDF files (ignore markdown documents)",
    )
    parser.add_argument(
        "--toc-limit",
        type=int,
        default=None,
        help="Optional limit on the number of TOC lines written to the output file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.corpus.exists():
        raise FileNotFoundError(f"Corpus path not found: {args.corpus}")

    args.sqlite.parent.mkdir(parents=True, exist_ok=True)
    # Ensure we start from a clean store so comparisons are deterministic.
    if args.sqlite.exists():
        args.sqlite.unlink()

    store = SQLiteHyperlinkStore(SQLiteHyperlinkConfig(db_path=args.sqlite))
    store.initialize()
    pipeline = IngestionPipeline(
        toc_builder=None,
        chunker=ParagraphChunker(),
        sqlite_store=store,
        vector_store=None,
        tenant_extractor=TenantExtractor(),
    )

    markdown_docs, pdf_docs = discover_documents(args.corpus)
    if args.only_pdf:
        markdown_docs = []
    if args.only_markdown:
        pdf_docs = []

    summary = IngestionSummary()
    result_md = ingest_markdown(pipeline, markdown_docs)
    if result_md:
        summary.add(result_md)
    result_pdf = ingest_pdfs(pipeline, pdf_docs)
    if result_pdf:
        summary.add(result_pdf)
    elif pdf_docs and not HAS_PYMUPDF:
        print("[warn] PyMuPDF not available; skipping PDF ingestion.")

    dump_toc(args.sqlite, args.toc_output, limit=args.toc_limit)
    store.close()

    print(
        "Ingestion complete",
        {
            "documents": summary.documents,
            "sections": summary.sections,
            "chunks": summary.chunks,
            "sqlite_db": str(args.sqlite),
            "toc_file": str(args.toc_output),
        },
    )


if __name__ == "__main__":
    main()
