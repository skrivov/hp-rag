from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from src.ingest import (
    IngestionPipeline,
    MarkdownTOCBuilder,
    MarkdownTOCBuilderConfig,
    ParagraphChunker,
)
from src.ingest.adapters.factory import adapter_choices, create_adapter
from src.hp_rag.storage import SQLiteHyperlinkConfig, SQLiteHyperlinkStore
from src.rag.storage import FaissVectorConfig, FaissVectorStore


def parse_args() -> argparse.Namespace:
    if Path(".env").exists():
        load_dotenv(".env", override=False)
    parser = argparse.ArgumentParser(description="Build corpus artifacts for HP-RAG experiments.")
    parser.add_argument("corpus", type=Path, help="Input corpus path (directory or dataset file)")
    parser.add_argument("output", type=Path, help="Directory to place generated artifacts")
    parser.add_argument(
        "--pattern",
        default="**/*.md",
        help="Glob pattern for markdown ingestion (default: **/*.md). Ignored when --dataset is set.",
    )
    parser.add_argument(
        "--sqlite-db",
        type=Path,
        default=None,
        help="Optional path for SQLite hyperlink store (default: <output>/hyperlink.db)",
    )
    parser.add_argument(
        "--faiss-dir",
        type=Path,
        default=None,
        help="Optional directory for FAISS index (default: <output>/faiss_index). If omitted or no embedding model is set, FAISS is skipped.",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL") or os.getenv("OPENAI_EMBEDDING_MODEL"),
        help="Embedding model for FAISS (e.g., text-embedding-3-small). If not set, FAISS build is skipped.",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(adapter_choices().keys()),
        help="Dataset adapter to use for structured corpora (beir, squad, hotpotqa, miracl).",
    )
    parser.add_argument(
        "--miracl-language",
        help="Language code for MIRACL adapter when the corpus path points to a directory.",
    )
    return parser.parse_args()


def discover_documents(root: Path, pattern: str) -> List[Path]:
    return sorted([path for path in root.glob(pattern) if path.is_file()])


def main() -> None:
    args = parse_args()
    if not args.corpus.exists():
        raise FileNotFoundError(f"Corpus path not found: {args.corpus}")

    args.output.mkdir(parents=True, exist_ok=True)
    sqlite_path = args.sqlite_db or args.output / "hyperlink.db"
    faiss_dir = args.faiss_dir or args.output / "faiss_index"

    sqlite_store = SQLiteHyperlinkStore(SQLiteHyperlinkConfig(db_path=sqlite_path))
    vector_store = None
    if args.embedding_model:
        vector_store = FaissVectorStore(
            FaissVectorConfig(index_path=faiss_dir, embed_model_name=args.embedding_model)
        )

    pipeline = IngestionPipeline(
        toc_builder=None,
        chunker=ParagraphChunker(),
        sqlite_store=sqlite_store,
        vector_store=vector_store,
    )

    if args.dataset:
        adapter_kwargs: dict[str, object] = {}
        if args.dataset == "miracl":
            if args.corpus.is_dir() and not args.miracl_language:
                raise ValueError(
                    "--miracl-language is required when using the MIRACL adapter with a directory corpus"
                )
            if args.miracl_language:
                adapter_kwargs["language"] = args.miracl_language
        adapter = create_adapter(args.dataset, args.corpus, **adapter_kwargs)
        result = pipeline.ingest_sections(adapter.iter_section_roots())
        documents_processed = result.documents_processed
    else:
        toc_builder = MarkdownTOCBuilder(MarkdownTOCBuilderConfig())
        pipeline.toc_builder = toc_builder
        docs = discover_documents(args.corpus, args.pattern)
        if not docs:
            raise RuntimeError("No documents found for ingestion")
        result = pipeline.ingest(docs)
        documents_processed = len(docs)

    print(
        "Ingestion complete",
        {
            "documents": documents_processed,
            "sections": result.sections_written,
            "chunks": result.chunks_written,
            "sqlite_db": str(sqlite_path),
            "faiss_dir": str(faiss_dir) if vector_store else None,
        },
    )


if __name__ == "__main__":
    main()
