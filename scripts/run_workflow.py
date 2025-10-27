from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.orchestration.workflow import BenchmarkWorkflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark workflow automation for HP-RAG experiments.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/datasets"),
        help="Directory where downloaded datasets are stored (default: data/datasets)",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts/datasets"),
        help="Directory for dataset-specific artifacts (default: artifacts/datasets)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List available datasets")

    download_parser = subparsers.add_parser("download", help="Download dataset assets")
    download_parser.add_argument("dataset", help="Dataset identifier")
    download_parser.add_argument("--force", action="store_true", help="Force re-download even if files exist")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest dataset into SQLite/FAISS stores")
    ingest_parser.add_argument("dataset", help="Dataset identifier")
    ingest_parser.add_argument("--embedding-model", help="Embedding model for FAISS index (optional)")
    ingest_parser.add_argument("--clean-stores", action="store_true", help="Remove existing stores before ingesting")

    evaluate_parser = subparsers.add_parser("evaluate", help="Run evaluations for a dataset")
    evaluate_parser.add_argument("dataset", help="Dataset identifier")
    evaluate_parser.add_argument("--suite", default="baseline", help="Evaluation suite name")
    evaluate_parser.add_argument("--questions", type=Path, help="Override questions file path")
    evaluate_parser.add_argument("--output", type=Path, help="Output path for evaluation JSON")
    evaluate_parser.add_argument("--top-k", type=int, default=3, help="Top-k contexts to retrieve per query")
    evaluate_parser.add_argument("--llm-model", default="gpt-4o-mini", help="LLM model name for selection")
    evaluate_parser.add_argument("--include-vector", action="store_true", help="Include FAISS baseline retriever")
    evaluate_parser.add_argument("--embedding-model", help="Embedding model used when include-vector is set")

    full_parser = subparsers.add_parser("full", help="Download, ingest, and evaluate in one command")
    full_parser.add_argument("dataset", help="Dataset identifier")
    full_parser.add_argument("--force-download", action="store_true", help="Force re-download")
    full_parser.add_argument("--clean-stores", action="store_true", help="Clean stores before ingesting")
    full_parser.add_argument("--embedding-model", help="Embedding model for FAISS index")
    full_parser.add_argument("--include-vector", action="store_true", help="Evaluate with vector retriever as well")
    full_parser.add_argument("--llm-model", default="gpt-4o-mini", help="LLM model name for selection")
    full_parser.add_argument("--top-k", type=int, default=3, help="Top-k contexts to retrieve per query")

    cleanup_parser = subparsers.add_parser("cleanup", help="Remove dataset downloads and/or artifacts")
    cleanup_parser.add_argument("dataset", help="Dataset identifier or 'all'")
    cleanup_parser.add_argument("--remove-artifacts", action="store_true", help="Also delete stored artifacts")

    return parser


def main(argv: list[str] | None = None) -> int:
    if Path(".env").exists():
        load_dotenv(".env", override=False)
    parser = build_parser()
    args = parser.parse_args(argv)

    workflow = BenchmarkWorkflow(data_root=args.data_root, artifact_root=args.artifacts_root)

    command = args.command

    if command == "list":
        datasets = workflow.list_datasets()
        for key, spec in sorted(datasets.items()):
            description = spec.description or ""
            print(f"{key}: {description}")
        return 0

    dataset = getattr(args, "dataset", None)
    if not dataset:
        parser.error("dataset argument is required for this command")

    if command == "download":
        path = workflow.download(dataset, force=args.force)
        print(f"Downloaded dataset '{dataset}' to {path}")
        return 0

    if command == "ingest":
        stats = workflow.ingest(dataset, clean_stores=args.clean_stores, embedding_model=args.embedding_model)
        print("Ingestion complete", stats)
        return 0

    if command == "evaluate":
        metrics = workflow.evaluate(
            dataset,
            suite_name=args.suite,
            questions_path=args.questions,
            output_path=args.output,
            top_k=args.top_k,
            include_vector=args.include_vector,
            llm_model=args.llm_model,
            embedding_model=args.embedding_model,
        )
        print("Evaluation metrics", metrics)
        return 0

    if command == "full":
        result = workflow.full_run(
            dataset,
            force_download=args.force_download,
            clean_stores=args.clean_stores,
            embedding_model=args.embedding_model,
            include_vector=args.include_vector,
            llm_model=args.llm_model,
            top_k=args.top_k,
        )
        print("Workflow complete", result)
        return 0

    if command == "cleanup":
        if dataset.lower() == "all":
            for key in workflow.list_datasets():
                workflow.remove_dataset(key, remove_artifacts=args.remove_artifacts)
                print(f"Removed dataset '{key}'")
        else:
            workflow.remove_dataset(dataset, remove_artifacts=args.remove_artifacts)
            print(f"Removed dataset '{dataset}'")
        return 0

    parser.error(f"Unknown command {command}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
