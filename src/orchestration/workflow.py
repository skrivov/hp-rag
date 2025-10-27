from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping

from dotenv import load_dotenv

from src.hp_rag.retriever import HyperlinkRetriever, HyperlinkRetrieverConfig
from src.hp_rag.storage import SQLiteHyperlinkConfig, SQLiteHyperlinkStore
from src.ingest import IngestionPipeline, MarkdownTOCBuilder, MarkdownTOCBuilderConfig, ParagraphChunker
from src.ingest.adapters.factory import create_adapter
from src.orchestration.datasets import DatasetSpec, available_datasets, download_dataset, get_dataset_spec
from src.rag.retriever import VectorRetriever, VectorRetrieverConfig
from src.rag.storage import FaissVectorConfig, FaissVectorStore


DEFAULT_DATA_ROOT = Path("data/datasets")
DEFAULT_ARTIFACT_ROOT = Path("artifacts/datasets")


class BenchmarkWorkflow:
    """High-level orchestration utilities for benchmark ingestion and evaluation."""

    def __init__(
        self,
        *,
        data_root: Path | str = DEFAULT_DATA_ROOT,
        artifact_root: Path | str = DEFAULT_ARTIFACT_ROOT,
    ) -> None:
        self.data_root = Path(data_root)
        self.artifact_root = Path(artifact_root)

    # ---- Dataset management -------------------------------------------------

    def list_datasets(self) -> Mapping[str, DatasetSpec]:
        return available_datasets()

    def dataset_dir(self, name: str) -> Path:
        spec = get_dataset_spec(name)
        return self.data_root / spec.name

    def artifacts_dir(self, name: str) -> Path:
        spec = get_dataset_spec(name)
        return self.artifact_root / spec.name

    def download(self, name: str, *, force: bool = False) -> Path:
        spec = get_dataset_spec(name)
        return download_dataset(spec, self.data_root, force=force)

    def remove_dataset(self, name: str, *, remove_artifacts: bool = False) -> None:
        dataset_path = self.dataset_dir(name)
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
        if remove_artifacts:
            artifacts_path = self.artifacts_dir(name)
            if artifacts_path.exists():
                shutil.rmtree(artifacts_path)

    # ---- Ingestion ----------------------------------------------------------

    def ingest(
        self,
        name: str,
        *,
        clean_stores: bool = False,
        embedding_model: str | None = None,
    ) -> dict[str, Any]:
        spec = get_dataset_spec(name)
        dataset_path = self.dataset_dir(name)
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset '{name}' not found at {dataset_path}. Run download() first."
            )

        corpus_path = self._resolve_corpus_path(dataset_path, spec)
        if not corpus_path.exists():
            raise FileNotFoundError(
                f"Corpus path '{corpus_path}' for dataset '{name}' does not exist."
            )

        artifacts_dir = self.artifacts_dir(name)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        sqlite_path = artifacts_dir / "hyperlink.db"
        faiss_dir = artifacts_dir / "faiss_index"

        if clean_stores:
            if sqlite_path.exists():
                sqlite_path.unlink()
            if faiss_dir.exists():
                shutil.rmtree(faiss_dir)

        sqlite_store = SQLiteHyperlinkStore(SQLiteHyperlinkConfig(db_path=sqlite_path))
        vector_store = None
        if embedding_model:
            vector_store = FaissVectorStore(
                FaissVectorConfig(index_path=faiss_dir, embed_model_name=embedding_model, overwrite=True)
            )

        pipeline = IngestionPipeline(
            toc_builder=None,
            chunker=ParagraphChunker(),
            sqlite_store=sqlite_store,
            vector_store=vector_store,
        )

        if spec.adapter_name:
            adapter = create_adapter(spec.adapter_name, corpus_path, **spec.adapter_kwargs)
            result = pipeline.ingest_sections(adapter.iter_section_roots())
        else:
            toc_builder = MarkdownTOCBuilder(MarkdownTOCBuilderConfig())
            pipeline.toc_builder = toc_builder
            docs = list(corpus_path.glob("**/*.md")) if corpus_path.is_dir() else [corpus_path]
            result = pipeline.ingest(docs)

        return {
            "documents": result.documents_processed,
            "sections": result.sections_written,
            "chunks": result.chunks_written,
            "sqlite": sqlite_path,
            "faiss": faiss_dir if embedding_model else None,
        }

    # ---- Evaluation ---------------------------------------------------------

    def evaluate(
        self,
        name: str,
        *,
        suite_name: str = "baseline",
        questions_path: str | Path | None = None,
        output_path: str | Path | None = None,
        top_k: int = 3,
        include_vector: bool = False,
        llm_model: str = "gpt-4o-mini",
        embedding_model: str | None = None,
        selector_llm: Any | None = None,
    ) -> dict[str, float]:
        spec = get_dataset_spec(name)
        artifacts_dir = self.artifacts_dir(name)
        sqlite_path = artifacts_dir / "hyperlink.db"
        faiss_dir = artifacts_dir / "faiss_index"

        if not sqlite_path.exists():
            raise FileNotFoundError(
                f"SQLite store not found at {sqlite_path}. Ingest dataset '{name}' first."
            )

        dataset_dir = self.dataset_dir(name)
        resolved_questions = self._resolve_questions_path(dataset_dir, spec, questions_path)

        output_path = Path(output_path) if output_path else artifacts_dir / "eval.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        retrievers: dict[str, object] = {}
        retriever_ids: list[str] = []

        load_dotenv(override=False)

        hp_config = HyperlinkRetrieverConfig(
            sqlite_config=SQLiteHyperlinkConfig(db_path=sqlite_path),
            selection_prompt_template=(
                "Return a JSON array of at most {limit} exact section paths most relevant to the query.\n"
                "Only include paths that appear in the TOC list. No commentary.\n\n"
                "Query: {query}\n\nTOC (path: title):\n{toc}"
            ),
            toc_limit=200,
            llm=selector_llm,
            llm_model=None if selector_llm else llm_model,
            max_sections=5,
            neighbor_window=1,
        )
        retrievers["hyperlink"] = HyperlinkRetriever(hp_config)
        retriever_ids.append("hyperlink")

        if include_vector:
            if not faiss_dir.exists():
                raise FileNotFoundError(
                    f"FAISS directory not found at {faiss_dir}. Re-ingest with an embedding model or set include_vector=False."
                )
            if not embedding_model:
                raise ValueError("embedding_model must be provided when include_vector=True")
            retrievers["vector"] = VectorRetriever(
                VectorRetrieverConfig(
                    faiss_config=FaissVectorConfig(
                        index_path=faiss_dir,
                        embed_model_name=embedding_model,
                    ),
                    similarity_top_k=top_k,
                )
            )
            retriever_ids.insert(0, "vector")

        from src.eval import EvaluationConfig, run_suite

        evaluation_config = EvaluationConfig(
            suite_name=suite_name,
            questions_path=str(resolved_questions),
            output_path=str(output_path),
            retriever_ids=retriever_ids,
            top_k=top_k,
        )

        result = run_suite(evaluation_config, retrievers)
        return dict(result.metrics)

    # ---- Combined convenience ----------------------------------------------

    def full_run(
        self,
        name: str,
        *,
        force_download: bool = False,
        clean_stores: bool = False,
        embedding_model: str | None = None,
        include_vector: bool = False,
        llm_model: str = "gpt-4o-mini",
        selector_llm: Any | None = None,
        top_k: int = 3,
    ) -> dict[str, Any]:
        self.download(name, force=force_download)
        ingest_stats = self.ingest(name, clean_stores=clean_stores, embedding_model=embedding_model)
        metrics = self.evaluate(
            name,
            include_vector=include_vector,
            llm_model=llm_model,
            embedding_model=embedding_model,
            selector_llm=selector_llm,
            top_k=top_k,
        )
        return {"ingestion": ingest_stats, "metrics": metrics}

    # ---- Helpers ------------------------------------------------------------

    @staticmethod
    def _resolve_corpus_path(dataset_dir: Path, spec: DatasetSpec) -> Path:
        if spec.corpus_relative_path:
            return dataset_dir / spec.corpus_relative_path
        return dataset_dir

    @staticmethod
    def _resolve_questions_path(
        dataset_dir: Path, spec: DatasetSpec, override: str | Path | None
    ) -> Path:
        if override is not None:
            return Path(override)
        if spec.questions_relative_path is None:
            raise ValueError(
                "questions_path must be provided because the dataset specification does not include a default."
            )
        default_path = dataset_dir / spec.questions_relative_path
        if not default_path.exists():
            raise FileNotFoundError(f"Question file not found at {default_path}")
        return default_path


__all__ = ["BenchmarkWorkflow"]
