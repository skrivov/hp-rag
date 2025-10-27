from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class DownloadConfig(BaseModel):
    url: str
    checksum: str | None = None
    filename: str | None = None


class ExtractConfig(BaseModel):
    archive_type: str | None = Field(default=None, description="zip, tar, tar.gz, tgz")
    subdir: str | None = None


class DatasetConfig(BaseModel):
    name: str
    data_dir: Path = Field(default=Path("./datasets"))
    download: DownloadConfig | None = None
    extract: ExtractConfig | None = None
    corpus_relative_path: Path | None = None
    questions_relative_path: Path | None = None
    corpus_path: Path | None = None
    questions_path: Path | None = None
    adapter: str
    adapter_kwargs: Dict[str, Any] = Field(default_factory=dict)
    keep_archive: bool = False
    force_download: bool = False
    description: str | None = None

    def resolve_paths(self, base_path: Path) -> "DatasetConfig":
        values = self.model_dump()
        for key in ("data_dir", "corpus_relative_path", "questions_relative_path", "corpus_path", "questions_path"):
            raw = values.get(key)
            if raw is None:
                continue
            values[key] = (base_path / raw).resolve() if not Path(raw).is_absolute() else Path(raw)
        return DatasetConfig.model_validate(values)


class RetrieverConfig(BaseModel):
    type: str
    llm_model: str | None = None
    embedding_model: str | None = None
    faiss_index: Path | None = None
    top_k: int | None = None
    max_sections: int | None = None
    neighbor_window: int | None = None
    toc_limit: int | None = None


class EvaluationRunConfig(BaseModel):
    suite: str = "baseline"
    questions: Path
    output: Path
    table_output: Path | None = None
    table_format: str | None = None


class ExperimentConfig(BaseModel):
    experiment_name: str
    dataset_configs: List[Path] = Field(default_factory=list)
    corpus: Path | None = None
    pattern: str | None = None
    dataset: str | None = None
    miracl_language: str | None = None
    sqlite_db: Path | None = None
    faiss_dir: Path | None = None
    embedding_model: str | None = None
    rebuild: bool = False
    include_vector: bool = False
    retrievers: List[RetrieverConfig] = Field(default_factory=list)
    hp_rag: Dict[str, Any] = Field(default_factory=dict)
    evaluation: EvaluationRunConfig

    def resolve_paths(self, base_path: Path) -> "ExperimentConfig":
        values = self.model_dump()
        path_fields = [
            "dataset_configs",
            "corpus",
            "sqlite_db",
            "faiss_dir",
        ]
        for field in path_fields:
            raw = values.get(field)
            if raw is None:
                continue
            if field == "dataset_configs":
                values[field] = [
                    (base_path / Path(p)).resolve() if not Path(p).is_absolute() else Path(p)
                    for p in raw
                ]
                continue
            values[field] = (base_path / Path(raw)).resolve() if not Path(raw).is_absolute() else Path(raw)

        eval_values = values.get("evaluation") or {}
        for key in ("questions", "output", "table_output"):
            p = eval_values.get(key)
            if p is None:
                continue
            eval_values[key] = (base_path / Path(p)).resolve() if not Path(p).is_absolute() else Path(p)
        values["evaluation"] = eval_values

        retrievers = []
        for item in values.get("retrievers", []):
            processed = item.copy()
            faiss_path = processed.get("faiss_index")
            if faiss_path is not None:
                processed["faiss_index"] = (
                    (base_path / Path(faiss_path)).resolve()
                    if not Path(faiss_path).is_absolute()
                    else Path(faiss_path)
                )
            retrievers.append(processed)
        values["retrievers"] = retrievers

        return ExperimentConfig.model_validate(values)


__all__ = [
    "DatasetConfig",
    "ExperimentConfig",
    "EvaluationRunConfig",
    "RetrieverConfig",
    "DownloadConfig",
    "ExtractConfig",
]
