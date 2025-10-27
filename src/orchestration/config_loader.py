from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from src.models.configs import DatasetConfig, ExperimentConfig
from src.orchestration.datasets import DatasetSpec, download_dataset


@dataclass(slots=True)
class PreparedDataset:
    config: DatasetConfig
    dataset_dir: Path
    corpus_path: Path
    questions_path: Path | None
    adapter_kwargs: Dict[str, Any]


def _load_structured_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    ext = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if ext in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    elif ext == ".toml":
        data = tomllib.loads(text)
    elif ext == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported config format '{ext}' for {path}")

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping at the top level")
    return data


def load_dataset_config(path: Path) -> DatasetConfig:
    raw = _load_structured_file(path)
    config = DatasetConfig.model_validate(raw)
    return config.resolve_paths(path.parent)


def load_experiment_config(path: Path) -> ExperimentConfig:
    raw = _load_structured_file(path)
    config = ExperimentConfig.model_validate(raw)
    return config.resolve_paths(path.parent)


def _to_dataset_spec(config: DatasetConfig) -> DatasetSpec:
    download_url = config.download.url if config.download else None
    archive_type = config.extract.archive_type if config.extract else None
    archive_subdir = config.extract.subdir if config.extract else None
    archive_filename = config.download.filename if config.download else None

    return DatasetSpec(
        name=config.name,
        download_url=download_url,
        archive_type=archive_type,
        archive_subdir=archive_subdir,
        corpus_relative_path=str(config.corpus_relative_path) if config.corpus_relative_path else None,
        questions_relative_path=str(config.questions_relative_path) if config.questions_relative_path else None,
        adapter_name=config.adapter,
        adapter_kwargs=config.adapter_kwargs,
        archive_filename=archive_filename,
        keep_archive=config.keep_archive,
        description=config.description,
    )


def prepare_dataset(config: DatasetConfig) -> PreparedDataset:
    if config.download:
        dataset_dir = download_dataset(
            _to_dataset_spec(config),
            config.data_dir,
            force=config.force_download,
        ).resolve()
    else:
        dataset_dir = (
            config.corpus_path.parent if config.corpus_path else config.data_dir
        ).resolve()
        dataset_dir.mkdir(parents=True, exist_ok=True)

    if config.corpus_path:
        corpus_path = config.corpus_path.resolve()
    elif config.corpus_relative_path:
        corpus_path = (dataset_dir / config.corpus_relative_path).resolve()
    else:
        corpus_path = dataset_dir

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus path {corpus_path} derived from dataset '{config.name}' does not exist")

    if config.questions_path:
        questions_path = config.questions_path.resolve()
    elif config.questions_relative_path:
        questions_path = (dataset_dir / config.questions_relative_path).resolve()
    else:
        questions_path = None

    if questions_path and not questions_path.exists():
        raise FileNotFoundError(
            f"Questions path {questions_path} derived from dataset '{config.name}' does not exist"
        )

    return PreparedDataset(
        config=config,
        dataset_dir=dataset_dir,
        corpus_path=corpus_path,
        questions_path=questions_path,
        adapter_kwargs=dict(config.adapter_kwargs),
    )


def prepare_datasets(configs: Iterable[DatasetConfig]) -> List[PreparedDataset]:
    return [prepare_dataset(cfg) for cfg in configs]


__all__ = [
    "PreparedDataset",
    "load_dataset_config",
    "load_experiment_config",
    "prepare_dataset",
    "prepare_datasets",
]
