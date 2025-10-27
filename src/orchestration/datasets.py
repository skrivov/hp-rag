from __future__ import annotations

import shutil
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping
from urllib.parse import urlparse
from urllib.request import urlopen


@dataclass(slots=True)
class DatasetSpec:
    """Represents a benchmark dataset available to the workflow."""

    name: str
    download_url: str | None
    archive_type: str | None = None  # "zip", "tar", "tar.gz", "tgz"
    archive_subdir: str | None = None
    corpus_relative_path: str | None = None
    questions_relative_path: str | None = None
    adapter_name: str | None = None
    adapter_kwargs: dict[str, object] = field(default_factory=dict)
    archive_filename: str | None = None
    keep_archive: bool = False
    description: str | None = None


_DATASETS: MutableMapping[str, DatasetSpec] = {}


def register_dataset(spec: DatasetSpec, *, override: bool = False) -> None:
    key = spec.name.lower()
    if not override and key in _DATASETS:
        raise KeyError(f"Dataset '{spec.name}' already registered")
    _DATASETS[key] = spec


def unregister_dataset(name: str) -> None:
    _DATASETS.pop(name.lower(), None)


def get_dataset_spec(name: str) -> DatasetSpec:
    key = name.lower()
    if key not in _DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Registered datasets: {', '.join(sorted(_DATASETS))}")
    return _DATASETS[key]


def available_datasets() -> Mapping[str, DatasetSpec]:
    return dict(_DATASETS)


def dataset_storage_path(spec: DatasetSpec, data_root: Path) -> Path:
    dataset_dir = data_root / spec.name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir


def download_dataset(spec: DatasetSpec, data_root: Path, *, force: bool = False) -> Path:
    dataset_dir = dataset_storage_path(spec, data_root)
    if spec.download_url is None:
        return dataset_dir

    archive_name = spec.archive_filename or Path(urlparse(spec.download_url).path).name
    archive_path = dataset_dir / archive_name

    if force or not archive_path.exists():
        _stream_download(spec.download_url, archive_path)

    corpus_target = dataset_dir / spec.corpus_relative_path if spec.corpus_relative_path else None
    needs_extract = spec.archive_type is not None and (
        force or corpus_target is None or not corpus_target.exists()
    )

    if needs_extract:
        _clear_dataset_contents(dataset_dir, keep={archive_path})
        _extract_archive(archive_path, dataset_dir, spec.archive_type, spec.archive_subdir)

    if not spec.keep_archive and archive_path.exists() and spec.archive_type:
        archive_path.unlink()

    return dataset_dir


def _stream_download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _clear_dataset_contents(dataset_dir: Path, *, keep: Iterable[Path] = ()) -> None:
    keep = {path.resolve() for path in keep}
    for child in dataset_dir.iterdir():
        if child.resolve() in keep:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _extract_archive(archive_path: Path, destination: Path, archive_type: str | None, subdir: str | None) -> None:
    if archive_type is None:
        return

    archive_type = archive_type.lower()
    if archive_type == "zip":
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(destination)
    elif archive_type in {"tar", "tar.gz", "tgz"}:
        mode = "r:gz" if archive_type in {"tar.gz", "tgz"} else "r:"
        with tarfile.open(archive_path, mode) as archive:
            archive.extractall(destination)
    else:
        raise ValueError(f"Unsupported archive type '{archive_type}' for {archive_path}")

    if subdir:
        extracted_root = destination / subdir
        if extracted_root.exists():
            for item in extracted_root.iterdir():
                target = destination / item.name
                if target.exists():
                    continue
                item.rename(target)
            shutil.rmtree(extracted_root, ignore_errors=True)


# Register default datasets -------------------------------------------------

register_dataset(
    DatasetSpec(
        name="beir-fiqa",
        download_url="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
        archive_type="zip",
        corpus_relative_path="fiqa/corpus.jsonl",
        questions_relative_path="fiqa/queries.jsonl",
        adapter_name="beir",
        description="BEIR FiQA financial QA corpus (question-answering over financial documents)",
    )
)

register_dataset(
    DatasetSpec(
        name="miracl-en",
        download_url="https://storage.googleapis.com/miracl/v1/datasets/miracl-v1.0-en-queries.tar",
        archive_type="tar.gz",
        corpus_relative_path="miracl-v1.0-en-queries/corpus.tsv",
        questions_relative_path="miracl-v1.0-en-queries/dev-queries.jsonl",
        adapter_name="miracl",
        adapter_kwargs={"language": "en"},
        description="MIRACL English multilingual retrieval dataset",
    )
)

register_dataset(
    DatasetSpec(
        name="hotpotqa",
        download_url="https://hotpotqa.s3.us-east-2.amazonaws.com/wiki/wiki.zip",
        archive_type="zip",
        corpus_relative_path="wiki/wiki.json",
        questions_relative_path=None,
        adapter_name="hotpotqa",
        description="HotpotQA supporting facts corpus",
    )
)
