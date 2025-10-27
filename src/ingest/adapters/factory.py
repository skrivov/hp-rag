from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Type

from src.ingest.adapters import (
    BeirCorpusAdapter,
    DatasetAdapter,
    HotpotQAAdapter,
    MiraclAdapter,
    SquadAdapter,
)


@dataclass(slots=True)
class AdapterSpec:
    name: str
    cls: Type[DatasetAdapter]


_ADAPTERS: Dict[str, AdapterSpec] = {
    "beir": AdapterSpec(name="beir", cls=BeirCorpusAdapter),
    "squad": AdapterSpec(name="squad", cls=SquadAdapter),
    "hotpotqa": AdapterSpec(name="hotpotqa", cls=HotpotQAAdapter),
    "miracl": AdapterSpec(name="miracl", cls=MiraclAdapter),
}


def adapter_choices() -> Mapping[str, AdapterSpec]:
    return _ADAPTERS


def create_adapter(name: str, source: Path, **kwargs) -> DatasetAdapter:
    key = name.lower()
    if key not in _ADAPTERS:
        raise KeyError(f"Unknown adapter '{name}'")
    spec = _ADAPTERS[key]
    return spec.cls(source, **kwargs)


__all__ = ["adapter_choices", "create_adapter", "AdapterSpec"]
