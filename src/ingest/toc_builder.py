from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.models.section import SectionNode


@dataclass(slots=True)
class TOCBuilderConfig:
    """Configuration for parsing documents into hierarchical sections."""

    max_depth: int = 4
    include_body: bool = True


class TOCBuilder(ABC):
    """Abstract base class for turning documents into TOC-aware SectionNode trees."""

    def __init__(self, config: TOCBuilderConfig | None = None) -> None:
        self.config = config or TOCBuilderConfig()

    @abstractmethod
    def build(self, document_path: Path) -> SectionNode:
        """Parse a single document and return the root SectionNode."""

    def build_many(self, paths: Iterable[Path]) -> Iterable[SectionNode]:
        """Utility for parsing multiple documents."""

        for path in paths:
            yield self.build(path)


__all__ = ["TOCBuilder", "TOCBuilderConfig"]
