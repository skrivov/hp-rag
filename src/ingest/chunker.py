from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

from src.models.section import SectionNode


@dataclass(slots=True)
class ChunkConfig:
    """Chunking strategy configuration."""

    max_tokens: int = 512
    overlap_tokens: int = 64


@dataclass(slots=True)
class SectionChunk:
    """Represents a chunk derived from a SectionNode."""

    path: str
    text: str
    order: int
    parent_title: str


class Chunker(ABC):
    """Abstract chunker for splitting SectionNode content."""

    def __init__(self, config: ChunkConfig | None = None) -> None:
        self.config = config or ChunkConfig()

    @abstractmethod
    def chunk(self, section: SectionNode) -> Iterable[SectionChunk]:
        """Yield ordered chunks from a SectionNode."""


__all__ = ["Chunker", "ChunkConfig", "SectionChunk"]
