from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from src.ingest.utils import slugify
from src.models.section import SectionNode


@dataclass(slots=True)
class SectionFragment:
    """Represents a logical subsection emitted by an adapter."""

    title: str
    text: str


def build_section_tree(
    document_id: str,
    title: str | None,
    fragments: Sequence[SectionFragment],
) -> SectionNode:
    """Construct a SectionNode tree from linear fragments."""

    root_title = title or document_id
    root = SectionNode(
        document_id=document_id,
        path=document_id,
        title=root_title,
        body="",
        level=0,
        parent_path=None,
        order=0,
    )

    for idx, fragment in enumerate(fragments):
        slug = slugify(fragment.title or f"section-{idx+1}")
        path = f"{root.path}/{slug}"
        section = SectionNode(
            document_id=document_id,
            path=path,
            title=fragment.title or f"Section {idx+1}",
            body=fragment.text,
            level=1,
            parent_path=root.path,
            order=idx,
        )
        root.add_child(section)

    return root


class DatasetAdapter(ABC):
    """Base interface for turning external benchmarks into SectionNodes."""

    def __init__(self, source: Path) -> None:
        self.source = source

    @abstractmethod
    def iter_section_roots(self) -> Iterable[SectionNode]:
        """Yield SectionNode trees representing the corpus."""

    def __iter__(self) -> Iterator[SectionNode]:
        return iter(self.iter_section_roots())


__all__ = ["DatasetAdapter", "SectionFragment", "build_section_tree"]
