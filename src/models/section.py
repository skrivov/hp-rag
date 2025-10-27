from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(slots=True)
class SectionNode:
    """Represents a TOC-aware section extracted from the corpus."""

    document_id: str
    path: str
    title: str
    body: str
    level: int
    parent_path: Optional[str] = None
    order: int = 0
    children: List["SectionNode"] = field(default_factory=list)

    def add_child(self, child: "SectionNode") -> None:
        """Attach a child node while keeping the TOC hierarchy consistent."""

        child.order = len(self.children)
        child.parent_path = self.path
        self.children.append(child)


__all__ = ["SectionNode"]
