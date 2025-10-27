from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from src.ingest.adapters.base import DatasetAdapter, SectionFragment, build_section_tree
from src.ingest.utils import slugify
from src.models.section import SectionNode


class SquadAdapter(DatasetAdapter):
    """Adapter for SQuAD v1.1/v2.0 style JSON files."""

    def __init__(self, source: Path) -> None:
        super().__init__(source)
        if not self.source.exists():
            raise FileNotFoundError(f"SQuAD dataset not found at {self.source}")

        with self.source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or "data" not in payload:
            raise ValueError("Invalid SQuAD payload: expected top-level 'data' list")
        self._dataset = payload

    def iter_section_roots(self) -> Iterable[SectionNode]:
        data = self._dataset.get("data", [])
        for entry in data:
            title = entry.get("title") or "Untitled"
            articles = entry.get("paragraphs", [])
            title_slug = slugify(title or "article")
            for idx, paragraph in enumerate(articles):
                context = paragraph.get("context", "").strip()
                fragments = self._build_fragments(context)
                doc_id = f"{title_slug}-{idx}"
                yield build_section_tree(doc_id, title, fragments)

    @staticmethod
    def _build_fragments(context: str) -> List[SectionFragment]:
        if not context:
            return [SectionFragment(title="Paragraph", text="")]
        return [SectionFragment(title="Paragraph", text=context)]


__all__ = ["SquadAdapter"]
