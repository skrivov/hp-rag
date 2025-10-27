from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from src.ingest.adapters.base import DatasetAdapter, SectionFragment, build_section_tree
from src.models.section import SectionNode


class BeirCorpusAdapter(DatasetAdapter):
    """Adapter for BEIR-style corpora stored as JSONL files."""

    def __init__(self, source: Path, *, corpus_filename: str = "corpus.jsonl") -> None:
        super().__init__(source)
        self.corpus_path = source if source.is_file() else source / corpus_filename
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"BEIR corpus not found at {self.corpus_path}")

    def iter_section_roots(self) -> Iterable[SectionNode]:
        with self.corpus_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                doc_id = str(record.get("_id") or record.get("id"))
                if not doc_id:
                    continue
                title = record.get("title")
                text = record.get("text") or ""
                fragments = self._build_fragments(text)
                yield build_section_tree(doc_id, title, fragments)

    @staticmethod
    def _build_fragments(text: str) -> List[SectionFragment]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [text.strip()] if text.strip() else []
        if not paragraphs:
            paragraphs = [""]
        return [
            SectionFragment(title=f"Passage {idx+1}", text=paragraph)
            for idx, paragraph in enumerate(paragraphs)
        ]


__all__ = ["BeirCorpusAdapter"]
