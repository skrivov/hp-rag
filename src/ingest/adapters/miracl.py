from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List

from src.ingest.adapters.base import DatasetAdapter, SectionFragment, build_section_tree
from src.ingest.utils import slugify
from src.models.section import SectionNode


class MiraclAdapter(DatasetAdapter):
    """Adapter for MIRACL per-language TSV corpora."""

    def __init__(self, source: Path, *, language: str | None = None) -> None:
        super().__init__(source)
        if source.is_dir():
            if language is None:
                raise ValueError("language must be provided when source is a directory")
            candidate = source / f"{language}.tsv"
            if not candidate.exists():
                raise FileNotFoundError(f"MIRACL corpus not found for language '{language}' at {candidate}")
            self.corpus_path = candidate
        else:
            self.corpus_path = source
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"MIRACL corpus not found at {self.corpus_path}")

    def iter_section_roots(self) -> Iterable[SectionNode]:
        with self.corpus_path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter="\t")
            for row in reader:
                if not row:
                    continue
                doc_id = row[0]
                title = row[2] if len(row) > 2 else None
                text = row[3] if len(row) > 3 else (row[1] if len(row) > 1 else "")
                fragments = self._build_fragments(text)
                doc_slug = slugify(doc_id)
                yield build_section_tree(doc_slug, title or doc_id, fragments)

    @staticmethod
    def _build_fragments(text: str) -> List[SectionFragment]:
        text = text.strip()
        if not text:
            return [SectionFragment(title="Passage", text="")]
        return [SectionFragment(title="Passage", text=text)]


__all__ = ["MiraclAdapter"]
