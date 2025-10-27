from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.ingest.adapters.base import DatasetAdapter, SectionFragment, build_section_tree
from src.ingest.utils import slugify
from src.models.section import SectionNode


class HotpotQAAdapter(DatasetAdapter):
    """Adapter for HotpotQA JSONL or JSON corpora."""

    def __init__(self, source: Path) -> None:
        super().__init__(source)
        if not self.source.exists():
            raise FileNotFoundError(f"HotpotQA data not found at {self.source}")

    def iter_section_roots(self) -> Iterable[SectionNode]:
        if self.source.suffix == ".json":
            with self.source.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            for entry in data:
                yield from self._build_from_entry(entry)
        else:
            with self.source.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    yield from self._build_from_entry(entry)

    def _build_from_entry(self, entry: dict) -> Iterable[SectionNode]:
        contexts = entry.get("context", [])
        entry_id = str(entry.get("_id") or entry.get("id") or "sample")
        for idx, ctx in enumerate(contexts):
            if not isinstance(ctx, (list, tuple)) or len(ctx) != 2:
                continue
            title, sentences = ctx
            title = title or f"context-{idx}"
            title_slug = slugify(title)
            text = " ".join(sentence.strip() for sentence in sentences if sentence.strip())
            if not text:
                continue
            fragments = [SectionFragment(title="Passage", text=text)]
            doc_id = f"{entry_id}-{title_slug}"
            yield build_section_tree(doc_id, title, fragments)


__all__ = ["HotpotQAAdapter"]
