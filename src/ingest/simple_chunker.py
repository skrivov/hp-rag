from __future__ import annotations

import re
from typing import Iterable, List, Sequence

from src.ingest.chunker import Chunker, SectionChunk
from src.models.section import SectionNode


class ParagraphChunker(Chunker):
    """Split section bodies into short, overlap-aware chunks."""

    def chunk(self, section: SectionNode) -> Iterable[SectionChunk]:
        body = section.body.strip()
        if not body:
            return []

        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        chunks: List[SectionChunk] = []
        order = 0

        for paragraph in paragraphs:
            for fragment in self._split_paragraph(paragraph):
                chunks.append(
                    SectionChunk(
                        path=section.path,
                        text=fragment,
                        order=order,
                        parent_title=section.title,
                    )
                )
                order += 1

        return chunks

    def _split_paragraph(self, paragraph: str) -> Sequence[str]:
        """Approximate token-aware splitting using whitespace boundaries."""

        max_tokens = max(1, self.config.max_tokens)
        overlap = max(0, min(self.config.overlap_tokens, max_tokens - 1))

        tokens = re.findall(r"\S+\s*", paragraph)
        if len(tokens) <= max_tokens:
            return [paragraph]

        fragments: List[str] = []
        start = 0
        while start < len(tokens):
            end = min(len(tokens), start + max_tokens)
            fragment = "".join(tokens[start:end]).strip()
            if fragment:
                fragments.append(fragment)
            if end == len(tokens):
                break
            start = end - overlap if overlap else end

        return fragments


__all__ = ["ParagraphChunker"]
