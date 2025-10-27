from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from src.ingest.toc_builder import TOCBuilder, TOCBuilderConfig
from src.ingest.utils import slugify
from src.models.section import SectionNode


_HEADING_PATTERN = re.compile(r"^(?P<hashes>#+)\s+(?P<title>.+?)\s*$")


@dataclass(slots=True)
class MarkdownTOCBuilderConfig(TOCBuilderConfig):
    """Configuration for markdown parsing."""

    document_title_fallback: str = "Document"


class MarkdownTOCBuilder(TOCBuilder):
    """Parse markdown headings (#, ##, ###, ...) into SectionNode trees."""

    config: MarkdownTOCBuilderConfig

    def __init__(self, config: MarkdownTOCBuilderConfig | None = None) -> None:
        super().__init__(config or MarkdownTOCBuilderConfig())
        assert isinstance(self.config, MarkdownTOCBuilderConfig)

    def build(self, document_path: Path) -> SectionNode:
        text = document_path.read_text(encoding="utf-8")
        document_id = document_path.stem

        root = SectionNode(
            document_id=document_id,
            path=document_id,
            title=document_id or self.config.document_title_fallback,
            body=text if self.config.include_body else "",
            level=0,
            parent_path=None,
            order=0,
        )

        stack: List[SectionNode] = [root]
        buffers: Dict[str, List[str]] = defaultdict(list)
        finalized: Dict[str, bool] = {}

        for line in text.splitlines():
            match = _HEADING_PATTERN.match(line)
            if not match:
                buffers[stack[-1].path].append(line)
                continue

            level = min(len(match.group("hashes")), self.config.max_depth)
            title = match.group("title").strip()

            while len(stack) > level:
                node = stack.pop()
                if not finalized.get(node.path):
                    node.body = "\n".join(buffers[node.path]).strip()
                    finalized[node.path] = True

            parent = stack[-1]
            slug = slugify(title)
            path = f"{parent.path}/{slug}" if parent.path else slug
            section = SectionNode(
                document_id=document_id,
                path=path,
                title=title,
                body="",
                level=level,
                parent_path=parent.path,
                order=len(parent.children),
            )
            parent.add_child(section)
            stack.append(section)

        # Finalize remaining buffers
        while stack:
            node = stack.pop()
            if not finalized.get(node.path):
                node.body = "\n".join(buffers[node.path]).strip()
                finalized[node.path] = True

        return root


__all__ = ["MarkdownTOCBuilder", "MarkdownTOCBuilderConfig"]
