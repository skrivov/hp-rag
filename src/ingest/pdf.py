from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import fitz  # PyMuPDF

from src.ingest.toc_builder import TOCBuilder, TOCBuilderConfig
from src.ingest.utils import slugify
from src.models.section import SectionNode


@dataclass(slots=True)
class PyMuPDFTOCBuilderConfig(TOCBuilderConfig):
    """Configuration options for PyMuPDF-based TOC parsing."""

    min_level: int = 1


class PyMuPDFTOCBuilder(TOCBuilder):
    """Build a SectionNode tree from a PDF's outline using PyMuPDF."""

    config: PyMuPDFTOCBuilderConfig

    def __init__(self, config: PyMuPDFTOCBuilderConfig | None = None) -> None:
        super().__init__(config or PyMuPDFTOCBuilderConfig())
        assert isinstance(self.config, PyMuPDFTOCBuilderConfig)

    def build(self, document_path: Path) -> SectionNode:
        if not document_path.exists():
            raise FileNotFoundError(f"PDF document not found: {document_path}")

        doc_id = document_path.stem

        root = SectionNode(
            document_id=doc_id,
            path=doc_id,
            title=doc_id,
            body="",
            level=0,
            parent_path=None,
            order=0,
        )

        with fitz.open(document_path) as document:
            toc_entries = document.get_toc()
            if not toc_entries:
                raise ValueError(f"PDF '{document_path}' has no outline to build a TOC from.")

            sections, page_map = self._build_tree(root, toc_entries)
            self._populate_bodies(document, sections, page_map)

        return root

    def _build_tree(
        self,
        root: SectionNode,
        toc_entries: Sequence[Sequence[int | str]],
    ) -> Tuple[List[SectionNode], List[Tuple[int, int]]]:
        stack: List[SectionNode] = [root]
        sections: List[SectionNode] = []
        page_map: List[Tuple[int, int]] = []

        for entry in toc_entries:
            if len(entry) < 3:
                continue
            level_raw, title_raw, page_raw = entry[:3]

            try:
                level = int(level_raw)
            except (TypeError, ValueError):
                continue
            level = max(self.config.min_level, level)
            level = min(level, self.config.max_depth)

            if not isinstance(title_raw, str):
                title = str(title_raw or f"Section {len(sections)+1}")
            else:
                title = title_raw.strip() or f"Section {len(sections)+1}"

            try:
                page = int(page_raw) - 1
            except (TypeError, ValueError):
                page = 0
            page = max(0, page)

            while len(stack) > level:
                stack.pop()

            parent = stack[-1]
            slug = slugify(title)
            path = f"{parent.path}/{slug}" if parent.path else slug
            section = SectionNode(
                document_id=root.document_id,
                path=path,
                title=title,
                body="",
                level=level,
                parent_path=parent.path,
                order=len(parent.children),
            )
            parent.add_child(section)
            stack.append(section)
            sections.append(section)
            page_map.append((level, page))

        return sections, page_map

    def _populate_bodies(
        self,
        document: fitz.Document,
        sections: Sequence[SectionNode],
        page_map: Sequence[Tuple[int, int]],
    ) -> None:
        page_count = document.page_count
        for idx, section in enumerate(sections):
            level, start_page = page_map[idx]
            end_page = page_count

            for jdx in range(idx + 1, len(page_map)):
                next_level, next_start = page_map[jdx]
                if next_level <= level:
                    end_page = next_start
                    break

            end_page = max(start_page, min(end_page, page_count))
            body_text = self._extract_text(document, start_page, end_page)
            section.body = body_text.strip()

    @staticmethod
    def _extract_text(document: fitz.Document, start_page: int, end_page: int) -> str:
        if start_page >= document.page_count or start_page >= end_page:
            return ""

        parts: List[str] = []
        for page_number in range(start_page, end_page):
            try:
                page = document.load_page(page_number)
            except ValueError:
                break
            parts.append(page.get_text("text").strip())
        return "\n".join(part for part in parts if part)


__all__ = ["PyMuPDFTOCBuilder", "PyMuPDFTOCBuilderConfig"]
