from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple

import fitz  # type: ignore

from src.ingest.toc_builder import TOCBuilder, TOCBuilderConfig
from src.ingest.utils import slugify
from src.models.section import SectionNode


@dataclass(slots=True)
class PyMuPDFTOCBuilderConfig(TOCBuilderConfig):
    """Configuration for PDF ingestion."""

    heading_tolerance: float = 0.75


class _Entry:
    __slots__ = ("page", "size", "text")

    def __init__(self, page: int, size: float, text: str) -> None:
        self.page = page
        self.size = size
        self.text = text


class PyMuPDFTOCBuilder(TOCBuilder):
    """Create TOC-aware sections from PDFs using outlines or font analysis."""

    config: PyMuPDFTOCBuilderConfig
    _body_font_size: float | None

    _PAGE_PATTERN = re.compile(r"^page\s+\d+(?:\s+of\s+\d+)?$", re.IGNORECASE)
    _DOT_LEADER_PATTERN = re.compile(r"\.{3,}\s*\d+$")
    _ENUM_PATTERN = re.compile(r"^(\d+(?:\.\d+)*)(?:\s*[:\-]?\s*)(.*)$")
    _SECTION_PATTERN = re.compile(r"^(section\s+\d+(?:\.\d+)*)(?:\s*[:\-]?\s*)(.*)$", re.IGNORECASE)
    _STOPWORDS = {
        "the",
        "this",
        "that",
        "these",
        "those",
        "shall",
        "should",
        "will",
        "buyer",
        "seller",
        "compensation",
        "contract",
        "within",
        "whereas",
        "including",
        "provided",
        "subject",
        "after",
        "before",
        "when",
        "with",
        "without",
        "if",
        "for",
        "and",
        "or",
        "of",
        "in",
        "on",
        "to",
        "at",
    }

    def __init__(self, config: PyMuPDFTOCBuilderConfig | None = None) -> None:
        super().__init__(config or PyMuPDFTOCBuilderConfig())
        self._body_font_size = None
        assert isinstance(self.config, PyMuPDFTOCBuilderConfig)

    # ------------------------------------------------------------------ public API
    def build(self, document_path: Path) -> SectionNode:
        if not document_path.exists():
            raise FileNotFoundError(f"PDF document not found: {document_path}")

        document_id = document_path.stem
        root = SectionNode(
            document_id=document_id,
            path=document_id,
            title=document_id or self.config.document_title_fallback,
            body="",
            level=0,
            parent_path=None,
            order=0,
        )

        with fitz.open(document_path) as pdf:
            outline = pdf.get_toc()
            if outline:
                self._build_from_outline(pdf, root, outline)
                return root

            entries = list(self._iter_entries(pdf))
            if not entries:
                raise ValueError("Unable to detect headings in PDF (no usable text).")

            level_sizes = self._heading_sizes(entries)
            self._build_from_fonts(root, entries, level_sizes)

        return root

    # ------------------------------------------------------------------ outline path
    def _build_from_outline(self, pdf: fitz.Document, root: SectionNode, toc: List[List[int | str]]) -> None:
        stack: List[SectionNode] = [root]
        slug_counts: DefaultDict[str, Dict[str, int]] = defaultdict(dict)

        def sanitize(title: str) -> str:
            cleaned = self._sanitize_title(title)
            return cleaned if cleaned else title.strip()

        for idx, entry in enumerate(toc):
            if len(entry) < 3:
                continue
            level_raw, title_raw, page_raw = entry[:3]
            try:
                level = max(1, int(level_raw))
            except (TypeError, ValueError):
                continue
            level = min(level, self.config.max_depth)

            title = sanitize(str(title_raw))
            if not title:
                continue

            try:
                start_page = max(int(page_raw) - 1, 0)
            except (TypeError, ValueError):
                start_page = 0

            while len(stack) > level:
                stack.pop()

            parent = stack[-1]
            slug_base = slugify(title)
            slug_map = slug_counts[parent.path]
            count = slug_map.get(slug_base, 0)
            slug_map[slug_base] = count + 1
            slug = f"{slug_base}-{count}" if count else slug_base
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

            end_page = pdf.page_count
            for future in toc[idx + 1 :]:
                if len(future) < 3:
                    continue
                try:
                    next_page = max(int(future[2]) - 1, 0)
                except (TypeError, ValueError):
                    continue
                if next_page > start_page:
                    end_page = next_page
                    break

            section.body = self._extract_text(pdf, start_page, end_page)

    # ------------------------------------------------------------------ font-based path
    def _build_from_fonts(
        self,
        root: SectionNode,
        entries: Sequence[_Entry],
        level_sizes: Sequence[float],
    ) -> None:
        stack: List[SectionNode] = [root]
        buffers: DefaultDict[str, List[str]] = defaultdict(list)
        slug_counts: DefaultDict[str, Dict[str, int]] = defaultdict(dict)
        doc_title_parts: List[str] = []
        doc_title_node: SectionNode | None = None

        def finalize(node: SectionNode) -> None:
            if buffers[node.path]:
                node.body = "\n".join(buffers[node.path]).strip()
                buffers[node.path] = []

        def ensure_doc_title() -> None:
            nonlocal doc_title_node
            if doc_title_node is not None or not doc_title_parts:
                return

            parent = stack[-1]
            title = " ".join(part for part in doc_title_parts if part).strip()
            if not title:
                title = self.config.document_title_fallback

            slug_base = slugify(title)
            slug_map = slug_counts[parent.path]
            count = slug_map.get(slug_base, 0)
            slug_map[slug_base] = count + 1
            slug = f"{slug_base}-{count}" if count else slug_base
            path = f"{parent.path}/{slug}" if parent.path else slug

            section = SectionNode(
                document_id=root.document_id,
                path=path,
                title=title,
                body="",
                level=1,
                parent_path=parent.path,
                order=len(parent.children),
            )
            parent.add_child(section)

            section_buffer = buffers[path]  # ensure key exists
            parent_buffer = buffers[parent.path]
            if parent_buffer:
                section_buffer.extend(parent_buffer)
                parent_buffer.clear()

            stack.append(section)
            doc_title_parts.clear()
            doc_title_node = section

        def is_doc_title_candidate(
            entry: _Entry,
            level: int | None,
            sanitized_title: str,
        ) -> bool:
            if level != 1 or doc_title_node is not None:
                return False
            if stack[-1] is not root:
                return False
            if entry.page > 0:
                return False
            if not sanitized_title:
                return False
            return not sanitized_title[0].isdigit()

        for entry in entries:
            level, title, remainder = self._determine_heading(entry, level_sizes)
            title_text = (title or entry.text).strip()
            if not title_text:
                buffers[stack[-1].path].append(entry.text.strip())
                continue

            sanitized_title = self._sanitize_title(title_text)
            candidate_title = title_text

            if is_doc_title_candidate(entry, level, sanitized_title):
                doc_title_parts.append(candidate_title)
                if remainder:
                    buffers[root.path].append(remainder)
                continue

            if doc_title_parts and doc_title_node is None:
                ensure_doc_title()

            if level is None:
                buffers[stack[-1].path].append(entry.text.strip())
                continue

            if doc_title_node is None and stack[-1] is root and not doc_title_parts:
                # no detected title but first heading matches root title slug – treat as body
                root_slug = slugify(root.title)
                if sanitized_title and slugify(sanitized_title) == root_slug:
                    if remainder:
                        buffers[root.path].append(remainder)
                    continue

            while len(stack) > level:
                finalize(stack.pop())

            parent = stack[-1]
            slug_seed = self._slug_source(candidate_title, sanitized_title)
            slug_base = slugify(slug_seed)
            slug_map = slug_counts[parent.path]
            count = slug_map.get(slug_base, 0)
            slug_map[slug_base] = count + 1
            slug = f"{slug_base}-{count}" if count else slug_base
            path = f"{parent.path}/{slug}" if parent.path else slug

            section = SectionNode(
                document_id=root.document_id,
                path=path,
                title=candidate_title,
                body="",
                level=level,
                parent_path=parent.path,
                order=len(parent.children),
            )
            parent.add_child(section)
            stack.append(section)
            buffers[section.path] = []
            if remainder:
                buffers[section.path].append(remainder)

        if doc_title_parts and doc_title_node is None:
            ensure_doc_title()

        while stack:
            finalize(stack.pop())

    # ------------------------------------------------------------------ entry scanning helpers
    def _iter_entries(self, pdf: fitz.Document) -> Iterable[_Entry]:
        for page_idx in range(pdf.page_count):
            page = pdf.load_page(page_idx)
            page_dict = page.get_text("dict")
            for block in page_dict.get("blocks", []):
                for line in block.get("lines", []):
                    spans = [span for span in line.get("spans", []) if span.get("text", "").strip()]
                    if not spans:
                        continue
                    text = " ".join(span.get("text", "").strip() for span in spans).strip()
                    if not text or self._skip_line(text):
                        continue
                    size = max(float(span.get("size", 0.0)) for span in spans)
                    yield _Entry(page_idx, size, text)

    def _skip_line(self, text: str) -> bool:
        normalized = text.strip()
        if not normalized:
            return True
        if self._PAGE_PATTERN.match(normalized):
            return True
        if self._DOT_LEADER_PATTERN.search(normalized):
            return True
        return False

    # ------------------------------------------------------------------ heading heuristics
    def _heading_sizes(self, entries: Sequence[_Entry]) -> List[float]:
        counter: Counter = Counter(round(entry.size, 2) for entry in entries)
        if not counter:
            return []

        body_size, body_count = counter.most_common(1)[0]
        self._body_font_size = body_size
        heading_sizes = sorted(
            [
                size
                for size, count in counter.items()
                if size > body_size + self.config.heading_tolerance and count < body_count
            ],
            reverse=True,
        )
        if not heading_sizes:
            heading_sizes = sorted(counter.keys(), reverse=True)
        return heading_sizes[: self.config.max_depth]

    def _determine_heading(self, entry: _Entry, level_sizes: Sequence[float]) -> Tuple[int | None, str, str]:
        text = entry.text.strip()

        level = self._match_level(entry.size, level_sizes)
        if level is not None:
            return level, text, ""

        match = self._SECTION_PATTERN.match(text)
        if match:
            numeric = re.sub(r"[^0-9.]", "", match.group(1))
            if not numeric:
                return None, "", ""
            if "." in numeric:
                return None, "", ""
            level = numeric.count(".") + 1
            rest = match.group(2).strip()
            title, remainder = self._split_title_body(rest)
            if not any(ch.isalpha() for ch in title):
                return None, "", ""
            remainder_text = remainder or ""
            display_title = text
            if remainder_text and display_title.endswith(remainder_text):
                display_title = display_title[: -len(remainder_text)].rstrip(" -:;.,")
            return min(level, self.config.max_depth), display_title or text, remainder_text

        match = self._ENUM_PATTERN.match(text)
        if match:
            index = match.group(1)
            if "." in index:
                return None, "", ""
            level = index.count(".") + 1
            rest = match.group(2).strip()
            if self._body_font_size is not None and abs(entry.size - self._body_font_size) <= self.config.heading_tolerance:
                if len(rest.split()) >= 6:
                    return None, "", ""
            title, remainder = self._split_title_body(rest)
            if rest and not any(ch.isalpha() for ch in title):
                return None, "", ""
            remainder_text = remainder or ""
            display_title = text
            if remainder_text and display_title.endswith(remainder_text):
                display_title = display_title[: -len(remainder_text)].rstrip(" -:;.,")
            return min(level, self.config.max_depth), display_title or text, remainder_text

        return None, "", ""

    def _match_level(self, size: float, levels: Sequence[float]) -> int | None:
        for index, candidate in enumerate(levels, start=1):
            if abs(candidate - size) <= self.config.heading_tolerance:
                return min(index, self.config.max_depth)
        return None

    def _split_title_body(self, text: str) -> Tuple[str, str]:
        tokens = text.split()
        if not tokens:
            return "", ""
        title_tokens: List[str] = [tokens[0]]
        remainder_tokens: List[str] = []

        for idx, token in enumerate(tokens[1:], start=1):
            normalized = token.strip(",.;:").lower()
            if normalized in self._STOPWORDS:
                remainder_tokens = tokens[idx:]
                break
            if idx <= 2:
                title_tokens.append(token)
            else:
                remainder_tokens = tokens[idx:]
                break

        if not remainder_tokens and len(tokens) > len(title_tokens):
            remainder_tokens = tokens[len(title_tokens):]

        title = " ".join(title_tokens).strip()
        remainder = " ".join(remainder_tokens).strip()
        return title, remainder

    def _sanitize_title(self, text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r"^(section\s+\d+(?:\.\d+)*)\s*[:\-]?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^(\d+(?:\.\d+)*)(?:\.)?\s*[:\-]?\s*", "", cleaned)
        cleaned = cleaned.lstrip(".:;-" + "\u2013\u2014\u2022" + " ")
        return cleaned.strip()

    def _slug_source(self, title: str, sanitized: str) -> str:
        base = title or sanitized or "section"
        base = re.sub(r"\s*\u2011\s*", "\u2011", base)
        base = re.sub(r"\s*\u2010\s*", "-", base)
        base = re.sub(r"\s*[\u2012-\u2015]\s*", " - ", base)
        base = re.sub(r"\s+", " ", base).strip()
        return base

    # ------------------------------------------------------------------ extraction helpers
    def _extract_text(self, pdf: fitz.Document, start_page: int, end_page: int) -> str:
        lines: List[str] = []
        for page_idx in range(start_page, max(start_page + 1, end_page)):
            page = pdf.load_page(page_idx)
            page_dict = page.get_text("dict")
            for block in page_dict.get("blocks", []):
                for line in block.get("lines", []):
                    spans = [span for span in line.get("spans", []) if span.get("text", "").strip()]
                    if not spans:
                        continue
                    text = " ".join(span.get("text", "").strip() for span in spans).strip()
                    if not text or self._skip_line(text):
                        continue
                    lines.append(text)
        return "\n".join(lines).strip()


__all__ = ["PyMuPDFTOCBuilder", "PyMuPDFTOCBuilderConfig"]
