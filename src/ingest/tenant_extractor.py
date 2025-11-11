from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence, Set

from src.ingest.utils import slugify
from src.models.section import SectionNode
from src.models.tenant import TenantRecord


@dataclass(slots=True)
class TenantExtractorConfig:
    """Configuration for heuristic tenant extraction."""

    max_preamble_chars: int = 4000
    max_lines: int = 80
    include_roles: Sequence[str] = ()


class TenantExtractor:
    """Extract contracting parties from markdown/PDF preambles."""

    _LINE_PATTERN = re.compile(r'^\*\*(?P<label>[^*]+)\*\*:\s*(?P<body>.+)$')
    _ALIAS_PATTERN = re.compile(r'\("([^"]+)"\)')

    def __init__(self, config: TenantExtractorConfig | None = None) -> None:
        self.config = config or TenantExtractorConfig()

    def extract(self, document: SectionNode) -> List[TenantRecord]:
        preamble = self._preamble(document)
        if not preamble:
            return []

        tenants: List[TenantRecord] = []
        seen_ids: Set[str] = set()

        for raw_line in preamble.splitlines():
            line = raw_line.strip()
            if not line or not line.startswith("**"):
                continue
            match = self._LINE_PATTERN.match(line)
            if not match:
                continue

            label = match.group("label").strip()
            body = match.group("body").strip()
            if not label or not body:
                continue
            if self.config.include_roles and label not in self.config.include_roles:
                continue

            name = self._extract_name(body)
            if not name:
                continue

            tenant_id = self._make_tenant_id(name, label, document.document_id, seen_ids)
            aliases = self._extract_aliases(body, label)
            tenants.append(
                TenantRecord(
                    tenant_id=tenant_id,
                    name=name,
                    role=label,
                    aliases=aliases,
                    source="document",
                )
            )
            seen_ids.add(tenant_id)

        return tenants

    def _preamble(self, document: SectionNode) -> str:
        blocks: List[str] = []
        if document.body:
            blocks.append(document.body)
        for child in document.children[:3]:
            if child.body:
                blocks.append(child.body)

        if not blocks:
            return ""

        combined = "\n".join(blocks)
        lines = combined.splitlines()[: self.config.max_lines]
        excerpt = "\n".join(lines)
        return excerpt[: self.config.max_preamble_chars]

    def _extract_name(self, body: str) -> str:
        candidate = body
        alias_index = candidate.find("(")
        if alias_index != -1:
            candidate = candidate[:alias_index]

        qualifier_match = re.search(r",\s+(an?|the)\s", candidate, flags=re.IGNORECASE)
        if qualifier_match:
            candidate = candidate[: qualifier_match.start()]

        candidate = candidate.strip().rstrip(" .,")
        if not candidate:
            return ""
        return candidate

    def _extract_aliases(self, body: str, label: str) -> List[str]:
        aliases = set(self._ALIAS_PATTERN.findall(body))
        cleaned_label = label.strip()
        if cleaned_label:
            aliases.add(cleaned_label)
        return sorted(alias for alias in aliases if alias)

    def _make_tenant_id(self, name: str, role: str, document_id: str, seen: Set[str]) -> str:
        base = slugify(name, fallback=f"{document_id}-{slugify(role or 'party')}")
        if base not in seen:
            return base
        suffix = 2
        while True:
            candidate = f"{base}-{suffix}"
            if candidate not in seen:
                return candidate
            suffix += 1


__all__ = ["TenantExtractor", "TenantExtractorConfig"]
