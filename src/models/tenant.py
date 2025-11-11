from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(slots=True)
class TenantRecord:
    """Represents a contracting party (tenant) extracted from the corpus."""

    tenant_id: str
    name: str
    role: str
    aliases: List[str] = field(default_factory=list)
    source: str = "document"
    attributes: Dict[str, str] = field(default_factory=dict)

    def alias_string(self) -> str:
        return ", ".join(sorted(set(self.aliases))) if self.aliases else ""


@dataclass(slots=True)
class TenantSelection:
    """Represents the active tenant context for a chat run."""

    tenant_id: str
    name: str
    role: str
    source: str
    confidence: Optional[float] = None
    reason: Optional[str] = None


__all__ = ["TenantRecord", "TenantSelection"]
