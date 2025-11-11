from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class DocumentStatus(str, Enum):
    QUEUED = "queued"
    INGESTING = "ingesting"
    READY = "ready"
    FAILED = "failed"


class PluginStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    READY = "ready"
    FAILED = "failed"


@dataclass(slots=True)
class PluginState:
    plugin_name: str
    status: PluginStatus
    stats: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass(slots=True)
class DocumentMetadata:
    id: str
    name: str
    source_path: str
    mime_type: Optional[str]
    size_bytes: int
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime
    error: Optional[str] = None
    section_count: int = 0
    chunk_count: int = 0
    plugin_states: List[PluginState] = field(default_factory=list)
    tenants: List["DocumentTenant"] = field(default_factory=list)


@dataclass(slots=True)
class DocumentTenant:
    tenant_id: str
    name: str
    role: Optional[str] = None


__all__ = [
    "DocumentTenant",
    "DocumentMetadata",
    "DocumentStatus",
    "PluginState",
    "PluginStatus",
]
