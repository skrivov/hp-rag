from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol

from fastapi import APIRouter

from src.hp_rag.storage import SQLiteHyperlinkStore
from src.models.document import PluginStatus
from src.rag.storage import FaissVectorConfig


@dataclass(slots=True)
class PluginContext:
    document_id: str
    sqlite_store: SQLiteHyperlinkStore
    faiss_config: FaissVectorConfig | None = None
    extra: Dict[str, Any] | None = None


class RetrievalPlugin(Protocol):
    name: str
    display_name: str
    capabilities: Dict[str, Any]

    async def ingest(self, context: PluginContext) -> Dict[str, Any]:
        """Run plugin-specific ingestion. Return stats dict."""

    def router(self) -> APIRouter:
        """Return plugin-specific FastAPI router."""

    async def cleanup_all(self) -> None:
        """Remove any plugin-scoped artifacts."""


class PluginRegistry:
    """Keeps track of registered retrieval plugins."""

    def __init__(self) -> None:
        self._plugins: Dict[str, RetrievalPlugin] = {}

    def register(self, plugin: RetrievalPlugin) -> None:
        self._plugins[plugin.name] = plugin

    @property
    def plugins(self) -> Dict[str, RetrievalPlugin]:
        return dict(self._plugins)

    def get(self, name: str) -> RetrievalPlugin | None:
        return self._plugins.get(name)


__all__ = ["PluginContext", "RetrievalPlugin", "PluginRegistry", "PluginStatus"]
