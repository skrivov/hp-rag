from __future__ import annotations

from fastapi import APIRouter, Depends

from src.server.documents import get_document_service_instance
from src.server.settings import Settings, get_settings

router = APIRouter(prefix="/api/navigation", tags=["navigation"])


@router.get("")
async def get_navigation(settings: Settings = Depends(get_settings)):
    document_service = get_document_service_instance(settings)
    document_plugins = [
        {
            "name": plugin.name,
            "label": plugin.display_name,
            "capabilities": plugin.capabilities,
        }
        for plugin in document_service.registry.plugins.values()
    ]
    # Chat plugins not implemented yet; placeholder data.
    chat_plugins: list[dict[str, str]] = []
    return {
        "documents": document_plugins,
        "chat": chat_plugins,
        "traces": {"list": "/api/traces", "compare": "/api/traces/compare"},
    }


__all__ = ["router"]
