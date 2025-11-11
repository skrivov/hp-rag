from __future__ import annotations

import json
from typing import Iterator, List

from fastapi import APIRouter, Depends, HTTPException, Query

from src.hp_rag.storage import SQLiteHyperlinkConfig, SQLiteHyperlinkStore
from src.server.settings import Settings, get_settings
from .models import TenantDetail, TenantListResponse, TenantModel


router = APIRouter(prefix="/api/tenants", tags=["tenants"])


def _alias_list(raw: str | None) -> List[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
        if isinstance(value, list):
            return [str(item) for item in value if item]
    except json.JSONDecodeError:
        return []
    return []


def _row_to_model(row) -> TenantModel:
    return TenantModel(
        tenant_id=row["tenant_id"],
        name=row["name"],
        role=row["role"],
        aliases=_alias_list(row["aliases"]),
        document_count=row["document_count"],
    )


def get_tenant_store(settings: Settings = Depends(get_settings)) -> Iterator[SQLiteHyperlinkStore]:
    store = SQLiteHyperlinkStore(SQLiteHyperlinkConfig(db_path=settings.sqlite_db_path))
    store.initialize()
    try:
        yield store
    finally:
        store.close()


@router.get("", response_model=TenantListResponse)
def list_tenants(
    q: str | None = Query(default=None, description="Filter tenants by name or role"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    store: SQLiteHyperlinkStore = Depends(get_tenant_store),
) -> TenantListResponse:
    rows = store.list_tenants(query=q, limit=limit, offset=offset)
    items = [_row_to_model(row) for row in rows]
    return TenantListResponse(items=items)


@router.get("/{tenant_id}", response_model=TenantDetail)
def get_tenant(
    tenant_id: str,
    store: SQLiteHyperlinkStore = Depends(get_tenant_store),
) -> TenantDetail:
    row = store.fetch_tenant(tenant_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Tenant not found")
    documents = store.list_documents_for_tenant(tenant_id)
    base = _row_to_model(row)
    return TenantDetail(**base.model_dump(), documents=documents)


__all__ = ["router"]
