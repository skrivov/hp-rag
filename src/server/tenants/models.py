from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class TenantModel(BaseModel):
    tenant_id: str
    name: str
    role: str | None = None
    aliases: List[str] = Field(default_factory=list)
    document_count: int = 0


class TenantDetail(TenantModel):
    documents: List[str] = Field(default_factory=list)


class TenantListResponse(BaseModel):
    items: List[TenantModel]


__all__ = ["TenantModel", "TenantDetail", "TenantListResponse"]
