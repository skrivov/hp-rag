from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from src.server.chat_service import get_run_snapshot, list_runs
from src.server.settings import Settings, get_settings

router = APIRouter(prefix="/api/traces", tags=["traces"])


@router.get("")
async def list_traces(
    q: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    settings: Settings = Depends(get_settings),
):
    runs = await list_runs(settings, query=q, limit=limit)
    return {"traces": runs}


@router.get("/{run_id}")
async def get_trace(run_id: str, settings: Settings = Depends(get_settings)):
    snapshot = await get_run_snapshot(run_id, settings)
    return snapshot


@router.get("/compare")
async def compare_traces(run_a: str, run_b: str, settings: Settings = Depends(get_settings)):
    trace_a = await get_run_snapshot(run_a, settings)
    trace_b = await get_run_snapshot(run_b, settings)
    if trace_a["run_id"] == trace_b["run_id"]:
        raise HTTPException(status_code=400, detail="Provide two different run_ids")
    return {"run_a": trace_a, "run_b": trace_b}


__all__ = ["router"]
