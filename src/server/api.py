from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette import EventSourceResponse

from .chat_service import get_run_snapshot, start_run, stream_run
from .documents import router as documents_router, get_document_service_instance
from .navigation.router import router as navigation_router
from .traces.router import router as traces_router
from .tenants import router as tenants_router
from .models import RunCreate
from .settings import Settings, get_settings

app = FastAPI(title="AgentLab API", version="0.1.0")
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(documents_router)
app.include_router(traces_router)
app.include_router(navigation_router)
app.include_router(tenants_router)

# Register plugin routers
document_service = get_document_service_instance(settings)
for plugin in document_service.registry.plugins.values():
    app.include_router(plugin.router())


@app.post("/api/chat/runs")
async def create_run(payload: RunCreate, settings: Settings = Depends(get_settings)) -> dict[str, str]:
    run_id = await start_run(payload, settings=settings)
    return {"run_id": run_id}


@app.get("/api/chat/stream")
async def chat_stream(
    request: Request,
    run_id: str | None = None,
    system: str | None = None,
    q: str | None = None,
    settings: Settings = Depends(get_settings),
):
    if not run_id:
        if not settings.allow_dev_params or not (system and q):
            raise HTTPException(status_code=400, detail="Provide run_id or (system & q)")
        payload = RunCreate(system=system, message=q)
        run_id = await start_run(payload, settings=settings)

    async def event_generator():
        try:
            async for event in stream_run(run_id, request):
                yield event
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Run not found: {exc.args[0]}") from exc

    return EventSourceResponse(event_generator())


@app.get("/api/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str, settings: Settings = Depends(get_settings)) -> dict[str, object]:
    try:
        return await get_run_snapshot(run_id, settings)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Run not found: {exc.args[0]}") from exc


__all__ = ["app"]
