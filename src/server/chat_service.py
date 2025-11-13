from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Dict, List

from openai import AsyncOpenAI
from starlette.requests import Request

from src.orchestration.query_runner import run_query_streaming
from .instrumentation import EventFactory
from .models import DoneEvent, RunConfig, RunCreate, RunState
from .settings import Settings
from .sse import EventBus
from .storage import RunStore


_RUNS: dict[str, RunState] = {}
_RUN_LOCK = asyncio.Lock()
_CLIENT: AsyncOpenAI | None = None
_CLIENT_LOCK = asyncio.Lock()
_RUN_STORE: RunStore | None = None


async def start_run(payload: RunCreate | Dict[str, Any], *, settings: Settings) -> str:
    """Validate payload, create run config, and kick off execution."""

    request_model = payload if isinstance(payload, RunCreate) else RunCreate.model_validate(payload)
    config = RunConfig.from_request(request_model, settings)
    bus = EventBus(heartbeat_interval=settings.sse_heartbeat_interval)
    state = RunState(config=config, bus=bus)

    async with _RUN_LOCK:
        _RUNS[config.run_id] = state

    store = _get_store(settings)
    await store.cleanup()
    await store.record_run(state.config)

    asyncio.create_task(_execute_run(state, settings, store))
    return config.run_id


async def stream_run(run_id: str, request: Request) -> AsyncIterator[Any]:
    state = await _get_state(run_id)
    async for event in state.bus.subscribe(request):
        yield event


async def get_run_snapshot(run_id: str, settings: Settings) -> dict[str, Any]:
    async with _RUN_LOCK:
        state = _RUNS.get(run_id)
    if state is not None:
        return state.to_public_dict()
    store = _get_store(settings)
    stored = await store.fetch_run(run_id)
    if stored is None:
        raise KeyError(run_id)
    return stored


async def list_runs(
    settings: Settings,
    *,
    query: str | None = None,
    limit: int = 50,
) -> List[dict[str, Any]]:
    store = _get_store(settings)
    return await store.list_runs(query=query, limit=limit)


async def _get_state(run_id: str) -> RunState:
    async with _RUN_LOCK:
        if run_id not in _RUNS:
            raise KeyError(run_id)
        return _RUNS[run_id]


async def _execute_run(state: RunState, settings: Settings, store: RunStore) -> None:
    factory = EventFactory(run_id=state.config.run_id)
    client = await _ensure_client(settings)
    summary_payload: dict[str, Any] | None = None

    run_start_event = factory.event(
        phase="system",
        type="start",
        name="system.run",
        summary="Run started",
        detail={
            "system": state.config.system,
            "top_k": state.config.top_k,
            "metadata": state.config.metadata,
            "answer_model": settings.answer_model,
        },
    )
    await state.bus.send("event", run_start_event)
    state.append_event(run_start_event)
    await store.record_event(state.config.run_id, run_start_event)

    try:
        async for item in run_query_streaming(state.config, factory, settings, client):
            if item.kind == "token" and item.text:
                await state.bus.send("token", {"text": item.text})
                state.append_token(item.text)
                await store.record_token(state.config.run_id, len(state.tokens) - 1, item.text)
            elif item.kind == "event" and item.payload:
                await state.bus.send("event", item.payload)
                state.append_event(item.payload)
                await store.record_event(state.config.run_id, item.payload)
            elif item.kind == "summary" and item.payload:
                summary_payload = item.payload
    except Exception as exc:
        error_event = factory.event(
            phase="system",
            type="error",
            name="system.run",
            summary=str(exc),
            detail={"exc_type": exc.__class__.__name__},
            severity="error",
        )
        await state.bus.send("event", error_event)
        state.append_event(error_event)
        await store.record_event(state.config.run_id, error_event)
        state.error = str(exc)
        summary_payload = summary_payload or {
            "finish_reason": "error",
            "usage": {},
            "timings": {},
            "error": {"message": str(exc), "type": exc.__class__.__name__},
        }
    else:
        run_end_event = factory.event(
            phase="system",
            type="end",
            name="system.run",
            summary="Run completed",
            detail={},
        )
        await state.bus.send("event", run_end_event)
        state.append_event(run_end_event)
        await store.record_event(state.config.run_id, run_end_event)
    finally:
        done_payload = summary_payload or {
            "finish_reason": "unknown",
            "usage": {},
            "timings": {},
        }
        await state.bus.send("done", DoneEvent(**done_payload).model_dump())
        state.done_payload = done_payload
        state.finished = True
        await state.bus.close()
        await store.mark_done(state.config.run_id, done_payload)
        async with _RUN_LOCK:
            _RUNS.pop(state.config.run_id, None)


async def _ensure_client(settings: Settings) -> AsyncOpenAI:
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    async with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = AsyncOpenAI(api_key=settings.openai_api_key)
    return _CLIENT


def _get_store(settings: Settings) -> RunStore:
    global _RUN_STORE
    if _RUN_STORE is None or _RUN_STORE.path != settings.runs_db_path:
        _RUN_STORE = RunStore(settings.runs_db_path, ttl_seconds=settings.run_ttl_seconds)
    return _RUN_STORE


__all__ = ["get_run_snapshot", "start_run", "stream_run", "list_runs"]
