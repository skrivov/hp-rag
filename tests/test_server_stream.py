from __future__ import annotations

import asyncio
import importlib
from dataclasses import dataclass
import json

import pytest

import src.server.api as server_api
import src.server.chat_service as chat_service
import src.server.settings as server_settings
from src.orchestration.query_runner import StreamItem
from src.server.models import RunCreate


async def _fake_run_query_streaming(cfg, factory, settings, client):
    yield StreamItem(
        "event",
        payload=factory.event(
            phase="retrieval",
            type="start",
            name="stub.start",
            summary="Stub retrieval start",
        ),
    )
    yield StreamItem("token", text="Hello ")
    yield StreamItem("token", text="World")
    await asyncio.sleep(0)
    yield StreamItem(
        "summary",
        payload={
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            "timings": {"llm_ms": 5},
        },
    )


async def _noop_client(settings):
    return object()


@dataclass
class _FakeRequest:
    disconnected: bool = False

    async def is_disconnected(self) -> bool:
        return self.disconnected

    def close(self) -> None:
        self.disconnected = True


def _reload_modules():
    importlib.reload(server_settings)
    importlib.reload(chat_service)
    importlib.reload(server_api)


@pytest.mark.asyncio
async def test_chat_stream_and_persistence(monkeypatch, tmp_path):
    tmp_db = tmp_path / "runs.db"
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "gpt-test")
    monkeypatch.setenv("SELECTOR_MODEL", "gpt-test")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("RUNS_DB_PATH", str(tmp_db))
    monkeypatch.setenv("RUN_TTL_SECONDS", "1")
    _reload_modules()

    monkeypatch.setattr(chat_service, "run_query_streaming", _fake_run_query_streaming)
    monkeypatch.setattr(chat_service, "_ensure_client", _noop_client)

    settings = server_settings.get_settings()

    run_id = await chat_service.start_run(
        RunCreate(system="hp-rag", message="Who are you?"),
        settings=settings,
    )

    async def consume():
        req = _FakeRequest()
        events = []
        async for sse in chat_service.stream_run(run_id, req):
            data = sse.data
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    pass
            events.append((sse.event, data))
            if sse.event == "done":
                req.close()
                break
        return events

    events = await asyncio.wait_for(consume(), timeout=5)
    event_names = [name for name, _ in events]
    assert "token" in event_names
    assert (
        "done",
        {
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            "timings": {"llm_ms": 5},
            "error": None,
        },
    ) in events

    snapshot = await chat_service.get_run_snapshot(run_id, settings)
    assert snapshot["tokens"] == "Hello World"
    assert snapshot["done"]["finish_reason"] == "stop"
    assert snapshot["done"]["timings"]["llm_ms"] == 5
    assert tmp_db.exists()
