from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from sse_starlette import JSONServerSentEvent
from starlette.requests import Request


class EventBus:
    """Minimal pub/sub queue that feeds Server-Sent Events."""

    def __init__(self, *, heartbeat_interval: float | None = None) -> None:
        self._queue: asyncio.Queue[JSONServerSentEvent | None] = asyncio.Queue()
        self._heartbeat = heartbeat_interval
        self._closed = asyncio.Event()

    async def send(self, event: str, data: Optional[dict]) -> None:
        await self._queue.put(JSONServerSentEvent(data=data, event=event))

    async def close(self) -> None:
        """Stop the stream and drain any waiters."""

        if not self._closed.is_set():
            self._closed.set()
            await self._queue.put(None)

    async def subscribe(self, request: Request) -> AsyncIterator[JSONServerSentEvent]:
        """Yield events until the client disconnects or the bus closes."""

        while True:
            if self._closed.is_set() and self._queue.empty():
                break

            timeout = self._heartbeat if self._heartbeat and self._heartbeat > 0 else None
            try:
                if timeout is None:
                    item = await self._queue.get()
                else:
                    item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                if self._heartbeat:
                    yield JSONServerSentEvent(
                        data={"ts": datetime.now(timezone.utc).isoformat()},
                        event="heartbeat",
                    )
                continue

            if item is None:
                if self._closed.is_set() and self._queue.empty():
                    break
                continue

            yield item

            if await request.is_disconnected():
                break

        await self.close()


__all__ = ["EventBus"]
