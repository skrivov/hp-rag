from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import RunConfig


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    """Persist run transcripts, events, and completion summaries for replay."""

    def __init__(self, path: Path, *, ttl_seconds: int | None = None) -> None:
        self.path = path
        self.ttl_seconds = ttl_seconds if ttl_seconds and ttl_seconds > 0 else None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _initialize(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    system TEXT NOT NULL,
                    message TEXT NOT NULL,
                    top_k INTEGER NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    finished_at TEXT,
                    finish_reason TEXT,
                    done_payload TEXT
                );

                CREATE TABLE IF NOT EXISTS tokens (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_tokens_run ON tokens(run_id, seq);
                CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, id);
                """
            )
            conn.commit()
        finally:
            conn.close()

    async def cleanup(self) -> None:
        if not self.ttl_seconds:
            return
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.ttl_seconds)
        await asyncio.to_thread(self._cleanup_sync, cutoff.isoformat())

    def _cleanup_sync(self, cutoff_iso: str) -> None:
        conn = self._connect()
        try:
            conn.execute("DELETE FROM runs WHERE created_at < ?", (cutoff_iso,))
            conn.commit()
        finally:
            conn.close()

    async def record_run(self, config: "RunConfig") -> None:
        payload = {
            "run_id": config.run_id,
            "system": config.system,
            "message": config.message,
            "top_k": config.top_k,
            "metadata": json.dumps(config.metadata or {}),
            "created_at": _ts(),
        }
        await asyncio.to_thread(self._record_run_sync, payload)

    def _record_run_sync(self, payload: Dict[str, Any]) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs(run_id, system, message, top_k, metadata, created_at)
                VALUES(:run_id, :system, :message, :top_k, :metadata, :created_at)
                """,
                payload,
            )
            conn.commit()
        finally:
            conn.close()

    async def record_event(self, run_id: str, event_payload: Dict[str, Any]) -> None:
        if not event_payload:
            return
        await asyncio.to_thread(self._record_event_sync, run_id, json.dumps(event_payload))

    def _record_event_sync(self, run_id: str, payload_json: str) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO events(run_id, payload, created_at) VALUES (?, ?, ?)",
                (run_id, payload_json, _ts()),
            )
            conn.commit()
        finally:
            conn.close()

    async def record_token(self, run_id: str, seq: int, text: str) -> None:
        await asyncio.to_thread(self._record_token_sync, run_id, seq, text)

    def _record_token_sync(self, run_id: str, seq: int, text: str) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO tokens(run_id, seq, text) VALUES (?, ?, ?)",
                (run_id, seq, text),
            )
            conn.commit()
        finally:
            conn.close()

    async def mark_done(self, run_id: str, done_payload: Dict[str, Any]) -> None:
        await asyncio.to_thread(
            self._mark_done_sync,
            run_id,
            json.dumps(done_payload),
            done_payload.get("finish_reason"),
        )

    def _mark_done_sync(self, run_id: str, done_json: str, finish_reason: str | None) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                UPDATE runs
                SET finished_at = :finished_at,
                    finish_reason = :finish_reason,
                    done_payload = :done_payload
                WHERE run_id = :run_id
                """,
                {
                    "run_id": run_id,
                    "finished_at": _ts(),
                    "done_payload": done_json,
                    "finish_reason": finish_reason,
                },
            )
            conn.commit()
        finally:
            conn.close()

    async def fetch_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        return await asyncio.to_thread(self._fetch_run_sync, run_id)

    def _fetch_run_sync(self, run_id: str) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        try:
            run_row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if not run_row:
                return None
            tokens = [
                row["text"]
                for row in conn.execute(
                    "SELECT text FROM tokens WHERE run_id = ? ORDER BY seq ASC",
                    (run_id,),
                )
            ]
            events = [
                json.loads(row["payload"])
                for row in conn.execute(
                    "SELECT payload FROM events WHERE run_id = ? ORDER BY id ASC",
                    (run_id,),
                )
            ]
            done_payload = json.loads(run_row["done_payload"]) if run_row["done_payload"] else None
            metadata = json.loads(run_row["metadata"]) if run_row["metadata"] else {}
            return {
                "run_id": run_row["run_id"],
                "system": run_row["system"],
                "created_at": run_row["created_at"],
                "message": run_row["message"],
                "metadata": metadata,
                "tokens": "".join(tokens),
                "events": events,
                "done": done_payload,
                "error": None if (done_payload or {}).get("finish_reason") != "error" else (done_payload or {}).get("error"),
            }
        finally:
            conn.close()

    async def list_runs(self, *, query: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._list_runs_sync, query, limit)

    def _list_runs_sync(self, query: str | None, limit: int) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            params: list[Any] = []
            where = ""
            if query:
                like = f"%{query.lower()}%"
                where = "WHERE LOWER(message) LIKE ? OR LOWER(run_id) LIKE ?"
                params.extend([like, like])
            params.append(limit)
            rows = conn.execute(
                f"""
                SELECT * FROM runs
                {where}
                ORDER BY datetime(created_at) DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
            results: list[dict[str, Any]] = []
            for row in rows:
                done_payload = json.loads(row["done_payload"]) if row["done_payload"] else {}
                metadata = json.loads(row["metadata"]) if row["metadata"] else {}
                status = "running"
                if row["finished_at"]:
                    status = "error" if row["finish_reason"] == "error" else "ready"
                results.append(
                    {
                        "run_id": row["run_id"],
                        "system": row["system"],
                        "message": row["message"],
                        "created_at": row["created_at"],
                        "status": status,
                        "finish_reason": row["finish_reason"],
                        "usage": done_payload.get("usage", {}),
                        "metadata": metadata,
                    }
                )
            return results
        finally:
            conn.close()


__all__ = ["RunStore"]
