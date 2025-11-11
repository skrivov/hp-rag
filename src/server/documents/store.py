from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from src.models.document import (
    DocumentMetadata,
    DocumentStatus,
    DocumentTenant,
    PluginState,
    PluginStatus,
)


class DocumentStore:
    """Persists document metadata and plugin status alongside the SQLite hyperlink DB."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._ensure_parent()
        self._initialize()

    def _ensure_parent(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    mime_type TEXT,
                    size_bytes INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS document_plugin_status (
                    plugin_name TEXT NOT NULL,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    status TEXT NOT NULL,
                    error TEXT,
                    stats TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (plugin_name, document_id)
                );

                CREATE TABLE IF NOT EXISTS document_runs (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stats TEXT,
                    started_at TEXT NOT NULL,
                    completed_at TEXT
                );
                """
            )

    # ------------------------------------------------------------------ document CRUD
    def upsert_document(
        self,
        *,
        document_id: str,
        name: str,
        source_path: Path,
        mime_type: Optional[str],
        size_bytes: int,
        status: DocumentStatus,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents(id, name, source_path, mime_type, size_bytes, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    source_path=excluded.source_path,
                    mime_type=excluded.mime_type,
                    size_bytes=excluded.size_bytes,
                    status=excluded.status,
                    updated_at=excluded.updated_at
                """,
                (document_id, name, str(source_path), mime_type, size_bytes, status.value, now, now),
            )

    def update_status(self, document_id: str, status: DocumentStatus, *, error: str | None = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE documents
                SET status = ?, error = ?, updated_at = ?
                WHERE id = ?
                """,
                (status.value, error, now, document_id),
            )

    # ------------------------------------------------------------------ plugin status
    def set_plugin_status(
        self,
        document_id: str,
        plugin_name: str,
        status: PluginStatus,
        *,
        stats: Optional[Dict[str, object]] = None,
        error: str | None = None,
    ) -> None:
        payload = json.dumps(stats or {})
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO document_plugin_status(plugin_name, document_id, status, error, stats, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(plugin_name, document_id) DO UPDATE SET
                    status=excluded.status,
                    error=excluded.error,
                    stats=excluded.stats,
                    updated_at=excluded.updated_at
                """,
                (plugin_name, document_id, status.value, error, payload, now),
            )

    # ------------------------------------------------------------------ listing helpers
    def list_documents(self, tenant_id: Optional[str] = None) -> List[DocumentMetadata]:
        with self._connect() as conn:
            has_tenants = self._table_exists(conn, "document_tenants") and self._table_exists(conn, "tenants")
            where_clause = ""
            params: List[object] = []
            if tenant_id:
                if not has_tenants:
                    return []
                where_clause = (
                    "WHERE EXISTS (SELECT 1 FROM document_tenants dt "
                    "WHERE dt.document_id = d.id AND dt.tenant_id = ?)"
                )
                params.append(tenant_id)

            try:
                rows = conn.execute(
                    f"""
                    SELECT d.*,
                           IFNULL((SELECT COUNT(*) FROM sections s WHERE s.document_id = d.id), 0) AS section_count,
                           IFNULL((SELECT COUNT(*) FROM chunks c
                                    JOIN sections s2 ON s2.path = c.section_path
                                   WHERE s2.document_id = d.id), 0) AS chunk_count
                    FROM documents d
                    {where_clause}
                    ORDER BY datetime(d.created_at) DESC
                    """,
                    params,
                ).fetchall()
            except sqlite3.OperationalError:
                rows = conn.execute(
                    f"""
                    SELECT d.*, 0 as section_count, 0 as chunk_count
                    FROM documents d
                    {where_clause}
                    ORDER BY datetime(d.created_at) DESC
                    """,
                    params,
                ).fetchall()

        documents = [self._row_to_metadata(row) for row in rows]
        document_ids = [doc.id for doc in documents]
        plugin_map = self._fetch_plugin_states(document_ids)
        tenant_map = self._fetch_document_tenants(document_ids) if documents else {}
        for doc in documents:
            doc.plugin_states = plugin_map.get(doc.id, [])
            doc.tenants = tenant_map.get(doc.id, [])
        return documents

    def get_document(self, document_id: str) -> Optional[DocumentMetadata]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
        if not row:
            return None
        doc = self._row_to_metadata(row)
        doc.plugin_states = self._fetch_plugin_states([document_id]).get(document_id, [])
        doc.tenants = self._fetch_document_tenants([document_id]).get(document_id, [])
        return doc

    def delete_all(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                DELETE FROM document_plugin_status;
                DELETE FROM document_runs;
                DELETE FROM documents;
                """
            )

    # ------------------------------------------------------------------ helpers
    def _fetch_plugin_states(self, document_ids: Iterable[str]) -> Dict[str, List[PluginState]]:
        ids = list(document_ids)
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM document_plugin_status WHERE document_id IN ({placeholders})",
                ids,
            ).fetchall()

        result: Dict[str, List[PluginState]] = {}
        for row in rows:
            stats = json.loads(row["stats" or 0] or "{}") if row["stats"] else {}
            state = PluginState(
                plugin_name=row["plugin_name"],
                status=PluginStatus(row["status"]),
                stats=stats,
                error=row["error"],
            )
            result.setdefault(row["document_id"], []).append(state)
        return result

    def _fetch_document_tenants(self, document_ids: Iterable[str]) -> Dict[str, List[DocumentTenant]]:
        ids = list(document_ids)
        if not ids:
            return {}
        with self._connect() as conn:
            if not (self._table_exists(conn, "document_tenants") and self._table_exists(conn, "tenants")):
                return {}
            placeholders = ",".join("?" for _ in ids)
            rows = conn.execute(
                f"""
                SELECT dt.document_id, t.tenant_id, t.name, t.role
                FROM document_tenants dt
                JOIN tenants t ON t.tenant_id = dt.tenant_id
                WHERE dt.document_id IN ({placeholders})
                ORDER BY t.name
                """,
                ids,
            ).fetchall()

        mapping: Dict[str, List[DocumentTenant]] = {}
        for row in rows:
            tenant = DocumentTenant(
                tenant_id=row["tenant_id"],
                name=row["name"],
                role=row["role"],
            )
            mapping.setdefault(row["document_id"], []).append(tenant)
        return mapping

    @staticmethod
    def _row_to_metadata(row: sqlite3.Row) -> DocumentMetadata:
        return DocumentMetadata(
            id=row["id"],
            name=row["name"],
            source_path=row["source_path"],
            mime_type=row["mime_type"],
            size_bytes=int(row["size_bytes"] or 0),
            status=DocumentStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            error=row["error"],
            section_count=int(row["section_count"] or 0) if "section_count" in row.keys() else 0,
            chunk_count=int(row["chunk_count"] or 0) if "chunk_count" in row.keys() else 0,
        )

    @staticmethod
    def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        return row is not None


__all__ = ["DocumentStore"]
