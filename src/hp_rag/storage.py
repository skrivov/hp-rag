from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, TYPE_CHECKING

from src.models.section import SectionNode
from src.models.tenant import TenantRecord

if TYPE_CHECKING:
    from src.ingest.chunker import SectionChunk


@dataclass(slots=True)
class SQLiteHyperlinkConfig:
    """Configuration for SQLite-based hyperlink store."""

    db_path: Path
    enable_wal: bool = True


class SQLiteHyperlinkStore:
    """Persists TOC-aware sections into SQLite + FTS5 for HP-RAG retrieval."""

    def __init__(self, config: SQLiteHyperlinkConfig) -> None:
        self.config = config
        self._conn: Optional[sqlite3.Connection] = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.config.db_path)
            self._conn.row_factory = sqlite3.Row
            if self.config.enable_wal:
                self._conn.execute("PRAGMA journal_mode=WAL;")
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def initialize(self) -> None:
        """Create tables and FTS indices if they do not exist."""

        cursor = self.conn.cursor()
        cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS sections (
                path TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                level INTEGER NOT NULL,
                parent_path TEXT,
                order_index INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                section_path TEXT NOT NULL,
                chunk_order INTEGER NOT NULL,
                text TEXT NOT NULL,
                FOREIGN KEY(section_path) REFERENCES sections(path)
            );

            CREATE TABLE IF NOT EXISTS tenants (
                tenant_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                role TEXT,
                aliases TEXT,
                source TEXT DEFAULT 'document'
            );

            CREATE TABLE IF NOT EXISTS document_tenants (
                document_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                PRIMARY KEY (document_id, tenant_id),
                FOREIGN KEY(document_id) REFERENCES sections(document_id),
                FOREIGN KEY(tenant_id) REFERENCES tenants(tenant_id)
            );

            CREATE INDEX IF NOT EXISTS idx_document_tenants_tenant
                ON document_tenants (tenant_id);

            CREATE VIRTUAL TABLE IF NOT EXISTS sections_fts USING fts5(
                path UNINDEXED,
                title,
                body,
                content='sections',
                content_rowid='rowid'
            );

            CREATE TRIGGER IF NOT EXISTS sections_ai AFTER INSERT ON sections BEGIN
                INSERT INTO sections_fts(rowid, path, title, body)
                VALUES (new.rowid, new.path, new.title, new.body);
            END;

            CREATE TRIGGER IF NOT EXISTS sections_ad AFTER DELETE ON sections BEGIN
                INSERT INTO sections_fts(sections_fts, rowid, path, title, body)
                VALUES ('delete', old.rowid, old.path, old.title, old.body);
            END;

            CREATE TRIGGER IF NOT EXISTS sections_au AFTER UPDATE ON sections BEGIN
                INSERT INTO sections_fts(sections_fts, rowid, path, title, body)
                VALUES ('delete', old.rowid, old.path, old.title, old.body);
                INSERT INTO sections_fts(rowid, path, title, body)
                VALUES (new.rowid, new.path, new.title, new.body);
            END;
            """
        )
        self.conn.commit()

    def upsert_sections(self, sections: Iterable[SectionNode]) -> None:
        """Insert or update SectionNodes recursively."""

        cursor = self.conn.cursor()
        for section in sections:
            self._upsert_section(cursor, section)
        self.conn.commit()

    def _upsert_section(self, cursor: sqlite3.Cursor, section: SectionNode) -> None:
        cursor.execute(
            """
            INSERT INTO sections(path, document_id, title, body, level, parent_path, order_index)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                document_id=excluded.document_id,
                title=excluded.title,
                body=excluded.body,
                level=excluded.level,
                parent_path=excluded.parent_path,
                order_index=excluded.order_index;
            """,
            (
                section.path,
                section.document_id,
                section.title,
                section.body,
                section.level,
                section.parent_path,
                section.order,
            ),
        )
        if section.children:
            for child in section.children:
                self._upsert_section(cursor, child)

    def add_chunks(self, chunks: Iterable["SectionChunk"]) -> None:
        """Persist chunk metadata for later retrieval or inspection."""

        cursor = self.conn.cursor()
        cursor.executemany(
            """
            INSERT INTO chunks(section_path, chunk_order, text)
            VALUES (?, ?, ?);
            """,
            ((chunk.path, chunk.order, chunk.text) for chunk in chunks),
        )
        self.conn.commit()

    def register_document_tenants(self, document_id: str, tenants: Iterable[TenantRecord]) -> None:
        """Persist tenants and link them to a document."""

        items = list(tenants)
        if not items:
            return

        cursor = self.conn.cursor()
        for tenant in items:
            aliases = json.dumps(sorted(set(tenant.aliases))) if tenant.aliases else None
            cursor.execute(
                """
                INSERT INTO tenants(tenant_id, name, role, aliases, source)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id) DO UPDATE SET
                    name=excluded.name,
                    role=COALESCE(excluded.role, tenants.role),
                    aliases=COALESCE(excluded.aliases, tenants.aliases),
                    source=COALESCE(excluded.source, tenants.source);
                """,
                (tenant.tenant_id, tenant.name, tenant.role, aliases, tenant.source or "document"),
            )
            cursor.execute(
                """
                INSERT OR IGNORE INTO document_tenants(document_id, tenant_id)
                VALUES (?, ?);
                """,
                (document_id, tenant.tenant_id),
            )
        self.conn.commit()

    def list_tenants(
        self,
        *,
        query: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[sqlite3.Row]:
        cursor = self.conn.cursor()
        params: List[object] = []
        where_clauses: List[str] = []

        if query:
            like = f"%{query.lower()}%"
            where_clauses.append(
                "(LOWER(name) LIKE ? OR LOWER(role) LIKE ? OR LOWER(COALESCE(aliases, '')) LIKE ?)"
            )
            params.extend([like, like, like])

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        cursor.execute(
            f"""
            SELECT t.*, (
                SELECT COUNT(*)
                FROM document_tenants dt
                WHERE dt.tenant_id = t.tenant_id
            ) AS document_count
            FROM tenants t
            {where_sql}
            ORDER BY t.name
            LIMIT ?
            OFFSET ?;
            """,
            (*params, limit, offset),
        )
        return cursor.fetchall()

    def fetch_tenant(self, tenant_id: str) -> Optional[sqlite3.Row]:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT t.*, (
                SELECT COUNT(*)
                FROM document_tenants dt
                WHERE dt.tenant_id = t.tenant_id
            ) AS document_count
            FROM tenants t
            WHERE tenant_id = ?;
            """,
            (tenant_id,),
        )
        return cursor.fetchone()

    def list_documents_for_tenant(self, tenant_id: str) -> List[str]:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT document_id
            FROM document_tenants
            WHERE tenant_id = ?
            ORDER BY document_id;
            """,
            (tenant_id,),
        )
        rows = cursor.fetchall()
        return [row["document_id"] for row in rows]

    def search(self, query: str, *, limit: int = 10) -> List[sqlite3.Row]:
        """Run a full-text search across sections."""

        safe = re.sub(r"[^\w\s]", " ", query)
        safe = re.sub(r"\s+", " ", safe).strip()
        if safe:
            tokens = [t for t in safe.split(" ") if t]
            match_query = " OR ".join(tokens) if len(tokens) > 1 else tokens[0]
        else:
            match_query = query

        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT s.*,
                   bm25(sections_fts) AS score
            FROM sections s
            JOIN sections_fts ON s.rowid = sections_fts.rowid
            WHERE sections_fts MATCH ?
            ORDER BY score
            LIMIT ?;
            """,
            (match_query, limit),
        )
        return cursor.fetchall()

    def fetch_by_path(self, path: str) -> Optional[sqlite3.Row]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM sections WHERE path = ?;",
            (path,),
        )
        return cursor.fetchone()

    def fetch_sections(self, paths: Iterable[str]) -> List[sqlite3.Row]:
        path_list = list(paths)
        if not path_list:
            return []
        cursor = self.conn.cursor()
        placeholders = ",".join("?" for _ in path_list)
        query = f"SELECT * FROM sections WHERE path IN ({placeholders}) ORDER BY order_index"
        cursor.execute(query, path_list)
        return cursor.fetchall()

    def iter_sections(
        self,
        *,
        max_level: Optional[int] = None,
        tenant_id: Optional[str] = None,
    ) -> List[sqlite3.Row]:
        cursor = self.conn.cursor()
        clauses: List[str] = []
        params: List[object] = []
        if max_level is not None:
            clauses.append("level <= ?")
            params.append(max_level)
        if tenant_id:
            clauses.append(
                "document_id IN (SELECT document_id FROM document_tenants WHERE tenant_id = ?)"
            )
            params.append(tenant_id)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cursor.execute(
            f"""
            SELECT path, title, level, parent_path, order_index
            FROM sections
            {where_sql}
            ORDER BY path;
            """,
            params,
        )
        return cursor.fetchall()

    # ------------------------------------------------------------------ document helpers
    def iter_sections_by_document(self, document_id: str) -> List[sqlite3.Row]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM sections WHERE document_id = ? ORDER BY order_index",
            (document_id,),
        )
        return cursor.fetchall()

    def iter_chunks(self, document_id: Optional[str] = None) -> List[sqlite3.Row]:
        cursor = self.conn.cursor()
        if document_id:
            cursor.execute(
                """
                SELECT c.*, s.document_id, s.title as section_title
                FROM chunks c
                JOIN sections s ON s.path = c.section_path
                WHERE s.document_id = ?
                ORDER BY s.order_index, c.chunk_order
                """,
                (document_id,),
            )
        else:
            cursor.execute(
                """
                SELECT c.*, s.document_id, s.title as section_title
                FROM chunks c
                JOIN sections s ON s.path = c.section_path
                ORDER BY s.document_id, s.order_index, c.chunk_order
                """
            )
        return cursor.fetchall()

    def delete_document(self, document_id: str) -> None:
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM sections WHERE document_id = ?", (document_id,))
        cursor.execute(
            "DELETE FROM document_tenants WHERE document_id = ?",
            (document_id,),
        )
        cursor.execute(
            """
            DELETE FROM chunks
            WHERE section_path NOT IN (SELECT path FROM sections)
            """
        )
        cursor.execute(
            "DELETE FROM sections_fts WHERE rowid NOT IN (SELECT rowid FROM sections)"
        )
        self.conn.commit()

    def clear_all(self) -> None:
        cursor = self.conn.cursor()
        cursor.executescript(
            """
            DELETE FROM sections;
            DELETE FROM chunks;
            DELETE FROM sections_fts;
            DELETE FROM tenants;
            DELETE FROM document_tenants;
            """
        )
        self.conn.commit()

    def fetch_neighbors(self, path: str, *, window: int = 1) -> List[sqlite3.Row]:
        """Return neighboring sections based on order index within the same parent."""

        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT parent_path, order_index
            FROM sections
            WHERE path = ?;
            """,
            (path,),
        )
        row = cursor.fetchone()
        if row is None:
            return []
        parent_path = row["parent_path"]
        order_index = row["order_index"]

        cursor.execute(
            """
            SELECT *
            FROM sections
            WHERE parent_path IS ?
              AND order_index BETWEEN ? AND ?
            ORDER BY order_index;
            """,
            (parent_path, order_index - window, order_index + window),
        )
        return cursor.fetchall()


__all__ = ["SQLiteHyperlinkStore", "SQLiteHyperlinkConfig"]
