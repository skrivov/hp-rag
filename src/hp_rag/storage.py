from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, TYPE_CHECKING

from src.models.section import SectionNode

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

    def iter_sections(self, *, max_level: Optional[int] = None) -> List[sqlite3.Row]:
        cursor = self.conn.cursor()
        if max_level is None:
            cursor.execute(
                "SELECT path, title, level, parent_path, order_index FROM sections ORDER BY path;"
            )
        else:
            cursor.execute(
                "SELECT path, title, level, parent_path, order_index FROM sections WHERE level <= ? ORDER BY path;",
                (max_level,),
            )
        return cursor.fetchall()

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

