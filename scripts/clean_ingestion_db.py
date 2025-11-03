from __future__ import annotations

import argparse
from pathlib import Path

import sqlite3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove tables/rows from the ingestion SQLite store.")
    parser.add_argument(
        "sqlite",
        type=Path,
        nargs="?",
        default=Path("artifacts/test_ingestion/contracts_hyperlink.db"),
        help="Path to the SQLite database to clean (default: artifacts/test_ingestion/contracts_hyperlink.db)",
    )
    parser.add_argument(
        "--drop-tables",
        action="store_true",
        help="Drop the tables entirely instead of deleting rows.",
    )
    return parser.parse_args()


def clean_db(path: Path, drop_tables: bool = False) -> None:
    if not path.exists():
        print(f"[info] SQLite database not found at {path}; nothing to clean.")
        return

    conn = sqlite3.connect(path)
    try:
        cursor = conn.cursor()
        if drop_tables:
            cursor.executescript(
                """
                DROP TABLE IF EXISTS chunks;
                DROP TABLE IF EXISTS sections;
                DROP TABLE IF EXISTS sections_fts;
                """
            )
            print(f"[info] Dropped ingestion tables in {path}.")
        else:
            cursor.executescript(
                """
                DELETE FROM chunks;
                DELETE FROM sections;
                DELETE FROM sections_fts;
                """
            )
            print(f"[info] Cleared ingestion rows in {path}.")
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    args = parse_args()
    clean_db(args.sqlite, drop_tables=args.drop_tables)


if __name__ == "__main__":
    main()
