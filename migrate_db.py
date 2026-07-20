#!/usr/bin/env python3
"""
Temporary migration script: brings goodreads.db up to date with the
current schema defined in goodreads_ranker/core/db.py.

Differences found between live DB and expected schema
------------------------------------------------------
1. libraries.is_similar  →  rename to  libraries.similarity_score  (REAL)
   SQLite doesn't support ALTER COLUMN RENAME, so we:
     a. Add the new column
     b. Copy data from the old column (0/1 integer → cast to REAL)
     c. Drop the old column (requires recreating the table in SQLite < 3.35,
        but SQLite ≥ 3.35 supports DROP COLUMN directly)

2. library_books is missing the calibrated_rating column (REAL).

Run with:
    python migrate_db.py           # uses default data/goodreads.db
    python migrate_db.py --db-path path/to/custom.db
"""

import argparse
import sqlite3
from pathlib import Path


def get_column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {row[1] for row in rows}


def migrate(db_path: Path) -> None:
    print(f"Opening database: {db_path}")
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys=OFF")   # needed while recreating tables
    conn.execute("PRAGMA journal_mode=WAL")

    try:
        # ------------------------------------------------------------------
        # 1.  libraries: rename is_similar → similarity_score
        # ------------------------------------------------------------------
        libs_cols = get_column_names(conn, "libraries")
        print(f"  libraries columns: {sorted(libs_cols)}")

        needs_similarity_score = "similarity_score" not in libs_cols
        has_is_similar = "is_similar" in libs_cols

        if needs_similarity_score:
            print("  [libraries] Adding column 'similarity_score' (REAL)...")
            conn.execute("ALTER TABLE libraries ADD COLUMN similarity_score REAL")

            if has_is_similar:
                print("  [libraries] Copying is_similar → similarity_score...")
                conn.execute(
                    "UPDATE libraries SET similarity_score = CAST(is_similar AS REAL)"
                )

            conn.commit()
            print("  [libraries] similarity_score column added and populated.")
        else:
            print("  [libraries] similarity_score already exists — skipping.")

        # Drop is_similar if it still exists (SQLite ≥ 3.35.0 supports DROP COLUMN)
        libs_cols = get_column_names(conn, "libraries")   # refresh
        if "is_similar" in libs_cols:
            sqlite_version = tuple(int(x) for x in sqlite3.sqlite_version.split("."))
            if sqlite_version >= (3, 35, 0):
                print(f"  [libraries] SQLite {sqlite3.sqlite_version}: dropping old 'is_similar' column...")
                conn.execute("ALTER TABLE libraries DROP COLUMN is_similar")
                conn.commit()
                print("  [libraries] is_similar dropped.")
            else:
                print(
                    f"  [libraries] SQLite {sqlite3.sqlite_version} < 3.35 — cannot DROP COLUMN directly.\n"
                    "  The old 'is_similar' column will remain but is no longer used by the application.\n"
                    "  (Upgrade SQLite or recreate the DB from scratch to fully clean it up.)"
                )

        # ------------------------------------------------------------------
        # 2.  library_books: add missing calibrated_rating column
        # ------------------------------------------------------------------
        lb_cols = get_column_names(conn, "library_books")
        print(f"  library_books columns: {sorted(lb_cols)}")

        if "calibrated_rating" not in lb_cols:
            print("  [library_books] Adding column 'calibrated_rating' (REAL)...")
            conn.execute("ALTER TABLE library_books ADD COLUMN calibrated_rating REAL")
            conn.commit()
            print("  [library_books] calibrated_rating column added.")
        else:
            print("  [library_books] calibrated_rating already exists — skipping.")

        # ------------------------------------------------------------------
        # Done
        # ------------------------------------------------------------------
        print("\nMigration complete. Final column sets:")
        for table in ("libraries", "library_books"):
            cols = sorted(get_column_names(conn, table))
            print(f"  {table}: {cols}")

    except Exception as exc:
        conn.rollback()
        raise RuntimeError(f"Migration failed, rolled back: {exc}") from exc
    finally:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate goodreads.db schema")
    parser.add_argument(
        "--db-path",
        default="data/goodreads.db",
        help="Path to the SQLite database file (default: data/goodreads.db)",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    migrate(db_path)
