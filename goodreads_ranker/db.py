import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np

DB_PATH = Path("data/goodreads.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS readers (
    list_id            INTEGER PRIMARY KEY,
    username           TEXT,
    user_id            INTEGER,
    is_self            INTEGER DEFAULT 0,
    scrape_complete    INTEGER DEFAULT 0,
    date_last_scraped  TEXT,
    scrape_error       TEXT
);

CREATE TABLE IF NOT EXISTS reader_libraries (
    list_id    INTEGER NOT NULL,
    book_id    INTEGER NOT NULL,
    rating     INTEGER,
    date_read  TEXT,
    date_added TEXT,
    PRIMARY KEY (list_id, book_id)
);

CREATE TABLE IF NOT EXISTS books (
    book_id           INTEGER PRIMARY KEY,
    title             TEXT,
    authors           TEXT,
    avg_rating        REAL,
    review_count      INTEGER,
    num_pages         INTEGER,
    lang              TEXT,
    star_1            INTEGER DEFAULT 0,
    star_2            INTEGER DEFAULT 0,
    star_3            INTEGER DEFAULT 0,
    star_4            INTEGER DEFAULT 0,
    star_5            INTEGER DEFAULT 0,
    genres            TEXT,
    series            TEXT,
    year              INTEGER,
    description       TEXT,
    similar_books     TEXT,
    primary_author    TEXT,
    author_followers  INTEGER DEFAULT 0,
    want_to_read      INTEGER DEFAULT 0,
    author_num_books  INTEGER DEFAULT 0,
    currently_reading INTEGER DEFAULT 0,
    date_last_scraped TEXT
);

CREATE TABLE IF NOT EXISTS elo_ratings (
    book_id          INTEGER PRIMARY KEY,
    original_rating  REAL,
    elo_score        REAL DEFAULT 1200.0,
    matches_played   INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS embeddings (
    book_id           INTEGER,
    embedding_model   TEXT,
    dim               INTEGER NOT NULL,
    vector            BLOB NOT NULL,
    text_hash         TEXT NOT NULL,
    PRIMARY KEY (book_id, embedding_model)
);

CREATE TABLE IF NOT EXISTS predictions (
    book_id                INTEGER PRIMARY KEY,
    solo_pred_rating       REAL,
    friend_pred_rating     REAL,
    count_adjusted_rating  REAL,
    pred_rating            REAL,
    final_rating           REAL,
    updated_at             TEXT
);

CREATE TABLE IF NOT EXISTS model_params (
    name TEXT PRIMARY KEY,
    params_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_readers_single_self
ON readers(is_self) WHERE is_self = 1;
"""


@contextmanager
def get_connection(db_path=None):
    path = Path(db_path) if db_path is not None else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    db_conn = sqlite3.connect(str(path))
    try:
        db_conn.execute("PRAGMA journal_mode=WAL")
        db_conn.execute("PRAGMA synchronous=NORMAL")
        db_conn.execute("PRAGMA foreign_keys=ON")
        db_conn.row_factory = sqlite3.Row
        yield db_conn
    finally:
        db_conn.close()


def init_db(db_path=None):
    with get_connection(db_path) as db_conn:
        ensure_schema_compat(db_conn)
        db_conn.executescript(SCHEMA)
        db_conn.commit()


def ensure_schema_compat(db_conn):
    db_conn.execute("DROP TABLE IF EXISTS my_library")
    db_conn.execute("DROP TABLE IF EXISTS user_library")

    existing = {row["name"] for row in db_conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

    if "friend_lists" in existing and "readers" not in existing:
        db_conn.execute("ALTER TABLE friend_lists RENAME TO readers")
    if "friend_ratings" in existing and "reader_libraries" not in existing:
        db_conn.execute("ALTER TABLE friend_ratings RENAME TO reader_libraries")

    existing_after_renaming = {
        row["name"] for row in db_conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }

    if "books" in existing_after_renaming:
        ensure_column(db_conn, "books", "date_last_scraped", "TEXT")

    if "readers" in existing_after_renaming:
        ensure_column(db_conn, "readers", "username", "TEXT")
        ensure_column(db_conn, "readers", "user_id", "INTEGER")
        ensure_column(db_conn, "readers", "is_self", "INTEGER DEFAULT 0")
        ensure_column(db_conn, "readers", "scrape_error", "TEXT")

    if "embeddings" in existing_after_renaming:
        columns = {row["name"] for row in db_conn.execute("PRAGMA table_info(embeddings)")}
        if "embedding_model" not in columns:
            db_conn.execute("DROP TABLE embeddings")


def ensure_column(db_conn, table, column, definition):
    columns = {row["name"] for row in db_conn.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        db_conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def vector_to_blob(vec):
    return vec.astype(np.float32).tobytes()


def is_valid_embedding_blob(blob, dim):
    if blob is None or dim is None:
        return False
    expected_bytes = int(dim) * np.dtype(np.float32).itemsize
    if len(blob) != expected_bytes:
        return False
    vector = np.frombuffer(blob, dtype=np.float32)
    return bool(vector.size and np.any(vector != 0))


def save_model_params(db_conn, name, params):
    db_conn.execute(
        """
        INSERT OR REPLACE INTO model_params (name, params_json, updated_at)
        VALUES (?, ?, ?)
        """,
        (name, json.dumps(params, sort_keys=True), datetime.now().isoformat()),
    )
    db_conn.commit()


def load_model_params(db_conn, name):
    row = db_conn.execute("SELECT params_json FROM model_params WHERE name = ?", (name,)).fetchone()
    if not row:
        return None
    try:
        return json.loads(row["params_json"])
    except json.JSONDecodeError:
        return None


def save_embeddings(db_conn, book_ids, vectors, model, text_hashes):
    dim = vectors.shape[1]
    rows = []
    for i, bid in enumerate(book_ids):
        bid_int = int(bid)
        h = text_hashes.get(bid_int) if isinstance(text_hashes, dict) else text_hashes[i]
        rows.append((bid_int, model, dim, vector_to_blob(vectors[i]), h))

    db_conn.executemany(
        """
        INSERT OR REPLACE INTO embeddings (book_id, embedding_model, dim, vector, text_hash)
        VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )
    db_conn.commit()


def upsert_rows(db_conn, table, rows, columns):
    if not rows:
        return
    placeholders = ",".join("?" for _ in columns)
    col_names = ",".join(columns)
    db_conn.executemany(
        f"INSERT OR REPLACE INTO {table} ({col_names}) VALUES ({placeholders})",
        rows,
    )
    db_conn.commit()


def get_self_list_id(db_conn):
    row = db_conn.execute("SELECT list_id FROM readers WHERE is_self = 1").fetchone()
    if row is None:
        raise RuntimeError("No self reader found. Run seeding first.")
    return row["list_id"]
