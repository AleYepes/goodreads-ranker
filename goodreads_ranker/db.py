import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np

DB_PATH = Path("data/goodreads.db")

SCHEMA = """
-- 1. COMPATIBILITY TABLES (legacy_id → book_id rename throughout)

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
    list_id  INTEGER NOT NULL,
    book_id  INTEGER NOT NULL,   -- renamed from legacy_id
    rating   INTEGER,
    date_read  TEXT,
    date_added TEXT,
    PRIMARY KEY (list_id, book_id)
);

CREATE TABLE IF NOT EXISTS elo_ratings (
    book_id          INTEGER PRIMARY KEY,   -- renamed from legacy_id
    original_rating  REAL,
    elo_score        REAL DEFAULT 1200.0,
    matches_played   INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS embeddings (
    book_id          INTEGER,              -- renamed from legacy_id
    embedding_model  TEXT,
    dim              INTEGER NOT NULL,
    vector           BLOB NOT NULL,
    text_hash        TEXT NOT NULL,
    PRIMARY KEY (book_id, embedding_model)
);

CREATE TABLE IF NOT EXISTS predictions (
    book_id                INTEGER PRIMARY KEY,   -- renamed from legacy_id
    solo_pred_rating       REAL,
    friend_pred_rating     REAL,
    count_adjusted_rating  REAL,
    pred_rating            REAL,
    final_rating           REAL,
    updated_at             TEXT
);

CREATE TABLE IF NOT EXISTS model_params (
    name       TEXT PRIMARY KEY,
    params_json TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_readers_single_self
ON readers(is_self) WHERE is_self = 1;

-- 2. PERSISTENT CRAWL QUEUE

CREATE TABLE IF NOT EXISTS crawl_queue (
    book_id           INTEGER PRIMARY KEY,   -- renamed from legacy_id
    status            TEXT NOT NULL DEFAULT 'pending',
    priority          REAL NOT NULL DEFAULT 0.0,
    error_count       INTEGER DEFAULT 0,
    last_error_message TEXT,
    discovered_via    TEXT,
    enqueued_at       TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    processed_at      TEXT
);

CREATE INDEX IF NOT EXISTS idx_crawl_queue_priority
ON crawl_queue(status, discovered_via, priority DESC);

-- 3. AUTHORS

CREATE TABLE IF NOT EXISTS authors (
    legacy_id       INTEGER PRIMARY KEY,   -- author's own GR legacy_id
    name            TEXT NOT NULL,
    web_url         TEXT,
    is_gr_author    INTEGER DEFAULT 0,
    works_count     INTEGER DEFAULT 0,
    followers_count INTEGER DEFAULT 0
);

-- 4. CORE ENTITY: BOOKS (merged books + works)

CREATE TABLE IF NOT EXISTS books (
    legacy_id                INTEGER PRIMARY KEY,   -- always the canonical (bestBook) edition
    kca_id                   TEXT,                  -- "kca://book/..." for API calls
    author_id                INTEGER REFERENCES authors(legacy_id),
    author_role              TEXT,
    title                    TEXT,
    title_complete           TEXT,
    description              TEXT,
    web_url                  TEXT,
    asin                     TEXT,
    isbn                     TEXT,
    isbn13                   TEXT,
    format                   TEXT,
    num_pages                INTEGER,
    language_name            TEXT,
    publisher                TEXT,
    publication_time         INTEGER,               -- this edition's publication date
    original_publication_time INTEGER,              -- work-level original publication date
    star_1                   INTEGER DEFAULT 0,
    star_2                   INTEGER DEFAULT 0,
    star_3                   INTEGER DEFAULT 0,
    star_4                   INTEGER DEFAULT 0,
    star_5                   INTEGER DEFAULT 0,
    currently_reading_count  INTEGER DEFAULT 0,
    to_read_count            INTEGER DEFAULT 0,
    fetched_at               TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- 5. SERIES (entity + junction)

CREATE TABLE IF NOT EXISTS series (
    id      INTEGER PRIMARY KEY,
    title   TEXT NOT NULL,
    web_url TEXT
);

CREATE TABLE IF NOT EXISTS book_series (
    book_id   INTEGER REFERENCES books(legacy_id) ON DELETE CASCADE,
    series_id INTEGER REFERENCES series(id) ON DELETE CASCADE,
    position  TEXT,
    PRIMARY KEY (book_id, series_id)
);

-- 6. GENRES (flat child table, no entity table)

CREATE TABLE IF NOT EXISTS genres (
    book_id INTEGER REFERENCES books(legacy_id) ON DELETE CASCADE,
    name    TEXT NOT NULL,
    PRIMARY KEY (book_id, name)
);

-- 7. AUXILIARY DATA

CREATE TABLE IF NOT EXISTS awards (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id     INTEGER REFERENCES books(legacy_id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    category    TEXT,
    designation TEXT,
    awarded_at  INTEGER,
    web_url     TEXT
);

CREATE TABLE IF NOT EXISTS book_editions (
    book_id           INTEGER REFERENCES books(legacy_id) ON DELETE CASCADE,
    edition_legacy_id INTEGER NOT NULL,   -- sibling edition's GR legacy_id, not a FK
    title             TEXT,
    discovered_at     TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (book_id, edition_legacy_id)
);

CREATE TABLE IF NOT EXISTS similar_books (
    book_id           INTEGER REFERENCES books(legacy_id) ON DELETE CASCADE,
    similar_legacy_id INTEGER NOT NULL,   -- similar book's GR legacy_id, not a FK
    rank              INTEGER,
    title             TEXT,
    average_rating    REAL,
    ratings_count     INTEGER,
    fetched_at        TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (book_id, similar_legacy_id)
);
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
    pass


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


def save_embeddings(db_conn, legacy_ids, vectors, model, text_hashes):
    dim = vectors.shape[1]
    rows = []
    for i, bid in enumerate(legacy_ids):
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
