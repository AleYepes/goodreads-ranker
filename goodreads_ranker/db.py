import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np

DB_PATH = Path("data/goodreads.db")

SCHEMA = """
-- 1. COMPATIBILITY TABLES

CREATE TABLE IF NOT EXISTS readers (
    list_id            INTEGER PRIMARY KEY,
    username           TEXT,
    user_id            INTEGER,
    is_self            INTEGER DEFAULT 0,
    scrape_complete    INTEGER DEFAULT 0,
    date_scraped       TEXT,
    scrape_error       TEXT
);

CREATE TABLE IF NOT EXISTS reader_libraries (
    list_id  INTEGER NOT NULL,
    book_id  INTEGER NOT NULL,
    rating   INTEGER,
    date_read  TEXT,
    date_added TEXT,
    PRIMARY KEY (list_id, book_id)
);

CREATE TABLE IF NOT EXISTS book_elo_ratings (
    book_id          INTEGER PRIMARY KEY,
    original_rating  REAL,
    elo_score        REAL DEFAULT 1200.0,
    matches_played   INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS book_embeddings (
    book_id          INTEGER,
    embedding_model  TEXT,
    vector           BLOB NOT NULL,
    text_hash        TEXT NOT NULL,
    PRIMARY KEY (book_id, embedding_model)
);

CREATE TABLE IF NOT EXISTS book_predictions (
    book_id                INTEGER PRIMARY KEY,
    solo_pred_rating       REAL,
    friend_pred_rating     REAL,
    count_adjusted_rating  REAL,
    pred_rating            REAL,
    final_rating           REAL,
    date_updated           TEXT
);

CREATE TABLE IF NOT EXISTS prediction_hyperparams (
    name        TEXT PRIMARY KEY,
    params_json TEXT NOT NULL,
    date_updated TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_readers_single_self
ON readers(is_self) WHERE is_self = 1;

-- 2. PERSISTENT CRAWL QUEUE

CREATE TABLE IF NOT EXISTS crawl_queue (
    book_id            INTEGER PRIMARY KEY,
    status             TEXT NOT NULL DEFAULT 'pending',
    priority           REAL NOT NULL DEFAULT 0.0,
    error_count        INTEGER DEFAULT 0,
    last_error_message TEXT,
    discovered_via     TEXT,
    date_enqueued      TEXT DEFAULT (strftime('%Y-%m-%d', 'now')),
    date_processed     TEXT
);

CREATE INDEX IF NOT EXISTS idx_crawl_queue_priority
ON crawl_queue(status, discovered_via, priority DESC);

-- 3. CONTRIBUTORS
CREATE TABLE IF NOT EXISTS contributors (
    legacy_id       INTEGER PRIMARY KEY,
    kca_id          TEXT,
    name            TEXT NOT NULL,
    web_url         TEXT,
    is_gr_author    INTEGER DEFAULT 0,
    works_count     INTEGER DEFAULT 0,
    followers_count INTEGER DEFAULT 0
);

-- 4. CORE ENTITY: BOOKS
CREATE TABLE IF NOT EXISTS books (
    legacy_id                INTEGER PRIMARY KEY,
    kca_id                   TEXT,
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
    publication_time         INTEGER,
    original_publication_time INTEGER,
    star_1                   INTEGER DEFAULT 0,
    star_2                   INTEGER DEFAULT 0,
    star_3                   INTEGER DEFAULT 0,
    star_4                   INTEGER DEFAULT 0,
    star_5                   INTEGER DEFAULT 0,
    currently_reading_count  INTEGER DEFAULT 0,
    to_read_count            INTEGER DEFAULT 0,
    date_fetched             TEXT DEFAULT (strftime('%Y-%m-%d', 'now'))
);

CREATE TABLE IF NOT EXISTS book_contributors (
    book_id        INTEGER REFERENCES books(legacy_id) ON DELETE CASCADE,
    contributor_id INTEGER REFERENCES contributors(legacy_id) ON DELETE CASCADE,
    role           TEXT,
    is_primary     INTEGER DEFAULT 0,
    PRIMARY KEY (book_id, contributor_id, role)
);

-- 5. SERIES
CREATE TABLE IF NOT EXISTS series (
    legacy_id INTEGER PRIMARY KEY,
    kca_id    TEXT,
    title     TEXT NOT NULL,
    web_url   TEXT
);

CREATE TABLE IF NOT EXISTS book_series (
    book_id   INTEGER REFERENCES books(legacy_id) ON DELETE CASCADE,
    series_id INTEGER REFERENCES series(legacy_id) ON DELETE CASCADE,
    position  INTEGER,
    PRIMARY KEY (book_id, series_id)
);

-- 6. GENRES
CREATE TABLE IF NOT EXISTS genres (
    legacy_id TEXT PRIMARY KEY,
    kca_id    TEXT,
    name      TEXT,
    web_url   TEXT
);

CREATE TABLE IF NOT EXISTS book_genres (
    book_id   INTEGER REFERENCES books(legacy_id) ON DELETE CASCADE,
    genre_id  TEXT REFERENCES genres(legacy_id) ON DELETE CASCADE,
    PRIMARY KEY (book_id, genre_id)
);

-- 7. AUXILIARY DATA
CREATE TABLE IF NOT EXISTS awards (
    legacy_id INTEGER PRIMARY KEY,
    name      TEXT NOT NULL,
    web_url   TEXT
);

CREATE TABLE IF NOT EXISTS book_awards (
    book_id      INTEGER REFERENCES books(legacy_id) ON DELETE CASCADE,
    award_id     INTEGER REFERENCES awards(legacy_id) ON DELETE CASCADE,
    category     TEXT,
    designation  TEXT,
    date_awarded TEXT,
    PRIMARY KEY (book_id, award_id)
);

CREATE TABLE IF NOT EXISTS book_editions (
    book_id           INTEGER REFERENCES books(legacy_id) ON DELETE CASCADE,
    edition_legacy_id INTEGER NOT NULL,
    edition_kca_id    TEXT,
    title             TEXT,
    date_discovered   TEXT DEFAULT (strftime('%Y-%m-%d', 'now')),
    PRIMARY KEY (book_id, edition_legacy_id)
);

CREATE TABLE IF NOT EXISTS book_similar_books (
    book_id           INTEGER REFERENCES books(legacy_id) ON DELETE CASCADE,
    similar_legacy_id INTEGER NOT NULL,
    title             TEXT,
    average_rating    REAL,
    ratings_count     INTEGER,
    date_fetched      TEXT DEFAULT (strftime('%Y-%m-%d', 'now')),
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
        db_conn.executescript(SCHEMA)
        db_conn.execute("DROP VIEW IF EXISTS book_id_resolved")
        db_conn.commit()


def vector_to_blob(vec):
    return vec.astype(np.float32).tobytes()


def is_valid_embedding_blob(blob):
    if blob is None:
        return False
    if len(blob) % np.dtype(np.float32).itemsize != 0:
        return False
    vector = np.frombuffer(blob, dtype=np.float32)
    return bool(vector.size and np.any(vector != 0))


def save_prediction_hyperparams(db_conn, name, params):
    db_conn.execute(
        """
        INSERT OR REPLACE INTO prediction_hyperparams (name, params_json, date_updated)
        VALUES (?, ?, ?)
        """,
        (name, json.dumps(params, sort_keys=True), datetime.now().strftime("%Y-%m-%d")),
    )
    db_conn.commit()


def load_prediction_hyperparams(db_conn, name):
    row = db_conn.execute("SELECT params_json FROM prediction_hyperparams WHERE name = ?", (name,)).fetchone()
    if not row:
        return None
    try:
        return json.loads(row["params_json"])
    except json.JSONDecodeError:
        return None


def save_embeddings(db_conn, legacy_ids, vectors, model, text_hashes):
    rows = []
    for i, bid in enumerate(legacy_ids):
        bid_int = int(bid)
        h = text_hashes.get(bid_int) if isinstance(text_hashes, dict) else text_hashes[i]
        rows.append((bid_int, model, vector_to_blob(vectors[i]), h))

    db_conn.executemany(
        """
        INSERT OR REPLACE INTO book_embeddings (book_id, embedding_model, vector, text_hash)
        VALUES (?, ?, ?, ?)
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
