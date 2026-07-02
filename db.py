"""
SQLite database module for goodreads-ranker.

Provides schema initialization, connection management, and memory-efficient
helpers for bulk reads/writes — especially for embedding BLOB storage.
"""

import sqlite3
from pathlib import Path

import numpy as np

DB_PATH = Path("data/goodreads.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS user_library (
    book_id       INTEGER PRIMARY KEY,
    title         TEXT,
    author        TEXT,
    author_lf     TEXT,
    additional_authors TEXT,
    isbn          TEXT,
    isbn13        TEXT,
    my_rating     REAL,
    publisher     TEXT,
    binding       TEXT,
    number_of_pages INTEGER,
    year_published INTEGER,
    original_publication_year INTEGER,
    date_read     TEXT,
    date_added    TEXT,
    bookshelves   TEXT,
    bookshelves_with_positions TEXT,
    exclusive_shelf TEXT,
    my_review     TEXT,
    spoiler       TEXT,
    private_notes TEXT,
    read_count    INTEGER,
    owned_copies  INTEGER
);

CREATE TABLE IF NOT EXISTS friend_lists (
    list_id            INTEGER PRIMARY KEY,
    scrape_complete    INTEGER DEFAULT 0,
    date_last_scraped  TEXT
);

CREATE TABLE IF NOT EXISTS friend_ratings (
    list_id    INTEGER NOT NULL,
    book_id    INTEGER NOT NULL,
    title      TEXT,
    rating     INTEGER,
    num_pages  INTEGER,
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
    currently_reading INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS elo_ratings (
    book_id          INTEGER PRIMARY KEY,
    original_rating  REAL,
    elo_score        REAL DEFAULT 1200.0,
    matches_played   INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS embeddings (
    book_id   INTEGER PRIMARY KEY,
    dim       INTEGER NOT NULL,
    vector    BLOB NOT NULL
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
"""


def get_connection(db_path=None):
    """Return a new SQLite connection with WAL mode and pragmas for performance."""
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=None):
    """Create all tables if they don't exist."""
    conn = get_connection(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def vector_to_blob(vec):
    """Serialize a numpy float32 array to bytes for SQLite BLOB storage."""
    return vec.astype(np.float32).tobytes()


def blob_to_vector(blob, dim):
    """Deserialize a SQLite BLOB back to a numpy float32 array."""
    return np.frombuffer(blob, dtype=np.float32, count=dim)


def save_embeddings(conn, book_ids, vectors):
    """
    Bulk-insert embeddings into the embeddings table.

    Parameters
    ----------
    conn : sqlite3.Connection
    book_ids : array-like of int
    vectors : np.ndarray of shape (n, dim), dtype float32
    """
    dim = vectors.shape[1]
    rows = [
        (int(bid), dim, vector_to_blob(vectors[i])) for i, bid in enumerate(book_ids)
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO embeddings (book_id, dim, vector) VALUES (?, ?, ?)",
        rows,
    )
    conn.commit()


def load_embeddings(conn, book_ids=None):
    """
    Load embeddings into a pre-allocated numpy array.

    Parameters
    ----------
    conn : sqlite3.Connection
    book_ids : list[int] or None
        If None, load all embeddings.

    Returns
    -------
    loaded_ids : np.ndarray of int
    matrix : np.ndarray of shape (n, dim), dtype float32
    """
    if book_ids is not None:
        placeholders = ",".join("?" for _ in book_ids)
        cursor = conn.execute(
            f"SELECT book_id, dim, vector FROM embeddings WHERE book_id IN ({placeholders})",
            list(book_ids),
        )
    else:
        cursor = conn.execute("SELECT book_id, dim, vector FROM embeddings")

    rows = cursor.fetchall()
    if not rows:
        return np.array([], dtype=int), np.empty((0, 0), dtype=np.float32)

    dim = rows[0]["dim"]
    loaded_ids = np.array([r["book_id"] for r in rows], dtype=int)
    matrix = np.empty((len(rows), dim), dtype=np.float32)
    for i, r in enumerate(rows):
        matrix[i] = blob_to_vector(r["vector"], dim)

    return loaded_ids, matrix


def load_embeddings_for_books(conn, ordered_book_ids):
    """
    Load embeddings aligned to a specific ordered list of book IDs.

    Returns a numpy float32 matrix with rows matching the order of
    `ordered_book_ids`. Books without embeddings get zero vectors.

    Parameters
    ----------
    conn : sqlite3.Connection
    ordered_book_ids : array-like of int

    Returns
    -------
    matrix : np.ndarray of shape (len(ordered_book_ids), dim)
    missing_ids : list[int]
    """
    loaded_ids, loaded_matrix = load_embeddings(conn)
    if loaded_matrix.size == 0:
        return np.empty((len(ordered_book_ids), 0), dtype=np.float32), list(
            ordered_book_ids
        )

    dim = loaded_matrix.shape[1]
    id_to_idx = {int(bid): i for i, bid in enumerate(loaded_ids)}

    matrix = np.zeros((len(ordered_book_ids), dim), dtype=np.float32)
    missing_ids = []
    for i, bid in enumerate(ordered_book_ids):
        idx = id_to_idx.get(int(bid))
        if idx is not None:
            matrix[i] = loaded_matrix[idx]
        else:
            missing_ids.append(int(bid))

    return matrix, missing_ids


def upsert_rows(conn, table, rows, columns):
    """
    Bulk upsert rows into a table using INSERT OR REPLACE.

    Parameters
    ----------
    conn : sqlite3.Connection
    table : str
    rows : list[tuple]
    columns : list[str]
    """
    if not rows:
        return
    placeholders = ",".join("?" for _ in columns)
    col_names = ",".join(columns)
    conn.executemany(
        f"INSERT OR REPLACE INTO {table} ({col_names}) VALUES ({placeholders})",
        rows,
    )
    conn.commit()


def normalise_library_columns(df):
    """
    Normalise the column names of a raw Goodreads library export DataFrame.

    Goodreads exports columns like "Title", "Author l-f", "My Rating" etc.
    This converts them to the snake_case names used by the ``user_library``
    table schema.
    """
    rename = {}
    for col in df.columns:
        normalised = col.lower().replace(" ", "_")
        # Goodreads uses "Author l-f" for last-name-first; normalise the
        # hyphen so it maps to the schema column name ``author_lf``.
        if normalised == "author_l-f":
            normalised = "author_lf"
        rename[col] = normalised
    df = df.rename(columns=rename)
    return df
