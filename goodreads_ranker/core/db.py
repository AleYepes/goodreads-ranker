import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

DB_PATH = Path("data/goodreads.db")

SCHEMA = """
-- 1. READER LIBRARIES
CREATE TABLE IF NOT EXISTS libraries (
    legacy_id          INTEGER PRIMARY KEY,
    username           TEXT,
    user_id            INTEGER,
    is_main            INTEGER DEFAULT 0,
    similarity_score   REAL,
    scrape_complete    INTEGER DEFAULT 0,
    date_scraped       TEXT,
    scrape_error       TEXT
);

-- 2. CORE ENTITY: BOOKS
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
    date_fetched             TEXT DEFAULT (strftime('%Y-%m-%d', 'now'))
);

-- 3. BOOK ELO RATINGS
CREATE TABLE IF NOT EXISTS book_elo_ratings (
    book_id          INTEGER PRIMARY KEY REFERENCES books(legacy_id) ON DELETE CASCADE,
    original_rating  INTEGER,
    elo_score        REAL DEFAULT 1200.0,
    matches_played   INTEGER DEFAULT 0
);

-- 4. BOOK EMBEDDINGS
CREATE TABLE IF NOT EXISTS book_embeddings (
    book_id          INTEGER REFERENCES books(legacy_id) ON DELETE CASCADE,
    embedding_model  TEXT,
    vector           BLOB NOT NULL,
    text_hash        TEXT NOT NULL,
    PRIMARY KEY (book_id, embedding_model)
);

-- 5. BOOK PREDICTIONS
CREATE TABLE IF NOT EXISTS book_predictions (
    book_id                INTEGER PRIMARY KEY REFERENCES books(legacy_id) ON DELETE CASCADE,
    solo_pred_rating       REAL,
    friend_pred_rating     REAL,
    count_adjusted_rating  REAL,
    final_rating           REAL,
    date_updated           TEXT
);

-- 6. PREDICTION HYPERPARAMS
CREATE TABLE IF NOT EXISTS prediction_hyperparams (
    name        TEXT PRIMARY KEY,
    params_json TEXT NOT NULL,
    date_updated TEXT NOT NULL
);

-- 7. LIBRARY BOOKS
CREATE TABLE IF NOT EXISTS library_books (
    library_id     INTEGER NOT NULL REFERENCES libraries(legacy_id) ON DELETE CASCADE,
    book_legacy_id INTEGER NOT NULL,
    rating         INTEGER,
    date_read      TEXT,
    date_added     TEXT,
    calibrated_rating REAL,
    PRIMARY KEY (library_id, book_legacy_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_libraries_single_main
ON libraries(is_main) WHERE is_main = 1;

-- 8. PERSISTENT CRAWL QUEUE
CREATE TABLE IF NOT EXISTS crawl_queue (
    book_legacy_id     INTEGER PRIMARY KEY,
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

-- 9. CONTRIBUTORS
CREATE TABLE IF NOT EXISTS contributors (
    legacy_id       INTEGER PRIMARY KEY,
    kca_id          TEXT,
    name            TEXT NOT NULL,
    web_url         TEXT,
    is_gr_author    INTEGER DEFAULT 0,
    works_count     INTEGER DEFAULT 0,
    followers_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS book_contributors (
    book_id        INTEGER REFERENCES books(legacy_id) ON DELETE CASCADE,
    contributor_id INTEGER REFERENCES contributors(legacy_id) ON DELETE CASCADE,
    role           TEXT,
    is_primary     INTEGER DEFAULT 0,
    PRIMARY KEY (book_id, contributor_id, role)
);

-- 10. SERIES
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

-- 11. GENRES
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

-- 12. AUXILIARY DATA
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
    date_discovered   TEXT DEFAULT (strftime('%Y-%m-%d', 'now')),
    PRIMARY KEY (book_id, edition_legacy_id)
);

CREATE TABLE IF NOT EXISTS similar_books (
    book_id           INTEGER REFERENCES books(legacy_id) ON DELETE CASCADE,
    similar_legacy_id INTEGER NOT NULL,
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
        db_conn.execute("PRAGMA busy_timeout=5000")
        db_conn.row_factory = sqlite3.Row
        yield db_conn
    finally:
        db_conn.close()


def init_db(db_path=None):
    with get_connection(db_path) as db_conn:
        db_conn.executescript(SCHEMA)
        db_conn.execute("DROP VIEW IF EXISTS best_book_lookup")
        db_conn.execute(
            """
            CREATE VIEW best_book_lookup AS
            SELECT edition_legacy_id AS raw_legacy_id, book_id AS best_book_id
            FROM book_editions
            UNION
            SELECT legacy_id AS raw_legacy_id, legacy_id AS best_book_id
            FROM books
            """
        )
        db_conn.commit()


# ---------------------------------------------------------------------------
# Generic persistence helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# SQL Centralization - Ingestion crawler & seeder operations
# ---------------------------------------------------------------------------


def save_book_core(db_conn, book: dict):
    db_conn.execute(
        """
        INSERT OR REPLACE INTO books (
            legacy_id, kca_id, title, title_complete, description, web_url,
            asin, isbn, isbn13, format, num_pages, language_name, publisher, publication_time,
            original_publication_time, star_1, star_2, star_3, star_4, star_5,
            date_fetched
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            book.get("legacy_id"),
            book.get("kca_id"),
            book.get("title"),
            book.get("title_complete"),
            book.get("description"),
            book.get("web_url"),
            book.get("asin"),
            book.get("isbn"),
            book.get("isbn13"),
            book.get("format"),
            book.get("num_pages"),
            book.get("language_name"),
            book.get("publisher"),
            book.get("publication_time"),
            book.get("original_publication_time"),
            book.get("star_1"),
            book.get("star_2"),
            book.get("star_3"),
            book.get("star_4"),
            book.get("star_5"),
            book.get("date_fetched"),
        ),
    )


def save_contributors(db_conn, book_legacy_id, contributors: list[dict]):
    for c in contributors:
        db_conn.execute(
            """
            INSERT OR REPLACE INTO contributors (
                legacy_id, kca_id, name, web_url, is_gr_author, works_count, followers_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                c.get("legacy_id"),
                c.get("kca_id"),
                c.get("name"),
                c.get("web_url"),
                c.get("is_gr_author"),
                c.get("works_count"),
                c.get("followers_count"),
            ),
        )
        db_conn.execute(
            """
            INSERT OR REPLACE INTO book_contributors (book_id, contributor_id, role, is_primary)
            VALUES (?, ?, ?, ?)
            """,
            (
                book_legacy_id,
                c.get("legacy_id"),
                c.get("role"),
                c.get("is_primary"),
            ),
        )


def save_series(db_conn, book_legacy_id, series_list: list[dict]):
    for s in series_list:
        db_conn.execute(
            """
            INSERT OR IGNORE INTO series (legacy_id, kca_id, title, web_url)
            VALUES (?, ?, ?, ?)
            """,
            (s.get("legacy_id"), s.get("kca_id"), s.get("title"), s.get("web_url")),
        )
        db_conn.execute(
            """
            INSERT OR REPLACE INTO book_series (book_id, series_id, position)
            VALUES (?, ?, ?)
            """,
            (book_legacy_id, s.get("legacy_id"), s.get("position")),
        )


def save_genres(db_conn, book_legacy_id, genres: list[dict]):
    for g in genres:
        db_conn.execute(
            """
            INSERT OR IGNORE INTO genres (legacy_id, kca_id, name, web_url)
            VALUES (?, ?, ?, ?)
            """,
            (g.get("legacy_id"), g.get("kca_id"), g.get("name"), g.get("web_url")),
        )
        db_conn.execute(
            """
            INSERT OR REPLACE INTO book_genres (book_id, genre_id)
            VALUES (?, ?)
            """,
            (book_legacy_id, g.get("legacy_id")),
        )


def save_awards(db_conn, book_legacy_id, awards: list[dict]):
    for a in awards:
        db_conn.execute(
            """
            INSERT OR IGNORE INTO awards (legacy_id, name, web_url)
            VALUES (?, ?, ?)
            """,
            (a.get("legacy_id"), a.get("name"), a.get("web_url")),
        )
        db_conn.execute(
            """
            INSERT OR REPLACE INTO book_awards (book_id, award_id, category, designation, date_awarded)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                book_legacy_id,
                a.get("legacy_id"),
                a.get("category"),
                a.get("designation"),
                a.get("date_awarded"),
            ),
        )


def save_editions(db_conn, book_legacy_id, editions: list[dict]):
    for ed in editions:
        db_conn.execute(
            """
            INSERT OR REPLACE INTO book_editions (
                book_id, edition_legacy_id, edition_kca_id
            ) VALUES (?, ?, ?)
            """,
            (book_legacy_id, ed.get("edition_legacy_id"), ed.get("edition_kca_id")),
        )


def save_similar_books_and_enqueue(db_conn, book_legacy_id, similar_list: list[dict], now):
    for sim in similar_list:
        sim_legacy_id = sim.get("legacy_id")
        avg_rating = sim.get("average_rating")
        ratings_count = sim.get("ratings_count")

        db_conn.execute(
            """
            INSERT OR REPLACE INTO similar_books (
                book_id, similar_legacy_id, average_rating, ratings_count, date_fetched
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                book_legacy_id,
                sim_legacy_id,
                avg_rating,
                ratings_count,
                now,
            ),
        )

        if sim_legacy_id and avg_rating is not None and ratings_count is not None:
            priority = avg_rating - avg_rating / math.log10(ratings_count + 10)
            db_conn.execute(
                """
                INSERT INTO crawl_queue (book_legacy_id, priority, status, discovered_via)
                VALUES (?, ?, 'pending', 'similar')
                ON CONFLICT(book_legacy_id) DO UPDATE SET
                    priority = MAX(priority, excluded.priority)
                WHERE status = 'pending'
                """,
                (sim_legacy_id, priority),
            )


def book_exists(db_conn, legacy_id) -> bool:
    row = db_conn.execute("SELECT 1 FROM books WHERE legacy_id = ?", (legacy_id,)).fetchone()
    return row is not None


def link_editions_to_canonical(db_conn, best_book_legacy_id, legacy_id, book_kca_id, editions):
    db_conn.execute(
        "INSERT OR IGNORE INTO book_editions (book_id, edition_legacy_id, edition_kca_id) VALUES (?, ?, ?)",
        (best_book_legacy_id, legacy_id, book_kca_id),
    )
    for edition in editions:
        db_conn.execute(
            """
            INSERT OR REPLACE INTO book_editions (
                book_id, edition_legacy_id, edition_kca_id
            ) VALUES (?, ?, ?)
            """,
            (best_book_legacy_id, edition.get("legacyId"), edition.get("id")),
        )


def mark_known_editions_skipped(db_conn, best_book_legacy_id, now):
    db_conn.execute(
        """
        UPDATE crawl_queue
        SET status = 'skipped_known_edition', date_processed = ?
        WHERE book_legacy_id IN (SELECT edition_legacy_id FROM book_editions WHERE book_id = ?)
          AND status = 'pending'
        """,
        (now, best_book_legacy_id),
    )


def get_pending_crawl_batch(db_conn, allowed_sources, limit) -> list[int]:
    placeholders = ",".join("?" for _ in allowed_sources)
    query = f"""
        SELECT book_legacy_id FROM crawl_queue
        WHERE status = 'pending'
          AND discovered_via IN ({placeholders})
        ORDER BY (discovered_via = 'seed') DESC, priority DESC
        LIMIT ?
    """
    cursor = db_conn.execute(query, allowed_sources + [limit])
    return [row["book_legacy_id"] for row in cursor.fetchall()]


def count_crawl_queue(db_conn, statuses, allowed_sources) -> int:
    status_placeholders = ",".join("?" for _ in statuses)
    source_placeholders = ",".join("?" for _ in allowed_sources)
    query = f"""
        SELECT COUNT(*) FROM crawl_queue
        WHERE status IN ({status_placeholders})
          AND discovered_via IN ({source_placeholders})
    """
    cursor = db_conn.execute(query, list(statuses) + list(allowed_sources))
    return cursor.fetchone()[0]


def populate_seeds(db_conn):
    cursor = db_conn.execute("SELECT DISTINCT book_legacy_id FROM library_books WHERE book_legacy_id IS NOT NULL")
    seeds = [row["book_legacy_id"] for row in cursor.fetchall()]
    for seed in seeds:
        db_conn.execute(
            """
            INSERT OR IGNORE INTO crawl_queue (book_legacy_id, status, priority, discovered_via)
            VALUES (?, 'pending', 0.0, 'seed')
            """,
            (seed,),
        )
    db_conn.commit()


def handle_force_crawl(db_conn):
    db_conn.execute(
        """
        UPDATE crawl_queue
        SET status = 'pending'
        WHERE status = 'done'
        AND EXISTS (
            SELECT 1
            FROM books
            WHERE books.legacy_id = crawl_queue.book_legacy_id
                AND books.date_fetched < date('now', '-1 days')
        );
        """
    )
    db_conn.commit()


def set_crawl_status(db_conn, legacy_id, status, error_count=0, last_error_message=None, date_processed=None):
    db_conn.execute(
        """
        INSERT INTO crawl_queue (book_legacy_id, status, error_count, last_error_message, date_processed)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(book_legacy_id) DO UPDATE SET
            status = excluded.status,
            error_count = excluded.error_count,
            last_error_message = excluded.last_error_message,
            date_processed = excluded.date_processed
        """,
        (legacy_id, status, error_count, last_error_message, date_processed),
    )
    db_conn.commit()


def get_crawl_error_count(db_conn, legacy_id: int) -> int:
    row = db_conn.execute("SELECT error_count FROM crawl_queue WHERE book_legacy_id = ?", (legacy_id,)).fetchone()
    return (row["error_count"] or 0) if row else 0


def get_elo_ratings(db_conn) -> list[dict]:
    cursor = db_conn.execute(
        "SELECT book_id AS legacy_id, original_rating, elo_score, matches_played FROM book_elo_ratings"
    )
    return [dict(r) for r in cursor.fetchall()]


def save_elo_ratings(db_conn, rows):
    upsert_rows(
        db_conn,
        "book_elo_ratings",
        rows,
        ["book_id", "original_rating", "elo_score", "matches_played"],
    )


def get_main_library_id(db_conn):
    row = db_conn.execute("SELECT legacy_id FROM libraries WHERE is_main = 1").fetchone()
    if row is None:
        raise RuntimeError("No main library found. Run seeding first.")
    return row["legacy_id"]


def get_main_library_ratings(db_conn, main_library_id) -> list[dict]:
    cursor = db_conn.execute(
        """
        SELECT b.legacy_id, b.title, MAX(lb.rating) AS rating
        FROM books b
        LEFT JOIN best_book_lookup bbl ON b.legacy_id = bbl.best_book_id
        LEFT JOIN library_books lb ON bbl.raw_legacy_id = lb.book_legacy_id AND lb.library_id = ?
        GROUP BY b.legacy_id
        ORDER BY b.legacy_id
        """,
        (main_library_id,),
    )
    return [dict(r) for r in cursor.fetchall()]


def save_friend_similarity_scores(db_conn, scores: dict[int, float]):
    for library_id, score in scores.items():
        db_conn.execute(
            "UPDATE libraries SET similarity_score = ? WHERE legacy_id = ?",
            (score, int(library_id)),
        )
    db_conn.commit()


def update_calibrated_ratings(db_conn, rows: list[tuple[int, int, float]]):
    db_conn.executemany(
        """
        UPDATE library_books
        SET calibrated_rating = ?
        WHERE library_id = ? AND book_legacy_id = ?
        """,
        [(r[2], r[0], r[1]) for r in rows],
    )
    db_conn.commit()


def get_friend_calibrated_ratings(db_conn, min_friend_similarity: float) -> list[dict]:
    cursor = db_conn.execute(
        """
        SELECT lb.library_id, lb.book_legacy_id, lb.calibrated_rating, l.similarity_score
        FROM library_books lb
        JOIN libraries l ON lb.library_id = l.legacy_id
        WHERE l.similarity_score >= ? AND lb.calibrated_rating IS NOT NULL
        """,
        (min_friend_similarity,),
    )
    return [dict(r) for r in cursor.fetchall()]


def get_books_for_prediction(db_conn, main_library_id) -> list[dict]:
    cursor = db_conn.execute(
        """
        SELECT b.*, MAX(lb.rating) AS my_rating
        FROM books b
        LEFT JOIN best_book_lookup bbl ON b.legacy_id = bbl.best_book_id
        LEFT JOIN library_books lb ON bbl.raw_legacy_id = lb.book_legacy_id AND lb.library_id = ?
        GROUP BY b.legacy_id
        ORDER BY b.legacy_id
        """,
        (main_library_id,),
    )
    return [dict(r) for r in cursor.fetchall()]


def get_similar_books_edges(db_conn) -> list[tuple[int, int]]:
    cursor = db_conn.execute(
        """
        SELECT sb.book_id, bbl.best_book_id AS similar_legacy_id
        FROM similar_books sb
        JOIN best_book_lookup bbl ON sb.similar_legacy_id = bbl.raw_legacy_id
        """
    )
    return [(int(row["book_id"]), int(row["similar_legacy_id"])) for row in cursor.fetchall()]


def get_friend_library_book_ratings(db_conn) -> list[dict]:
    cursor = db_conn.execute(
        """
        SELECT lb.library_id, bbl.best_book_id AS legacy_id, MAX(lb.rating) AS rating
        FROM library_books lb
        JOIN libraries l ON lb.library_id = l.legacy_id
        JOIN best_book_lookup bbl ON lb.book_legacy_id = bbl.raw_legacy_id
        WHERE l.is_main = 0
        GROUP BY lb.library_id, bbl.best_book_id
        """
    )
    return [dict(r) for r in cursor.fetchall()]


def get_embeddings_by_model(db_conn, model: str) -> dict[int, bytes]:
    cursor = db_conn.execute(
        """
        SELECT book_id, vector
        FROM book_embeddings
        WHERE embedding_model = ?
        """,
        (model,),
    )
    return {int(row["book_id"]): row["vector"] for row in cursor.fetchall()}


def save_book_predictions(db_conn, rows):
    upsert_rows(
        db_conn,
        "book_predictions",
        rows,
        [
            "book_id",
            "solo_pred_rating",
            "friend_pred_rating",
            "count_adjusted_rating",
            "final_rating",
            "date_updated",
        ],
    )


def prune_book_predictions(db_conn, keep_ids: list[int]):
    if not keep_ids:
        return
    placeholders = ",".join("?" for _ in keep_ids)
    db_conn.execute(
        f"DELETE FROM book_predictions WHERE book_id NOT IN ({placeholders})",
        keep_ids,
    )
    db_conn.commit()


def get_prediction_hyperparams(db_conn, name):
    row = db_conn.execute("SELECT params_json FROM prediction_hyperparams WHERE name = ?", (name,)).fetchone()
    if not row:
        return None
    try:
        return json.loads(row["params_json"])
    except json.JSONDecodeError:
        return None


def save_prediction_hyperparams(db_conn, name, params):
    db_conn.execute(
        """
        INSERT OR REPLACE INTO prediction_hyperparams (name, params_json, date_updated)
        VALUES (?, ?, ?)
        """,
        (name, json.dumps(params, sort_keys=True), datetime.now().strftime("%Y-%m-%d")),
    )
    db_conn.commit()


def get_book_metadata_for_embedding(db_conn) -> list[dict]:
    cursor = db_conn.execute(
        """
        SELECT b.legacy_id, b.title, c.name AS author_name, b.description
        FROM books b
        LEFT JOIN book_contributors bc ON bc.book_id = b.legacy_id AND bc.is_primary = 1
        LEFT JOIN contributors c ON c.legacy_id = bc.contributor_id
        ORDER BY b.legacy_id
        """
    )
    rows = [dict(r) for r in cursor.fetchall()]

    cursor_genres = db_conn.execute(
        """
        SELECT bg.book_id, g.name
        FROM book_genres bg
        JOIN genres g ON bg.genre_id = g.legacy_id
        """
    )
    genres_by_book: dict[int, list[str]] = {}
    for r in cursor_genres.fetchall():
        bid = int(r["book_id"])
        genres_by_book.setdefault(bid, []).append(r["name"])

    results = []
    for r in rows:
        bid = int(r["legacy_id"])
        results.append(
            {
                "legacy_id": bid,
                "title": r["title"] or "",
                "author_name": r["author_name"] or "",
                "description": r["description"] or "",
                "genres": genres_by_book.get(bid, []),
            }
        )
    return results


def get_existing_embeddings(db_conn, model) -> dict[int, dict]:
    cursor = db_conn.execute(
        """
        SELECT book_id, vector, text_hash
        FROM book_embeddings
        WHERE embedding_model = ?
        """,
        (model,),
    )
    return {int(row["book_id"]): {"vector": row["vector"], "text_hash": row["text_hash"]} for row in cursor.fetchall()}


def save_embeddings(db_conn, legacy_ids, vectors, model, text_hashes):
    rows = []
    for i, bid in enumerate(legacy_ids):
        bid_int = int(bid)
        h = text_hashes.get(bid_int) if isinstance(text_hashes, dict) else text_hashes[i]
        vector_blob = vectors[i].astype("float32").tobytes()
        rows.append((bid_int, model, vector_blob, h))

    db_conn.executemany(
        """
        INSERT OR REPLACE INTO book_embeddings (book_id, embedding_model, vector, text_hash)
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )
    db_conn.commit()


def is_valid_embedding_blob(blob):
    if blob is None:
        return False
    import numpy as np

    if len(blob) % np.dtype(np.float32).itemsize != 0:
        return False
    vector = np.frombuffer(blob, dtype=np.float32)
    return bool(vector.size and np.any(vector != 0))


# ---------------------------------------------------------------------------
# Seeder-specific DB operations
# ---------------------------------------------------------------------------


def upsert_libraries(db_conn, main_user: dict, friends: list[dict]):
    db_conn.execute(
        "UPDATE libraries SET is_main = 0 WHERE is_main = 1 AND legacy_id != ?",
        (main_user["library_id"],),
    )

    db_conn.execute(
        """
        INSERT OR IGNORE INTO libraries (legacy_id, user_id, username, is_main, scrape_complete)
        VALUES (?, ?, ?, 1, 0)
        """,
        (main_user["library_id"], main_user["user_id"], main_user["username"]),
    )
    db_conn.execute(
        """
        UPDATE libraries
        SET user_id = ?, username = ?, is_main = 1
        WHERE legacy_id = ?
        """,
        (main_user["user_id"], main_user["username"], main_user["library_id"]),
    )

    db_conn.executemany(
        """
        INSERT OR IGNORE INTO libraries (legacy_id, user_id, username, is_main, scrape_complete)
        VALUES (?, ?, ?, 0, 0)
        """,
        [(f["library_id"], f["user_id"], f["username"]) for f in friends],
    )
    db_conn.commit()


def update_friend_info(db_conn, library_id, username, user_id):
    db_conn.execute(
        """
        UPDATE libraries
        SET username = ?,
            user_id = ?
        WHERE legacy_id = ?
        """,
        (username, user_id, library_id),
    )
    db_conn.commit()


def load_existing_library_rows(db_conn, library_id) -> dict:
    rows = db_conn.execute(
        """
        SELECT library_id, book_legacy_id, rating, date_read, date_added
        FROM library_books
        WHERE library_id = ?
        """,
        (library_id,),
    ).fetchall()
    return {
        (int(row["library_id"]), int(row["book_legacy_id"])): {
            "library_id": int(row["library_id"]),
            "book_legacy_id": int(row["book_legacy_id"]),
            "rating": int(row["rating"] or 0),
            "date_read": row["date_read"] or "",
            "date_added": row["date_added"] or "",
        }
        for row in rows
    }


def mark_library_complete(db_conn, library_id):
    today = datetime.now().strftime("%Y-%m-%d")
    db_conn.execute(
        """
        UPDATE libraries
        SET scrape_complete = 1,
            date_scraped = ?,
            scrape_error = NULL
        WHERE legacy_id = ?
        """,
        (today, library_id),
    )
    db_conn.commit()


def mark_library_failed(db_conn, library_id, error):
    db_conn.execute(
        """
        UPDATE libraries
        SET scrape_complete = 0,
            scrape_error = ?
        WHERE legacy_id = ?
        """,
        (str(error)[:1000], library_id),
    )
    db_conn.commit()


def upsert_library_books(db_conn, rows: list[dict]):
    if not rows:
        return
    upsert_rows(
        db_conn,
        "library_books",
        [
            (
                row["library_id"],
                row["book_legacy_id"],
                row["rating"],
                row["date_read"],
                row["date_added"],
            )
            for row in rows
        ],
        ["library_id", "book_legacy_id", "rating", "date_read", "date_added"],
    )


def bootstrap_libraries(db_conn, library_ids: list[int]):
    for lid in library_ids:
        db_conn.execute(
            "INSERT OR IGNORE INTO libraries (legacy_id, is_main, scrape_complete) VALUES (?, 0, 0)",
            (lid,),
        )
        db_conn.execute(
            "UPDATE libraries SET scrape_complete = 0 WHERE legacy_id = ?",
            (lid,),
        )
    db_conn.commit()


def get_total_books_count(db_conn) -> int:
    cursor = db_conn.execute("SELECT COUNT(*) FROM books")
    return cursor.fetchone()[0]


def get_libraries_to_scrape(db_conn, force_seed: bool) -> list[int]:
    if force_seed:
        cursor = db_conn.execute("SELECT legacy_id FROM libraries")
    else:
        cursor = db_conn.execute(
            """
            SELECT legacy_id
            FROM libraries
            WHERE scrape_complete != 1
            """
        )
    return [int(row["legacy_id"]) for row in cursor.fetchall()]


def get_all_book_ids(db_conn) -> list[int]:
    cursor = db_conn.execute("SELECT legacy_id FROM books ORDER BY legacy_id")
    return [int(row["legacy_id"]) for row in cursor.fetchall()]
