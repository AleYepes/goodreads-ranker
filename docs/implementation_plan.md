# Schema Refactoring: Merge Works into Books

Collapse the separate `works` and `books` tables into a single `books` entity, clean up all junction tables, standardise FK column naming, and remove secondary contributor logic. No migration — the DB will be deleted and repopulated from scratch.

---

## Decisions Summary

| Decision | Outcome |
|---|---|
| Merge `works` + `books` | ✅ Single `books` table |
| `work_id` / `work.legacyId` | ❌ Dropped entirely |
| `work.id` (KCA) | ❌ Dropped entirely |
| `book.id` (KCA) | ✅ Stored as `kca_id TEXT` on `books` |
| `books.legacy_id` PK | ✅ Unchanged, always the bestBook edition |
| FK column convention | ✅ `book_id` everywhere, including compatibility tables |
| Raw GR IDs (non-FK) | ✅ Prefixed: `edition_legacy_id`, `similar_legacy_id` |
| `social_signals` table | ❌ Dropped, folded into `books` as two columns |
| `contributors` + `book_contributors` | ❌ Dropped, replaced by flat `authors` table + FK on `books` |
| Secondary contributors | ❌ Dropped entirely |
| `work_series` → `book_series` | ✅ Renamed, FK updated |
| `work_genres` entity + junction | ❌ Collapsed into flat `genres` child table |
| `work_awards` → `awards` | ✅ Renamed, FK updated |
| `known_editions` → `book_editions` | ✅ Renamed, FK + column names updated |
| `series.web_url` | ✅ Kept |
| `genres.web_url` | ❌ Dropped |
| `books.image_url` | ❌ Dropped |
| `books.title` + `books.title_complete` | ✅ Both kept with original column names |
| Star ratings | ✅ `star_1`–`star_5` (from works, kept) |
| `language_name` | ✅ Kept as-is |
| `publication_time` (book) | ✅ Kept as-is (edition pub date) |
| `original_publication_time` | ✅ Added (was `works.publication_time`) |
| Dropped from works | `original_title`, `web_url`, `shelves_url`, `average_rating`, `ratings_count`, `text_reviews_count`, `text_reviews_language_counts`, `editions_total_count`, `editions_coverage_complete` |

---

## Proposed Changes

### `db.py`

#### [MODIFY] [db.py](file:///Users/alex/Documents/goodreads-ranker/goodreads_ranker/db.py)

Replace the entire `SCHEMA` constant with the schema below.

**New `SCHEMA`:**

```sql
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
```

Also update `get_self_list_id` and any other helper queries that reference the old column names.

---

### `crawler.py`

#### [MODIFY] [crawler.py](file:///Users/alex/Documents/goodreads-ranker/goodreads_ranker/crawler.py)

**`BOOK_QUERY` — fields to remove:**
- `imageUrl`
- `secondaryContributorEdges { ... }` (entire block)
- `work { id legacyId ... }` — remove `id` and `legacyId` from the work subselection
- `work.stats.averageRating`, `work.stats.ratingsCount`, `work.stats.textReviewsCount`, `work.stats.textReviewsLanguageCounts`
- `work.details.webUrl`, `work.details.shelvesUrl`, `work.details.originalTitle`

**`BOOK_QUERY` — fields to keep:**
- `id` (book-level KCA id → stored as `kca_id`)
- `legacyId`, `title`, `titleComplete`, `description`, `webUrl`
- `primaryContributorEdge { node { id legacyId name isGrAuthor webUrl followers { totalCount } works { totalCount } } role }`
- `bookSeries { userPosition series { id title webUrl } }`
- `bookGenres { genre { name } }`
- `details { asin isbn isbn13 format numPages publisher publicationTime language { name } }`
- `work { stats { ratingsCountDist } details { publicationTime awardsWon { name webUrl awardedAt category designation } } editions(pagination: {limit: 20}) { totalCount edges { node { legacyId title } } } }`

**`resolve_and_save_book` — DB write changes:**

1. **Remove** the `INSERT OR REPLACE INTO works (...)` block entirely.

2. **Modify** the `INSERT OR REPLACE INTO books (...)` to use the new unified schema:
   - Add `kca_id` (from `book_node.get("id")`)
   - Add `author_id` + `author_role` (from `primaryContributorEdge`)
   - Add `original_publication_time` (from `work_details.get("publicationTime")`)
   - Add `star_1`–`star_5` (from `work_stats.get("ratingsCountDist")`)
   - Add `currently_reading_count` / `to_read_count` (from `social_list`)
   - Remove `image_url`, `work_id`

3. **Remove** all `secondary_edges` / `secondaryContributorEdges` processing.

4. **Keep** the `INSERT OR REPLACE INTO authors (...)` (replaces `contributors`) + note: only insert the primary contributor node.

5. **Remove** `INSERT OR REPLACE INTO book_contributors (...)`. Instead, the author FK lives on the `books` row itself.

6. **Rename** junction table writes:
   - `work_series` → `book_series`, column `work_id` → `book_id`, `user_position` → `position`
   - `work_genres` → `genres`, column `work_id` → `book_id`; remove `web_url` insert
   - `work_awards` → `awards`, column `work_id` → `book_id`
   - `known_editions` → `book_editions`, columns `work_id` → `book_id`, `legacy_id` → `edition_legacy_id`
   - `social_signals` insert block → **removed** (data goes into `books` row directly)
   - `similar_books`: column `work_id` → `book_id`, `similar_legacy_id` unchanged

7. **Sibling pruning query** — update column/table names:
   ```sql
   UPDATE crawl_queue
   SET status = 'skipped_known_edition', processed_at = ?
   WHERE book_id IN (SELECT edition_legacy_id FROM book_editions WHERE book_id = ?)
     AND status = 'pending'
   ```

8. **`populate_seeds`** — update column name:
   ```sql
   SELECT DISTINCT book_id FROM reader_libraries WHERE book_id IS NOT NULL
   ```
   Insert seeds into `crawl_queue (book_id, ...)`.

9. **`handle_force_recrawl`** — update column name:
   ```sql
   UPDATE crawl_queue SET status = 'pending'
   WHERE book_id IN (
       SELECT legacy_id FROM books WHERE datetime(fetched_at) < datetime('now', '-30 days')
   )
   AND status = 'done'
   ```

10. **`resolve_and_save_book` canonical check** — no change needed (still `SELECT 1 FROM books WHERE legacy_id = ?`).

---

### `seeder.py`

#### [MODIFY] [seeder.py](file:///Users/alex/Documents/goodreads-ranker/goodreads_ranker/seeder.py)

- Rename all references to `reader_libraries.legacy_id` → `reader_libraries.book_id`.
- This affects: `load_existing_rows`, `upsert_extracted`, and the key tuple `(list_id, legacy_id)` → `(list_id, book_id)`.

---

### Out of Scope (downstream scripts)

`embedder.py`, `ranker.py` — these reference `legacy_id` columns in the compatibility tables. They will need updating separately once the schema lands.

---

## Verification Plan

### Manual
1. Run `python main.py seed --list_ids=<your_list_id>` — confirm `reader_libraries` populates with `book_id` column.
2. Run `python main.py crawl` — confirm `books`, `authors`, `genres`, `series`, `book_series`, `awards`, `book_editions`, `similar_books` all populate correctly.
3. Spot-check a few rows: verify `kca_id` is populated, `star_1`–`star_5` sum matches expectations, `original_publication_time` differs from `publication_time` for classic works, `currently_reading_count` / `to_read_count` are non-zero for popular books.
4. Confirm no `works`, `contributors`, `book_contributors`, or `social_signals` tables exist.
