# Design & Refactoring Document: GraphQL Crawler and Database Engine

## 1. Design Philosophy & Core Principles

This document outlines the architecture for refactoring the `goodreads_ranker` crawler. The project is transitioning from a high-overhead browser-automation script (Playwright/Chromium) to a lightweight, highly concurrent API crawler utilizing the AWS AppSync GraphQL endpoint.

The architecture is guided by three principles:
1. **Pristine Raw Storage (Bronze Layer):** Raw API payloads are stored in their native, unaltered form using an upsert pattern. No synthetic client-side JSON merges are performed.
2. **Unified Parsing and Execution Flow:** Code duplication is avoided by routing both canonical and non-canonical editions through the same processing logic.
3. **Database-Backed State Management:** Complex in-memory queues, custom date parsers, and custom scheduling threads are removed from the Python runtime and managed via clean, highly indexed relational database queries.

---

## 2. Refactored Architecture & Pipeline Flow

The execution model replaces the resource-heavy Playwright loop with a non-blocking asynchronous orchestrator. Below is the step-by-step pipeline for resolving a book ID popped from the queue.

### Step 1: GraphQL Fetch & Raw Payload Upsert
When an ID is popped, a single parameterized GraphQL query (`getBookByLegacyId`) is executed. The raw, unaltered JSON payload is immediately saved to the `raw_payloads` table.
* **On Conflict:** If the payload is already present, it is overwritten with the latest snapshot. This serves as an immutable cache of the most recent network query.

### Step 2: Canonical Resolution & Unified Recursion
The GraphQL response is analyzed to find `work.bestBook.legacyId`. 
* **If `seed_id == best_book_id`:** This is the canonical edition of the work. The system extracts and inserts the core work statistics, the canonical book attributes, and initiates the concurrent auxiliary crawls (sibling editions, similar books, and social signals).
* **If `seed_id != best_book_id`:** This is a non-canonical edition. The system writes the edition-specific book attributes (title, publisher, language, format) to the database with `is_canonical = 0`, marks the seed ID as `mapped_to_canonical` in the crawl queue, and recursively invokes the exact same processing function for the `best_book_id`. 
* *Note:* Because the canonical book’s `bestBook` points back to itself, the maximum recursion depth is guaranteed to be **1**. No duplicate parsing blocks are written.

### Step 3: Cascade-Pruning & Sibling Deduplication
Once the canonical book is resolved, the system fetches up to 10 pages (200 sibling editions) from the GraphQL resolver and writes them to the `known_editions` table. 
* To prevent the crawler from ever wasting API calls on already-discovered paperback/hardcover siblings, a post-crawl SQL update runs inside the same transaction, immediately cancelling any matching pending IDs in the `crawl_queue` table.

```sql
UPDATE crawl_queue
SET status = 'skipped_known_edition', processed_at = datetime('now')
WHERE legacy_book_id IN (
    SELECT legacy_id FROM known_editions WHERE work_id = :work_id
)
AND status = 'pending';
```

---

## 3. Queue Mechanics & Priority Calculations

The in-memory Python `heapq` is completely replaced by a persistent, indexed SQLite database table called `crawl_queue`. 

### Priority Score Calculation
To maintain search quality, we use your continuous Bayesian regularization approximation to score and order discovered books:

$$\text{Priority} = \text{average\_rating} - \frac{\text{average\_rating}}{\log_{10}(\text{ratings\_count} + 10)}$$

This continuous function naturally penalizes low-data outliers (bringing low-count items down to `0.0`) without requiring arbitrary thresholds or global database averages.

### Queue Expansion on Conflict
When similar books are returned from the API, they are scored in Python using the formula above and written to the database in real-time. To ensure a book discovered from multiple paths always uses its best priority score, SQLite handles priority maximization:

```sql
INSERT INTO crawl_queue (legacy_book_id, priority, status, discovered_via)
VALUES (:similar_id, :calculated_score, 'pending', 'similar')
ON CONFLICT(legacy_book_id) DO UPDATE SET
    priority = CASE WHEN status = 'pending' THEN MAX(priority, excluded.priority) ELSE priority END
WHERE status = 'pending';
```

### Staleness & Error Recovery
* **Staleness (`force_recrawl`):** Instead of looping, reading, and parsing date strings in Python, stale crawls are re-queued instantly at startup via a single SQLite command:
  ```sql
  UPDATE crawl_queue
  SET status = 'pending'
  WHERE legacy_book_id IN (
      SELECT legacy_id FROM books WHERE datetime(fetched_at) < datetime('now', '-30 days')
  )
  AND status = 'done';
  ```
* **Persistent Error States:** Temporary errors or persistent blocks increments an `error_count` column. If a book hits a hard 404 or fails 3 times, its status is permanently set to `error`. This ensures the crawler skips bad IDs upon restart without maintaining complex in-memory tracking structures.

---

## 4. Entity Modeling & SQLite Schema Design

The schema has been rebuilt to ensure compliance with SQLite's characteristics, eliminating circular references and minimizing redundant indices.

```
       +-------------------+
       |    crawl_queue    |
       +-------------------+
                 |
                 v
       +-------------------+
       |       works       | <---------------+
       +-------------------+                 |
         |               |                   |
         | (1:many)      | (1:many)          | (1:many)
         v               v                   |
  +------------+   +--------------+   +--------------+
  |   books    |   | work_genres  |   | work_series  |
  +------------+   +--------------+   +--------------+
    | (is_canonical = 1 / 0)
    |
    | (1:many)
    v
+-------------------+
| book_contributors |
+-------------------+
```

### 1. Eliminating Circular Dependencies
The `works` table no longer references `books.legacy_id` via a circular foreign key. Instead, the hierarchy is strictly linear: `books` references `works` via `work_id`. We identify which edition is the canonical edition via an `is_canonical` flag on the `books` table, enforced by a SQLite partial unique index:
```sql
CREATE UNIQUE INDEX uq_books_one_canonical_per_work
    ON books(work_id) WHERE is_canonical = 1;
```

### 2. Junction Table Normalization (Work vs. Book Level)
To prevent extreme data duplication across thousands of matching paperback/hardback editions, the schema categorizes junction tables based on their true logical scope:
* **`work_genres` and `work_series`:** Since genres and series placement never change between editions, they are mapped to the `work_id`. Any paperback can instantly find its genres and series by checking its parent work.
* **`book_contributors`:** Contributors can vary by edition (e.g., translators and audiobook narrators). We keep this table mapped to `book_id`, populating it for both canonical and non-canonical book runs to preserve these unique roles.

### 3. Native Type Mapping for SQLite
* PostgreSQL array structures like `ratings_count_dist INT[]` are mapped to explicit integer columns (`star_1` through `star_5`) on the `works` table. This allows models to run calculations without relying on expensive JSON extraction scripts.
* SQLite lacks native `BOOLEAN` and `TIMESTAMPTZ` data types. We standardize on `INTEGER` (using `0` and `1` values) and ISO-8601 text timestamps (`TEXT`) generated natively via SQLite's database-level datetime triggers.

### 4. Non-Blocking Event-Loop Execution
To maintain massive concurrent network throughput inside the asyncio loop, standard synchronous SQLite operations in your helper files are offloaded to background worker threads using Python's native standard library:
```python
await asyncio.to_thread(db.upsert_rows, db_conn, "books", ...)
```

---

## 5. Unified SQLite Schema Implementation

Below is the SQLite compatible schema, designed to replace your existing schema layout [schema.sql] [goodreads_ranker/db.py].

```sql
-- SQLite Schema Compatibility File
-- Database Target: data/goodreads.db
-- Always execute "PRAGMA foreign_keys = ON;" when initializing SQLite connections.

-- =========================================================================
-- 0. BRONZE LAYER (RAW PAYLOAD LANDING ZONE)
-- =========================================================================

CREATE TABLE IF NOT EXISTS raw_payloads (
    legacy_book_id INTEGER NOT NULL,
    operation_name TEXT NOT NULL,
    variables_json TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    scraped_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (legacy_book_id, operation_name, variables_json)
);

-- =========================================================================
-- 1. PERSISTENT CRAWL QUEUE
-- =========================================================================

CREATE TABLE IF NOT EXISTS crawl_queue (
    legacy_book_id INTEGER PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'pending', -- 'pending' | 'done' | 'mapped_to_canonical' | 'skipped_known_edition' | 'error'
    priority REAL NOT NULL DEFAULT 0.0,     -- Computed: avg - avg/log10(count + 10)
    error_count INTEGER DEFAULT 0,
    last_error_message TEXT,
    discovered_via TEXT,                   -- 'seed' | 'similar'
    enqueued_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    processed_at TEXT
);

-- Index prioritized pending items for high-concurrency O(1) pulls
CREATE INDEX IF NOT EXISTS idx_crawl_queue_priority ON crawl_queue(status, priority DESC);

-- =========================================================================
-- 2. CORE SYSTEM ENTITIES
-- =========================================================================

CREATE TABLE IF NOT EXISTS works (
    id TEXT PRIMARY KEY,                       -- kca://work/...
    legacy_id INTEGER UNIQUE,
    original_title TEXT,
    publication_time INTEGER,                  -- Unix millisecond timestamp
    web_url TEXT,
    shelves_url TEXT,
    average_rating REAL,
    ratings_count INTEGER,
    star_1 INTEGER DEFAULT 0,
    star_2 INTEGER DEFAULT 0,
    star_3 INTEGER DEFAULT 0,
    star_4 INTEGER DEFAULT 0,
    star_5 INTEGER DEFAULT 0,
    text_reviews_count INTEGER,
    text_reviews_language_counts TEXT,         -- Raw serialized JSON array
    editions_total_count INTEGER,
    editions_coverage_complete INTEGER DEFAULT 1, -- Boolean flag (0 or 1)
    updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS books (
    id TEXT PRIMARY KEY,                       -- kca://book/...
    legacy_id INTEGER UNIQUE NOT NULL,
    work_id TEXT REFERENCES works(id) ON DELETE SET NULL,
    is_canonical INTEGER NOT NULL DEFAULT 0,   -- Boolean flag (0 or 1)
    title TEXT,
    title_complete TEXT,
    description TEXT,
    description_stripped TEXT,
    image_url TEXT,
    web_url TEXT,
    asin TEXT,
    isbn TEXT,
    isbn13 TEXT,
    format TEXT,
    num_pages INTEGER,
    publisher TEXT,
    publication_time INTEGER,                  -- Unix millisecond timestamp
    language_name TEXT,
    fetched_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Enforce exactly one canonical book per work
CREATE UNIQUE INDEX IF NOT EXISTS uq_books_one_canonical_per_work
    ON books(work_id) WHERE is_canonical = 1;

CREATE TABLE IF NOT EXISTS contributors (
    id TEXT PRIMARY KEY,                       -- kca://author/...
    legacy_id INTEGER UNIQUE,
    name TEXT NOT NULL,
    description TEXT,
    profile_image_url TEXT,
    web_url TEXT,
    is_gr_author INTEGER DEFAULT 0,            -- Boolean flag (0 or 1)
    works_count INTEGER DEFAULT 0,
    followers_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS series (
    id TEXT PRIMARY KEY,                       -- kca://series/...
    title TEXT NOT NULL,
    web_url TEXT
);

CREATE TABLE IF NOT EXISTS genres (
    name TEXT PRIMARY KEY,
    web_url TEXT
);

-- =========================================================================
-- 3. INTERMEDIATE JUNCTION RELATIONSHIPS
-- =========================================================================

-- Normalized work-level links (identical across sibling editions)
CREATE TABLE IF NOT EXISTS work_series (
    work_id TEXT REFERENCES works(id) ON DELETE CASCADE,
    series_id TEXT REFERENCES series(id) ON DELETE CASCADE,
    user_position TEXT,
    PRIMARY KEY (work_id, series_id)
);

CREATE TABLE IF NOT EXISTS work_genres (
    work_id TEXT REFERENCES works(id) ON DELETE CASCADE,
    genre_name TEXT REFERENCES genres(name) ON DELETE CASCADE,
    PRIMARY KEY (work_id, genre_name)
);

-- Book-level links to catch translator or narrator modifications
CREATE TABLE IF NOT EXISTS book_contributors (
    book_id TEXT REFERENCES books(id) ON DELETE CASCADE,
    contributor_id TEXT REFERENCES contributors(id) ON DELETE CASCADE,
    role TEXT,
    PRIMARY KEY (book_id, contributor_id, role)
);

-- =========================================================================
-- 4. AUXILIARY STRUCTS & ANALYTICAL ARRAYS
-- =========================================================================

CREATE TABLE IF NOT EXISTS work_awards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    work_id TEXT REFERENCES works(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    category TEXT,
    designation TEXT,
    awarded_at INTEGER,                        -- Unix millisecond timestamp
    web_url TEXT
);

CREATE TABLE IF NOT EXISTS known_editions (
    work_id TEXT REFERENCES works(id) ON DELETE CASCADE,
    legacy_id INTEGER NOT NULL,
    title TEXT,
    discovered_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (work_id, legacy_id)
);

CREATE TABLE IF NOT EXISTS social_signals (
    work_id TEXT REFERENCES works(id) ON DELETE CASCADE,
    signal_name TEXT NOT NULL,                 -- CURRENTLY_READING | TO_READ
    count INTEGER NOT NULL,
    updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (work_id, signal_name)
);

CREATE TABLE IF NOT EXISTS similar_books (
    work_id TEXT REFERENCES works(id) ON DELETE CASCADE,
    similar_legacy_id INTEGER NOT NULL,
    rank INTEGER,
    title TEXT,
    average_rating REAL,
    ratings_count INTEGER,
    fetched_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (work_id, similar_legacy_id)
);
```