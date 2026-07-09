# Design & Refactoring Document: GraphQL Crawler and Database Engine

## 1. Design Philosophy & Core Principles

This document outlines the architecture for refactoring the `goodreads_ranker` crawler. This is a **full replacement**: the existing Playwright/Chromium browser-automation crawler (`crawler.py`) is deleted entirely and replaced with a lightweight, concurrent `httpx`-based crawler against the AWS AppSync GraphQL endpoint discovered during exploration (see `goodreads_api_findings.md`). There is no hybrid or fallback mode.

The architecture is guided by one overriding principle, applied at every decision point below: **only as complicated as absolutely necessary.** Where a proposed mechanism didn't have a concrete, present need, it was cut rather than spec'd "for later."

Two principles from the prior draft of this document were dropped outright:
- **"Pristine Raw Storage (Bronze Layer)"** — removed. No `raw_payloads` table. Nothing in this design reprocesses old API responses, so there's no current need to retain them raw.
- Everything else (unified canonical-resolution flow, database-backed queue state replacing in-memory structures) is retained and detailed below.

---

## 2. Data Model

### 2.1 Naming: `legacy_id`, not `book_id`

Every place the schema previously said `book_id` (a Goodreads legacy integer ID) is renamed to `legacy_id`, to keep it unambiguous that this is the Goodreads-assigned integer ID, not an internal surrogate key. This rename applies to `works`, `books`, and is expected to propagate into `reader_libraries`, `elo_ratings`, `embeddings`, and `predictions` — those four tables are otherwise **untouched** by this refactor and will be updated separately downstream (out of scope for this document).

### 2.2 Primary keys: legacy integer IDs, not opaque `kca://` IDs

`works.legacy_id` and `books.legacy_id` are the real primary keys. The opaque `kca://work/...` / `kca://book/...` IDs returned by the API are **not stored** — they're never used as a lookup key anywhere in the crawl flow (every fetch is by `legacyId`), and finding #3 in `goodreads_api_findings.md` already recommends not exposing `work.id`/`work.legacyId` anywhere it could be mistaken for a book reference. This keeps every downstream table (`reader_libraries`, `elo_ratings`, `embeddings`, `predictions`) joining directly on the same integer ID they already use today — no indirection, no schema change required on their end beyond the `book_id` → `legacy_id` rename.

### 2.3 Only canonical books are stored

Only the **canonical** book (i.e. `work.bestBook`) for a given work is ever written to the `books` table. There is no `is_canonical` column — every row in `books` is canonical by construction, so the flag would be redundant.

When a seeded/discovered `legacy_id` turns out to be a non-canonical edition, its edition-specific data (title, publisher, format, etc.) is simply **not stored**. Instead:
- `crawl_queue.status` for that non-canonical `legacy_id` is set to `'mapped_to_canonical'`. This status *is* the durable "we've handled this ID" record — it replaces what the old Playwright crawler achieved implicitly by writing a `books` row for every ID it touched.
- The canonical `bestBook` is fetched and fully processed as normal.
- Any future crawl-queue insert attempt for that same non-canonical `legacy_id` (e.g. rediscovered later as a "similar book") is a no-op against its existing `'mapped_to_canonical'` status.

Separately, `known_editions` (populated from the canonical book's `work.editions` connection) gives a join path from *any* sibling edition's `legacy_id` — canonical or not — back to its `work`. This is what lets `reader_libraries` rows that reference a non-canonical edition still resolve to the work's aggregate data, without needing a stored `books` row for that edition.

### 2.4 Normalization: genres, series, contributors stay as separate tables

`genres`, `series`, and `contributors` remain standalone tables with junction tables (`work_genres`, `work_series`, `book_contributors`), rather than collapsing to flat delimited columns. This is a deliberate choice to support future relational queries (e.g. "all works tagged Fantasy") even though nothing in the current `embedder.py`/`ranker.py` requires it today.

### 2.5 Full SQLite schema

```sql
-- =========================================================================
-- 1. PERSISTENT CRAWL QUEUE (replaces in-memory heapq)
-- =========================================================================

CREATE TABLE IF NOT EXISTS crawl_queue (
    legacy_id INTEGER PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'pending', -- 'pending' | 'done' | 'mapped_to_canonical' | 'skipped_known_edition' | 'error'
    priority REAL NOT NULL DEFAULT 0.0,     -- avg_rating - avg_rating / log10(ratings_count + 10). Pure function of rating data only — never adjusted for seed/expansion status.
    error_count INTEGER DEFAULT 0,
    last_error_message TEXT,
    discovered_via TEXT,                    -- 'seed' | 'similar'
    enqueued_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    processed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_crawl_queue_priority ON crawl_queue(status, discovered_via, priority DESC);

-- =========================================================================
-- 2. CORE ENTITIES
-- =========================================================================

CREATE TABLE IF NOT EXISTS works (
    legacy_id INTEGER PRIMARY KEY,
    original_title TEXT,
    publication_time INTEGER,
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
    text_reviews_language_counts TEXT,          -- serialized JSON
    editions_total_count INTEGER,
    editions_coverage_complete INTEGER DEFAULT 1, -- 0/1; false once editions_total_count > 200 (see §4)
    updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS books (
    legacy_id INTEGER PRIMARY KEY,               -- always the canonical (bestBook) edition; see §2.3
    work_id INTEGER REFERENCES works(legacy_id) ON DELETE SET NULL,
    title TEXT,
    title_complete TEXT,
    description TEXT,
    image_url TEXT,
    web_url TEXT,
    asin TEXT,
    isbn TEXT,
    isbn13 TEXT,
    format TEXT,
    num_pages INTEGER,
    publisher TEXT,
    publication_time INTEGER,
    language_name TEXT,
    fetched_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS contributors (
    legacy_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    web_url TEXT,
    is_gr_author INTEGER DEFAULT 0,
    works_count INTEGER DEFAULT 0,
    followers_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS series (
    id INTEGER PRIMARY KEY,   -- surrogate; Goodreads series don't reliably expose a stable legacy int id in probed fields
    title TEXT NOT NULL,
    web_url TEXT
);

CREATE TABLE IF NOT EXISTS genres (
    name TEXT PRIMARY KEY,
    web_url TEXT
);

-- =========================================================================
-- 3. JUNCTIONS
-- =========================================================================

CREATE TABLE IF NOT EXISTS work_series (
    work_id INTEGER REFERENCES works(legacy_id) ON DELETE CASCADE,
    series_id INTEGER REFERENCES series(id) ON DELETE CASCADE,
    user_position TEXT,
    PRIMARY KEY (work_id, series_id)
);

CREATE TABLE IF NOT EXISTS work_genres (
    work_id INTEGER REFERENCES works(legacy_id) ON DELETE CASCADE,
    genre_name TEXT REFERENCES genres(name) ON DELETE CASCADE,
    PRIMARY KEY (work_id, genre_name)
);

CREATE TABLE IF NOT EXISTS book_contributors (
    book_id INTEGER REFERENCES books(legacy_id) ON DELETE CASCADE,
    contributor_id INTEGER REFERENCES contributors(legacy_id) ON DELETE CASCADE,
    role TEXT,
    PRIMARY KEY (book_id, contributor_id, role)
);

-- =========================================================================
-- 4. AUXILIARY DATA
-- =========================================================================

CREATE TABLE IF NOT EXISTS work_awards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    work_id INTEGER REFERENCES works(legacy_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    category TEXT,
    designation TEXT,
    awarded_at INTEGER,
    web_url TEXT
);

CREATE TABLE IF NOT EXISTS known_editions (
    work_id INTEGER REFERENCES works(legacy_id) ON DELETE CASCADE,
    legacy_id INTEGER NOT NULL,
    title TEXT,
    discovered_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (work_id, legacy_id)
);

CREATE TABLE IF NOT EXISTS social_signals (
    work_id INTEGER REFERENCES works(legacy_id) ON DELETE CASCADE,
    signal_name TEXT NOT NULL,   -- CURRENTLY_READING | TO_READ
    count INTEGER NOT NULL,
    updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (work_id, signal_name)
);

CREATE TABLE IF NOT EXISTS similar_books (
    work_id INTEGER REFERENCES works(legacy_id) ON DELETE CASCADE,
    similar_legacy_id INTEGER NOT NULL,
    rank INTEGER,
    title TEXT,
    average_rating REAL,
    ratings_count INTEGER,
    fetched_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (work_id, similar_legacy_id)
);
```

### 2.6 Migration: none — clean break

This ships against a **fresh database**. No migration or backfill logic is written for the old flat `books` table. `reader_libraries` (which drives *what* gets seeded as a crawl target) is untouched, so nothing about *which* books need crawling is lost — only previously-fetched book detail, which is simply re-fetched under the new schema.

---

## 3. Canonical Resolution Flow

For a `legacy_id` popped from `crawl_queue`:

1. **Probe fetch.** Issue a lightweight GraphQL query requesting only `legacyId` and `work.bestBook.legacyId` — not the full `BOOK_QUERY`. This avoids paying for genres/contributors/editions payload on every seed, most of which is thrown away for non-canonical seeds.
   ```graphql
   query getBookByLegacyId($legacyBookId: Int!) {
     getBookByLegacyId(legacyId: $legacyBookId) {
       legacyId
       work { bestBook { legacyId } }
     }
   }
   ```
2. **Branch on canonicality:**
   - **`legacy_id == bestBook.legacyId`:** this is canonical. Proceed to the full fetch (step 3) directly on this ID.
   - **`legacy_id != bestBook.legacyId`:** this is a non-canonical edition. Set `crawl_queue.status = 'mapped_to_canonical'` for this `legacy_id` (no `books` row written for it — see §2.3), then recurse: run this same resolution flow on `bestBook.legacyId`. Because `bestBook` always points to itself, recursion depth is guaranteed to be 1.
3. **Full fetch on the canonical ID** — the full `BOOK_QUERY` (title, contributors, genres, series, `details`, first page of editions, work stats/awards).
4. **Concurrent auxiliary fetches** (all fanned out via `asyncio.gather`, unbounded per book — see §5): remaining editions pages (up to the 10-page / 200-edition cap — see §4), similar books, social signals.
5. **Write.** Upsert `works`, `books`, `contributors`/`genres`/`series` + junctions, `known_editions`, `similar_books`, `social_signals`, `work_awards`. Set `crawl_queue.status = 'done'` for the canonical `legacy_id`.
6. **Sibling pruning.** Cancel any still-`pending` `crawl_queue` rows whose `legacy_id` now appears in `known_editions` for this work:
   ```sql
   UPDATE crawl_queue
   SET status = 'skipped_known_edition', processed_at = datetime('now')
   WHERE legacy_id IN (SELECT legacy_id FROM known_editions WHERE work_id = :work_id)
   AND status = 'pending';
   ```
7. **Queue expansion.** Each discovered similar-book is inserted (or priority-maximized on conflict) into `crawl_queue` with `discovered_via = 'similar'` — see §4 for the exact upsert.

Only two GraphQL query shapes are needed per canonical resolution in the common case (probe + full fetch), matching finding #4's "1–2 requests" framing, plus the auxiliary fan-out.

---

## 4. Queue Mechanics & Priority

### 4.1 Priority formula — single, non-cycling

```
priority = average_rating - average_rating / log10(ratings_count + 10)
```

This is the only scoring function. The prior draft's dual "Rating" / "Count" algorithms and the cycling between them (present in the old `crawler.py` as `SCORING_FUNCTIONS`) are **dropped** — simpler, and nothing in the current design needs a second prioritization pass.

`priority` is a pure function of rating data. It is never adjusted with a hardcoded offset for seed vs. discovered books — see §4.3 for how seed-first ordering is achieved instead.

### 4.2 Queue expansion on conflict

```sql
INSERT INTO crawl_queue (legacy_id, priority, status, discovered_via)
VALUES (:similar_id, :calculated_score, 'pending', 'similar')
ON CONFLICT(legacy_id) DO UPDATE SET
    priority = CASE WHEN status = 'pending' THEN MAX(priority, excluded.priority) ELSE priority END
WHERE status = 'pending';
```

### 4.3 Seed-first ordering without magic numbers

`run_pipeline`'s default crawl behavior must be preserved exactly:
- **`limit=None` (default):** seeds only — no expansion.
- **`limit=0` or `limit=-1`:** expansion included, no cap, runs until `crawl_queue` is exhausted.
- **`limit=N>0`:** expansion included, capped so total scraped books (already-scraped + newly-scraped) reaches at most `N`. Within this cap, **seeds still get first claim** over expansion books — this is a deliberate behavior preserved from the old `(group, score)` heap tuple, not something that should be lost as a side effect of the queue redesign.

Rather than baking a hardcoded priority offset into the stored `priority` column (a magic number polluting an otherwise-pure rating-based value), seed-first ordering is expressed as a **sort-key tiebreaker** at query time:

```sql
SELECT legacy_id, priority, discovered_via
FROM crawl_queue
WHERE status = 'pending'
  AND discovered_via IN (:allowed_sources)   -- ('seed',) or ('seed', 'similar') depending on mode
ORDER BY (discovered_via = 'seed') DESC, priority DESC
LIMIT :n                                      -- omitted entirely for unlimited modes
```

This single query, varying only `:allowed_sources` and `:n`, covers all three modes:
- **seeds-only:** `allowed_sources = ('seed',)`, no `LIMIT`.
- **unlimited:** `allowed_sources = ('seed', 'similar')`, no `LIMIT`.
- **capped at N:** same `allowed_sources`, `LIMIT (N - already_scraped)`.

`priority` itself is never mutated for this purpose — it stays purely diagnostic/inspectable as "how good is this book," independent of why it's in the queue.

### 4.4 Editions cap — accepted limitation

Per finding #5, only the first 10 pages (200 editions) of a work's editions are ever retrievable via GraphQL, regardless of method; there is no fallback to HTML scraping of the SSR editions page. `works.editions_coverage_complete` is set to `0` when `editions_total_count > 200`, so it's visible in the data which works have incomplete sibling-edition coverage. Any sibling edition beyond the cap that gets crawled "cold" via a different discovery path (e.g. as a `reader_libraries` seed) is still correctly resolved to its work through the probe-fetch → `bestBook` flow in §3 — it's a minor crawl-efficiency cost for rare high-edition-count outliers, not a correctness issue.

---

## 5. Concurrency Model

`concurrency` (CLI flag, default `2`, unchanged from `main.py`) means **books in flight simultaneously** — this preserves the existing flag's meaning with no `main.py` changes required.

Each in-flight book's *internal* fan-out (remaining editions pages, similar books, social signals — all via `asyncio.gather`, per §3 step 4) is **unbounded** and not counted against `concurrency`. There is no empirical rate-limit data for the GraphQL endpoint at real concurrency (see §6), so rather than guess a safe proactive global request cap, concurrency is controlled at the coarser "books in flight" level and paired with reactive error handling.

The old `RESTART_THRESHOLD` / periodic-browser-restart cycle structure from `crawler.py` is **removed entirely**. It existed to work around Playwright-specific browser memory growth and stuck session/modal state — none of which applies to `httpx`. There is also no more in-memory heap to periodically rebuild from the DB: `crawl_queue` is the single source of truth, and newly-discovered similar-books are inserted directly into it (§4.2), immediately visible to the very next queue pull. The crawler is a single continuous loop: pull up to `concurrency` pending rows per §4.3's query, fan them out as concurrent tasks, write results as they complete, and keep pulling until `crawl_queue` has no eligible pending rows (or the budget from §4.3 is exhausted).

---

## 6. Error Handling

### 6.1 Confirmed: invalid/missing `legacyId` signature

A `legacyId` that doesn't correspond to a real book returns HTTP `200` with `data: null` and an `errors` array containing:

```json
{"path":null,"locations":[{"line":2,"column":41,"sourceName":null}],"message":"Variable 'legacyBookId' has an invalid value."}
```

This exact message (`errors[0].message == "Variable 'legacyBookId' has an invalid value."`) is treated as the sole "permanent, do not retry" signal — on match, `crawl_queue.status` is set to `'error'` immediately (no retries needed), analogous to the old `bad_book_ids` / 404 handling.

### 6.2 Everything else: uniform retry-with-backoff

No other GraphQL error signature has been confirmed (no rate-limiting was encountered during exploration). Until real rate-limit behavior is observed, **every other failure is treated identically**: increment `crawl_queue.error_count`, retry with backoff, and only transition to permanent `'error'` status after repeated failures (3, matching the old crawler's `MAX_RATE_LIMIT_RETRIES` convention) — rather than trying to pre-classify failures into rate-limit vs. transient-network vs. other categories we don't yet have evidence for.

### 6.3 Open items (explicitly deferred, not yet designed)

- **Rate-limit detection.** Unknown whether AppSync throttling (if/when encountered) surfaces as a transport-level HTTP `429`/`5xx`, or as a GraphQL-level `errors` entry (200 status), or both. No backoff strategy specific to rate-limiting exists yet beyond the generic retry in §6.2. Revisit once throttling is actually observed.
- **Message-match robustness.** The invalid-`legacyId` message match in §6.1 has only been confirmed for a clearly-out-of-range ID. It's unconfirmed whether the same exact message fires for other malformed inputs (e.g. `legacyId = -1` or `0`) that might arise from a bug rather than a genuinely missing book. Not currently a blocker; worth a quick check later if `error` classifications look suspicious in practice.

---

## 7. Staleness & Recrawl

Recrawling is gated on **both** conditions, matching current CLI semantics exactly (`main.py`'s `crawl(force_recrawl=False)` default is unchanged):

```sql
UPDATE crawl_queue
SET status = 'pending'
WHERE legacy_id IN (
    SELECT legacy_id FROM books WHERE datetime(fetched_at) < datetime('now', '-30 days')
)
AND status = 'done';
```

This update only runs when `force_recrawl=True` is explicitly passed — it is **not** automatic at startup. Books are otherwise left alone indefinitely once scraped, exactly as today. The 30-day window is unchanged from the existing `is_stale_scrape` default in `crawler.py`; nothing about this refactor changes how quickly the underlying Goodreads aggregates move, so there's no basis for tightening it.

---

## 8. Secrets

The AppSync `x-api-key` moves out of source (it was a hardcoded constant in the `test_httpx.py` exploration script) and into `.env`, read via `os.environ`, matching the existing `dotenv.load_dotenv()` pattern already used for `seed` in `main.py`:

```
X_API_KEY=da2-xpgsdydkbregjhpr6ejzqdhuwy
```

This is a static key discovered via inspecting Goodreads' own frontend traffic, not an issued credential — moving it to `.env` means a rotation only requires an environment change, not a code change, and keeps it out of git history.

---

## 9. CLI Surface (`main.py`) — unchanged

`crawl(limit=None, concurrency=2, force_recrawl=False)` keeps its existing signature and semantics exactly as implemented today:
- `limit=None` → seeds only, no cap.
- `limit=0` or `limit=-1` → expansion included, uncapped.
- `limit=N>0` → expansion included, capped at `N` total scraped books, seeds still prioritized first (§4.3).
- `force_recrawl` → gates staleness requeue (§7); does nothing on its own without the 30-day-stale condition also being true.

`run_pipeline`'s default behavior (seed → crawl seeds-only → embed → rank) is unchanged by this refactor.
