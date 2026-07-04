# Refactor Plan: Retire `my_library`, Unify Seeding via `reader_libraries`

## 1. Goal

Eliminate the CSV-export code path (`download_my_library`) entirely. The
main user's own ratings will be captured by the *same* review-list scraping
mechanism already used for friends (`process_list` / `reader_libraries`),
by including the main user's own list in the scrape scope. This removes an
entire parallel data pipeline (CSV download → pandas import → `my_library`
table) in favor of one unified mechanism.

## 2. Accepted tradeoff (confirmed)

The CSV export captured rich per-book personal metadata that the review-list
scrape does not: `isbn`, `isbn13`, `publisher`, `binding`, `exclusive_shelf`,
`my_review`, `private_notes`, `read_count`, `owned_copies`,
`bookshelves`/`bookshelves_with_positions`, `author`/`author_lf`/
`additional_authors`, `number_of_pages`, `year_published`,
`original_publication_year`. **None of this is used anywhere in the
pipeline** (verified against `ranker.py`, `seeder.py`, and the
`prep_crawl_heapq` snippet from `crawler.py`) — the only fields ever
consumed downstream are `book_id`, `rating`, and `title` (the last only for
interactive-ranking display, and `title` is available from the `books`
table instead). **Confirmed: dropping this data permanently is acceptable.**
No migration of `my_library` contents into `reader_libraries` will be
performed — the next `seed` run after upgrading simply re-scrapes the
self-list from Goodreads directly, exactly as it would for a fresh DB.

## 3. Key design decisions (all confirmed during interview)

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Drop `my_library` table and all CSV-export logic. No content migration. | Nothing downstream depends on the extra fields; Goodreads remains source of truth. |
| 2 | `get_lists_to_scrape` scrapes **both** self and friends — remove the `WHERE is_self = 0` filter. | Same underlying mechanism now serves both. |
| 3 | `get_similar_friend_ratings` **must** exclude `is_self = 1` rows via a join. | Without this, "you" would appear as a friend correlating perfectly with yourself, corrupting the friend-similarity model. This was an actual bug in the naive version of the refactor, not just a rename. |
| 4 | `run_ranking`'s book-loading step becomes a single `LEFT JOIN` between `books` and the `is_self=1` slice of `reader_libraries`, parameterized by a new `db.get_self_list_id()` helper. `gr_export` disappears entirely. | `reader_libraries` has PK `(list_id, book_id)`, so joining against one fixed `list_id` is a true one-to-one relationship — no fan-out risk. Titles for interactive ranking come from `books_df` instead of a separate table. |
| 5 | Add a **partial unique index** `idx_readers_single_self` on `readers(is_self) WHERE is_self = 1`, plus a `db.get_self_list_id(db_conn)` helper that raises `RuntimeError` if no self row exists. | Makes "at most one self reader" a hard DB invariant instead of an assumption; gives callers a clear, centralized way to get the self `list_id` instead of repeating a scalar subquery. A `NULL`-returning subquery would otherwise silently zero out all `my_rating` values instead of erroring. |
| 6 | `upsert_readers` **demotes** any existing `is_self=1` row (sets `is_self=0`) before inserting/replacing the new one, rather than deleting it. | Preserves the old self-user's `readers` row and all their `reader_libraries` history — they simply become an ordinary friend going forward. Order matters: demote first, then insert, so the partial unique index is never violated mid-operation. Handling of stale ELO history for a demoted self-user is explicitly out of scope (edge case, unlikely to occur). |
| 7 | `main.py`: collapse `seed(user=, friends=, force=, list_ids=)` into `seed(force=, list_ids=)`. `run_pipeline`: drop `seed_user`/`seed_friends` params outright (no deprecation/no-op shim). | Self-discovery already happens as a side effect of the existing login flow (`extract_main_user` runs whenever `list_ids is None`) — separate toggles no longer map to anything real. This is a personal tool; a loud CLI error on a removed flag is preferable to silently-ignored dead parameters. |
| 8 | `ensure_schema_compat` drops `my_library` (and the now-pointless `user_library` → `my_library` rename branch) and creates the new partial unique index, for both fresh and existing DBs. | Existing users' `is_self=1` `readers` row already exists (created by a prior `upsert_readers` run) but has `scrape_complete != 1`, since it never went through `process_list`. Once the `is_self=0` filter is removed from `get_lists_to_scrape`, it is automatically picked up and scraped on the very next `seed` run — no explicit backfill needed. |
| 9 | Full dead-code cleanup: remove `clean_isbn()`, `normalise_library_columns()`, and any imports that become unused as a result (verify each import's other usages before removing, don't assume). | These only ever existed to support `download_my_library`; leaving them in place would be exactly the kind of orphaned code this refactor is meant to remove. |
| 10 | `prep_crawl_heapq` in `crawler.py`: remove the separate `my_library` query; `reader_libraries` (which now includes self) is the sole seed-id source. | Provided directly by you — collapses two queries into one `SELECT DISTINCT book_id FROM reader_libraries` equivalent. |
| 11 | Explicitly out of scope for this pass: the Python-2-syntax bug on `ranker.py` line ~499 (`except ValueError, IndexError:`), and any broader pandas→SQL migration beyond the `books_df`/`gr_export` join. | Confirmed as separate, unrelated concerns not to bundle into this refactor. |

## 4. File-by-file change list

### `db.py`

- **`SCHEMA`**: remove the `my_library` table definition entirely. Add:
  ```sql
  CREATE UNIQUE INDEX IF NOT EXISTS idx_readers_single_self
  ON readers(is_self) WHERE is_self = 1;
  ```
- **`ensure_schema_compat`**:
  - Remove the `user_library` → `my_library` rename branch (pointless once
    `my_library` is being dropped in the same pass).
  - Add `DROP TABLE IF EXISTS my_library;` and `DROP TABLE IF EXISTS user_library;`
    (covers both current and very-old schema naming) — run before
    `executescript(SCHEMA)`.
- **Remove** `normalise_library_columns()` — no remaining caller.
- **Add** new helper:
  ```python
  def get_self_list_id(db_conn):
      row = db_conn.execute("SELECT list_id FROM readers WHERE is_self = 1").fetchone()
      if row is None:
          raise RuntimeError("No self reader found. Run seeding first.")
      return row["list_id"]
  ```

### `seeder.py`

- **Remove**: `download_my_library()`, `clean_isbn()`, and the CSV/pandas
  import block (`pandas`, `numpy`, `tempfile`) — verify each import has no
  other remaining use in the file before deleting the `import` line.
- **`get_lists_to_scrape`**: remove `is_self = 0` from both branches of the
  `WHERE` clause (the `force_all` branch and the default branch), so self
  and friends are scraped uniformly.
- **`upsert_readers`**: add the demote-then-insert logic:
  ```python
  def upsert_readers(db_conn, main_user: dict, friends: list[dict]):
      db_conn.execute(
          "UPDATE readers SET is_self = 0 WHERE is_self = 1 AND list_id != ?",
          (main_user["list_id"],),
      )
      db_conn.execute(
          """
          INSERT OR REPLACE INTO readers (list_id, user_id, username, is_self)
          VALUES (?, ?, ?, 1)
          """,
          (main_user["list_id"], main_user["user_id"], main_user["username"]),
      )
      db_conn.executemany(
          """
          INSERT OR IGNORE INTO readers (list_id, user_id, username, is_self)
          VALUES (?, ?, ?, 0)
          """,
          [(f["list_id"], f["user_id"], f["username"]) for f in friends],
      )
      db_conn.commit()
  ```
- **`scrape_reader_libraries`**: signature drops nothing structurally (it
  already takes `list_ids` and `force_all`), but it is now the sole entry
  point called by `main.py`'s `seed()` — no other change needed here beyond
  what `get_lists_to_scrape` already covers.

### `ranker.py`

- **`get_similar_friend_ratings`**: change the query at line ~315 to exclude
  self:
  ```sql
  SELECT rl.list_id, rl.book_id, rl.rating
  FROM reader_libraries rl
  JOIN readers r ON rl.list_id = r.list_id
  WHERE r.is_self = 0
  ```
- **`run_ranking`**: replace the two-step "load `books_df`, separately load
  `gr_export`, then merge" sequence with a single join:
  ```python
  self_list_id = db.get_self_list_id(db_conn)  # may raise RuntimeError

  cursor = db_conn.execute(
      """
      SELECT b.*, l.rating AS my_rating
      FROM books b
      LEFT JOIN reader_libraries l
          ON b.book_id = l.book_id AND l.list_id = ?
      ORDER BY b.book_id
      """,
      (self_list_id,),
  )
  books_df = pd.DataFrame([dict(r) for r in cursor.fetchall()])
  books_df["my_rating"] = pd.to_numeric(books_df["my_rating"], errors="coerce").replace(0, np.nan)
  ```
  Wrap the `get_self_list_id` call in a `try/except RuntimeError`, print a
  friendly message ("No self reader found. Run seed first."), and return
  early — consistent with the existing early-exit style used elsewhere in
  `run_ranking` (e.g. the "no book records" / "no library records" guards).
  The old "no library records found" guard is replaced by checking
  `books_df["my_rating"].notna().any()` after the join.
- **`refine_ratings`**: drop the `title_col="title"` parameter and the
  `gr_export`-based title lookup. Pass `books_df` itself in as `target_df`
  (it now carries both `my_rating` and `title` from the join). Titles dict
  for interactive mode should be built from `books_df` and use a fallback:
  ```python
  titles = dict(zip(books_df["book_id"], books_df["title"]))
  ```
  and in `run_interactive_ranking`, look up titles defensively:
  ```python
  title_a = titles.get(elo_df.at[index_a, "book_id"]) or f"Book {elo_df.at[index_a, 'book_id']}"
  ```
  (handles books present in the ELO table but not yet crawled.)
- All downstream references to `gr_export` and `my_rating_count` (line
  ~764) update to read from `books_df` directly instead of the now-removed
  `gr_export`.
- **No change** to the Python-2 syntax bug on line ~499 — explicitly out of
  scope.
- **No change** to the pandas-based merges inside `get_similar_friend_ratings`
  itself (overlap/calibration merges) — explicitly out of scope beyond the
  one join described above.

### `crawler.py`

- **`prep_crawl_heapq`**: remove the `my_library` query entirely; keep only
  the `reader_libraries` query for seed IDs (it now implicitly includes the
  self list once seeded):
  ```python
  cursor = db_conn.execute("SELECT book_id FROM reader_libraries WHERE book_id IS NOT NULL")
  seed_ids = {int(row["book_id"]) for row in cursor.fetchall()}
  ```
  Everything else in that function is unchanged.

### `main.py`

- **`seed`**: collapse to:
  ```python
  def seed(self, force=False, list_ids=None):
      from dotenv import load_dotenv
      from goodreads_ranker import seeder

      load_dotenv()
      db.init_db()

      parsed_ids = None
      if list_ids is not None:
          if isinstance(list_ids, str):
              parsed_ids = [int(x.strip()) for x in list_ids.split(",") if x.strip()]
          else:
              parsed_ids = [int(x) for x in list_ids]

      asyncio.run(seeder.scrape_reader_libraries(list_ids=parsed_ids, force_all=as_bool(force)))
  ```
  (`GOODREADS_EMAIL`/`GOODREADS_PASSWORD` env lookups move into
  `scrape_reader_libraries` itself if not already read there — confirm
  during implementation that `seeder.scrape_reader_libraries` already reads
  them via `os.getenv` internally, which it does per the current code, so no
  change needed there.)
- **`run_pipeline`**: drop `seed_user`/`seed_friends` params entirely:
  ```python
  def run_pipeline(self, seed=True, limit=None, force_recrawl=False, optimize=False, model=None):
      ...
      if seed:
          print("STEP 1: Seeding database")
          self.seed(force=False)
      else:
          print("STEP 1: Seeding skipped")
      ...
  ```

## 5. Migration / upgrade behavior for existing local DBs

No explicit backfill migration is written. On first run after upgrading:

1. `init_db()` → `ensure_schema_compat()` drops `my_library` (and
   `user_library` if present from very old schemas) and creates the new
   partial unique index.
2. The existing `is_self = 1` row in `readers` (created by a prior seed run)
   already exists but has `scrape_complete != 1` (it was never scraped via
   `process_list`).
3. On the next `seed` call, `get_lists_to_scrape` (no longer filtering on
   `is_self`) picks up the self row as needing a scrape, exactly like any
   incomplete friend list.
4. That one run will be slightly slower (one extra list — your own — walked
   via Playwright), then `reader_libraries` has your ratings and everything
   downstream (`ranker.py`, `crawler.py`) works unchanged.

No `elo_ratings` data is affected by any of this (keyed independently by
`book_id`).

## 6. Explicit non-goals for this pass

- No fix for the `except ValueError, IndexError:` Python 2 syntax bug in
  `build_adjacency_matrix` (`ranker.py` ~line 499).
- No further pandas → SQL migration beyond the `books_df`/`my_rating` join
  described above (the friend-overlap/calibration/aggregation logic in
  `get_similar_friend_ratings` stays as pandas — it operates on computed,
  in-memory state, not raw table data, and isn't a good SQL candidate).
- No handling of stale/orphaned ELO ratings if a demoted former self-user's
  old ratings become misleading as "friend" data — deferred as an unlikely
  edge case.
- No migration of `my_library`'s richer historical fields into any other
  table.

## 7. Suggested implementation/verification order

1. `db.py` — schema + `ensure_schema_compat` + `get_self_list_id` helper
   (foundational; everything else depends on it).
2. `seeder.py` — `get_lists_to_scrape`, `upsert_readers`, removal of
   `download_my_library`/`clean_isbn`.
3. `crawler.py` — `prep_crawl_heapq` simplification.
4. `ranker.py` — the `run_ranking` join, `get_similar_friend_ratings`
   self-exclusion, `refine_ratings` title handling.
5. `main.py` — `seed`/`run_pipeline` signature changes.
6. Sanity check: run `seed()` against a copy of an existing local DB and
   confirm (a) `my_library` is gone, (b) the self list gets scraped, (c)
   `reader_libraries` contains your own `list_id`'s rows, (d) `run_ranking`
   completes and `similar_friends` in its printed output does **not**
   include your own list_id.
