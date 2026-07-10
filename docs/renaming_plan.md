# Rename book_id to legacy_id Implementation Plan

Rename all database column and code mentions of `book_id` to `legacy_id` for consistency and clarity across the codebase.

## User Review Required

> [!WARNING]
> This is a breaking change for the database schema. All occurrences of `book_id` inside tables (`reader_libraries`, `elo_ratings`, `embeddings`, `predictions`, `book_contributors`, `known_editions`, etc.) will be renamed to `legacy_id`. We will be starting with a fresh database.

## Proposed Changes

### 1. Database Component

#### [MODIFY] [db.py](file:///Users/alex/Documents/goodreads-ranker/goodreads_ranker/db.py)
- Rename `book_id` column to `legacy_id` in all schema definitions:
  - `reader_libraries`
  - `elo_ratings`
  - `embeddings`
  - `predictions`
  - `book_contributors`
- Rename parameters and variables in helper functions (e.g. `save_embeddings` parameters `book_ids` -> `legacy_ids`).

---

### 2. Crawler Component

#### [MODIFY] [crawler.py](file:///Users/alex/Documents/goodreads-ranker/goodreads_ranker/crawler.py)
- Update any SQL insertions or queries referencing `book_id` (like `reader_libraries.book_id`) to `legacy_id`.
- Update the table definition names inside `crawler.py` (e.g., column names when referencing junctions).

---

### 3. Seeder Component

#### [MODIFY] [seeder.py](file:///Users/alex/Documents/goodreads-ranker/goodreads_ranker/seeder.py)
- Update SQL insert/select statements on `reader_libraries` to use `legacy_id` instead of `book_id`.
- Update internal variables and parsing dictionaries referencing `book_id` to `legacy_id`.

---

### 4. Embedder Component

#### [MODIFY] [embedder.py](file:///Users/alex/Documents/goodreads-ranker/goodreads_ranker/embedder.py)
- Update all selects and left joins referencing `book_id` to `legacy_id`.
- Rename variables `book_id` -> `legacy_id`, `book_ids` -> `legacy_ids`.

---

### 5. Ranker Component

#### [MODIFY] [ranker.py](file:///Users/alex/Documents/goodreads-ranker/goodreads_ranker/ranker.py)
- Update all dataframes (`books_df`, `elo_df`, `friends_df`, `my_books`) indexing or column selection on `book_id` to `legacy_id`.
- Update all SQLite queries joining tables on `book_id` to join on `legacy_id`.

## Verification Plan

### Automated Tests
- Run `PYTHONPATH=. python3 /Users/alex/.gemini/antigravity/brain/c3522a35-082b-4a05-87bc-76953ef0e4ad/scratch/test_crawl.py` to verify that the crawler works correctly under the fully updated schema.
- Run `python3 main.py crawl --limit 5` (which runs the main pipeline crawler) to confirm no execution issues.
