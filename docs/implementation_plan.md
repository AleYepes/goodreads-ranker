# Goodreads Ranker SQLite Refactor and Decoupling Plan

This plan details the migration of the Goodreads book crawler, embedding generator, and ranker pipeline from CSV files to a SQLite database. It also outlines the structural split of the codebase into lean, decoupled scripts orchestrated by a central CLI.

## User Review Required

> [!IMPORTANT]
> **Memory Optimization for Embeddings**
> To avoid out-of-memory crashes on your machine when handling 4096-dimensional embeddings for 50,000+ books, the refactored code will:
> - Store vectors as binary float32 arrays (`BLOB`) in SQLite using `np.ndarray.tobytes()`.
> - Load embeddings directly from SQLite into pre-allocated NumPy arrays using `np.frombuffer()`, completely bypassing Pandas and any high-overhead Python object allocations.
> - This keeps the memory footprint under ~850MB for 50,000 books during the ML/ranking step.

> [!NOTE]
> **Data Migration & Backwards Compatibility**
> - The new schema will keep the exact columns from the original CSV files.
> - We will provide a migration helper (a one-off command or run step) to load all your existing CSV data (`books.csv`, `friend_ratings.csv`, `elo_ratings.csv`, etc.) into the SQLite database.
> - The `recommendations` output will be stored in a SQLite table `predictions`. We will also add a simple flag to export it to `data/recommendations.csv` if you still want to view it directly in a spreadsheet.

## Proposed Changes

We will introduce a shared database module `db.py`, partition the codebase into five distinct scripts, and create a CLI wrapper.

### Database Module

#### [NEW] [db.py](file:///Users/alex/Documents/goodreads-ranker/db.py)
A lightweight helper script containing table definitions, SQLite initialization, and memory-efficient helpers for bulk reads/writes (especially for embeddings BLOB serialization).

- **Tables to Create:**
  1. `user_library`: User's exported books.
  2. `friend_lists`: Friend lists tracker.
  3. `friend_ratings`: Ratings scraped from friends.
  4. `books`: Detailed book metadata scraped from Goodreads.
  5. `elo_ratings`: Interactive ELO comparison rankings.
  6. `embeddings`: Book IDs mapped to float32 binary vector BLOBs.
  7. `predictions`: Resulting ratings computed by the ensemble model.

---

### Seeding Logic

#### [NEW] [seeder.py](file:///Users/alex/Documents/goodreads-ranker/seeder.py)
Extracts the seeding logic from `crawler.py` and sync logic from `crawler_lists.py` into a single script.
- **Functions:**
  - `download_user_library`: Playwright automation to download the user's library CSV, parse it, and upsert it into the `user_library` table.
  - `scrape_friend_ratings`: Reads untracked or stale list IDs from `friend_lists`, scrapes reviews via Playwright, and upserts them into `friend_ratings` and `friend_lists`.
  - Supports loading seeds from a seed configuration (e.g. your list of friend list IDs).

---

### Book Crawler

#### [MODIFY] [crawler.py](file:///Users/alex/Documents/goodreads-ranker/crawler.py)
Focuses strictly on detailed book crawling and metadata collection.
- Reads seed book IDs directly from `user_library` and `friend_ratings`.
- Constructs the crawl queue by reading already scraped books from the `books` table, and parsing the `similar_books` fields.
- Scrapes book details using async Playwright and writes them straight to the SQLite `books` table (upserting to avoid duplicates).
- Accepts a `--limit` parameter to cap the number of books processed in a single run (crucial for pipeline orchestration).

---

### Embedding Generator

#### [NEW] [embedder.py](file:///Users/alex/Documents/goodreads-ranker/embedder.py)
Extracts the embedding logic from `analysis.ipynb` into a dedicated script.
- Queries `books` table to find book descriptions/metadata.
- Compares against `embeddings` table to identify books missing embeddings.
- Calls Ollama (`qwen3-embedding:8b` or as configured in env) in batches (e.g., 128).
- Saves vectors into `embeddings` table as BLOBs (`np.ndarray.tobytes()`).

---

### Machine Learning & Ranking

#### [NEW] [ranker.py](file:///Users/alex/Documents/goodreads-ranker/ranker.py)
Extracts the models and analysis logic from `analysis.ipynb` into a clean Python script.
- **Process Steps:**
  1. **ELO Ratings:** Runs the interactive ELO prompt if selected, and updates the `elo_ratings` table.
  2. **Friend taste calibration:** Matches overlaps, calculates slopes/intercepts, and computes weighted friend ratings.
  3. **Graph propagation:** Loads graph edges from `similar_books` in `books` table and propagates embeddings.
  4. **Ensemble model & optimization:** Tunes parameters using Nevergrad (or loads saved parameters) and runs Bayesian Ridge, SVR, and KNN regressors.
  5. **Prediction upsert:** Computes final predicted scores and saves all prediction outputs to the SQLite `predictions` table.

---

### Orchestration CLI

#### [NEW] [cli.py](file:///Users/alex/Documents/goodreads-ranker/cli.py)
A CLI tool using `google-fire` to orchestrate pipeline stages.
- **Commands:**
  - `seed [--user] [--friends] [--force]`: Pull user library and/or friend review lists.
  - `crawl [--limit L] [--concurrency C]`: Run the book metadata crawler.
  - `embed [--batch-size B]`: Generate embeddings for newly crawled books.
  - `rank [--optimize] [--interactive]`: Calibrate ratings, run GCN, ensemble prediction, and save predictions.
  - `run-pipeline [--crawl-limit L]`: Sequentially runs seed, crawl (with a hard cap), embed, and rank.
  - `migrate-csv`: Helper command to migrate your existing CSVs (`books.csv`, `friend_ratings.csv`, etc.) into SQLite so you do not lose any scraped data.

---

## Verification Plan

### Automated Verification
- We will write a small migration run to verify that all CSV files are loaded correctly into SQLite without data loss.
- Run dry-runs of the CLI command segments:
  - `python cli.py migrate-csv`
  - `python cli.py embed --batch-size 10`
  - `python cli.py rank --interactive=False`

### Manual Verification
- Verify that `goodreads.db` contains all tables and is populated.
- Compare predictions from the SQLite `predictions` table with the original `data/recommendations.csv` for consistency.
- Test ELO interactive prompt via CLI to ensure keyboard inputs write to database.
