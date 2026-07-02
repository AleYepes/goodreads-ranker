# Goodreads Ranker Stabilization Plan

This plan defines the cleanup work required before adding new features. The goal is to make the SQLite refactor reliable, faithful to the intended pipeline, and usable through `run_pipeline` as the main entrypoint.

This is a stabilization pass only. Do not add recommendation CSV export, notebook viewing tools, scraping-time analytics, a migration framework, or new recommendation features in this pass.

## Target Workflow

`run_pipeline` is the primary entrypoint.

Default behavior:

1. Seed the user's Goodreads library.
2. Seed friend ratings.
3. Crawl remaining unscraped seed book IDs.
4. Embed scraped books that need embeddings.
5. Rank books using stored/default hyperparameters.
6. Verify pipeline state.

The pipeline should support a fresh database populated from scratch. Existing CSV migration is not part of the acceptance path.

## Schema Changes

Add only the fields needed for correctness.

### `books`

Add:

- `date_last_scraped TEXT`
- `verified_embedding INTEGER DEFAULT 0`

Rules:

- `date_last_scraped` is updated after a successful book scrape.
- `verified_embedding = 1` means the current embedding row matches the current embedding input text for that book.
- `verified_embedding = 0` means the book needs embedding generation or regeneration.

### `friend_lists`

Add:

- `scrape_error TEXT`

Rules:

- Successful scrape clears `scrape_error`.
- Failed scrape records the error and must not mark the list complete.

### `model_params`

Add a new table:

```sql
CREATE TABLE IF NOT EXISTS model_params (
    name TEXT PRIMARY KEY,
    params_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

Rules:

- Store `friend_params` and `solo_params` as JSON.
- Default ranking loads from this table.
- If no stored params exist, use the current hardcoded defaults and persist them.
- Nevergrad optimization updates these rows.

## Seeding

`seeder.py` owns user-library and friend-rating seeding.

### User Library

Keep the current user-library download/import behavior, but make sure it works as a stage called by `run_pipeline`.

### Friend Ratings

Port the robust behavior from `crawler_lists.py` into `seeder.py`, writing to SQLite instead of CSV.

Required behavior:

- Use a single browser session/page by default.
- Handle Goodreads login/session recovery for private lists.
- Open each list URL with the expected sort/order parameters.
- Parse rows into `friend_ratings`.
- Upsert rows by `(list_id, book_id)`.
- Stop early only after confirming the current page is valid and unchanged.
- Mark a list complete only after at least one valid page was parsed.
- If navigation, login, selector lookup, or parsing fails before a valid page is parsed, keep `scrape_complete = 0` and write `scrape_error`.
- If a list page is valid and rows are parsed, set `scrape_complete = 1`, clear `scrape_error`, and update `date_last_scraped`.

Do not optimize friend scraping concurrency in this pass.

## Crawling

`crawler.py` owns Goodreads book metadata crawling.

Seed IDs are all book IDs from:

- `user_library`
- `friend_ratings`

The crawler must prioritize seed IDs before non-seed expansion IDs.

### Limit Semantics

- `crawl_limit = None`: crawl remaining unscraped seed IDs only.
- `crawl_limit = N > 0`: crawl up to `N` books, seed IDs first. If remaining seed count is smaller than `N`, continue into non-seed expansion books until the limit is reached.
- `crawl_limit <= 0`: crawl indefinitely, including non-seed expansion books.

### Default Pipeline Crawl

By default, `run_pipeline` crawls remaining unscraped seed IDs only. It must not crawl non-seed similar-book expansion IDs unless the supplied limit semantics allow it.

### Forced Recrawl

Add `force_recrawl`.

Rules:

- Forced recrawl includes already-scraped books only when `date_last_scraped` is older than one month.
- Books scraped less than one month ago are skipped.
- Rescraping updates the existing `books` row by `book_id`.
- Rescraping must not create duplicate rows.
- After successful scrape, update `date_last_scraped`.

### Embedding Verification During Recrawl

Before updating a rescraped book row, compare the old embedding input components with the new ones:

- `title`
- `authors`
- `genres`
- `description`

Rules:

- If any component changed, set `verified_embedding = 0`.
- If all components are unchanged, set `verified_embedding = 1`.
- For newly scraped books, set `verified_embedding = 0`.

The crawler should not call Ollama or generate embeddings.

## Embeddings

`embedder.py` owns embedding generation.

The embedder should avoid loading the Ollama model whenever no work is required.

Queue a book for embedding when any of these are true:

- no embedding row exists,
- embedding vector byte length does not match `dim * 4`,
- embedding vector is all zeros,
- `books.verified_embedding = 0`.

After a successful embedding write:

- save the embedding BLOB,
- set `books.verified_embedding = 1`.

If there is no queued work, print a clear "nothing to embed" message and exit before calling Ollama.

## Ranking

`ranker.py` owns ELO refinement, friend calibration, graph propagation, regression, optional optimization, and prediction snapshot writing.

### Hyperparameters

Default ranking:

- skip Nevergrad,
- load `friend_params` and `solo_params` from `model_params`,
- if params are missing, use the current hardcoded defaults and persist them.

Optimization mode:

- run Nevergrad,
- update `model_params`,
- rank with the newly stored best params.

Fix the current optimization bug by ensuring optimization uses the same embedded/scored subset as modeling. Do not pass full `books_df` masks against embedding arrays built from `embedded_books_df`.

### Valid Model Inputs

Use only books with valid embeddings for modeling and prediction.

Invalid embeddings are:

- missing row,
- wrong byte length,
- all-zero vector,
- `verified_embedding = 0`.

Report excluded counts clearly.

### Predictions Table

`predictions` is a fresh snapshot table.

Rules:

- Clear `predictions` at the start of a successful prediction write.
- Insert only books with non-null `solo_pred_rating`, `friend_pred_rating`, `pred_rating`, and `final_rating`.
- Do not insert rows for books that cannot be scored.
- Do not preserve stale prediction rows.
- Do not export predictions to CSV.

## CLI

Keep individual stage commands, but make `run_pipeline` the main workflow command.

### `run_pipeline`

Arguments:

- `seed=True`
- `seed_user=True`
- `seed_friends=True`
- `crawl_limit=None`
- `force_recrawl=False`
- `optimize=False`

Behavior:

- `seed=False` skips both user and friend seeding.
- `seed_user=False` skips only user-library seeding.
- `seed_friends=False` skips only friend-rating seeding.
- `crawl_limit=None` crawls remaining seed IDs only.
- `crawl_limit=N > 0` crawls up to `N` queued books, seed-first.
- `crawl_limit <= 0` crawls indefinitely, including expansion books.
- `force_recrawl=True` enables stale-book recrawling with the one-month rule.
- `optimize=True` runs Nevergrad and updates stored params.

Default `run_pipeline` sequence:

1. initialize database,
2. seed user library if enabled,
3. seed friend ratings if enabled,
4. crawl according to seed-first limit semantics,
5. embed any scraped books needing embeddings,
6. rank with stored/default params unless optimization is enabled,
7. verify state.

### `verify`

Add a read-only verification command.

It should report:

- row counts for core tables,
- friend lists incomplete or with `scrape_error`,
- seed books missing from `books`,
- scraped books missing embeddings,
- invalid embeddings,
- books with `verified_embedding = 0`,
- prediction row count,
- null prediction-field count,
- unread scored count.

It must not show top recommendations, write CSVs, scrape, embed, crawl, or rank.

### `migrate_csv`

Leave the existing helper alone unless it interferes with the cleaned pipeline.

It is out of scope for this stabilization pass and is not part of the acceptance path.

## Out of Scope

Do not include the following in this pass:

- recommendation CSV export,
- `view.ipynb`,
- recommendation display commands,
- scraping-time analytics,
- CSV migration hardening,
- migration framework,
- new ranking features,
- UI or reporting layer.

## Acceptance Criteria

The cleanup is complete when:

- `run_pipeline` is the documented main workflow.
- A fresh database can be initialized and populated through the pipeline.
- Friend scrape failures do not mark lists complete.
- Seed books are crawled before expansion books.
- Default pipeline crawling does not crawl expansion books.
- Crawling limits follow the agreed seed-first semantics.
- Forced recrawl updates existing rows and respects the one-month skip rule.
- Rescrape invalidates embeddings only when embedding input components changed.
- Ollama is not called when all scraped books have valid verified embeddings.
- Invalid or zero-vector embeddings are detected and regenerated.
- Ranking defaults to stored/default params and does not run Nevergrad unless requested.
- Optimization stores updated params and uses the correct embedded subset.
- `predictions` is a clean snapshot of successfully scored books only.
- `verify` reports coherent state after a successful run.
