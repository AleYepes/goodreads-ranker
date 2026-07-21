# Refactoring Implementation Plan v2: Goodreads Ranker

## 0. How to Use This Document

This is the authoritative, refined version of the original refactoring plan, produced after a detailed review of the actual codebase (`main.py`, `goodreads_ranker/config.py`, `goodreads_ranker/utils.py`, `goodreads_ranker/embedder.py`, `goodreads_ranker/db.py`, `goodreads_ranker/crawler.py`, `goodreads_ranker/ranker.py`, `goodreads_ranker/seeder.py`) and a design interview that resolved every ambiguous or incorrect assumption in the original draft. It supersedes that draft entirely — where the two disagree, this document wins.

Section 1 restates the objective. Sections 2–7 are the phase-by-phase spec, written to be executable by an agent with no prior context on this conversation, only the codebase. Section 8 is a decisions log explaining *why* each non-obvious choice was made, for anyone auditing the result later. Section 9 covers small cleanup items that don't need their own phase.

---

## 1. Context & Objectives

This codebase is an end-to-end pipeline that scrapes Goodreads data, interacts with a GraphQL API, generates local LLM embeddings (via Ollama), and runs ML models (GNN, SVR, KNN) to predict book ratings. The goal is to decouple it into a database-mediated, unidirectional architecture: `core/` holds the single source of truth (schema, config, pure helpers) and `ingestion/`/`ml/` modules talk to each other **only** through the database, never through direct imports or in-memory handoffs.

---

## 2. Target Directory Structure

```text
goodreads_ranker/
├── core/
│   ├── db.py               (Strict DB access layer — the ONLY file with raw SQL)
│   ├── config.py           (Environment variables)
│   └── utils.py            (Pure, stateless helper functions)
├── ingestion/
│   ├── api_client.py       (GraphQL network layer: HTTP, headers, rate-limit backoff)
│   ├── crawler.py          (Queue orchestrator)
│   └── seeder.py           (Playwright scraping)
└── ml/
    ├── elo_calibration.py  (Interactive terminal rating — the `rate` command)
    ├── friend_similarity.py (Correlation + calibration — the `friend-similarity` command)
    ├── embedder.py          (LLM prompt prep & embedding generation)
    └── predictor.py         (GNN/ensemble prediction — the `predict` command, replaces ranker.py)
main.py                      (CLI entrypoint, repo root)
```
```
goodreads_ranker/
├── assets/
├── data/
├── docs/
├── notebooks/
├── old/
├── goodreads_ranker/
│   ├── __init__.py
│   ├── core/
│   │   ├── db.py
│   │   ├── config.py
│   │   └── utils.py
│   ├── ingestion/
│   │   ├── api_client.py
│   │   ├── crawler.py
│   │   └── seeder.py
│   └── ml/
│       ├── elo_calibration.py
│       ├── friend_similarity.py
│       ├── embedder.py
│       └── predictor.py
├── .env
├── main.py
├── pyproject.toml
└── ...
```

**Import boundary rule (strict):** `ingestion/` and `ml/` modules may import from `core/` only — never from each other, and never from a sibling module in the other package. `core/` modules may never import from `ingestion/` or `ml/`. `main.py` sits outside the package and may import anything. This is achievable cleanly given the current call graph — no module today needs a cross-boundary import once the DB-mediated handoffs described below are in place.

---

## 3. Phase 1 — Directory Reorganization

Mechanical: move each existing file into its target location per Section 2, splitting `crawler.py` into `ingestion/api_client.py` + `ingestion/crawler.py`, and `ranker.py` into `ml/elo_calibration.py` + `ml/friend_similarity.py` + `ml/predictor.py`. Add `__init__.py` to `core/`, `ingestion/`, `ml/` as needed. No behavior changes in this phase — it's pure setup for Phases 2–5.

---

## 4. Phase 2 — Database Layer (`core/db.py`)

### 4.1 Schema Changes

1. **Reorder table declarations** in the schema string so every table appears *after* the tables it holds a foreign key against. Concretely: `books` must be declared before `book_elo_ratings` (which references `books(legacy_id)`). This was previously backwards — harmless at runtime (SQLite doesn't validate FK targets until enforcement), but confusing to read top-to-bottom.
2. **`libraries` table:** replace `is_similar INTEGER DEFAULT 0` with `similarity_score REAL` (nullable, no default). This is **not** a data migration — `is_similar` is written by `set_similar_libraries` but never read anywhere in the current codebase, so there is no existing consumer to preserve. Just drop the old column and add the new one; no backfill needed. `similarity_score` becomes meaningful for the first time once `friend_similarity.py` (Section 6.3) starts writing it and `predictor.py` (Section 6.4) starts reading it.
3. **`library_books` table:** add `calibrated_rating REAL` (nullable). Currently this value is computed in-memory inside `ranker.py` and never persisted — this column makes it a real, queryable output of `friend_similarity.py`.

### 4.2 SQL Centralization

**Design constraint carried over from the original plan:** `core/db.py` is the only file allowed to contain raw SQL. Every other module calls a named `db.py` function with plain Python objects (dicts, lists, primitives) — never raw GraphQL node shapes, never numpy arrays or PyTorch tensors. This means `ingestion/crawler.py` must **flatten** each GraphQL node into a flat dict before calling the corresponding `db.py` save function (this was implied by the original plan's "no GraphQL schema knowledge in `db.py`" rule, so it's carried forward explicitly here even though it wasn't a separate interview question).

Extract these functions into `db.py`, at the same granularity as today's `crawler.py` helpers (one function per entity — **not** one coarse "save everything" call, since canonical-edition resolution has real branching logic that belongs in the orchestrator, not buried in a save function):

- `save_book_core(db_conn, book: dict)` — flat dict with columns matching today's `_save_book_core`: `legacy_id, kca_id, title, title_complete, description, web_url, asin, isbn, isbn13, format, num_pages, language_name, publisher, publication_time, original_publication_time, star_1..star_5, date_fetched`.
- `save_contributors(db_conn, book_legacy_id, contributors: list[dict])` — each dict: `{legacy_id, kca_id, name, web_url, is_gr_author, works_count, followers_count, role, is_primary}`.
- `save_series(db_conn, book_legacy_id, series_list: list[dict])` — `{legacy_id, kca_id, title, web_url, position}`.
- `save_genres(db_conn, book_legacy_id, genres: list[dict])` — `{legacy_id, kca_id, name, web_url}`.
- `save_awards(db_conn, book_legacy_id, awards: list[dict])` — `{legacy_id, name, web_url, category, designation, date_awarded}`.
- `save_editions(db_conn, book_legacy_id, editions: list[dict])` — `{edition_legacy_id, edition_kca_id}`.
- `save_similar_books_and_enqueue(db_conn, book_legacy_id, similar_list: list[dict], now)` — keeps the priority-score computation inline (as today), since it's simple arithmetic over fields already present in `similar_list` and immediately consumed by the same INSERT; splitting it out would add indirection for no benefit.
- `book_exists(db_conn, legacy_id) -> bool`, `link_editions_to_canonical(db_conn, best_book_legacy_id, legacy_id, book_kca_id, editions)`, `mark_known_editions_skipped(db_conn, best_book_legacy_id, now)` — support `_resolve_canonical_edition`'s branching, which **stays in `crawler.py`** as orchestration logic, not a `db.py` function.
- `get_pending_crawl_batch(db_conn, allowed_sources, limit) -> list[int]` — powers the new batched concurrency loop (Section 5.2).
- `count_crawl_queue(db_conn, statuses, allowed_sources) -> int` — consolidates the near-identical `completed_count`/`pending_count` queries into one parameterized function.
- `populate_seeds(db_conn)` and `handle_force_crawl(db_conn)` — relocate as-is (already close to pure SQL); **delete the commented-out dead `db_conn.execute` block** inside `handle_force_crawl` while moving it (see Section 9).
- **`set_crawl_status(db_conn, legacy_id, status, error_count=0, last_error_message=None, date_processed=None)`** — new consolidated function replacing the six near-duplicate `INSERT ... ON CONFLICT DO UPDATE` blocks scattered across today's `crawler.py` (`mapped_to_canonical`, `error` with invalid-ID message, `error` with incrementing count, `done`, `skipped_known_edition`, etc.). Every call site in the new `crawler.py` uses this one function.
- `get_elo_ratings(db_conn) -> list[dict]`, `save_elo_ratings(db_conn, rows)` — raw SQL only for `book_elo_ratings`; the merge/compute logic that currently lives alongside this SQL in `refine_ratings` moves to `core/utils.py` (Section 6.1).
- `get_main_library_id(db_conn)` — already exists, keep as-is.
- `get_main_library_ratings(db_conn, main_library_id) -> list[dict]` with `{legacy_id, title, rating}` — used independently by `elo_calibration.py` (needs `title` for interactive prompts) and by `friend_similarity.py`/`predictor.py` (only need `legacy_id`/`rating`, ignore `title`).
- `save_friend_similarity_scores(db_conn, scores: dict[library_id, float])` — writes `libraries.similarity_score`.
- `update_calibrated_ratings(db_conn, rows: list[tuple[library_id, book_legacy_id, calibrated_rating]])` — writes `library_books.calibrated_rating` for every friend-rated book (not just ones that pass any threshold — `friend_similarity.py` writes unconditionally for all friends, per Section 6.3).
- `get_friend_calibrated_ratings(db_conn, min_friend_similarity: float) -> list[dict]` — flat `WHERE libraries.similarity_score >= ?` join against `library_books`, returning `{library_id, book_legacy_id, calibrated_rating, similarity_score}`. **No minimum-friend-count fallback** — if fewer than N friends clear the threshold, `predictor.py` just works with fewer (or zero) friends. This is a deliberate simplification versus today's behavior; see decision log (Section 8).
- `get_books_for_prediction(db_conn, main_library_id) -> list[dict]` — replaces the big join query at the top of today's `run_ranking` (books + `best_book_lookup` + `library_books` + `my_rating`).
- `get_similar_books_edges(db_conn) -> list[tuple[int, int]]` — replaces the raw query inside `build_adjacency_matrix`; resolves through `best_book_lookup` in SQL as today, returns plain edge pairs for `predictor.py` to turn into a tensor.
- `get_friend_library_book_ratings(db_conn) -> list[dict]` — relocates the friends query (library_id, best_book_id via `best_book_lookup`, rating) currently inline near the top of `get_similar_friend_ratings`.
- `get_embeddings_by_model(db_conn, model: str) -> dict[legacy_id, bytes]` — raw blob fetch only, no numpy. Used independently by both `friend_similarity.py` and `predictor.py` (see Section 8 for why this is duplicated work by design, not an oversight).
- `save_book_predictions(db_conn, rows)`, `prune_book_predictions(...)`, `get_prediction_hyperparams`/`save_prediction_hyperparams` — already exist in roughly this shape, keep as-is, just relocate.
- `get_book_metadata_for_embedding(db_conn) -> list[dict]` — replaces `build_embedding_inputs`'s SQL half. Returns raw metadata per book: `{legacy_id, title, author_name, description, genres: list[str]}`, with genres **pre-aggregated in `db.py`** (grouping a join result by parent ID is query-shaping, not embedding-domain logic, and matches where this aggregation already happens today).
- `get_existing_embeddings(db_conn, model) -> dict[legacy_id, {vector: bytes, text_hash: str}]` — replaces the `LEFT JOIN` half of today's `find_stale_or_missing_embeddings`.
- `save_embeddings(db_conn, ids, vectors, model, hashes)` — already exists, keep.

---

## 5. Phase 3 — Ingestion Layer

### 5.1 `ingestion/api_client.py`

Move the pure GraphQL/HTTP logic here, kept as **module-level functions** (not a class — the current code is functional in style, and wrapping this in an object hierarchy would be exactly the kind of unnecessary abstraction the review is meant to eliminate):

- `gql()`, `fetch_similar_books()`, `_fetch_book_node()`, the `BOOK_QUERY`/`SIMILAR_QUERY` constants, `make_after_token()`, `InvalidLegacyIdError`.
- `build_headers()` — assembles the request headers (`X_API_KEY` from `core.config`, content-type, user-agent from `core.utils.USER_AGENT`).
- Rate-limit backoff (429/403 handling) stays here, using a `cooldown: list[float]` passed in by the caller and mutated in place. **This does not need any new synchronization primitive** — asyncio is single-threaded and cooperative, so a plain list mutated between `await` points cannot race the way it would under real threads. (The original plan's "thread-safe backoff" concern conflated this with the actual concurrency risk, which is entirely about SQLite connection sharing — see Section 5.2.)

### 5.2 `ingestion/crawler.py`

Acts purely as an orchestrator. Keeps: `resolve_and_save_book` (now flattening each GraphQL node into a flat dict before calling the matching `db.py` save function), `_resolve_canonical_edition` (orchestration/branching, stays here), `run_crawler`.

**Concurrency fix — one connection per concurrent task.** Today's `while` loop is sequential despite an unused `asyncio.Semaphore(3)`. Fix:

1. `run_crawler` opens one long-lived `db_conn` via `db.get_connection(db_path)` for orchestration-only reads: `populate_seeds`, `handle_force_crawl`, progress-bar counts, and `db.get_pending_crawl_batch(db_conn, allowed_sources, limit=5)`.
2. For each ID in that batch, spawn a task that opens **its own** short-lived connection (also via `db.get_connection(db_path)`, which already sets `PRAGMA journal_mode=WAL` — designed for exactly this multi-connection pattern) scoped to just that task's `resolve_and_save_book` call. Also set `PRAGMA busy_timeout=5000` on these connections so a task that loses a brief write race waits and retries instead of raising `database is locked`.
3. Run the batch with `asyncio.gather(*tasks)`. The existing `asyncio.Semaphore(3)` continues to bound concurrency *within* each task's own recursive fetch chain, as it does today.

Sketch:

```python
async def run_crawler(limit=None, force_crawl=False, db_path=None):
    sem = asyncio.Semaphore(3)
    cooldown = [0.0]
    db.init_db(db_path)
    headers = api_client.build_headers()
    with db.get_connection(db_path) as db_conn:
        db.populate_seeds(db_conn)
        if force_crawl:
            db.handle_force_crawl(db_conn)
        async with httpx.AsyncClient() as client:
            while True:
                # ... limit / stop-condition checks using db_conn ...
                batch_ids = db.get_pending_crawl_batch(db_conn, allowed_sources, limit=5)
                if not batch_ids:
                    break

                async def process_one(legacy_id):
                    with db.get_connection(db_path) as task_conn:
                        await resolve_and_save_book(
                            client, headers, task_conn, legacy_id, allowed_sources, sem, cooldown
                        )

                await asyncio.gather(*(process_one(lid) for lid in batch_ids))
                # ... refresh progress bar using db_conn ...
    pbar.close()
```

### 5.3 `ingestion/seeder.py`

Relocate as-is (already close to pure orchestration + Playwright scraping + `db.py` calls). Tuning constants (`CONCURRENCY`, `NAV_RETRY_ATTEMPTS`, `JITTER_MIN`/`MAX`) stay as module-level constants next to the code that uses them — they're implementation tuning, not user-facing settings, and `core/config.py`'s stated scope is environment variables; mixing "read from `.env`" with "hardcoded retry count" would blur that boundary rather than clarify it.

---

## 6. Phase 4 — Machine Learning Layer

Split `ranker.py` into three files that share **zero in-memory state**, handing off data strictly via the database. All three independently call `db.py`/`utils.py` as needed — none of them import each other.

### 6.1 `core/utils.py` additions

- `compute_continuous(elo_df)` — relocated from `ranker.py`, unchanged, already pure.
- `merge_elo_state(existing_rows: list[dict], target_ratings: dict[legacy_id, rating]) -> DataFrame` — the merge logic currently inline in `refine_ratings` (updates `original_rating` for books with existing Elo rows, adds new rows at `elo_score=1200` for books that don't have one yet). Pure pandas, no SQL, no I/O.
- `assemble_embedding_matrix(legacy_ids: list[int], vectors_by_id: dict[int, bytes]) -> tuple[np.ndarray, np.ndarray]` — the numpy assembly/validity-check half of today's `load_valid_embeddings_for_books` (frombuffer, dimension checks, mask building). Takes the raw blob dict from `db.get_embeddings_by_model` and returns `(valid_mask, matrix)`. Used independently by both `friend_similarity.py` and `predictor.py`.
- Move `format_string_for_embedding` and `join_embedding_parts` **out of here** and into `ml/embedder.py` (Section 6.5) — they're LLM-prompt-specific, not general-purpose.

Together, `db.get_elo_ratings` + `utils.merge_elo_state` + `utils.compute_continuous` give any caller a read-only "my_refined" rating series from whatever Elo state currently exists, with **no interactivity required** — this is what resolves the earlier gap where `friend_similarity.py` and `predictor.py` need refined ratings but `rate` is not guaranteed to have run first. Fresh books default gracefully to their plain star rating, exactly as today.

### 6.2 `ml/elo_calibration.py` (the `rate` command)

The **only** module that writes to `book_elo_ratings`, and the only place `input()`/`print()` calls exist in the ML layer.

- `run_interactive_ranking(...)`, `update_elo(...)`, `get_expected_score(...)` — relocated unchanged; these stay local to this file (not promoted to `core/utils.py`) since nothing else needs them.
- Top-level entrypoint `run_calibration(db_path=None)`: reads via `db.get_main_library_ratings`/`db.get_elo_ratings`, merges via `utils.merge_elo_state`, runs `run_interactive_ranking`, writes back via `db.save_elo_ratings`. Always interactive — that's this command's entire purpose, so no `interactive` flag is needed.

### 6.3 `ml/friend_similarity.py` (the `friend-similarity` command)

**Behavior change (confirmed):** compute the Spearman correlation (`similarity_score`) for **all** friends, and the `calibrated_rating` for **all** of their books — no filtering at this stage. Filtering happens downstream, in `predictor.py`.

- `safe_spearman(...)`, `calibrate_friend_ratings(...)` — relocated, unchanged, pure.
- Entrypoint `run_friend_similarity(embedding_model=None, db_path=None)`:
  1. `db.get_friend_library_book_ratings(db_conn)` for the raw friend/book/rating rows.
  2. `db.get_main_library_ratings` + `db.get_elo_ratings` + `utils.merge_elo_state` + `utils.compute_continuous` for "my_refined" (independent computation, not shared with `predictor.py` in memory).
  3. `db.get_embeddings_by_model(db_conn, model)` + `utils.assemble_embedding_matrix(...)` to support the KNN synthetic-rating step used when imputing missing overlap.
  4. For each friend: run the correlation + calibration math (unchanged), including the **ephemeral KNN synthetic ratings** — these stay in-memory only, used solely to compute the correlation score, and are **never persisted** (unchanged from today).
  5. Write results: `db.save_friend_similarity_scores(db_conn, scores)` and `db.update_calibrated_ratings(db_conn, rows)` for every friend, unconditionally.

### 6.4 `ml/predictor.py` (the `predict` command, replaces `ranker.py`)

- Loads embeddings via `db.get_embeddings_by_model` + `utils.assemble_embedding_matrix` — **no staleness check, no hash comparison, no fallback warning.** Predictor trusts whatever is in `book_embeddings`. This is a deliberate simplification (see Section 8) that assumes `embed` generally runs before `predict` (true by construction in `run_pipeline`; true by convention for anyone running commands individually).
- Gets "my_refined" independently via `db.get_main_library_ratings` + `db.get_elo_ratings` + `utils.merge_elo_state` + `utils.compute_continuous` — its own computation, not shared with `friend_similarity.py`.
- Gets friend data via `db.get_friend_calibrated_ratings(db_conn, min_friend_similarity)` — flat threshold, **no minimum-friend-count fallback**. Does its own weighted-average aggregation in Python (the same math as today's tail end of `get_similar_friend_ratings`: weighted sum / total weight, grouped by book).
- Everything downstream — `build_adjacency_matrix` (now using `db.get_similar_books_edges` instead of inline SQL), PyTorch Geometric graph construction, nevergrad optimization, and scikit-learn ensembling — is **relocated as-is**. Do not alter model parameters, nevergrad search spaces, or scikit-learn configuration; only move them.
- Entrypoint: `run_prediction(optimize=False, embedding_model=None, min_friend_similarity=0.3, db_path=None)`. No `interactive` parameter — that responsibility now belongs entirely to `elo_calibration.py`.

### 6.5 `ml/embedder.py`

- Owns `format_string_for_embedding` and `join_embedding_parts` (moved from `core/utils.py`).
- `db.get_book_metadata_for_embedding(db_conn)` gives raw metadata (genres pre-aggregated); `embedder.py` assembles the text, computes MD5 hashes, and does its own staleness comparison against `db.get_existing_embeddings(db_conn, model)` — this logic (currently the pure-Python half of `find_stale_or_missing_embeddings`) is fully relocated here.
- `generate_embeddings(batch_size, embedding_model)` — Ollama lifecycle management (`_ensure_ollama`) and the batch loop are otherwise unchanged, still calling `db.save_embeddings`.

---

## 7. Phase 5 — CLI Integration (`main.py`)

1. **Commands:**
   - `init` — unchanged.
   - `seed` — unchanged, calls `ingestion.seeder`.
   - `rate` — new, calls `ml.elo_calibration.run_calibration`.
   - `crawl` — calls `ingestion.crawler.run_crawler`.
   - `embed` — calls `ml.embedder.generate_embeddings`.
   - `friend_similarity` — new, calls `ml.friend_similarity.run_friend_similarity`. (Fire exposes underscore method names as dash-separated on the CLI, i.e. `friend-similarity`.)
   - `predict` — new, replaces `rank`, calls `ml.predictor.run_prediction`, exposes `min_friend_similarity` (default `0.3`).
   - `run_pipeline` — see below.
2. **`run_pipeline`:** calls every subcommand directly and unconditionally, in this order: **`seed → rate → crawl → embed → friend_similarity → predict`**. `rate` is placed right after `seed` because Elo calibration is most closely tied to `library_books` data, which `seed` populates — not because of any data dependency on `crawl`/`embed`. `rate`'s interactive blocking behavior inside an automated pipeline run is accepted as-is for now; the user will add opt-out flags later once they've exercised the full pipeline manually. `run_pipeline`'s signature keeps flattening every subcommand's kwargs as its own top-level parameter, exactly as it does today — including the new `min_friend_similarity`. **No `run_*` boolean toggles are introduced** — every step always runs when `run_pipeline` is called.

---

## 8. Decisions Log (rationale for non-obvious choices)

| Area | Decision | Why |
|---|---|---|
| `is_similar`/`similarity_score` | Treated as a new capability, not a migrated filter | No code today reads `is_similar` — it's write-only. There is no existing query to "update." |
| Friend inclusion in `predictor.py` | Flat `WHERE similarity_score >= ?`, **no** minimum-friend-count fallback | Today's fallback-to-top-N-if-under-threshold can't be expressed as a flat SQL filter without extra logic; explicitly dropped for simplicity rather than pushed into a `db.py` helper. |
| "My refined" rating computation | Pure logic (`compute_continuous`, `merge_elo_state`) lives in `core/utils.py`; interactivity stays only in `elo_calibration.py` | `friend_similarity.py` and `predictor.py` need refined ratings but must not depend on `rate` having run in the same session — a pure, shared function resolves this without violating "zero shared state between ML modules." |
| DB save function granularity | One function per entity (matches today), not one coarse "save everything" call | `_resolve_canonical_edition`'s branching is orchestration, not persistence, and belongs in `crawler.py`. |
| Crawl-queue status writes | Consolidated into one `db.set_crawl_status(...)` | Six near-duplicate `INSERT ... ON CONFLICT` blocks were redundant. |
| Crawler concurrency fix | Per-task SQLite connection (WAL + `busy_timeout`), not a shared connection + lock | `gather()`-ing tasks that share one connection risks interleaved/incomplete transactions; WAL mode is designed for exactly this multi-connection pattern. |
| Cooldown backoff state | Left as a plain shared `list[float]`, no lock added | asyncio is cooperative/single-threaded; this was never actually at risk, unlike the DB connection sharing. |
| Embedding staleness in `predictor.py` | Removed entirely, **accepted silently** (no warning, no doc comment) | Deliberate simplification per the plan; `embed` is expected to generally precede `predict`. |
| Embedding loading duplication | `friend_similarity.py` and `predictor.py` each independently query + assemble the embedding matrix | Direct consequence of "zero shared in-memory state, DB-mediated handoff only" — an accepted, computationally cheap tradeoff, not an oversight. |
| `min_friend_similarity` | Promoted to a CLI flag (default `0.3`), threaded through `predict` and `run_pipeline` | Persisting a continuous `similarity_score` only pays off if the cutoff is tunable without recomputing anything. |
| `run_pipeline` shape | Flattened kwargs for every subcommand, no `run_*` toggles | Consistent with today's pattern; toggles deferred until the user has exercised the pipeline manually. |
| `rate`'s position in the pipeline | Right after `seed` | Elo calibration is conceptually tied to `library_books`, populated by `seed`, not to crawled metadata. |
| Goodreads API key default | Left hardcoded in `config.py` | It's Goodreads' own public, anonymous AppSync key embedded in every visitor's browser bundle — not a secret, unlike `GOODREADS_EMAIL`/`GOODREADS_PASSWORD`. |

---

## 9. Cleanup Items (fold in while touching nearby code)

- Delete the commented-out dead SQL block inside `handle_force_crawl` (in `crawler.py` today) while relocating it to `db.py`.
- Reorder the schema string so `books` is declared before `book_elo_ratings`.
- No other dead code, unused imports, or commented-out blocks were found elsewhere in the reviewed files.

---

## 10. Strict Implementation Rules (unchanged from original plan)

1. **No circular imports:** `ingestion`/`ml` modules import only from `core`, never from each other.
2. **No SQL outside `core/db.py`:** any `SELECT`/`INSERT`/`UPDATE`/`DELETE` found elsewhere must be moved into a named `db.py` function.
3. **Preserve logic/math:** do not alter scikit-learn parameters, PyTorch Geometric configuration, or nevergrad search spaces — only move them.
4. **Preserve type hinting:** maintain or improve existing type hints across all touched function signatures.
