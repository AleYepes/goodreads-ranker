# Goodreads Ranker — Refactoring Plan

Synthesized from a decision-by-decision review of `main.py` and every module in
`goodreads_ranker/` (seeder.py, crawler.py, embedder.py, ranker.py, db.py, utils.py,
config.py). Every dead-code claim below was verified against actual call sites, not
assumed.

## Architecture rule (clarified)

Original rule: *"db.py and utils.py should be the only shared packages."*
**Clarified rule: db.py, utils.py, and config.py are the three shared packages.**
config.py was already a de facto shared package (used by main, seeder, embedder,
ranker) — crawler.py was the sole holdout, hardcoding values instead. No pipeline
module (seeder/crawler/embedder/ranker) may import another pipeline module directly.

Scope boundary for db.py: **not** a strict repository pattern. Only SQL that is
(a) duplicated across module boundaries, or (b) inconsistent with a pattern the
same file already follows, moves into db.py. Single-call-site SQL local to one
module (crawler.py's crawl_queue transitions, ranker.py's elo/friends/adjacency
queries) stays where it is — extracting it would just relocate complexity, not
reduce it.

---

## Phase 1 — Mechanical, low-risk

| File | Change | Why |
|---|---|---|
| `config.py` | Add `API_URL` and `X_API_KEY` (or `get_api_url()` / `get_api_key()`) | Currently hardcoded in crawler.py; every other module gets its settings from config.py |
| `crawler.py` | Import URL/key from config.py instead of module-level constants | Consistency with the rest of the codebase — not a secrets fix (this key is Goodreads' own public frontend key), a config-layering fix |
| `utils.py` | `parse_date_str`: catch `dateutil.parser.ParserError`/`ValueError`/`OverflowError` specifically instead of bare `Exception`; log a warning before falling back to the raw string | Currently fails silently — no way to distinguish "unusual format" from a real bug |

## Phase 2 — db.py consolidation

| File | Change | Why |
|---|---|---|
| `db.py` | Replace unused `load_embedding_text_hashes` with a real helper, e.g. `find_stale_or_missing_embeddings(db_conn, inputs, model) -> list[int]`, that both embedder.py and ranker.py call | embedder.py's `find_books_needing_embeddings` and ranker.py's `load_valid_embeddings_for_books` independently reimplement the same hash-and-validity check; `load_embedding_text_hashes` looks like an abandoned first attempt at exactly this |
| `embedder.py` | Delete `find_books_needing_embeddings`; call the new db.py helper | — |
| `ranker.py` | Delete `load_valid_embeddings_for_books`'s duplicated logic; call the new db.py helper; **remove `from .embedder import build_embedding_inputs`** | This import is the one clear architecture-boundary violation in the codebase |
| `db.py` | Add `bootstrap_libraries(db_conn, library_ids)` and `get_libraries_to_scrape(db_conn, force_seed)` (or similar) | seeder.py already delegates most library persistence to db.py (`upsert_libraries`, `mark_library_complete`, `load_existing_library_rows`, etc.) but bypasses it for the initial ID-seeding insert and the scrape-candidates query — same file, same concern, inconsistent layering |
| `seeder.py` | Call the new db.py functions instead of raw `db_conn.execute(...)` | — |
| `db.py` | Re-section the module with accurate header comments (e.g. move `book_elo_ratings` / `book_embeddings` / `book_predictions` / `prediction_hyperparams` out from under the "READER LIBRARIES" heading into their own section) | Cosmetic but cheap; db.py stays a single file — splitting into a package at ~400–500 lines would be premature structure |

## Phase 3 — main.py CLI cleanup

| File | Change | Why |
|---|---|---|
| `main.py` | Delete `command_helps`, `pipeline_flags`/`seen_flags` module-level loop, `print_help()`, the `-h`/`--help` interception block in `main()` | ~80 lines duplicating what `fire.Fire()` already generates from docstrings via `--help`; two sources of truth for the same flag descriptions is a maintenance trap |
| `main.py` | **Keep** the `PAGER` environment hack, repurposed to protect Fire's own default help (which uses a pager and can hang non-interactively) rather than the deleted custom one | Same underlying problem the hack originally solved still exists once Fire's built-in help takes over |
| `main.py` | Rewrite `run_pipeline` as five explicit calls (`self.init(...)`, `self.seed(...)`, etc. with named args) instead of `inspect.signature`-based kwarg filtering | The dynamic dispatch hides what the method does and silently drops typo'd flag names instead of erroring; explicit calls are self-documenting |
| `main.py` | `seed()`'s inline library-ID file/string parsing stays as-is | Decided: it's CLI-input parsing and belongs with the CLI, not extracted |

## Phase 4 — crawler.py decomposition

| File | Change | Why |
|---|---|---|
| `crawler.py` | Break `resolve_and_save_book` (~470 lines, single function) into named internal steps — e.g. `_fetch_book_node`, `_resolve_canonical_edition`, `_save_book_core`, `_save_contributors`, `_save_series`, `_save_genres`, `_save_awards`, `_save_editions`, `_save_similar_books_and_enqueue` | The clearest "too much responsibility in one place" finding in the codebase: GraphQL fetch, a recursive canonical-edition redirect, and eight distinct write paths, all inline with no internal structure |
| `crawler.py` | While decomposing, consolidate the near-duplicate primary/secondary contributor-saving logic into one parameterized helper | Falls out naturally from the above split |

## Phase 5 — Type hints (do last, once structure has settled)

Full pass, every function in every file: `main.py`, `seeder.py`, `crawler.py`,
`embedder.py`, `ranker.py` get brought up to the standard `db.py`/`utils.py`/`config.py`
already mostly meet. Doing this last avoids re-annotating code that Phases 1–4 are
about to move or rewrite.

---

## Explicitly decided *not* to change

- Ollama server auto-start/stop/poll lifecycle in `embedder._ensure_ollama` — kept for first-run convenience, a deliberate trade-off, not an oversight.
- Per-item resilience `except Exception` blocks in embedder's batch loop, seeder's per-library scraping, crawler's per-book processing — intentional "skip this one, keep the pipeline running" pattern for long jobs; only `parse_date_str`'s silent swallow was a genuine smell.
- `ranker.run_ranking` — long, but already decomposed into named helper functions (`refine_ratings`, `get_similar_friend_ratings`, `build_adjacency_matrix`, `prep_optimization`, `run_optimized`); the remaining bulk is ordinary pandas/numpy score-blending, not a god-function.
- `utils.parse_id_from_slug` vs `parse_slug` — confirmed both are used for genuinely different purposes (numeric ID extraction vs. string slug extraction), not redundant.
- Raw SQL local to crawler.py's crawl_queue transitions and ranker.py's elo/friends/adjacency queries — single call site each, no cross-module duplication, extracting them would add indirection without reducing complexity.

## Suggested implementation order

1. **Phase 1** — isolated, no cross-file dependencies, quick wins.
2. **Phase 2** — db.py changes are a prerequisite for removing the ranker→embedder import cleanly.
3. **Phase 3** — main.py is self-contained, can happen anytime after Phase 1.
4. **Phase 4** — the biggest single diff; do it once the data layer underneath it (Phase 2) is stable.
5. **Phase 5** — last, so type hints land on the final shape of the code rather than code that's about to be moved.
