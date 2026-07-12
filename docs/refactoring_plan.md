# goodreads_ranker — Refactoring & Debugging Plan

Compiled from a design-review interview after the GraphQL migration of `crawler.py`. Target: fresh database (no migration/backfill needed — starting from scratch). This document is intended to let another agent implement all agreed changes without re-litigating the decisions.

---

## 0. Root Cause Summary

The GraphQL crawler introduced **book canonicalization**: every scraped book ID is resolved to its "bestBook" (canonical) edition, and only canonical IDs get a row in `books`. However, `reader_libraries.book_id` (populated by `seeder.py` from Goodreads shelf pages) still contains whatever edition ID the user actually shelved — which is frequently **not** canonical.

Since `ranker.py` joined `reader_libraries` to `books` on exact ID equality, ~21% of shelved books (measured: 1,076 of 5,089 rows) silently fail to join. This starves `run_ranking` of ratings data, tripping one of two silent early-return paths (`print(...); return`, no exception, no non-zero exit) — which looked like "pipeline finishes cleanly but `predictions` stays empty."

The fix has three parts: (1) crawler must *record* the edition→canonical mapping it already discovers but currently discards, (2) every consumer that assumes an ID is canonical must resolve through that mapping, (3) failures must stop being silent so this class of bug is caught immediately next time.

---

## 1. `crawler.py` — Canonicalization & Mapping Fixes

### 1.1 Persist edition→canonical mapping (was: silently discarded)
In `resolve_and_save_book`, the non-canonical branch currently marks `crawl_queue` as `mapped_to_canonical` but never records *what* it mapped to anywhere queryable. `book_editions` only gets populated later, incidentally, via the canonical book's paginated `work.editions` connection (capped at ~200, and only if the sibling happens to be included) — so many mappings are lost entirely.

**Fix:** Add `title` to `PROBE_QUERY`. Immediately after computing `best_book_legacy_id` (before the `canonical_row` early-return check — this ordering matters, since skipping it there is exactly the common case once most of the library is seeded), write:

```python
if legacy_id != best_book_legacy_id:
    db_conn.execute(
        "INSERT OR IGNORE INTO book_editions (book_id, edition_legacy_id, edition_kca_id, title) "
        "VALUES (?, ?, ?, ?)",
        (best_book_legacy_id, legacy_id, probe_kca_id, book_node.get("title")),
    )
    ...
```
Use `INSERT OR IGNORE` (not `REPLACE`) — the mapping is immutable, no reason to overwrite.

### 1.2 Fix `crawl_queue` column typo (real crash bug)
The "no best book legacy ID resolved" branch inserts into a nonexistent `processed_at` column:
```sql
INSERT OR REPLACE INTO crawl_queue (book_id, status, error_count, last_error_message, processed_at) ...
```
Schema column is `date_processed`. This raises an unhandled `sqlite3.OperationalError` today. **Fix: rename `processed_at` → `date_processed` in this one insert.**

### 1.3 User-Agent bug
`crawler.py`'s inline GraphQL request header UA string is truncated (missing `(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36`), differing from `seeder.py`'s UA and reading as bot-like. **Fix: both files import a single shared `USER_AGENT` constant from `utils.py` (see §5).**

### 1.4 `limit` / similar-books-discovery semantics (confirmed intentional — consolidate, don't change behavior)
Confirmed design: no `--limit` → seed IDs only. With `--limit` → seeds first (by priority, all seeds ahead of all similar books, then priority within group — **already implemented correctly** in the `ORDER BY (discovered_via = 'seed') DESC, priority DESC` query, verified against worked examples). `--limit 0` or negative → unbounded crawl, seeds + similar books both eligible.

This behavior is correct today but is encoded via the same `limit > 0` / `limit is not None` comparisons repeated independently in three places (source-eligibility gate, pre-loop skip check, in-loop stop check). **Fix: consolidate into one place at the top of `run_crawler`:**
```python
expand_similar = limit is not None
effective_limit = limit if (limit is not None and limit > 0) else None  # 0 or negative = unbounded
```
Use `effective_limit` for all cap checks, `expand_similar` for `allowed_sources`. Add a one-line comment documenting the `0`/negative = unbounded convention. This is a pure refactor — no behavior change.

Note: the crawl limit is a **soft cap** (bounded overshoot of at most `concurrency - 1` books past `limit`, since the budget check counts completed books, not in-flight dispatches). **Confirmed acceptable as-is** — `limit` is a cost/budget dial against a rate-limited API, not a correctness-critical invariant. No change needed here.

### 1.5 Rate-limit visibility
`gql()`'s retry loop currently backs off silently on HTTP 403/429. **Add:**
```python
tqdm.write(f"  Rate limited (status {resp.status_code}) on {operation_name}, attempt {attempt+1}/3 — backing off {delay:.1f}s")
```

### 1.6 Concurrency / shared SQLite connection (no fix — document only)
`run_crawler` shares one synchronous `sqlite3` connection across concurrently-dispatched tasks. This is safe today only because sqlite3 calls are blocking and asyncio is single-threaded (no `await` occurs mid-write in any current code path). **No structural change** — add a comment at the connection's point of use stating this invariant explicitly: *shared across concurrent tasks; safe only because calls are synchronous — do not introduce `await` between related writes, and revisit if ever migrating to `aiosqlite`.*

### 1.7 `Series`/`Award`/`Genre`/`book_editions` id fields — GraphQL schema notes
- `book_editions.edition_kca_id`: available via `edges { node { id legacyId title } }` in both `BOOK_QUERY`'s `work.editions` and `EDITIONS_PAGE_QUERY` — the earlier `editions/id`/`editions/legacyId` field errors were from querying the connection object directly rather than `edges.node`; not a real schema gap. Add `id` to both queries' `node` selection.
- `Series`: GraphQL exposes `id` (kca_id) but no `legacyId` field at all. Must extract numeric `legacy_id` from `series.webUrl` slug (e.g. `goodreads.com/series/44427-his-dark-materials`) via `parse_id_from_slug` (moved to `utils.py`, see §5).
- `Award`: GraphQL exposes **no id field of any kind** (`id` and `legacyId` both invalid on `Award`). Must extract `legacy_id` from `award.webUrl` slug (e.g. `goodreads.com/award/show/3572-audie-award`) — this is the *only* available identifier. If `webUrl` is missing/unparseable for a given award, skip inserting that award row rather than inventing a fallback id.
- `Genre`: `bookGenres` currently only fetches `name`; expand to `bookGenres { genre { id name webUrl } }` (§2.4) to get `kca_id` and `web_url`. Genre slugs are pure text (`young-adult`) with no numeric prefix, so `legacy_id` extraction uses a plain last-path-segment helper, not `parse_id_from_slug` (§5).
- `Contributor` (formerly `Author`): expand `BOOK_QUERY` to add `secondaryContributorEdges { node { id legacyId name isGrAuthor webUrl followers { totalCount } works { totalCount } } role }` alongside the existing `primaryContributorEdge` fetch (§2.6). Crawler must tag rows from `primaryContributorEdge` with `is_primary = 1` and rows from `secondaryContributorEdges` with `is_primary = 0` at write time — this cannot be inferred later from `role`, since secondary contributors can carry the same `role` value (e.g. `"Author"`) as the primary contributor (§2.7).

---

## 2. `db.py` — Schema Changes

### 2.1 ID naming convention (confirmed)
Keep `legacy_id` (numeric Goodreads ID, used in all slugs/URLs — except genres, see §2.4) and `kca_id` (new hex ID) as the two id-type names, applied **consistently across every table that has both**. This is distinct from **foreign key columns**, which keep relational names (`book_id`, `contributor_id`, `series_id`, `genre_id`, `award_id`) — never renamed to `legacy_id`, since that would create ambiguity about *whose* legacy_id it is. Non-FK columns that hold a raw external id of a different entity than the row's own PK get an entity-prefixed name (`edition_legacy_id`, `edition_kca_id`, `similar_legacy_id`, `similar_kca_id`).

### 2.2 Table renames (confirmed — apply to schema, all SQL, and every Python reference/function name)
| Old name | New name |
|---|---|
| `model_params` | `prediction_hyperparams` |
| `predictions` | `book_predictions` |
| `embeddings` | `book_embeddings` |
| `genres` (junction) | → restructured, see §2.4 |
| `similar_books` | `book_similar_books` |
| `elo_ratings` | `book_elo_ratings` |

Rename cascades into Python: `save_model_params`/`load_model_params` → `save_prediction_hyperparams`/`load_prediction_hyperparams`, and every `db_conn.execute` referencing these table names, across `db.py`, `ranker.py`, `embedder.py`, `crawler.py`. Grep for each old table name as a final check — a name surviving in a comment or string literal is exactly the kind of drift this session started with.

### 2.3 `series` / `book_series` junction pattern (confirmed: natural key + junction table)
```sql
CREATE TABLE series (
    legacy_id INTEGER PRIMARY KEY,  -- natural key, extracted from web_url slug
    kca_id    TEXT,                 -- "kca://series/..."
    title     TEXT NOT NULL,
    web_url   TEXT
);
CREATE TABLE book_series (
    book_id   INTEGER REFERENCES books(legacy_id) ON DELETE CASCADE,
    series_id INTEGER REFERENCES series(legacy_id) ON DELETE CASCADE,
    position  INTEGER,              -- was TEXT; parse to int
    PRIMARY KEY (book_id, series_id)
);
```
If `web_url` is missing/unparseable, skip the row (same policy as §2.4/§2.5).
This also fixes a live bug: `series` dedup is currently broken today — `INSERT OR IGNORE INTO series (title, web_url) VALUES (...)` never triggers `IGNORE` because there's no `UNIQUE` constraint and the old PK was autoincrement, so **every crawl silently inserted duplicate series rows**. The natural `legacy_id` PK fixes this as a side effect.

### 2.4 `genres` / `book_genres` junction pattern (confirmed)
```sql
CREATE TABLE genres (
    legacy_id TEXT PRIMARY KEY,     -- natural key from web_url slug; TEXT because genre slugs
                                     -- (e.g. 'young-adult', 'non-fiction') have no numeric prefix —
                                     -- the only non-integer legacy_id in the schema
    kca_id    TEXT,                 -- "kca://genre/..."
    name      TEXT,
    web_url   TEXT
);
CREATE TABLE book_genres (
    book_id   INTEGER REFERENCES books(legacy_id) ON DELETE CASCADE,
    genre_id  TEXT REFERENCES genres(legacy_id) ON DELETE CASCADE,   -- TEXT, matching genres.legacy_id
    PRIMARY KEY (book_id, genre_id)
);
```
Query expansion needed: `BOOK_QUERY`'s `bookGenres` selection must expand to `bookGenres { genre { id name webUrl } }` to obtain `kca_id` (`id`) and `web_url` (for slug extraction) — currently only `name` is fetched.
`genres.legacy_id` extraction needs a **new, separate** helper from `parse_id_from_slug` — genre slugs are pure text with no leading numeric id (unlike series/awards), so this needs a plain "last path segment" extractor. See §5.

### 2.5 `awards` / `book_awards` junction pattern (confirmed)
```sql
CREATE TABLE awards (
    legacy_id INTEGER PRIMARY KEY,  -- natural key, numeric prefix extracted from web_url slug
    name      TEXT NOT NULL,
    web_url   TEXT
);
CREATE TABLE book_awards (
    book_id      INTEGER REFERENCES books(legacy_id) ON DELETE CASCADE,
    award_id     INTEGER REFERENCES awards(legacy_id) ON DELETE CASCADE,
    category     TEXT,
    designation  TEXT,
    date_awarded TEXT,              -- normalize to strftime('%Y-%m-%d') format for consistency
                                     -- with other date_* columns in the schema
    PRIMARY KEY (book_id, award_id)
);
```
`awards.legacy_id` is `INTEGER` (award slugs do have a numeric prefix, e.g. `3572-audie-award` — unlike genres), extracted via the existing `parse_id_from_slug`. No `kca_id` column — confirmed in §1.7 that `Award` has no id field of any kind in the GraphQL schema, numeric or kca. If `web_url` is missing/unparseable, skip the row (this is already the only way to get an id for an award at all, so an unparseable slug means the award can't be stored — acceptable, matches the policy already agreed for series).

### 2.6 `authors` → `contributors` / `book_contributors`, secondary contributors reintegrated (confirmed, with one clarification — see §2.7)
```sql
CREATE TABLE contributors (
    legacy_id       INTEGER PRIMARY KEY,   -- fetched via primaryContributorEdge/secondaryContributorEdges node.legacyId
    kca_id          TEXT,                  -- "kca://author/..."
    name            TEXT NOT NULL,
    web_url         TEXT,
    is_gr_author    INTEGER DEFAULT 0,
    works_count     INTEGER DEFAULT 0,
    followers_count INTEGER DEFAULT 0
);
CREATE TABLE book_contributors (
    book_id        INTEGER REFERENCES books(legacy_id) ON DELETE CASCADE,
    contributor_id INTEGER REFERENCES contributors(legacy_id) ON DELETE CASCADE,
    role           TEXT,
    is_primary     INTEGER DEFAULT 0,      -- see §2.7 — required, not optional
    PRIMARY KEY (book_id, contributor_id, role)
);
```
`books` table: **drop `author_id` and `author_role` columns** — this relationship now lives entirely in `book_contributors`.

Query expansion needed in `BOOK_QUERY`:
```python
secondaryContributorEdges {
    node { id legacyId name isGrAuthor webUrl followers { totalCount } works { totalCount } }
    role
}
```
(mirrors the fields already fetched for `primaryContributorEdge`).

### 2.7 `is_primary` is required, not just role-based (clarification on the above)
`role` alone cannot distinguish primary from secondary — the sample GraphQL payload used to confirm this design shows secondary contributors can carry `role: "Author"` too (e.g. genuine co-authors), identical to the primary contributor's role string. `book_contributors.is_primary` must be set explicitly by the crawler based on **which edge the row came from** (`primaryContributorEdge` → `1`, each `secondaryContributorEdges` entry → `0`), not inferred from `role` at query time or read time.

### 2.8 `book_editions` table
Add `edition_kca_id` column (see §1.7). `book_id` FK stays as-is (correct relational name — do not rename to `legacy_id`; this was explicitly discussed and rejected to avoid ambiguity with `edition_legacy_id`).

### 2.9 Canonical-ID resolution view (confirmed: SQL view, not a new table or Python helper)
```sql
CREATE VIEW IF NOT EXISTS book_id_resolved AS
SELECT edition_legacy_id AS raw_id, book_id AS canonical_id FROM book_editions
UNION
SELECT legacy_id AS raw_id, legacy_id AS canonical_id FROM books;
```
This is the single mechanism used everywhere a raw/possibly-non-canonical ID needs to be resolved to its canonical `books.legacy_id` for joining. Do **not** build a separate mapping table or a Python-side dict-based resolver — this reuses `book_editions` (now correctly populated per §1.1) as intended, keeping "raw scrape data" and "resolved knowledge" cleanly separated at the DB layer rather than duplicating resolution logic in application code.

### 2.10 Embeddings stay primary-contributor-only (confirmed — does NOT change with §2.6)
Reintegrating secondary contributors into the schema is a **data-model completeness change, not a reversal of the embedding design**. `embedder.py`'s `build_embedding_inputs` continues to use only the `is_primary = 1` contributor for the "Written by:" line — see §4, unchanged from the earlier decision. Explicitly deferred: whether/how secondary contributors might factor into embedding text later, pending investigation into how secondary-contributor `role` values actually vary in practice (translator vs. illustrator vs. genuine co-author, etc.) — out of scope for this pass.

---

## 3. `ranker.py` — Join Fixes & Reliability

### 3.0 Table-rename fallout (see §2.2)
Update every query referencing `predictions`, `embeddings`, `elo_ratings`, `similar_books`, or `model_params` to the new names (`book_predictions`, `book_embeddings`, `book_elo_ratings`, `book_similar_books`, `prediction_hyperparams`). Also update `authors`/`author_id` references to `contributors`/`book_contributors` — the `books_df` construction's author lookup needs to become a join through `book_contributors WHERE is_primary = 1` instead of `books.author_id`, since that column is being dropped (§2.6).

### 3.1 Apply `book_id_resolved` at all three affected call sites (confirmed scope: all three, uniform treatment)
1. **`run_ranking`'s main `books_df` construction** — join `reader_libraries` through `book_id_resolved` instead of direct `ON b.legacy_id = l.book_id`.
2. **`get_similar_friend_ratings`'s friend-overlap query** — same fix; friend ratings were being undercounted by the same mechanism, and since `min_overlap=5` / `min_correlation=0.3` are hard thresholds, this could silently drop otherwise-qualifying friends from the model.
3. **`build_adjacency_matrix`'s GCN edges** — `book_similar_books.similar_legacy_id` has the identical bug (edges to non-canonical similar-book IDs currently vanish since `book_ids_set` only contains canonical ids). Resolve through the same view.

### 3.2 Orphan visibility
Add a `print` in `run_ranking` reporting the count of `reader_libraries` rows that remain unresolved even after the `book_id_resolved` join (i.e., not found in `books` directly nor via `book_editions`) — e.g. crawler hasn't reached them yet, or they errored. Cheap, and gives early warning if this gap ever reopens.

### 3.3 `book_predictions` table: don't destroy last-known-good data on failure (confirmed: both defer-delete AND raise-on-failure)
Currently `DELETE FROM book_predictions` (renamed from `predictions`, §2.2) runs **unconditionally at the top** of `run_ranking`, before any validity checks — so a run that hits a silent early-return wipes existing predictions without replacing them. Two changes, both required:

- **Defer the delete.** Compute `predictions_data` fully first (already how it works), then at the end replace with `INSERT OR REPLACE` plus a final `DELETE FROM book_predictions WHERE book_id NOT IN (<computed ids>)` (to still prune books that dropped out of scope) — instead of a blanket delete at the start. Do this inside the same transaction as the inserts.
- **Raise instead of silently returning.** Convert every `print("..."); return` early-exit in `run_ranking` (e.g. "No self rating records found", "Not enough rated books...") into a raised `RuntimeError` with the same message. Propagate up through `main.py` so `run_pipeline` gets a non-zero exit / visible failure instead of printing "Pipeline run finished successfully!" regardless of whether anything meaningful happened. `run_pipeline` should **not** catch and continue past a stage failure — each stage depends on the previous stage's output, so abort the rest of the pipeline on first failure.

Combined effect: a bad run fails loudly, and never destroys the last good `predictions` data in the process.

### 3.4 Device selection bug
```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```
never checks CUDA. **Fix:**
```python
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
```

### 3.5 Out of scope for this pass (explicitly deferred, not forgotten)
`ranker.py`'s modeling choices are **not** part of this refactor and should be raised as a separate, future discussion:
- `final_rating = count_adjusted_scaled * friend_pred * solo_pred` — multiplying three [0,1]-normalized signals is an aggressive combination (any one weak signal drags down the whole score); worth revisiting whether this is the intended blending strategy.
- Hardcoded `mrl_dimensions` starting value (32) and doubling logic.
- Minor naming nit: `skf` variable name for a `KFold` (not stratified) split object.

---

## 4. `embedder.py` — Embedding Input Cleanup

### 4.1 Remove dead multi-author logic (confirmed: secondary contributors intentionally excluded from embedding text, not a regression — reconfirmed after §2.6's schema reintegration)
- Build `authors_post` directly from the single primary-contributor name (no list-wrapping round-trip through `format_string_for_embedding`).
- Remove the now-unreachable `truncate` parameter and its `elif n > truncate > 1: ... "and N others"` branch from `format_string_for_embedding` — no remaining caller passes `truncate`.
- `format_string_for_embedding` itself stays (genres still use its list-joining behavior); only the dead parameter/branch and the author call site's unnecessary list-wrapping are removed.

### 4.2 Query update for `contributors`/`book_contributors` rename (see §2.6, §2.10)
`build_embedding_inputs`'s author lookup changes from `LEFT JOIN authors a ON b.author_id = a.legacy_id` to a join through `book_contributors bc ON bc.book_id = b.legacy_id AND bc.is_primary = 1`, then `JOIN contributors c ON c.legacy_id = bc.contributor_id`. Embedding text remains primary-contributor-only, per §2.10 — this is a rename/rewire of the existing query, not a behavior change to what gets embedded.

### 4.3 Table-rename fallout (see §2.2)
Update `embeddings` references to `book_embeddings` throughout (`find_books_needing_embeddings`, `generate_embeddings`'s save call, etc.).

---

## 5. New Module: `goodreads_ranker/utils.py`

Purpose: dependency-free, cross-cutting helpers not specific to persistence (which stays scoped to `db.py`) or to any single pipeline stage. Confirmed contents:

- `parse_id_from_slug` (moved from `seeder.py`) — needed by `crawler.py` for `series`/`awards` numeric slug-extraction (§1.7, §2.3, §2.5); can't import from `seeder.py` directly since it pulls in `playwright.async_api` at module load.
- `parse_slug` (new — plain "last path segment" extractor, no numeric-prefix assumption) — needed for `genres.legacy_id` (§2.4), since genre slugs are pure text with no leading digits and `parse_id_from_slug` won't match them.
- `clean_text` (moved from `seeder.py`) — generic string cleanup, moved on principle even though currently single-use.
- `USER_AGENT` (single shared constant) — fixes the `seeder.py`/`crawler.py` UA-string drift (§1.3); both modules import from here.

**Explicitly NOT moved:**
- `as_bool` / `parse_optional_int` (`main.py`) — stay put; these solve a CLI-argument-parsing problem specific to Fire's string-typed args, not a general-purpose concern. Avoid preemptive abstraction for a need that doesn't currently exist elsewhere.
- `format_string_for_embedding` / `join_embedding_parts` (`embedder.py`) — embedding-text-shape-specific.
- `safe_spearman` (`ranker.py`) — only used within `ranker.py`.
- `vector_to_blob` / `is_valid_embedding_blob` / `infer_dim_from_blob` (`db.py`) — about embedding storage format, correctly scoped to `db.py`.

---

## 6. Explicit Non-Goals for This Pass

- **No migration/backfill logic** — starting from a fresh database, so no need to reprocess previously `mapped_to_canonical` `crawl_queue` rows or repair historical `book_editions` gaps.
- **No hard crawl-limit enforcement** — soft cap (§1.4) confirmed acceptable.
- **No `aiosqlite` migration or per-worker connection pool** — current shared synchronous connection is safe as-is; document only (§1.6).
- **No `ranker.py` modeling/hyperparameter changes** — deferred to a future session (§3.5).
- **No CLI restructuring** — `limit`-gates-similar-discovery coupling stays (§1.4), by design, to avoid CLI argument bloat.

---

## 7. Implementation Checklist

- [ ] `db.py`:
  - [ ] Rename tables per §2.2 (`model_params`→`prediction_hyperparams`, `predictions`→`book_predictions`, `embeddings`→`book_embeddings`, `similar_books`→`book_similar_books`, `elo_ratings`→`book_elo_ratings`) and their Python function names (`save_model_params`→`save_prediction_hyperparams`, etc.).
  - [ ] Replace `authors` with `contributors` (add `kca_id`), drop `books.author_id`/`books.author_role`, add `contributors`/`book_contributors` tables (`book_contributors` includes `role` and `is_primary`).
  - [ ] Replace surrogate-keyed `series` with natural-keyed `series` (`legacy_id` PK, add `kca_id`) + new `book_series` junction (fixes the series-dedup bug for free).
  - [ ] Replace flat `genres` junction with natural-keyed `genres` (`legacy_id TEXT` PK, add `kca_id`) + new `book_genres` junction.
  - [ ] Replace surrogate-keyed `awards` with natural-keyed `awards` (`legacy_id` PK, no `kca_id` — field doesn't exist) + new `book_awards` junction; normalize `date_awarded` to `strftime('%Y-%m-%d')`.
  - [ ] Add `edition_kca_id` to `book_editions`.
  - [ ] Add `book_id_resolved` view.
- [ ] `utils.py`: new file — `parse_id_from_slug`, `parse_slug`, `clean_text`, `USER_AGENT`.
- [ ] `seeder.py`: import `parse_id_from_slug`, `clean_text`, `USER_AGENT` from `utils.py` instead of defining locally.
- [ ] `crawler.py`:
  - [ ] Add `title`/`id` to `PROBE_QUERY`; write `book_editions` mapping immediately on canonicalization branch, before `canonical_row` early return.
  - [ ] Fix `processed_at` → `date_processed` typo.
  - [ ] Add `id` to `work.editions.edges.node` selections in `BOOK_QUERY` and `EDITIONS_PAGE_QUERY`; populate `edition_kca_id`.
  - [ ] Expand `bookGenres` to `bookGenres { genre { id name webUrl } }`; extract `genres.legacy_id` via `parse_slug`.
  - [ ] Add `secondaryContributorEdges { node { id legacyId name isGrAuthor webUrl followers { totalCount } works { totalCount } } role }` to `BOOK_QUERY`; write `book_contributors` rows for both primary (`is_primary=1`) and secondary (`is_primary=0`) contributors.
  - [ ] Extract and store `series.legacy_id`/`kca_id` (via `book_series`), `awards.legacy_id` (via `book_awards`) using `parse_id_from_slug` on `webUrl`; skip rows where extraction fails.
  - [ ] Import `USER_AGENT` from `utils.py` for GraphQL request headers.
  - [ ] Consolidate `limit` semantics into `expand_similar`/`effective_limit` at top of `run_crawler`; add explanatory comment.
  - [ ] Add `tqdm.write` on 403/429 rate-limit retry in `gql`.
  - [ ] Add comment documenting the shared-synchronous-connection invariant.
- [ ] `embedder.py`:
  - [ ] Rewire author lookup to `book_contributors JOIN contributors WHERE is_primary = 1` (still primary-only in the embedded text — §2.10).
  - [ ] Simplify author string handling; remove `truncate` param/branch from `format_string_for_embedding`.
  - [ ] Update `embeddings` → `book_embeddings` references.
- [ ] `ranker.py`:
  - [ ] Update all renamed-table references (`book_predictions`, `book_embeddings`, `book_elo_ratings`, `book_similar_books`, `prediction_hyperparams`, `contributors`/`book_contributors`).
  - [ ] Join through `book_id_resolved` in `run_ranking`'s `books_df` query, `get_similar_friend_ratings`'s friend query, and `build_adjacency_matrix`'s edge query.
  - [ ] Add orphan-count print after resolved joins.
  - [ ] Defer `DELETE FROM book_predictions` to end of run; replace with `INSERT OR REPLACE` + targeted prune of out-of-scope book_ids.
  - [ ] Convert silent `print(...); return` early exits to `raise RuntimeError(...)`.
  - [ ] Fix device selection to check `torch.cuda.is_available()`.
- [ ] `main.py`: ensure `run_pipeline` does not catch/continue past a stage exception — let it propagate and abort remaining stages.