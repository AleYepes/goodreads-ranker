# Implementation Plan: Per-Model Hyperparams & Predictions via `embedding_models` Table

## 0. Context (read this first)

This repo is a pipeline that crawls Goodreads' internal GraphQL API to pull book
metadata, generates text embeddings from that metadata via Ollama, and feeds the
embeddings into ML models (KNN + Bayesian Ridge + SVR ensemble) that predict how
a user would rate books, blended with a friend-taste signal derived from
similarity to other Goodreads users' libraries.

Relevant directory structure:

```
goodreads_ranker/
├── goodreads_ranker/
│   ├── core/
│   │   ├── db.py        # all SQL lives here — other modules never write raw SQL
│   │   ├── config.py    # env-based config, incl. get_embedding_model()
│   │   └── utils.py
│   ├── ingestion/        # NOT part of this task — crawler.py, seeder.py, api_client.py
│   └── ml/
│       ├── elo_calibration.py   # NOT part of this task
│       ├── friend_similarity.py
│       ├── embedder.py
│       └── predictor.py
├── data/goodreads.db      # SQLite DB, WAL mode, PRAGMA foreign_keys=ON
├── main.py                 # Google Fire CLI entrypoint
```

Every ML/DB command in the CLI (`embed`, `friend_similarity`, `predict`) accepts an
`embedding_model` argument — an Ollama model name string like
`"qwen3-embedding:0.6b"`. Multiple embedding models can coexist: `book_embeddings`
already stores one row per `(book_id, embedding_model)` so you never have to
regenerate embeddings for a model you've already used. `prediction_hyperparams`
and `book_predictions`, however, currently have **no notion of embedding model at
all** — they store one single global set of data regardless of which model
produced it. This plan fixes that, and normalizes the model-name string into a
proper lookup table in the process.

**Important constraint carried through this whole plan:** the ML modules
(`embedder.py`, `friend_similarity.py`, `predictor.py`) and the CLI
(`main.py`) should keep passing `embedding_model` around as a plain string,
exactly as they do today. All resolution of that string to a numeric ID
happens *inside* `db.py`, at the point each function touches SQL. Only
`predictor.py` needs a real logic rewrite (see §5) — everything else is a
narrow, mechanical signature change.

---

## 1. Goals

1. Add a normalized `embedding_models` lookup table (`id`, `name`).
2. `book_embeddings`, `book_predictions`, and `prediction_hyperparams` all key
   into it via `embedding_model_id` instead of a raw text column — so multiple
   embedding models can each have their own predictions and hyperparams living
   side by side, with no cross-model overwriting.
3. Remove the hardcoded `DEFAULT_FRIEND_PARAMS` / `DEFAULT_SOLO_PARAMS` from
   `predictor.py`. Optimization now runs automatically, per model, the first
   time it's needed.
4. Rename the `optimize` flag to `force_optimize` throughout the call chain,
   with new semantics (see §5.2).
5. Add a `training_set_size` column to `prediction_hyperparams` (observational
   metadata only — nothing reads it to trigger anything automatically).

## 2. Explicit non-goals (do not do these)

- Do **not** touch `utils.py`'s `except ValueError, OverflowError, OSError:`
  line. It was raised and the maintainer confirmed it's fine as-is on their
  end — leave it alone.
- Do **not** touch the ingestion layer (`seeder.py`, `crawler.py`,
  `api_client.py`) or any date-formatting concerns there. Out of scope.
- Do **not** fix the `min_friend_similarity` default mismatch between
  `predict` (0.3) and `run_pipeline` (0.4) in `main.py`. Confirmed
  intentional-enough to leave alone.
- Do **not** turn `book_predictions` or `prediction_hyperparams` into
  historical/append-only tables. Each `(name/book_id, embedding_model_id)`
  pair should hold exactly one row — the latest — via upsert. No history.
- Do **not** build a general schema-versioning/migration framework. This repo
  has none today (`CREATE TABLE IF NOT EXISTS` only), and one narrow
  idempotent migration function is all this task calls for.
- Do **not** add automatic re-optimization triggered by library growth.
  `training_set_size` is observational only.

---

## 3. Current state (for reference — do not skip, the details matter)

### 3.1 Current schema (relevant tables only)

```sql
CREATE TABLE IF NOT EXISTS book_embeddings (
    book_id          INTEGER REFERENCES books(legacy_id) ON DELETE CASCADE,
    embedding_model  TEXT,
    vector           BLOB NOT NULL,
    text_hash        TEXT NOT NULL,
    PRIMARY KEY (book_id, embedding_model)
);

CREATE TABLE IF NOT EXISTS book_predictions (
    book_id                INTEGER PRIMARY KEY REFERENCES books(legacy_id) ON DELETE CASCADE,
    solo_pred_rating       REAL,
    friend_pred_rating     REAL,
    count_adjusted_rating  REAL,
    final_rating           REAL,
    date_updated           TEXT
);

CREATE TABLE IF NOT EXISTS prediction_hyperparams (
    name        TEXT PRIMARY KEY,
    params_json TEXT NOT NULL,
    date_updated TEXT NOT NULL
);
```

### 3.2 Current `db.py` functions that touch these tables

- `get_existing_embeddings(db_conn, model)` — reads `book_embeddings` filtered
  by `embedding_model = ?`
- `save_embeddings(db_conn, legacy_ids, vectors, model, text_hashes)` —
  writes `book_embeddings`
- `get_embeddings_by_model(db_conn, model)` — reads `book_embeddings`
- `save_book_predictions(db_conn, rows)` — writes `book_predictions` via
  `upsert_rows`, columns `["book_id", "solo_pred_rating",
  "friend_pred_rating", "count_adjusted_rating", "final_rating",
  "date_updated"]`
- `prune_book_predictions(db_conn, keep_ids)` — `DELETE FROM
  book_predictions WHERE book_id NOT IN (...)`, **no model scoping today**
- `get_prediction_hyperparams(db_conn, name)` — reads `prediction_hyperparams`
- `save_prediction_hyperparams(db_conn, name, params)` — writes
  `prediction_hyperparams`

### 3.3 Current `predictor.py` logic (what's being replaced)

```python
DEFAULT_FRIEND_PARAMS = { ... }   # hardcoded dict, ~11 keys
DEFAULT_SOLO_PARAMS = { ... }     # hardcoded dict, ~11 keys

def get_or_create_prediction_hyperparams(db_conn, name, defaults):
    params = db.get_prediction_hyperparams(db_conn, name)
    if params is None:
        params = dict(defaults)
        db.save_prediction_hyperparams(db_conn, name, params)
        return params
    return normalize_model_params(params)
```

Inside `run_prediction(..., optimize=False, ...)`:

```python
if optimize:
    friend_params = prep_optimization(..., training_col="training_ratings", budget=200, ...)
    solo_params = prep_optimization(..., training_col="my_refined", budget=200, ...)
    friend_params = normalize_model_params(friend_params)
    solo_params = normalize_model_params(solo_params)
    db.save_prediction_hyperparams(db_conn, "friend_params", friend_params)
    db.save_prediction_hyperparams(db_conn, "solo_params", solo_params)
else:
    friend_params = get_or_create_prediction_hyperparams(db_conn, "friend_params", DEFAULT_FRIEND_PARAMS)
    solo_params = get_or_create_prediction_hyperparams(db_conn, "solo_params", DEFAULT_SOLO_PARAMS)
```

`train_size` and `solo_train_size` (row counts of the two different training
sets used by the friend vs. solo models) are already computed earlier in
`run_prediction`, right before the embeddings tensor is built:

```python
train_size = (~pd.isna(embedded_books_df["training_ratings"])).sum()
solo_train_size = (~pd.isna(embedded_books_df["my_refined"])).sum()
```

These are exactly the values that should be written into the new
`training_set_size` column — friend's optimize run gets `train_size`, solo's
gets `solo_train_size`.

At the very end of `run_prediction`:

```python
if predictions_data:
    db.save_book_predictions(db_conn, predictions_data)
    db.prune_book_predictions(db_conn, [x[0] for x in predictions_data])
```

Each element of `predictions_data` is currently a 6-tuple:
`(legacy_id, solo_pred_rating, friend_pred_rating, count_adjusted_rating, final_rating, now_str)`.

---

## 4. New schema

Add this new table (place it near the top of the `SCHEMA` string in `db.py`,
before section "4. BOOK EMBEDDINGS"):

```sql
-- 0. EMBEDDING MODELS (lookup table)
CREATE TABLE IF NOT EXISTS embedding_models (
    id   INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);
```

Replace the three affected table definitions with:

```sql
-- 4. BOOK EMBEDDINGS
CREATE TABLE IF NOT EXISTS book_embeddings (
    book_id             INTEGER REFERENCES books(legacy_id) ON DELETE CASCADE,
    embedding_model_id  INTEGER NOT NULL REFERENCES embedding_models(id),
    vector              BLOB NOT NULL,
    text_hash           TEXT NOT NULL,
    PRIMARY KEY (book_id, embedding_model_id)
);

-- 5. BOOK PREDICTIONS
CREATE TABLE IF NOT EXISTS book_predictions (
    book_id                INTEGER REFERENCES books(legacy_id) ON DELETE CASCADE,
    embedding_model_id     INTEGER NOT NULL REFERENCES embedding_models(id),
    solo_pred_rating       REAL,
    friend_pred_rating     REAL,
    count_adjusted_rating  REAL,
    final_rating           REAL,
    date_updated           TEXT,
    PRIMARY KEY (book_id, embedding_model_id)
);

-- 6. PREDICTION HYPERPARAMS
CREATE TABLE IF NOT EXISTS prediction_hyperparams (
    name                TEXT NOT NULL,
    embedding_model_id  INTEGER NOT NULL REFERENCES embedding_models(id),
    params_json         TEXT NOT NULL,
    training_set_size   INTEGER,
    date_updated        TEXT NOT NULL,
    PRIMARY KEY (name, embedding_model_id)
);
```

Note what stayed the same: `date_updated` was **not** renamed (a
`date_optimized` alternative was considered and rejected — it would've been a
pure synonym once defaults are removed, since this column is now only ever
written after a real optimize run).

---

## 5. `db.py` changes

### 5.1 New resolver helper

Add this near the top of the "Generic persistence helpers" section (it's used
by nearly everything below):

```python
def get_or_create_embedding_model_id(db_conn, name: str) -> int:
    row = db_conn.execute("SELECT id FROM embedding_models WHERE name = ?", (name,)).fetchone()
    if row is not None:
        return int(row["id"])
    cursor = db_conn.execute("INSERT INTO embedding_models (name) VALUES (?)", (name,))
    db_conn.commit()
    return int(cursor.lastrowid)
```

Design decision already made: **get-or-create on every path, reads included.**
A read for a never-used model name will insert a (harmless, empty) row for
it. This was chosen deliberately for simplicity over a stricter
read-vs-write split — don't second-guess it.

### 5.2 The idempotent migration function

This must run **before** `db_conn.executescript(SCHEMA)` inside `init_db()`,
because `CREATE TABLE IF NOT EXISTS` is a no-op against a table that already
exists in the old shape — it will not add the new columns on its own.

```python
def _needs_embedding_model_migration(db_conn, table_name: str) -> bool:
    cursor = db_conn.execute(f"PRAGMA table_info({table_name})")
    columns = {row[1] for row in cursor.fetchall()}
    if not columns:
        return False  # table doesn't exist yet (fresh install) — nothing to migrate
    return "embedding_model_id" not in columns and (
        "embedding_model" in columns or table_name in ("book_predictions", "prediction_hyperparams")
    )


def _migrate_embedding_model_schema(db_conn):
    # Lookup table must exist first — the book_embeddings backfill below joins against it.
    db_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embedding_models (
            id   INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE
        )
        """
    )

    # book_embeddings: REAL migration. This holds real generated data (Ollama
    # calls, potentially many/slow) — do not drop it. Rename, rebuild, backfill, copy, drop.
    if _needs_embedding_model_migration(db_conn, "book_embeddings"):
        db_conn.execute("ALTER TABLE book_embeddings RENAME TO book_embeddings_old")
        db_conn.execute(
            """
            CREATE TABLE book_embeddings (
                book_id             INTEGER REFERENCES books(legacy_id) ON DELETE CASCADE,
                embedding_model_id  INTEGER NOT NULL REFERENCES embedding_models(id),
                vector              BLOB NOT NULL,
                text_hash           TEXT NOT NULL,
                PRIMARY KEY (book_id, embedding_model_id)
            )
            """
        )
        db_conn.execute(
            "INSERT OR IGNORE INTO embedding_models (name) SELECT DISTINCT embedding_model FROM book_embeddings_old"
        )
        db_conn.execute(
            """
            INSERT INTO book_embeddings (book_id, embedding_model_id, vector, text_hash)
            SELECT o.book_id, m.id, o.vector, o.text_hash
            FROM book_embeddings_old o
            JOIN embedding_models m ON m.name = o.embedding_model
            """
        )
        db_conn.execute("DROP TABLE book_embeddings_old")
        db_conn.commit()
        print("✓ Migrated book_embeddings to embedding_model_id schema (data preserved).")

    # prediction_hyperparams: cheap to regenerate (a couple minutes of nevergrad
    # search). Drop and let it repopulate on next predict run.
    if _needs_embedding_model_migration(db_conn, "prediction_hyperparams"):
        db_conn.execute("DROP TABLE prediction_hyperparams")
        db_conn.commit()
        print("✓ Dropped prediction_hyperparams (old schema) — will repopulate on next predict run.")

    # book_predictions: cheap to regenerate (no Ollama calls, just sklearn
    # fits on cached embeddings). Drop and let it repopulate.
    if _needs_embedding_model_migration(db_conn, "book_predictions"):
        db_conn.execute("DROP TABLE book_predictions")
        db_conn.commit()
        print("✓ Dropped book_predictions (old schema) — will repopulate on next predict run.")
```

Wire it into `init_db()`, before the `executescript` call:

```python
def init_db(db_path=None):
    with get_connection(db_path) as db_conn:
        _migrate_embedding_model_schema(db_conn)
        db_conn.executescript(SCHEMA)
        db_conn.execute("DROP VIEW IF EXISTS best_book_lookup")
        ...
```

This is safe to run on every single `init_db()` call (i.e. every CLI
invocation) — the `PRAGMA table_info` check makes the whole thing a no-op
once already migrated.

**Before testing this**, back up the existing DB file:
`cp data/goodreads.db data/goodreads.db.bak`.

### 5.3 Updated CRUD functions

Replace each of the following in full.

```python
def get_existing_embeddings(db_conn, model) -> dict[int, dict]:
    model_id = get_or_create_embedding_model_id(db_conn, model)
    cursor = db_conn.execute(
        "SELECT book_id, vector, text_hash FROM book_embeddings WHERE embedding_model_id = ?",
        (model_id,),
    )
    return {int(row["book_id"]): {"vector": row["vector"], "text_hash": row["text_hash"]} for row in cursor.fetchall()}


def save_embeddings(db_conn, legacy_ids, vectors, model, text_hashes):
    model_id = get_or_create_embedding_model_id(db_conn, model)
    rows = []
    for i, bid in enumerate(legacy_ids):
        bid_int = int(bid)
        h = text_hashes.get(bid_int) if isinstance(text_hashes, dict) else text_hashes[i]
        vector_blob = vectors[i].astype("float32").tobytes()
        rows.append((bid_int, model_id, vector_blob, h))

    db_conn.executemany(
        "INSERT OR REPLACE INTO book_embeddings (book_id, embedding_model_id, vector, text_hash) VALUES (?, ?, ?, ?)",
        rows,
    )
    db_conn.commit()


def get_embeddings_by_model(db_conn, model: str) -> dict[int, bytes]:
    model_id = get_or_create_embedding_model_id(db_conn, model)
    cursor = db_conn.execute(
        "SELECT book_id, vector FROM book_embeddings WHERE embedding_model_id = ?",
        (model_id,),
    )
    return {int(row["book_id"]): row["vector"] for row in cursor.fetchall()}


def save_book_predictions(db_conn, rows, embedding_model: str):
    model_id = get_or_create_embedding_model_id(db_conn, embedding_model)
    full_rows = [(r[0], model_id, r[1], r[2], r[3], r[4], r[5]) for r in rows]
    upsert_rows(
        db_conn,
        "book_predictions",
        full_rows,
        [
            "book_id",
            "embedding_model_id",
            "solo_pred_rating",
            "friend_pred_rating",
            "count_adjusted_rating",
            "final_rating",
            "date_updated",
        ],
    )


def prune_book_predictions(db_conn, keep_ids: list[int], embedding_model: str):
    if not keep_ids:
        return
    model_id = get_or_create_embedding_model_id(db_conn, embedding_model)
    placeholders = ",".join("?" for _ in keep_ids)
    db_conn.execute(
        f"DELETE FROM book_predictions WHERE embedding_model_id = ? AND book_id NOT IN ({placeholders})",
        [model_id, *keep_ids],
    )
    db_conn.commit()


def get_prediction_hyperparams(db_conn, name, embedding_model):
    model_id = get_or_create_embedding_model_id(db_conn, embedding_model)
    row = db_conn.execute(
        "SELECT params_json FROM prediction_hyperparams WHERE name = ? AND embedding_model_id = ?",
        (name, model_id),
    ).fetchone()
    if not row:
        return None
    try:
        return json.loads(row["params_json"])
    except json.JSONDecodeError:
        return None


def save_prediction_hyperparams(db_conn, name, embedding_model, params, training_set_size):
    model_id = get_or_create_embedding_model_id(db_conn, embedding_model)
    db_conn.execute(
        """
        INSERT OR REPLACE INTO prediction_hyperparams
            (name, embedding_model_id, params_json, training_set_size, date_updated)
        VALUES (?, ?, ?, ?, ?)
        """,
        (name, model_id, json.dumps(params, sort_keys=True), int(training_set_size), datetime.now().strftime("%Y-%m-%d")),
    )
    db_conn.commit()
```

**Critical bug fix baked into the above:** `prune_book_predictions` now scopes
its `DELETE` by `embedding_model_id`. Without this, running `predict` for
model A would delete model B's already-computed rows for any book that
happens to lack an embedding under model A — silently destroying another
model's data. Do not drop the `embedding_model_id = ?` clause.

### 5.4 `embedder.py` and `friend_similarity.py` — no changes needed

Both files call `db.get_existing_embeddings(...)`, `db.save_embeddings(...)`,
and `db.get_embeddings_by_model(...)` passing `embedding_model` as a plain
string, exactly as before. Since the ID resolution now happens inside those
`db.py` functions, **the call sites in `embedder.py` and
`friend_similarity.py` require zero changes.** Verify this after the `db.py`
edits by re-reading both files — if you find yourself wanting to edit them,
stop and re-check the `db.py` signatures instead.

---

## 6. `predictor.py` changes

### 6.1 Remove

- The entire `DEFAULT_FRIEND_PARAMS = {...}` and `DEFAULT_SOLO_PARAMS = {...}`
  dicts.
- The entire `get_or_create_prediction_hyperparams` function.

### 6.2 Add a helper to replace the removed function

```python
def _get_or_optimize_hyperparams(
    db_conn,
    name,
    embedding_model,
    training_col,
    embedded_books_df,
    precomputed_embeddings,
    mrl_dimensions,
    max_propagations,
    train_set_size,
    force_optimize,
):
    existing = None if force_optimize else db.get_prediction_hyperparams(db_conn, name, embedding_model)
    if existing is not None:
        return normalize_model_params(existing)

    desc = "Optimizing friend-taste model" if name == "friend_params" else "Optimizing solo-taste model"
    params = prep_optimization(
        embedded_books_df,
        precomputed_embeddings,
        training_col,
        mrl_dimensions,
        max_propagations,
        budget=200,
        desc=desc,
    )
    params = normalize_model_params(params)
    db.save_prediction_hyperparams(db_conn, name, embedding_model, params, int(train_set_size))
    return params
```

### 6.3 Replace the `if optimize: ... else: ...` block

Find this in `run_prediction` (it comes right after `precomputed_embeddings`
is built, and `train_size`/`solo_train_size` are already in scope from
earlier in the function):

```python
if optimize:
    friend_params = prep_optimization(...)
    solo_params = prep_optimization(...)
    ...
    db.save_prediction_hyperparams(db_conn, "friend_params", friend_params)
    db.save_prediction_hyperparams(db_conn, "solo_params", solo_params)
else:
    print("Using stored/default hyperparameters...")
    friend_params = get_or_create_prediction_hyperparams(db_conn, "friend_params", DEFAULT_FRIEND_PARAMS)
    solo_params = get_or_create_prediction_hyperparams(db_conn, "solo_params", DEFAULT_SOLO_PARAMS)
```

Replace it with:

```python
friend_params = _get_or_optimize_hyperparams(
    db_conn,
    "friend_params",
    embedding_model,
    "training_ratings",
    embedded_books_df,
    precomputed_embeddings,
    mrl_dimensions,
    max_propagations,
    train_size,
    force_optimize,
)
solo_params = _get_or_optimize_hyperparams(
    db_conn,
    "solo_params",
    embedding_model,
    "my_refined",
    embedded_books_df,
    precomputed_embeddings,
    mrl_dimensions,
    max_propagations,
    solo_train_size,
    force_optimize,
)
```

**Behavior this produces** (confirmed design — implement exactly this):

| `force_optimize` | friend/solo cached for this model? | Behavior |
|---|---|---|
| `False` (default) | both cached | use cached values, no search |
| `False` (default) | one or both missing | auto-run optimize only for the missing one(s) |
| `True` | any state | re-run optimize for **both**, overwrite whatever's cached |

### 6.4 Rename `optimize` → `force_optimize`

In `run_prediction`'s signature:

```python
def run_prediction(db_path=None, embedding_model=None, min_friend_similarity=0.3, optimize=False):
```
becomes
```python
def run_prediction(db_path=None, embedding_model=None, min_friend_similarity=0.3, force_optimize=False):
```

### 6.5 Update the final save calls

```python
if predictions_data:
    db.save_book_predictions(db_conn, predictions_data)
    db.prune_book_predictions(db_conn, [x[0] for x in predictions_data])
    print("✓ Ranking predictions complete and saved.")
```
becomes
```python
if predictions_data:
    db.save_book_predictions(db_conn, predictions_data, embedding_model)
    db.prune_book_predictions(db_conn, [x[0] for x in predictions_data], embedding_model)
    print("✓ Ranking predictions complete and saved.")
```

(`predictions_data` tuples themselves are unchanged — still 6-tuples of
`(legacy_id, solo_pred_rating, friend_pred_rating, count_adjusted_rating,
final_rating, now_str)`. The model gets injected inside `save_book_predictions`
now, per §5.3.)

---

## 7. `main.py` changes

Rename `optimize` → `force_optimize` in two places, updating docstrings to
match:

```python
def predict(self, force_optimize=False, embedding_model=None, min_friend_similarity=0.3):
    """Run the ensemble prediction model and write predictions to the database.

    Args:
        force_optimize (bool): Force re-optimization of hyperparameters, even if
            already cached for this embedding model. If not set, optimization
            runs automatically only for whichever of friend/solo hyperparameters
            are missing for the current model.
        embedding_model (str): Ollama embedding model name (overrides configured model).
        min_friend_similarity (float): Minimum friend taste correlation threshold.
    """
    from goodreads_ranker.ml import predictor

    print("\nRunning models and predictions")
    db.init_db()
    predictor.run_prediction(
        force_optimize=as_bool(force_optimize),
        embedding_model=embedding_model or None,
        min_friend_similarity=float(min_friend_similarity),
    )
```

```python
def run_pipeline(
    self,
    force_init=False,
    force_seed=False,
    library_ids=None,
    interactive=False,
    limit=None,
    force_crawl=False,
    batch_size=1,
    embedding_model=None,
    force_optimize=False,
    min_friend_similarity=0.4,
):
    """..."""
    self.init(force_init=force_init)
    self.seed(force_seed=force_seed, library_ids=library_ids)
    self.rate(interactive=interactive)
    self.crawl(limit=limit, force_crawl=force_crawl)
    self.embed(batch_size=batch_size, embedding_model=embedding_model)
    self.friend_similarity(embedding_model=embedding_model)
    self.predict(force_optimize=force_optimize, embedding_model=embedding_model, min_friend_similarity=min_friend_similarity)
    print("\n✓ Pipeline run finished successfully!")
```

Also update the `Args:` docstring entry for `optimize` → `force_optimize` in
`run_pipeline`'s docstring block.

---

## 8. Testing / validation checklist

Do these in order, against a **copy** of the real DB first if possible:

1. `cp data/goodreads.db data/goodreads.db.bak`
2. Run any command that calls `db.init_db()` (e.g. `python main.py embed`).
   Confirm in the printed output that the `book_embeddings` migration message
   appears exactly once, and re-running the same command produces no
   migration message the second time (idempotency check).
3. Sanity check row counts survived the `book_embeddings` migration:
   ```sql
   SELECT COUNT(*) FROM book_embeddings;      -- compare to goodreads.db.bak
   SELECT * FROM embedding_models;             -- should list your existing model name(s)
   ```
4. Run `python main.py predict` (no `--force_optimize`) with no cached
   hyperparams for the current model. Confirm both friend and solo
   optimization runs happen (two progress bars), and:
   ```sql
   SELECT name, embedding_model_id, training_set_size, date_updated FROM prediction_hyperparams;
   ```
   shows two rows with sane `training_set_size` values (friend's should be
   ≥ solo's, since friend-taste training data is a superset).
5. Run `python main.py predict` again, same model, no flag. Confirm it's
   fast (no optimization progress bars — cached hyperparams were reused).
6. Run `python main.py predict --force_optimize`. Confirm both optimizations
   re-run even though hyperparams were already cached, and `date_updated`
   changes.
7. Run `python main.py embed --embedding_model=<a different model>` then
   `python main.py predict --embedding_model=<that different model>`.
   Confirm:
   - A second row appears in `embedding_models`.
   - `prediction_hyperparams` now has *four* rows total (two per model), not
     two overwritten rows.
   - `book_predictions` now has rows for both models coexisting — spot check
     that a book's prediction row for the *first* model wasn't deleted by
     running `predict` for the second model (this is the
     `prune_book_predictions` scoping fix — verify it actually works, not
     just that it compiles).
8. Run project lint/format (e.g. `ruff check .` / `ruff format .`) if
   configured, and fix anything it flags in the touched files.

## 9. Definition of done

- [ ] `embedding_models` table exists and is populated correctly after migration
- [ ] `book_embeddings` migrated with all rows preserved (row count matches pre-migration)
- [ ] `prediction_hyperparams` and `book_predictions` have the new composite PKs
- [ ] `DEFAULT_FRIEND_PARAMS` / `DEFAULT_SOLO_PARAMS` / `get_or_create_prediction_hyperparams` fully removed from `predictor.py`
- [ ] `force_optimize` renamed consistently across `predictor.py`, `main.py`'s `predict`, and `main.py`'s `run_pipeline`, matching the fill-vs-force matrix in §6.3
- [ ] `prune_book_predictions` is model-scoped (§5.3 critical bug fix)
- [ ] `embedder.py` and `friend_similarity.py` are untouched (confirm by diff — if either changed, something went wrong)
- [ ] Full checklist in §8 passes
- [ ] Nothing in §2 (non-goals) was touched
