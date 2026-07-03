# Refactoring Embeddings Table to Support Multiple Models and Implicit Validation

We will refactor the database schema and application logic to:
1. Support storing embeddings for multiple models in the `embeddings` table.
2. Use **implicit verification via hashing** (`text_hash`) rather than an explicit `verified_embedding` flag column on the `books` table. This completely decouples the crawler from the embedding logic.
3. Add a `--model` CLI argument to both `embed` and `rank` commands (defaulting to the default model `qwen3-embedding:8b` / `OLLAMA_EMBEDDING_MODEL`).

## Proposed Changes

### Database Layer
#### [MODIFY] [db.py](file:///Users/alex/Documents/goodreads-ranker/db.py)
*   Update the `SCHEMA` to define the new `embeddings` table:
    ```sql
    CREATE TABLE IF NOT EXISTS embeddings (
        book_id           INTEGER,
        embedding_model   TEXT,
        dim               INTEGER NOT NULL,
        vector            BLOB NOT NULL,
        text_hash         TEXT NOT NULL,
        PRIMARY KEY (book_id, embedding_model)
    );
    ```
*   Remove the `verified_embedding` column from the `books` table definition.
*   Update `ensure_schema_compat(conn)`:
    - If the `embeddings` table exists and doesn't have the `embedding_model` column, drop it (recreating it with the new schema). Since it is currently empty, no data migration is needed.
*   Update `save_embeddings` to accept `model` and `text_hashes` (or a dictionary of book_id -> text_hash) and write them to the table.
*   Update `load_embeddings` and `load_embeddings_for_books` to accept a `model` parameter.

---

### Crawler Layer
#### [MODIFY] [crawler.py](file:///Users/alex/Documents/goodreads-ranker/crawler.py)
*   Remove `"verified_embedding"` from `field_names`.
*   Remove the `embedding_components_changed` helper function entirely.
*   Remove the code in `fetch_wrapper` that selects `old_row`, runs `embedding_components_changed`, and computes `verified_embedding`. The crawler now simply saves the newly scraped book details.

---

### Embedding Layer
#### [MODIFY] [embedder.py](file:///Users/alex/Documents/goodreads-ranker/embedder.py)
*   Update `find_books_needing_embeddings(conn, all_inputs, model)`:
    - Query the database joining `books` and `embeddings` on `e.book_id = b.book_id AND e.embedding_model = ?`.
    - Compute MD5 hashes of the current `all_inputs` formatting strings.
    - Check if the stored `text_hash` matches the current hash to determine if an embedding needs to be generated or regenerated.
*   Update `generate_embeddings`:
    - Pass the selected model (from argument or default env/constant) to `find_books_needing_embeddings`.
    - Compute hashes for the batch of generated embeddings and pass them to `db.save_embeddings`.
    - Remove the update query that was setting `verified_embedding = 1` on `books`.

---

### Ranking & Command Line Layer
#### [MODIFY] [ranker.py](file:///Users/alex/Documents/goodreads-ranker/ranker.py)
*   Update `load_valid_embeddings_for_books(conn, books_df, model=None)`:
    - Query embeddings from `embeddings` table where `embedding_model = ?`.
    - Generate current metadata hashes for all books.
    - Compare `text_hash` with the current hash to verify correctness.
*   Update `run_ranking` to accept `model=None`.
*   Update the CLI parsing in `__main__` to accept `--model`.

#### [MODIFY] [cli.py](file:///Users/alex/Documents/goodreads-ranker/cli.py)
*   Update `rank(self, interactive=False, optimize=False, model=None)` to accept the model parameter and pass it to `ranker.run_ranking`.
*   Update `run_pipeline` to accept `model=None` and pass it to `self.embed` and `self.rank`.
*   Update `verify(self, model=None)`:
    - Check for missing/invalid/outdated embeddings for the selected model specifically.
    - Compute current hashes and compare them to `text_hash` in `embeddings`.
    - Remove any checks of `verified_embedding` on the `books` table.

---

## Verification Plan

### Manual Verification
1. Run `python cli.py verify` to check the status of the empty database.
2. Run `python cli.py embed --model qwen3-embedding:8b` (or run pipeline) to generate embeddings for a subset of books and verify they are stored with their hashes in the database.
3. Verify that `python cli.py verify` shows 0 missing/unverified embeddings for that model.
4. Modify a scraped book's title or description directly in sqlite to simulate a rescrape metadata change, run `python cli.py verify`, and confirm it detects a text_hash mismatch (unverified embedding).
5. Run `python cli.py embed` again to confirm the outdated embedding gets regenerated.
