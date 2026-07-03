# Pipeline Output Cleanup

Refine the `run_pipeline` terminal output across all modules: reduce noise, improve
consistency, and make the calibration summary actually informative. No functional
logic changes — logging only.

## Proposed Changes

### cli.py

#### [MODIFY] [cli.py](file:///Users/alex/Documents/goodreads-ranker/cli.py)

No structural changes to STEP labels — they stay as:
```
STEP 1: Seeding database
STEP 2: Crawling book details
...
```

---

### seeder.py

#### [MODIFY] [seeder.py](file:///Users/alex/Documents/goodreads-ranker/seeder.py)

**`download_user_library`** — rephrase the skip notice and indent all sub-lines:

| Before | After |
|--------|-------|
| `user_library table already has 661 rows. Skipping download. Use --force to override.` | `  Library already seeded (661 books). Use --force to re-download.` |
| `Logging into Goodreads to download library export...` | `  Logging into Goodreads to download library export...` |
| `Export downloaded. Importing into database...` | `  Export downloaded. Importing into database...` |
| `Successfully imported 661 books from your library...` | `  Imported 661 books into library.` |

**`scrape_friend_ratings`** — rephrase and indent:

| Before | After |
|--------|-------|
| `All friend lists are already scraped. Use --force-all to re-scrape.` | `  Friend lists already scraped. Use --force-all to re-scrape.` |
| `Preparing to scrape N friend lists...` | `  Scraping N friend lists...` |

Inner per-list prints (`Scraping list ...`, `Finished ...`, `Processed list ...`, etc.) already exist and will get two-space indent where they don't have one yet.

---

### embedder.py

#### [MODIFY] [embedder.py](file:///Users/alex/Documents/goodreads-ranker/embedder.py)

**`_ensure_ollama` context manager** — trim lifecycle noise:

| Before | After |
|--------|-------|
| `Ollama server not detected — starting 'ollama serve'...` | `  Ollama server not detected — starting 'ollama serve'...` (kept, indented) |
| `Ollama server started.` | *(dropped)* |
| `Model 'X' not found locally — pulling...` | `  Model 'X' not found locally — pulling (this may take a while)...` (kept, indented) |
| `Model 'X' ready.` | `  Model 'X' ready.` (kept, indented) |
| `Stopping Ollama server (started by this process)...` | *(dropped — silent cleanup)* |

**`generate_embeddings`**:

| Before | After |
|--------|-------|
| `Generating embeddings for N books using Ollama model 'X'...` | *(dropped — tqdm bar is self-describing)* |
| `Embedding generation process finished.` | *(dropped — tqdm completion is the signal)* |
| `Nothing to embed: all scraped books have valid verified embeddings...` | `  Nothing to embed: all books have valid embeddings for model 'X'.` (kept, indented) |
| `No books found in database...` | `  No books found. Run crawler first.` (kept, indented) |
| Batch error print | `  Error generating embeddings for batch starting at book_id N: ...` (indented) |

---

### ranker.py

#### [MODIFY] [ranker.py](file:///Users/alex/Documents/goodreads-ranker/ranker.py)

**Taste calibration block** (~line 841–862) — replace the summary print with a
per-friend breakdown. Requires a DB lookup to resolve `list_id → username` from
`friend_lists`. `similar_friends` (list of included list_ids) and `friend_scores`
(DataFrame with `list_id`, `overlap_count`) are both in scope at that point.

New output:
```
  Taste calibration
    My ratings (55)
    alice - 12345 (210 books)
    bob   - 67890 (180 books)
```

Where the count is `overlap_count` from `friend_scores` — real shared ratings only,
no synthetic data.

> [!IMPORTANT]
> `similar_friends` is the filtered list (correlation threshold passed). Only these appear in the printout.

**All other inner prints in `run_ranking`** — add two-space indent:

| Before | After |
|--------|-------|
| `Running ELO ratings refinement...` | `  Running ELO ratings refinement...` |
| `Loading valid embeddings...` | `  Loading valid embeddings...` |
| `Excluding N books from model inputs...` | `  Excluding N books from model inputs (...)` |
| `No valid embeddings found...` | `  No valid embeddings found. Run embedder before ranking.` |
| `Running graph GCN propagation...` | `  Running graph GCN propagation...` |
| `Tuning hyperparameters via Nevergrad...` | `  Tuning hyperparameters via Nevergrad (budget=200)...` |
| `Using stored/default hyperparameters...` | `  Using stored/default hyperparameters...` |
| `Running ensemble models...` | `  Running ensemble models...` |
| `Formulating final recommendations...` | `  Formulating final recommendations...` |
| `Saving predictions to database...` | `  Saving predictions to database...` |
| `Successfully computed and saved predictions for N books.` | `  Saved predictions for N books.` |

---

## Verification Plan

### Manual Verification
- Run `python3 cli.py run_pipeline --limit 99` and visually confirm the new output
  matches the agreed format (indented sub-lines, concise skip messages, calibration
  breakdown with usernames).
- Run `python3 cli.py embed` on a fresh DB to confirm the Ollama startup lines
  still appear (with indent) and the dropped lines are gone.
- Run `python3 cli.py seed` with data already seeded to confirm the new skip messages.
