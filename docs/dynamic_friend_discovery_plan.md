# Dynamic Friend Discovery — Final Implementation Plan

Replace the hard-coded `DEFAULT_FRIEND_LIST_IDS` with a live scraping flow: login →
extract main user from homepage → paginate through `/friend/user/{user_id}` → populate
`readers` table dynamically.

---

## Decisions Locked In

| Decision | Resolution |
|---|---|
| Main-user storage | `is_self INTEGER DEFAULT 0` flag column in `readers` |
| `href` → `user_id` | Replace; IDs stored as plain integers (numeric prefix only) |
| Migration of old `href` data | Leave as-is; re-run of `seed --friends` fills in `user_id` |
| Friends-page pagination | Fully paginate using `a.next_page` |
| New friends on re-run | `INSERT OR IGNORE` — new friends inserted, scrape metadata preserved |
| Removed friends | Never deleted — their ratings remain valid model data |
| Homepage parse failure | Hard `RuntimeError` — no silent fallback |
| Friends with 0 books | Skip — do not insert into `readers` |
| CLI override | `--list-ids` (comma-separated) bypasses discovery entirely |

---

## Table Renames

| Old name | New name |
|---|---|
| `friend_lists` | `readers` |
| `friend_ratings` | `reader_libraries` |
| `user_library` | `my_library` |

---

## Proposed Changes

### DB Layer — `goodreads_ranker/db.py`

#### [MODIFY] [db.py](file:///Users/alex/Documents/goodreads-ranker/goodreads_ranker/db.py)

**1. Rename tables in `SCHEMA`:**

```diff
-CREATE TABLE IF NOT EXISTS user_library (
+CREATE TABLE IF NOT EXISTS my_library (
     book_id       INTEGER PRIMARY KEY,
     ...
 );

-CREATE TABLE IF NOT EXISTS friend_lists (
+CREATE TABLE IF NOT EXISTS readers (
     list_id            INTEGER PRIMARY KEY,
     username           TEXT,
-    href               TEXT,
+    user_id            INTEGER,
+    is_self            INTEGER DEFAULT 0,
     scrape_complete    INTEGER DEFAULT 0,
     date_last_scraped  TEXT,
     scrape_error       TEXT
 );

-CREATE TABLE IF NOT EXISTS friend_ratings (
+CREATE TABLE IF NOT EXISTS reader_libraries (
     list_id    INTEGER NOT NULL,
     book_id    INTEGER NOT NULL,
     rating     INTEGER,
     date_read  TEXT,
     date_added TEXT,
     PRIMARY KEY (list_id, book_id)
 );
```

**2. `ensure_schema_compat` — renames + new columns:**

```python
# Rename tables if old names still exist
if "user_library" in existing and "my_library" not in existing:
    db_conn.execute("ALTER TABLE user_library RENAME TO my_library")
if "friend_lists" in existing and "readers" not in existing:
    db_conn.execute("ALTER TABLE friend_lists RENAME TO readers")
if "friend_ratings" in existing and "reader_libraries" not in existing:
    db_conn.execute("ALTER TABLE friend_ratings RENAME TO reader_libraries")

# Add new columns to readers (old href stays as dead column — no DROP COLUMN)
if "readers" in existing or "friend_lists" in existing:
    ensure_column(db_conn, "readers", "username", "TEXT")
    ensure_column(db_conn, "readers", "user_id", "INTEGER")
    ensure_column(db_conn, "readers", "is_self", "INTEGER DEFAULT 0")
    ensure_column(db_conn, "readers", "scrape_error", "TEXT")
```

> [!NOTE]
> `user_id` is stored as `INTEGER` — the non-numeric postfix (e.g. `-alex-y-shutov`) is stripped before storage. `list_id` and `user_id` are kept as separate columns even though they coincide for most users.

**3. Update all hardcoded table references** in helper functions (`upsert_rows`, `save_embeddings`, etc. — none reference table names directly; all callers pass the table name). Audit all callers in `seeder.py`, `ranker.py`, `embedder.py`, `crawler.py`.

---

### Seeder — `goodreads_ranker/seeder.py`

#### [MODIFY] [seeder.py](file:///Users/alex/Documents/goodreads-ranker/goodreads_ranker/seeder.py)

**1. Remove `DEFAULT_FRIEND_LIST_IDS`** — deleted entirely.

**2. Add `extract_main_user(page) -> (user_id: int, list_id: int, username: str)`**

Called immediately after login while still on the Goodreads homepage.

- `user_id`: parse `href` from `.dropdown__trigger--profileMenu` → `/user/show/{slug}` → strip to leading integer.
- `username`: `alt` attribute of the `img` inside that same anchor.
- `list_id`: parse `href` from `li.siteHeader__topLevelItem a` where text == "My Books" → `/review/list/{slug}` → strip to leading integer.
- Raises `RuntimeError` if any of the three values cannot be extracted.

Helper for stripping slug to int:
```python
def parse_id_from_slug(slug: str) -> int:
    """Extracts leading integer from slugs like '89208838-alex-y-shutov'."""
    return int(re.match(r"(\d+)", slug.lstrip("/").split("/")[-1]).group(1))
```

**3. Add `fetch_friends(page, user_id: int) -> list[dict]`**

- Navigates to `https://www.goodreads.com/friend/user/{user_id}` (page 1).
- Scrapes `#friendTable tbody tr`:
  - `user_id`: leading int from first `td a[href^="/user/show/"]`.
  - `username`: text of the `rel="acquaintance"` anchor.
  - `list_id`: leading int from `a[href^="/review/list/"]`.
  - `book_count`: integer parsed from the text of the `/review/list/` anchor (e.g. `"33 books"` → `33`).
  - **Skip rows where `book_count == 0`.**
- Follows `a.next_page` pagination until no next link (same pattern as `process_list`).
- Returns `list[dict]` with keys `user_id`, `username`, `list_id`.

**4. Add `upsert_readers(db_conn, main_user: dict, friends: list[dict])`**

```python
# Main user row — INSERT OR REPLACE so user_id/username stay current
db_conn.execute("""
    INSERT OR REPLACE INTO readers (list_id, user_id, username, is_self)
    VALUES (?, ?, ?, 1)
""", (main_user["list_id"], main_user["user_id"], main_user["username"]))

# Friend rows — INSERT OR IGNORE to preserve existing scrape metadata
db_conn.executemany("""
    INSERT OR IGNORE INTO readers (list_id, user_id, username, is_self)
    VALUES (?, ?, ?, 0)
""", [(f["list_id"], f["user_id"], f["username"]) for f in friends])
db_conn.commit()
```

**5. Refactor `scrape_friend_ratings` → new signature:**

```python
async def scrape_friend_ratings(db_path=None, list_ids=None, force_all=False):
```

**Dynamic mode** (`list_ids is None`):
1. Launch browser, login via `login_to_goodreads`.
2. Call `extract_main_user(page)` → raises on failure.
3. Call `fetch_friends(page, main_user["user_id"])`.
4. Call `upsert_readers(db_conn, main_user, friends)`.
5. Query `readers WHERE is_self = 0` for IDs to scrape (filtered by `force_all` / `scrape_complete`).

**Override mode** (`list_ids` provided):
- Insert those IDs directly via `INSERT OR IGNORE INTO readers (list_id, is_self) VALUES (?, 0)`.
- Skip discovery entirely.
- Then proceed to scrape as before.

**6. Update all internal table name references:**

- `INSERT OR IGNORE INTO friend_lists` → `readers`
- `SELECT ... FROM friend_lists` → `readers`
- `UPDATE friend_lists` → `readers`
- `upsert_rows(..., "friend_ratings", ...)` → `"reader_libraries"`

---

### CLI — `main.py`

#### [MODIFY] [main.py](file:///Users/alex/Documents/goodreads-ranker/main.py)

`seed()` gains `list_ids` parameter:

```python
def seed(self, user=None, friends=None, force=False, list_ids=None):
    ...
    if friends:
        parsed_ids = None
        if list_ids is not None:
            if isinstance(list_ids, str):
                parsed_ids = [int(x.strip()) for x in list_ids.split(",") if x.strip()]
            else:
                parsed_ids = [int(x) for x in list_ids]
        asyncio.run(seeder.scrape_friend_ratings(list_ids=parsed_ids, force_all=force))
```

---

### Audit — All other files

#### `crawler.py`, `embedder.py`, `ranker.py`

Search for hardcoded references to `user_library`, `friend_lists`, `friend_ratings` and update to `my_library`, `readers`, `reader_libraries`.

---

## Verification Plan

### Manual Verification

1. **Fresh DB**: Delete `data/goodreads.db`, run `python main.py seed`, confirm:
   - `readers` has one `is_self=1` row with correct `list_id`, `user_id`, `username`.
   - Friend rows have correct data; friends with 0 books are absent.
2. **Override**: Run `python main.py seed --list-ids=104343033,104945614`, confirm only those IDs are inserted.
3. **Migration**: Keep existing DB, run `init_db()`, confirm `readers`, `my_library`, `reader_libraries` tables exist with data intact.
4. **Idempotency**: Run `seed --friends` twice; confirm scrape metadata is not overwritten for existing rows.
5. **Pagination**: Verify a user with >30 friends gets all pages scraped.
