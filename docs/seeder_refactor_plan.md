# seeder.py Refactor: Concurrent Tabs + Session Persistence + Resilience

## Objective

`seeder.py` must keep using Playwright (real browser) because Goodreads sits behind an
AWS WAF JS proof-of-work challenge that only a JS-capable client can solve. Within that
constraint, refactor `seeder.py` to:

1. Parallelize friend-library scraping across multiple tabs in one authenticated
   browser context.
2. Persist the authenticated session across CLI runs to avoid repeated logins.
3. Make page navigation resilient via a shared retry helper.
4. Reduce request burstiness and page-load cost.
5. Simplify progress reporting to fit a concurrent model.

`crawler.py` and `db.py` are not modified by this plan. `main.py` is not modified —
no new CLI flags are being added; concurrency is a hardcoded constant tuned by hand.

## Non-Goals / Explicitly Deferred

Do not implement these. They were discussed and deliberately deferred:

- No `--concurrency` CLI flag — concurrency is a hardcoded module constant.
- No dedicated WAF-challenge detection function (e.g. `is_waf_challenge_page()`).
  Recovery is handled generically by the navigation retry helper.
- No page-level checkpointing / mid-list resume. A failed list still restarts from
  page 1 on the next run (existing behavior via `readers.scrape_complete` /
  `scrape_error`).
- No `--force_login` flag to bypass a saved session. If a saved session needs to be
  discarded, delete `data/storage_state.json` manually.
- No proactive session-validity check after loading `storage_state`. Validity is
  discovered reactively via the existing `is_login_page()` checks the first time a
  page is navigated.
- No changes to headless vs. headed browser launch (`headless=True` stays).
- No retry wrapping of the click-triggered pagination navigation in `process_list`
  (the `next_button.click()` + `page.expect_navigation()` pattern). Only plain
  `page.goto(...)` call sites get the retry helper in this pass.
- No changes to `main.py` or the DB schema.

## New Module-Level Constants (seeder.py)

```python
import random  # new import

DATA_DIR = Path("data")                        # same convention as db.DB_PATH
STORAGE_STATE_PATH = DATA_DIR / "storage_state.json"

CONCURRENCY = 3                                 # hardcoded; tune by hand, no CLI flag

NAV_RETRY_ATTEMPTS = 3
NAV_RETRY_BASE_DELAY = 2.0                      # seconds; multiplied by attempt number

JITTER_MIN = 0.5                                # seconds
JITTER_MAX = 2.0                                # seconds

BLOCKED_RESOURCE_TYPES = {"image", "font", "media"}
```

## New Helper Functions

### `async def goto_with_retry(page, url, wait_until="domcontentloaded", retries=NAV_RETRY_ATTEMPTS)`

Replaces bare `await page.goto(url, wait_until=...)` calls. Retries on
`PlaywrightTimeoutError` (and any other navigation-related exception `page.goto` can
raise, e.g. `playwright.async_api.Error` for network errors) with linear backoff
(`NAV_RETRY_BASE_DELAY * (attempt + 1)` seconds between attempts). Re-raises the last
exception if all attempts are exhausted. This is the mechanism that absorbs transient
WAF re-checks and network blips — no separate challenge-detection logic needed; giving
the page more time/attempts is sufficient since Playwright's JS engine solves the
proof-of-work automatically once the page has loaded.

Apply this helper at every plain `page.goto(...)` call site (see "Modified Functions"
below for the full list). Do **not** apply it to the click+`expect_navigation()`
pagination pattern in `process_list`.

### `async def save_storage_state(context)`

```python
async def save_storage_state(context):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    await context.storage_state(path=str(STORAGE_STATE_PATH))
```

Called once, immediately after `login_to_goodreads()` succeeds (see below) — this is
the only place it needs to be called, since every login path (initial sign-in and any
reactive re-login) routes through `login_to_goodreads()`.

### Resource-blocking route handler

Not a standalone function necessarily, but a small `route()` callback registered once
on the **context** (not per-page — `context.route()` applies to all current and future
pages created from that context, so one registration covers the setup page and all
worker pages):

```python
async def _block_heavy_resources(route):
    if route.request.resource_type in BLOCKED_RESOURCE_TYPES:
        await route.abort()
    else:
        await route.continue_()
```

Register with `await context.route("**/*", _block_heavy_resources)` immediately after
`context` is created, before any page navigates anywhere.

## Modified Functions

### `login_to_goodreads(page, email, password)`

No signature change. After `wait_for_post_login(page)` completes successfully (i.e. at
the very end of the function, on the success path), add:

```python
await save_storage_state(page.context)
```

Any internal plain `await page.goto(...)` calls in this function (there's currently one,
to `SIGNIN_URL`) should be routed through `goto_with_retry`.

### `fetch_friends(page, user_id, email=None, password=None)`

- Replace `await page.goto(url, wait_until="domcontentloaded")` (both occurrences —
  initial navigation and the reactive re-login retry navigation) with
  `await goto_with_retry(page, url)`.
- Replace `await asyncio.sleep(1)` in the pagination loop with
  `await asyncio.sleep(random.uniform(JITTER_MIN, JITTER_MAX))`.
- No other logic changes.

### `open_list_page(page, list_id, email, password)`

- Replace both `await page.goto(url, ...)` calls (initial navigation and the
  post-reactive-login navigation) with `await goto_with_retry(page, url)`.
- No other logic changes.

### `ensure_list_page(page, target_url, email, password)`

- Replace `await page.goto(target_url, ...)` with
  `await goto_with_retry(page, target_url)`.
- No other logic changes.

### `process_list(db_conn, page, list_id, email, password, force_seed=False)`

- Replace `await asyncio.sleep(1)` in the pagination loop with
  `await asyncio.sleep(random.uniform(JITTER_MIN, JITTER_MAX))`.
- **Remove** the per-list nested `tqdm(...)` progress bar entirely (the
  `with tqdm(total=total_pages, ...) as pbar:` block and its `pbar.update(1)` /
  `pbar.leave = False` calls). This bar doesn't generalize to N concurrent workers
  scraping different lists at once.
- Replace the page-level detail it used to convey via the progress bar with plain
  `tqdm.write(...)` log lines, e.g. on each page processed:
  `tqdm.write(f"  [{list_id}] page {page_num} done ({len(page_rows)} rows)")`.
  Keep the existing `tqdm.write("  P1 unchanged, stopping early")` and error-write
  calls as-is.
- The click+`expect_navigation()` pagination navigation stays unwrapped (see
  Non-Goals).
- Signature and return behavior (raises on failure) unchanged — the caller (the new
  worker function, see below) is responsible for catching exceptions per list, same
  as today's top-level loop does.

### `scrape_reader_libraries(db_path=None, list_ids=None, force_seed=False)`

This function is substantially rewritten. New flow:

1. `load_dotenv()`, `db.init_db(db_path)`, read `GOODREADS_EMAIL` / `GOODREADS_PASSWORD`
   — unchanged.
2. Open `db.get_connection(db_path)` — unchanged.
3. `async with async_playwright() as p:` launch `browser = await p.chromium.launch(headless=True)`
   — unchanged.
4. Create the context with conditional `storage_state`:
   ```python
   context_kwargs = {"user_agent": USER_AGENT}
   if STORAGE_STATE_PATH.exists():
       context_kwargs["storage_state"] = str(STORAGE_STATE_PATH)
   context = await browser.new_context(**context_kwargs)
   await context.route("**/*", _block_heavy_resources)
   ```
5. Create the setup page: `setup_page = await context.new_page()`.
6. Branch on `list_ids` exactly as today:
   - If `list_ids is not None`: upsert them into `readers` as before (unchanged DB
     logic). No login/friends-fetch step is forced here — if a page later turns out
     to require login (loaded session invalid or no session existed), the existing
     reactive `is_login_page()` checks inside `open_list_page`/`ensure_list_page`
     will call `login_to_goodreads`, which will itself save a fresh
     `storage_state` on success.
   - Else (no explicit `list_ids`): do the full flow on `setup_page` —
     `goto_with_retry(setup_page, SIGNIN_URL)`, `login_to_goodreads(setup_page, email, password)`,
     `extract_main_user(setup_page)`, `fetch_friends(setup_page, ...)`,
     `upsert_readers(...)`. Unchanged logic, just routed through `goto_with_retry`
     where it navigates.
7. `to_scrape = get_lists_to_scrape(db_conn, force_seed)` — unchanged.
8. If `not to_scrape`: print the existing "no new friends" message, close the
   browser, return — unchanged.
9. **New: build the work queue and worker pool.**
   ```python
   queue = asyncio.Queue()
   for list_id in to_scrape:
       queue.put_nowait(list_id)

   worker_count = min(CONCURRENCY, len(to_scrape))

   # setup_page is reused as worker #1 — do not close/discard it.
   worker_pages = [setup_page]
   for _ in range(worker_count - 1):
       worker_pages.append(await context.new_page())

   pbar = tqdm(total=len(to_scrape), desc="Scraping friend libraries", unit="list")

   async def list_worker(page):
       while True:
           try:
               list_id = queue.get_nowait()
           except asyncio.QueueEmpty:
               return
           try:
               await process_list(db_conn, page, list_id, email, password, force_seed=force_seed)
           except Exception as e:
               tqdm.write(f"  Failed list {list_id}: {e}")
               mark_list_failed(db_conn, list_id, e)
               with contextlib.suppress(Exception):
                   await page.goto("about:blank")
           finally:
               pbar.update(1)

   await asyncio.gather(*(list_worker(p) for p in worker_pages))
   pbar.close()
   ```
   This replaces the current flat `for list_id in to_scrape:` loop and its inline
   try/except — that per-list error isolation logic moves into `list_worker` verbatim
   (same behavior: catch, log, `mark_list_failed`, blank the page, continue).
10. `await browser.close()` — unchanged, at the very end.

The existing `print(f"  Scraping {len(to_scrape)} list(s)...")` line can be dropped
since the new `pbar` conveys the same information.

## Concurrency & Correctness Notes (for the implementer, not action items)

- `db_conn` (a single `sqlite3.Connection`) is shared across all worker coroutines.
  This is safe as-is: Python's `sqlite3` calls in this codebase are synchronous and
  none of `db.py`'s helper functions `await` mid-transaction, so under asyncio's
  single-threaded cooperative scheduling no two workers can interleave a partial
  DB write. No locks, queues, or connection-per-worker changes are needed for the DB
  layer.
- `context.route()` registered before any `new_page()` calls covers pages created
  later (including the worker pages spun up in step 9), so it does not need to be
  re-registered per page.
- Cookies/session state are shared automatically across all pages in the same
  `context` — no explicit cookie-copying is needed between the setup page and worker
  pages.

## File/Data Changes

- New file created at runtime: `data/storage_state.json` (git-ignored the same way
  `data/goodreads.db` presumably already is — confirm `.gitignore` covers `data/` or
  add an entry for this file specifically).
- No SQLite schema changes.

## Testing / Validation Checklist

1. **Fresh run, no saved session**: delete `data/storage_state.json` if present, run
   `seed` with no `list_ids`. Confirm login happens once on the setup page, friends
   are fetched, `data/storage_state.json` is created, and worker pages successfully
   scrape lists concurrently (watch for interleaved `tqdm.write` output from
   different `list_id`s).
2. **Warm run, saved session valid**: run `seed` again immediately after (1). Confirm
   no login/signin navigation occurs and scraping proceeds directly (or only rescans
   for `scrape_complete != 1`, per existing semantics).
3. **Stale session**: manually corrupt/expire `data/storage_state.json` (or wait for
   real expiry) and run `seed`. Confirm `is_login_page()` correctly detects this on
   first navigation, `login_to_goodreads` re-runs, and a fresh `storage_state.json`
   is written.
4. **`--list_ids` path**: run `seed --list_ids=...` explicitly. Confirm it skips the
   friends-fetch flow, still uses the worker pool, and still reactively logs in if
   the loaded/absent session requires it.
5. **Transient navigation failure**: verify `goto_with_retry` actually retries (e.g.
   temporarily point at an invalid host or simulate a timeout) and that it raises
   after `NAV_RETRY_ATTEMPTS` exhausted rather than hanging indefinitely.
6. **Concurrency edge case**: run with `to_scrape` containing fewer entries than
   `CONCURRENCY` (e.g. 1-2 lists) and confirm only `min(CONCURRENCY, len(to_scrape))`
   pages are created — no idle/unused worker pages.
7. **Resource blocking sanity check**: confirm pages still render the DOM elements
   `process_list`/`fetch_friends` depend on (`#booksBody`, `#friendTable`, etc.) with
   images/fonts/media blocked — i.e. blocking doesn't inadvertently break a selector
   that depends on a blocked resource loading first.
8. **Progress bar output**: confirm the single overall `tqdm` bar advances correctly
   as each of the `worker_count` workers completes lists, and that `tqdm.write` lines
   from concurrent workers don't corrupt the bar rendering.
9. **Full pipeline**: run `run_pipeline` end-to-end (`seed` → `crawl` → `embed` →
   `rank`) to confirm nothing downstream broke.
