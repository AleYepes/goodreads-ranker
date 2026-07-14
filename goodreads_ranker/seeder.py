import asyncio
import contextlib
import random
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright
from tqdm import tqdm

from . import config, db
from .utils import USER_AGENT, clean_text, parse_id_from_slug

SIGNIN_URL = "https://www.goodreads.com/ap/signin?language=en_US&openid.assoc_handle=amzn_goodreads_web_na&openid.claimed_id=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.identity=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.mode=checkid_setup&openid.ns=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0&openid.pape.max_auth_age=0&openid.return_to=https%3A%2F%2Fwww.goodreads.com%2Fap-handler%2Fsign-in"

RATING_MAP = {
    "it was amazing": 5,
    "really liked it": 4,
    "liked it": 3,
    "it was ok": 2,
    "did not like it": 1,
}

DATA_DIR = Path("data")
STORAGE_STATE_PATH = DATA_DIR / "storage_state.json"

CONCURRENCY = 3

NAV_RETRY_ATTEMPTS = 3
NAV_RETRY_BASE_DELAY = 2.0

JITTER_MIN = 0.5
JITTER_MAX = 2.0

BLOCKED_RESOURCE_TYPES = {"image", "font", "media"}


async def goto_with_retry(page, url, wait_until="domcontentloaded", retries=NAV_RETRY_ATTEMPTS):
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            await page.goto(url, wait_until=wait_until)
            return
        except (PlaywrightTimeoutError, PlaywrightError) as exc:
            last_exc = exc
            delay = NAV_RETRY_BASE_DELAY * (attempt + 1)
            tqdm.write(f"Attempt {attempt + 1}/{retries} failed for {url!r}: {exc}. Retrying in {delay:.1f}s...")
            await asyncio.sleep(delay)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Navigation failed without an exception")


async def save_storage_state(context):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    await context.storage_state(path=str(STORAGE_STATE_PATH))


async def _block_heavy_resources(route):
    if route.request.resource_type in BLOCKED_RESOURCE_TYPES:
        await route.abort()
    else:
        await route.continue_()


async def is_login_page(page):
    return (
        "/ap/signin" in page.url
        or await page.locator('input[type="email"]').count() > 0
        or await page.locator("button.authPortalSignInButton").count() > 0
    )


async def login_to_goodreads(page, email, password):
    async def wait_for_post_login(page):
        for selector in (".homePrimaryColumn",):
            try:
                await page.wait_for_selector(selector, timeout=15000)
                return
            except PlaywrightTimeoutError:
                continue

        if await is_login_page(page):
            raise RuntimeError("Goodreads login did not complete successfully.")

    if not email or not password:
        raise RuntimeError("GOODREADS_EMAIL and GOODREADS_PASSWORD must be set to scrape private review lists.")

    if await page.locator("button.authPortalSignInButton").count() > 0:
        async with page.expect_navigation(wait_until="domcontentloaded"):
            await page.locator("button.authPortalSignInButton").click()
    elif await page.locator('input[type="email"]').count() == 0:
        await goto_with_retry(page, SIGNIN_URL)

    await page.wait_for_selector('input[type="email"]', timeout=30000)
    await page.fill('input[type="email"]', email)
    await page.fill('input[type="password"]', password)
    await page.click('input[type="submit"]')
    await page.wait_for_load_state("domcontentloaded")
    await wait_for_post_login(page)

    await save_storage_state(page.context)


async def extract_main_user(page):
    profile_el = page.locator("a.dropdown__trigger--profileMenu").first
    if await profile_el.count() == 0:
        raise RuntimeError("Could not find user profile on homepage.")

    my_books_el = page.locator('li.siteHeader__topLevelItem a:has-text("My Books")').first
    if await my_books_el.count() == 0:
        raise RuntimeError("Could not find book links on homepage.")

    profile_href = await profile_el.get_attribute("href") or ""
    my_books_href = await my_books_el.get_attribute("href") or ""
    img_el = profile_el.locator("img")
    username = await img_el.get_attribute("alt") if await img_el.count() > 0 else ""

    if not profile_href or not my_books_href or not username:
        raise RuntimeError("User profile metadata extraction failed.")

    user_id = parse_id_from_slug(profile_href)
    library_id = parse_id_from_slug(my_books_href)

    return {"user_id": user_id, "library_id": library_id, "username": username}


async def fetch_friends(page, user_id: int, email: str | None = None, password: str | None = None) -> list[dict]:
    friends = []
    page_num = 1
    target_url = f"https://www.goodreads.com/friend/user/{user_id}"

    while True:
        url = f"{target_url}?page={page_num}"
        await goto_with_retry(page, url)

        if await is_login_page(page):
            print("Redirected to login page — re-authenticating...")
            if not email or not password:
                raise RuntimeError("Session lost on friends page and no credentials available to re-login.")
            await login_to_goodreads(page, email, password)
            await goto_with_retry(page, url)

        try:
            await page.wait_for_selector("#friendTable", timeout=10000)
        except PlaywrightTimeoutError:
            print(f"#friendTable element not found on page {page_num} (current URL: {page.url}). Stopping pagination.")
            break

        rows = await page.locator("#friendTable tbody tr").all()
        if not rows:
            break

        for row in rows:
            user_anchor = row.locator('a[href^="/user/show/"][rel="acquaintance"]').first
            list_anchor = row.locator('a[href^="/review/list/"]').first

            if await user_anchor.count() == 0 or await list_anchor.count() == 0:
                continue

            user_href = await user_anchor.get_attribute("href")
            list_href = await list_anchor.get_attribute("href")
            username = clean_text(await user_anchor.inner_text())

            if not user_href or not list_href:
                continue

            book_text = await list_anchor.inner_text()
            book_text_clean = book_text.replace(",", "")
            book_match = re.search(r"(\d+)\s+books?", book_text_clean, re.IGNORECASE)
            book_count = int(book_match.group(1)) if book_match else 0

            if book_count == 0:
                continue

            friends.append(
                {
                    "user_id": parse_id_from_slug(user_href),
                    "library_id": parse_id_from_slug(list_href),
                    "username": username,
                }
            )

        next_btn = page.locator("a.next_page").first
        if await next_btn.count() > 0 and "disabled" not in (await next_btn.get_attribute("class") or ""):
            page_num += 1
            await asyncio.sleep(random.uniform(JITTER_MIN, JITTER_MAX))
        else:
            break

    print(f"Found {len(friends)} friend(s) with books.")
    return friends


def upsert_libraries(db_conn, main_user: dict, friends: list[dict]):
    db_conn.execute(
        "UPDATE libraries SET is_main = 0 WHERE is_main = 1 AND legacy_id != ?",
        (main_user["library_id"],),
    )

    db_conn.execute(
        """
        INSERT OR IGNORE INTO libraries (legacy_id, user_id, username, is_main, scrape_complete)
        VALUES (?, ?, ?, 1, 0)
        """,
        (main_user["library_id"], main_user["user_id"], main_user["username"]),
    )
    db_conn.execute(
        """
        UPDATE libraries
        SET user_id = ?, username = ?, is_main = 1
        WHERE legacy_id = ?
        """,
        (main_user["user_id"], main_user["username"], main_user["library_id"]),
    )

    db_conn.executemany(
        """
        INSERT OR IGNORE INTO libraries (legacy_id, user_id, username, is_main, scrape_complete)
        VALUES (?, ?, ?, 0, 0)
        """,
        [(f["library_id"], f["user_id"], f["username"]) for f in friends],
    )
    db_conn.commit()


def update_friend_info(db_conn, library_id, username, user_id):
    db_conn.execute(
        """
        UPDATE libraries
        SET username = ?,
            user_id = ?
        WHERE legacy_id = ?
        """,
        (username, user_id, library_id),
    )
    db_conn.commit()


def load_existing_rows(db_conn, library_id):
    rows = db_conn.execute(
        """
        SELECT library_id, book_legacy_id, rating, date_read, date_added
        FROM library_books
        WHERE library_id = ?
        """,
        (library_id,),
    ).fetchall()
    return {
        (int(row["library_id"]), int(row["book_legacy_id"])): {
            "library_id": int(row["library_id"]),
            "book_legacy_id": int(row["book_legacy_id"]),
            "rating": int(row["rating"] or 0),
            "date_read": row["date_read"] or "",
            "date_added": row["date_added"] or "",
        }
        for row in rows
    }


def mark_library_complete(db_conn, library_id):
    today = datetime.now().strftime("%Y-%m-%d")
    db_conn.execute(
        """
        UPDATE libraries
        SET scrape_complete = 1,
            date_scraped = ?,
            scrape_error = NULL
        WHERE legacy_id = ?
        """,
        (today, library_id),
    )
    db_conn.commit()


def mark_library_failed(db_conn, library_id, error):
    db_conn.execute(
        """
        UPDATE libraries
        SET scrape_complete = 0,
            scrape_error = ?
        WHERE legacy_id = ?
        """,
        (str(error)[:1000], library_id),
    )
    db_conn.commit()


def upsert_extracted(db_conn, rows):
    if not rows:
        return
    db.upsert_rows(
        db_conn,
        "library_books",
        [
            (
                row["library_id"],
                row["book_legacy_id"],
                row["rating"],
                row["date_read"],
                row["date_added"],
            )
            for row in rows
        ],
        [
            "library_id",
            "book_legacy_id",
            "rating",
            "date_read",
            "date_added",
        ],
    )


async def open_list_page(page, list_id, email, password):
    url = f"https://www.goodreads.com/review/list/{list_id}?print=true"
    await goto_with_retry(page, url)

    if await is_login_page(page):
        print("Login required. Authenticating session...")
        await login_to_goodreads(page, email, password)
        await goto_with_retry(page, url)

    return url


async def ensure_list_page(page, target_url, email, password):
    if await is_login_page(page):
        print("Login required. Authenticating session...")
        await login_to_goodreads(page, email, password)
        await goto_with_retry(page, target_url)


async def extract_friend_row(row, library_id):
    title_el = await row.query_selector(".field.title a")
    href = await title_el.get_attribute("href") if title_el else ""
    legacy_id_match = re.search(r"/book/show/(\d+)", href)
    legacy_id = int(legacy_id_match.group(1)) if legacy_id_match else None
    if not legacy_id:
        return None

    rating_container = await row.query_selector(".field.rating")
    rating = 0

    if rating_container:
        static_stars = await rating_container.query_selector(
            ".staticStars"
        )  # for read-only staticStars (friend's rating)
        if static_stars:
            rating_text = await static_stars.get_attribute("title") or ""
            rating = RATING_MAP.get(rating_text, 0)
        else:
            stars_el = await rating_container.query_selector(".stars")  # for editable stars (own rating)
            if stars_el:
                data_rating = await stars_el.get_attribute("data-rating")
                if data_rating and data_rating.isdigit():
                    rating = int(data_rating)

    date_read_el = await row.query_selector(".field.date_read .date_read_value")
    date_read = clean_text(await date_read_el.inner_text()) if date_read_el else ""

    date_added_el = await row.query_selector(".field.date_added span")
    date_added = ""
    if date_added_el:
        date_added = await date_added_el.get_attribute("title") or ""
        if not date_added:
            date_added = clean_text(await date_added_el.inner_text())

    def format_date(dt_str):
        if not dt_str:
            return ""
        try:
            from dateutil import parser

            return parser.parse(dt_str.strip()).strftime("%Y-%m-%d")
        except Exception:
            return dt_str.strip()

    return {
        "library_id": int(library_id),
        "book_legacy_id": legacy_id,
        "rating": rating,
        "date_read": format_date(date_read),
        "date_added": format_date(date_added),
    }


async def process_library(db_conn, page, library_id, email, password, force_seed=False):
    page_num = 1
    total_rows = 0
    valid_page_parsed = False
    target_url = await open_list_page(page, library_id, email, password)

    with contextlib.suppress(PlaywrightTimeoutError):
        await page.wait_for_selector("h1", timeout=10000)

    h1_el = await page.query_selector("h1")
    username = None
    user_id = None
    if h1_el:
        links = await h1_el.query_selector_all("a")
        for link in links:
            link_href = await link.get_attribute("href")
            if link_href and "/user/show/" in link_href:
                if not user_id:
                    user_id = parse_id_from_slug(link_href)
                text = clean_text(await link.inner_text())
                if text:
                    username = text
    if username or user_id:
        update_friend_info(db_conn, library_id, username, user_id)

    existing_rows = load_existing_rows(db_conn, library_id)

    while True:
        await ensure_list_page(page, target_url, email, password)
        await page.wait_for_selector("#booksBody", timeout=10000)

        rows = await page.query_selector_all("tr.bookalike.review")
        page_rows = []
        page_all_known = bool(rows)
        for row in rows:
            try:
                extracted = await extract_friend_row(row, library_id)
                if not extracted:
                    page_all_known = False
                    continue

                key = (extracted["library_id"], extracted["book_legacy_id"])
                if existing_rows.get(key) != extracted:
                    page_all_known = False
                existing_rows[key] = extracted
                page_rows.append(extracted)
            except Exception as e:
                page_all_known = False
                tqdm.write(f"Error parsing book in library {library_id}: {e}")

        if not page_rows:
            raise RuntimeError(f"No valid rows parsed on page {page_num}")

        valid_page_parsed = True
        total_rows += len(page_rows)
        upsert_extracted(db_conn, page_rows)

        if page_all_known and not force_seed:
            break

        next_button = await page.query_selector("a.next_page")
        next_class = await next_button.get_attribute("class") if next_button else ""
        next_href = await next_button.get_attribute("href") if next_button else None
        if next_button and next_href and "disabled" not in next_class:
            target_url = urljoin(page.url, next_href)
            async with page.expect_navigation():
                await next_button.click()
            page_num += 1
            await asyncio.sleep(random.uniform(JITTER_MIN, JITTER_MAX))
            continue

        break

    if not valid_page_parsed:
        raise RuntimeError("No valid list page was parsed")

    mark_library_complete(db_conn, library_id)


def get_libraries_to_scrape(db_conn, force_seed):
    if force_seed:
        cursor = db_conn.execute("SELECT legacy_id FROM libraries")
    else:
        cursor = db_conn.execute(
            """
            SELECT legacy_id
            FROM libraries
            WHERE scrape_complete != 1
            """
        )
    return [row["legacy_id"] for row in cursor.fetchall()]


async def scrape_libraries(db_path=None, library_ids=None, force_seed=False):
    db.init_db(db_path)
    email = config.get_goodreads_email()
    password = config.get_goodreads_password()

    with db.get_connection(db_path) as db_conn:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)

            if STORAGE_STATE_PATH.exists():
                context = await browser.new_context(
                    user_agent=USER_AGENT,
                    storage_state=str(STORAGE_STATE_PATH),
                )
            else:
                context = await browser.new_context(user_agent=USER_AGENT)
            await context.route("**/*", _block_heavy_resources)

            setup_page = await context.new_page()

            if library_ids is not None:
                for lid in library_ids:
                    db_conn.execute(
                        "INSERT OR IGNORE INTO libraries (legacy_id, is_main, scrape_complete) VALUES (?, 0, 0)",
                        (lid,),
                    )
                    db_conn.execute(
                        "UPDATE libraries SET scrape_complete = 0 WHERE legacy_id = ?",
                        (lid,),
                    )
                db_conn.commit()
            else:
                await goto_with_retry(setup_page, SIGNIN_URL)
                await login_to_goodreads(setup_page, email, password)
                main_user = await extract_main_user(setup_page)
                friends = await fetch_friends(setup_page, main_user["user_id"], email=email, password=password)
                upsert_libraries(db_conn, main_user, friends)

            to_scrape = get_libraries_to_scrape(db_conn, force_seed)

            if not to_scrape:
                print("No new friends or incomplete libraries to scrape. Use --force_seed to re-scrape.")
                await browser.close()
                return

            worker_count = min(CONCURRENCY, len(to_scrape))

            worker_pages = [setup_page]
            for _ in range(worker_count - 1):
                worker_pages.append(await context.new_page())

            queue = asyncio.Queue()
            for library_id in to_scrape:
                queue.put_nowait(library_id)

            pbar = tqdm(total=len(to_scrape), desc="Scraping friend libraries", unit="list")

            async def library_worker(page):
                while True:
                    try:
                        library_id = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    try:
                        await process_library(db_conn, page, library_id, email, password, force_seed=force_seed)
                    except Exception as e:
                        tqdm.write(f"Failed library {library_id}: {e}")
                        mark_library_failed(db_conn, library_id, e)
                        with contextlib.suppress(Exception):
                            await page.goto("about:blank")
                    finally:
                        pbar.update(1)

            await asyncio.gather(*(library_worker(wp) for wp in worker_pages))
            pbar.close()

            await browser.close()
