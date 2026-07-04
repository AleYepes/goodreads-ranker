import asyncio
import contextlib
import os
import re
from datetime import datetime
import select
from urllib.parse import urljoin

from dotenv import load_dotenv
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

from . import db

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
SIGNIN_URL = "https://www.goodreads.com/ap/signin?language=en_US&openid.assoc_handle=amzn_goodreads_web_na&openid.claimed_id=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.identity=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.mode=checkid_setup&openid.ns=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0&openid.pape.max_auth_age=0&openid.return_to=https%3A%2F%2Fwww.goodreads.com%2Fap-handler%2Fsign-in"

RATING_MAP = {
    "it was amazing": 5,
    "really liked it": 4,
    "liked it": 3,
    "it was ok": 2,
    "did not like it": 1,
}

LIST_SORT = "date_added"
LIST_ORDER = "d"


def clean_text(text):
    return text.strip().replace("\n", "") if text else ""


def parse_id_from_slug(slug: str) -> int:
    match = re.search(r"(\d+)", slug.strip("/").split("/")[-1])
    if not match:
        raise ValueError(f"Could not parse numeric ID from slug: {slug}")
    return int(match.group(1))


async def is_login_page(page):
    return (
        "/ap/signin" in page.url
        or await page.locator('input[type="email"]').count() > 0
        or await page.locator("button.authPortalSignInButton").count() > 0
    )


async def login_to_goodreads(page, email, password):

    async def wait_for_post_login(page):
        for selector in (
            ".homePrimaryColumn",
        ):
            try:
                await page.wait_for_selector(selector, timeout=15000)
                return
            except PlaywrightTimeoutError:
                print(f'wait_for_post_login - {selector}')
                continue

        if await is_login_page(page):
            raise RuntimeError("Goodreads login did not complete successfully.")

    if not email or not password:
        raise RuntimeError("GOODREADS_EMAIL and GOODREADS_PASSWORD must be set to scrape private review lists.")

    if await page.locator("button.authPortalSignInButton").count() > 0:
        async with page.expect_navigation(wait_until="domcontentloaded"):
            await page.locator("button.authPortalSignInButton").click()
    elif await page.locator('input[type="email"]').count() == 0:
        await page.goto(SIGNIN_URL, wait_until="domcontentloaded")

    await page.wait_for_selector('input[type="email"]', timeout=30000)
    await page.fill('input[type="email"]', email)
    await page.fill('input[type="password"]', password)
    await page.click('input[type="submit"]')
    await page.wait_for_load_state("domcontentloaded")
    await wait_for_post_login(page)


async def extract_main_user(page):
    profile_el = page.locator("a.dropdown__trigger--profileMenu").first
    if await profile_el.count() == 0:
        raise RuntimeError("Could not find user profile on homepage.")

    my_books_el = page.locator('li.siteHeader__topLevelItem a:has-text("My Books")').first
    if await my_books_el.count() == 0:
        raise RuntimeError("Could not find book links on homepage.")

    profile_href = await profile_el.get_attribute("href")
    my_books_href = await my_books_el.get_attribute("href")
    img_el = profile_el.locator("img")
    username = await img_el.get_attribute("alt") if await img_el.count() > 0 else None

    if not profile_href or not my_books_href or not username:
        raise RuntimeError("User profile metadata extraction failed.")

    user_id = parse_id_from_slug(profile_href)
    list_id = parse_id_from_slug(my_books_href)

    return {"user_id": user_id, "list_id": list_id, "username": username}


async def fetch_friends(page, user_id: int) -> list[dict]:
    friends = []
    page_num = 1
    target_url = f"https://www.goodreads.com/friend/user/{user_id}"

    while True:
        await page.goto(f"{target_url}?page={page_num}", wait_until="domcontentloaded")
        rows = await page.locator("#friendTable tbody tr").all()
        if not rows:
            break

        for row in rows:
            user_anchor = row.locator('td a[href^="/user/show/"]').first
            list_anchor = row.locator('a[href^="/review/list/"]').first

            if await user_anchor.count() == 0 or await list_anchor.count() == 0:
                continue

            user_href = await user_anchor.get_attribute("href")
            list_href = await list_anchor.get_attribute("href")
            username = await user_anchor.get_attribute("rel")

            if not user_href or not list_href:
                continue

            book_text = await list_anchor.inner_text()
            book_match = re.search(r"(\d+)\s+books?", book_text, re.IGNORECASE)
            book_count = int(book_match.group(1)) if book_match else 0

            if book_count == 0:
                continue

            friends.append(
                {
                    "user_id": parse_id_from_slug(user_href),
                    "list_id": parse_id_from_slug(list_href),
                    "username": username or "",
                }
            )

        next_btn = page.locator("a.next_page").first
        if await next_btn.count() > 0 and "disabled" not in (await next_btn.get_attribute("class") or ""):
            page_num += 1
            await asyncio.sleep(1)
        else:
            break

    return friends


def upsert_readers(db_conn, main_user: dict, friends: list[dict]):
    db_conn.execute(
        "UPDATE readers SET is_self = 0 WHERE is_self = 1 AND list_id != ?",
        (main_user["list_id"],),
    )
    db_conn.execute(
        """
        INSERT OR REPLACE INTO readers (list_id, user_id, username, is_self)
        VALUES (?, ?, ?, 1)
        """,
        (main_user["list_id"], main_user["user_id"], main_user["username"]),
    )
    db_conn.executemany(
        """
        INSERT OR IGNORE INTO readers (list_id, user_id, username, is_self)
        VALUES (?, ?, ?, 0)
        """,
        [(f["list_id"], f["user_id"], f["username"]) for f in friends],
    )
    db_conn.commit()


def update_friend_info(db_conn, list_id, username, user_id):
    db_conn.execute(
        """
        UPDATE readers
        SET username = ?,
            user_id = ?
        WHERE list_id = ?
        """,
        (username, user_id, list_id),
    )
    db_conn.commit()


def load_existing_rows(db_conn, list_id):
    rows = db_conn.execute(
        """
        SELECT list_id, book_id, rating, date_read, date_added
        FROM reader_libraries
        WHERE list_id = ?
        """,
        (list_id,),
    ).fetchall()
    return {
        (int(row["list_id"]), int(row["book_id"])): {
            "list_id": int(row["list_id"]),
            "book_id": int(row["book_id"]),
            "rating": int(row["rating"] or 0),
            "date_read": row["date_read"] or "",
            "date_added": row["date_added"] or "",
        }
        for row in rows
    }


def mark_list_complete(db_conn, list_id):
    today = datetime.now().strftime("%Y-%m-%d")
    db_conn.execute(
        """
        UPDATE readers
        SET scrape_complete = 1,
            date_last_scraped = ?,
            scrape_error = NULL
        WHERE list_id = ?
        """,
        (today, list_id),
    )
    db_conn.commit()


def mark_list_failed(db_conn, list_id, error):
    db_conn.execute(
        """
        UPDATE readers
        SET scrape_complete = 0,
            scrape_error = ?
        WHERE list_id = ?
        """,
        (str(error)[:1000], list_id),
    )
    db_conn.commit()


def upsert_extracted(db_conn, rows):
    if not rows:
        return
    db.upsert_rows(
        db_conn,
        "reader_libraries",
        [
            (
                row["list_id"],
                row["book_id"],
                row["rating"],
                row["date_read"],
                row["date_added"],
            )
            for row in rows
        ],
        [
            "list_id",
            "book_id",
            "rating",
            "date_read",
            "date_added",
        ],
    )


async def open_list_page(page, list_id, email, password):
    url = f"https://www.goodreads.com/review/list/{list_id}?print=true&sort={LIST_SORT}&order={LIST_ORDER}&view=reviews"
    await page.goto(url, wait_until="domcontentloaded")

    if await is_login_page(page):
        print("Login required. Authenticating session...")
        await login_to_goodreads(page, email, password)
        await page.goto(url, wait_until="domcontentloaded")

    return url


async def ensure_list_page(page, target_url, email, password):
    if await is_login_page(page):
        print("Login required. Authenticating session...")
        await login_to_goodreads(page, email, password)
        await page.goto(target_url, wait_until="domcontentloaded")


async def extract_friend_row(row, list_id):
    title_el = await row.query_selector(".field.title a")
    href = await title_el.get_attribute("href") if title_el else ""
    book_id_match = re.search(r"/book/show/(\d+)", href)
    book_id = int(book_id_match.group(1)) if book_id_match else None
    if not book_id:
        return None

    rating_el = await row.query_selector(".field.rating .staticStars")
    rating_text = await rating_el.get_attribute("title") if rating_el else ""
    rating = RATING_MAP.get(rating_text, 0)

    date_read_el = await row.query_selector(".field.date_read .date_read_value")
    date_read = clean_text(await date_read_el.inner_text()) if date_read_el else ""

    date_added_el = await row.query_selector(".field.date_added span")
    date_added = ""
    if date_added_el:
        date_added = await date_added_el.get_attribute("title") or ""
        if not date_added:
            date_added = clean_text(await date_added_el.inner_text())

    return {
        "list_id": int(list_id),
        "book_id": book_id,
        "rating": rating,
        "date_read": date_read,
        "date_added": date_added,
    }


async def process_list(db_conn, page, list_id, email, password, metadata_only=False):
    print(f"  Scraping list {list_id} (metadata_only={metadata_only})...")
    page_num = 1
    total_rows = 0
    valid_page_parsed = False
    target_url = await open_list_page(page, list_id, email, password)

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
        update_friend_info(db_conn, list_id, username, user_id)
        print(f"    Updated metadata: {username} ({user_id})")

    if metadata_only:
        print(f"    Finished metadata-only scrape for {list_id}")
        return

    existing_rows = load_existing_rows(db_conn, list_id)
    while True:
        await ensure_list_page(page, target_url, email, password)
        await page.wait_for_selector("#booksBody", timeout=10000)

        rows = await page.query_selector_all("tr.bookalike.review")
        page_rows = []
        page_all_known = bool(rows)
        for row in rows:
            try:
                extracted = await extract_friend_row(row, list_id)
                if not extracted:
                    page_all_known = False
                    continue

                key = (extracted["list_id"], extracted["book_id"])
                if existing_rows.get(key) != extracted:
                    page_all_known = False
                existing_rows[key] = extracted
                page_rows.append(extracted)
            except Exception as e:
                page_all_known = False
                print(f"    Error parsing book in list {list_id}: {e}")

        if not page_rows:
            raise RuntimeError(f"No valid rows parsed on page {page_num}")

        valid_page_parsed = True
        total_rows += len(page_rows)
        upsert_extracted(db_conn, page_rows)

        if page_all_known:
            print(f"      P{page_num} unchanged, stopping early")
            break

        next_button = await page.query_selector("a.next_page")
        next_class = await next_button.get_attribute("class") if next_button else ""
        next_href = await next_button.get_attribute("href") if next_button else None
        if next_button and next_href and "disabled" not in next_class:
            target_url = urljoin(page.url, next_href)
            async with page.expect_navigation():
                await next_button.click()
            print(f"      P{page_num}")
            page_num += 1
            await asyncio.sleep(1)
            continue

        print(f"    Finished {list_id}")
        break

    if not valid_page_parsed:
        raise RuntimeError("No valid list page was parsed")

    mark_list_complete(db_conn, list_id)
    print(f"  Processed list {list_id} ({total_rows} books)")


def get_lists_to_scrape(db_conn, force_all):
    if force_all:
        cursor = db_conn.execute("SELECT list_id, 0 as metadata_only FROM readers")
    else:
        cursor = db_conn.execute(
            """
            SELECT list_id,
                   (scrape_complete = 1 AND (username IS NULL OR user_id IS NULL)) as metadata_only
            FROM readers
            WHERE scrape_complete != 1 OR username IS NULL OR user_id IS NULL
            """
        )
    return [(row["list_id"], bool(row["metadata_only"])) for row in cursor.fetchall()]


async def scrape_reader_libraries(db_path=None, list_ids=None, force_all=False):
    load_dotenv()
    db.init_db(db_path)
    email = os.getenv("GOODREADS_EMAIL")
    password = os.getenv("GOODREADS_PASSWORD")

    with db.get_connection(db_path) as db_conn:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context(user_agent=USER_AGENT)
            page = await context.new_page()

            if list_ids is not None:
                for lid in list_ids:
                    db_conn.execute(
                        "INSERT OR IGNORE INTO readers (list_id, is_self, scrape_complete) VALUES (?, 0, 0)",
                        (lid,),
                    )
                db_conn.commit()
            else:
                await page.goto(SIGNIN_URL)
                await login_to_goodreads(page, email, password)
                main_user = await extract_main_user(page)
                friends = await fetch_friends(page, main_user["user_id"])
                upsert_readers(db_conn, main_user, friends)

            to_scrape = get_lists_to_scrape(db_conn, force_all)

            if not to_scrape:
                print("  Friend lists already scraped. Use --force-all to re-scrape.")
                await browser.close()
                return

            print(f"  Scraping {len(to_scrape)} friend lists...")

            for list_id, metadata_only in to_scrape:
                try:
                    await process_list(
                        db_conn,
                        page,
                        list_id,
                        email,
                        password,
                        metadata_only=metadata_only,
                    )
                except Exception as e:
                    print(f"  Failed list {list_id}: {e}")
                    mark_list_failed(db_conn, list_id, e)
                    with contextlib.suppress(Exception):
                        await page.goto("about:blank")

            await browser.close()
