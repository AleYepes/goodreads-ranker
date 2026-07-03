import argparse
import asyncio
import contextlib
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

import db

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
SIGNIN_URL = "https://www.goodreads.com/ap/signin?language=en_US&openid.assoc_handle=amzn_goodreads_web_na&openid.claimed_id=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.identity=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.mode=checkid_setup&openid.ns=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0&openid.pape.max_auth_age=0&openid.return_to=https%3A%2F%2Fwww.goodreads.com%2Fap-handler%2Fsign-in"


DEFAULT_FRIEND_LIST_IDS = [
    104343033,  # 104343033-santiago-lecumberri **
    104945614,  # 104945614-aadil-kumar
    105258888,  # 105258888-zach-saylor
    11116469,  # 11116469-sebastian-gebski
    113185438,  # 113185438-taiyr
    115764833,  # 115764833-austin-george
    118560285,  # 118560285-bichons-and-books-nz
    124720847,  # 124720847-abrish
    129155685,  # 129155685-ignacio-mu-oz-lanza **
    13448447,  # 13448447-lyda
    13647498,  # 13647498.Oana_David
    13737030,  # 13737030-maddy
    156484926,  # 156484926-victoria
    160516894,  # 160516894-janine
    166997642,  # 166997642-carson-cummins
    1713956,  # 1834894.Manny_Rayner ## Different list for some reason
    174792571,  # 174792571-annie
    18922126,  # 18922126-ella-park
    18913667,  # 18913667.Edward_Vass
    21397146,  # 21397146-stefy
    22482559,  # 22482559-mathi-fonseca
    22726983,  # 22726983-gast-n-mousqu-s
    23161382,  # 23161382-vanesa
    24885719,  # 24885719-fran-oise
    267189,  # 267189-todd-n
    26052616,  # 26052616-margherita
    27115955,  # 27115955-catherine-wood
    22978411,  # 22978411-cristina
    31565140,  # 31565140-irina-toledo
    33074940,  # 33074940-anca-e-milea
    34518408,  # 34518408-dawood
    40426330,  # 40426330-sabrina-li
    41797321,  # 41797321-cristina-cojocaru
    41944053,  # 8136076.Cosmin_Leucu_a ## Different list for some reason
    42001957,  # 42001957-matty-van-hoof
    43400637,  # 43400637-fay-pretty
    46459461,  # 46459461-an-fech
    51281420,  # 51281420-daniela-g-mez
    54115664,  # 54115664-mandy
    5868084,  # 5868084-mairi
    65139494,  # 65139494-daniel-castro
    70012245,  # 70012245-till-chen
    7043947,  # 7043947-andra-enache
    75706676,  # 75706676-steve-abreu
    76860332,  # 76860332-cecilia
    90649237,  # 90649237-sara
    91998392,  # 91998392-daniel-prelipcean
]

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


def clean_isbn(val):
    if isinstance(val, str):
        if val.startswith('="') and val.endswith('"'):
            return val[2:-1]
        return val.strip()
    return val


async def is_login_page(page):
    return (
        "/ap/signin" in page.url
        or await page.locator('input[type="email"]').count() > 0
        or await page.locator("button.authPortalSignInButton").count() > 0
    )


async def wait_for_post_login(page):
    for selector in ("#booksBody", "#books", "#reviewPagination", ".homePrimaryColumn"):
        try:
            await page.wait_for_selector(selector, timeout=15000)
            return
        except PlaywrightTimeoutError:
            continue

    if await is_login_page(page):
        raise RuntimeError("Goodreads login did not complete successfully.")


async def login_to_goodreads(page, email, password):
    if not email or not password:
        raise RuntimeError(
            "GOODREADS_EMAIL and GOODREADS_PASSWORD must be set to scrape private review lists."
        )

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


async def open_list_page(page, list_id, email, password):
    url = (
        f"https://www.goodreads.com/review/list/{list_id}"
        f"?print=true&sort={LIST_SORT}&order={LIST_ORDER}&view=reviews"
    )
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


async def download_user_library(email, password, db_path=None, force=False):
    """Download user's library export using Playwright and import into user_library table."""
    db.init_db(db_path)
    conn = db.get_connection(db_path)

    if not force:
        # Check if table already has data
        cursor = conn.execute("SELECT COUNT(*) FROM user_library")
        count = cursor.fetchone()[0]
        if count > 0:
            print(
                f"  Library already seeded ({count} books). Use --force to re-download."
            )
            conn.close()
            return

    if not email or not password:
        raise RuntimeError(
            "GOODREADS_EMAIL and GOODREADS_PASSWORD must be set to download user library."
        )

    print("  Logging into Goodreads to download library export...")

    temp_dir = tempfile.mkdtemp()
    temp_csv_path = Path(temp_dir) / "library_export.csv"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=USER_AGENT,
            accept_downloads=True,
        )
        page = await context.new_page()

        await page.goto(SIGNIN_URL)
        await page.fill('input[type="email"]', email)
        await page.fill('input[type="password"]', password)
        await page.click('input[type="submit"]')

        # Prep export
        await page.wait_for_selector(".homePrimaryColumn", timeout=60000)
        await page.goto(
            "https://www.goodreads.com/review/import", wait_until="domcontentloaded"
        )
        await page.wait_for_selector(".js-LibraryExport", timeout=10000)

        export_button = page.locator(".js-LibraryExport").first
        while True:
            if await export_button.is_visible():
                await export_button.click()
                break
            await page.wait_for_timeout(500)

        prepped_export_list = page.locator(".fileList")
        for _ in range(240):
            if (
                await prepped_export_list.count() > 0
                and await prepped_export_list.locator("a").count() > 0
            ):
                break
            await page.wait_for_timeout(500)

        async with page.expect_download() as download_info:
            list_link = prepped_export_list.locator("a").first
            await list_link.click()

        download = await download_info.value
        await download.save_as(str(temp_csv_path))
        await browser.close()

    print("  Export downloaded. Importing into database...")

    df = pd.read_csv(temp_csv_path)
    df = db.normalise_library_columns(df)

    # Clean ISBNs
    if "isbn" in df.columns:
        df["isbn"] = df["isbn"].apply(clean_isbn)
    if "isbn13" in df.columns:
        df["isbn13"] = df["isbn13"].apply(clean_isbn)

    df = df.replace({np.nan: None})

    columns = [
        "book_id",
        "title",
        "author",
        "author_lf",
        "additional_authors",
        "isbn",
        "isbn13",
        "my_rating",
        "publisher",
        "binding",
        "number_of_pages",
        "year_published",
        "original_publication_year",
        "date_read",
        "date_added",
        "bookshelves",
        "bookshelves_with_positions",
        "exclusive_shelf",
        "my_review",
        "spoiler",
        "private_notes",
        "read_count",
        "owned_copies",
    ]

    # Extract only matching columns and format rows
    rows = []
    for _, row in df.iterrows():
        row_tuple = tuple(row.get(col) for col in columns)
        rows.append(row_tuple)

    db.upsert_rows(conn, "user_library", rows, columns)

    print(f"  Imported {len(rows)} books into library.")

    try:
        os.remove(temp_csv_path)
        os.rmdir(temp_dir)
    except OSError:
        pass

    conn.close()


async def scrape_friend_ratings(db_path=None, friend_list_ids=None, force_all=False):
    """Scrape friend libraries/ratings lists and insert into friend_ratings table."""
    load_dotenv()
    db.init_db(db_path)
    email = os.getenv("GOODREADS_EMAIL")
    password = os.getenv("GOODREADS_PASSWORD")
    conn = db.get_connection(db_path)

    if friend_list_ids is None:
        friend_list_ids = DEFAULT_FRIEND_LIST_IDS

    # Ensure friend_lists table is initialized
    for lid in friend_list_ids:
        conn.execute(
            "INSERT OR IGNORE INTO friend_lists (list_id, scrape_complete) VALUES (?, 0)",
            (lid,),
        )
    conn.commit()

    if force_all:
        cursor = conn.execute("SELECT list_id, 0 as metadata_only FROM friend_lists")
    else:
        cursor = conn.execute(
            """
            SELECT list_id,
                   (scrape_complete = 1 AND (username IS NULL OR href IS NULL)) as metadata_only
            FROM friend_lists
            WHERE scrape_complete != 1 OR username IS NULL OR href IS NULL
            """
        )

    to_scrape = [
        (row["list_id"], bool(row["metadata_only"])) for row in cursor.fetchall()
    ]
    conn.close()

    if not to_scrape:
        print("  Friend lists already scraped. Use --force-all to re-scrape.")
        return

    print(f"  Scraping {len(to_scrape)} friend lists...")

    def update_friend_info(list_id, username, href):
        db_conn = db.get_connection(db_path)
        try:
            db_conn.execute(
                """
                UPDATE friend_lists
                SET username = ?,
                    href = ?
                WHERE list_id = ?
                """,
                (username, href, list_id),
            )
            db_conn.commit()
        finally:
            db_conn.close()

    def load_existing_rows(list_id):
        db_conn = db.get_connection(db_path)
        try:
            rows = db_conn.execute(
                """
                SELECT list_id, book_id, rating, date_read, date_added
                FROM friend_ratings
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
        finally:
            db_conn.close()

    def mark_list_complete(list_id):
        db_conn = db.get_connection(db_path)
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            db_conn.execute(
                """
                UPDATE friend_lists
                SET scrape_complete = 1,
                    date_last_scraped = ?,
                    scrape_error = NULL
                WHERE list_id = ?
                """,
                (today, list_id),
            )
            db_conn.commit()
        finally:
            db_conn.close()

    def mark_list_failed(list_id, error):
        db_conn = db.get_connection(db_path)
        try:
            db_conn.execute(
                """
                UPDATE friend_lists
                SET scrape_complete = 0,
                    scrape_error = ?
                WHERE list_id = ?
                """,
                (str(error)[:1000], list_id),
            )
            db_conn.commit()
        finally:
            db_conn.close()

    def upsert_extracted(rows):
        if not rows:
            return
        db_conn = db.get_connection(db_path)
        try:
            db.upsert_rows(
                db_conn,
                "friend_ratings",
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
        finally:
            db_conn.close()

    async def process_list(page, list_id, metadata_only=False):
        print(f"  Scraping list {list_id} (metadata_only={metadata_only})...")
        page_num = 1
        total_rows = 0
        valid_page_parsed = False
        target_url = await open_list_page(page, list_id, email, password)

        await ensure_list_page(page, target_url, email, password)

        with contextlib.suppress(PlaywrightTimeoutError):
            await page.wait_for_selector("h1", timeout=10000)

        h1_el = await page.query_selector("h1")
        username = None
        href = None
        if h1_el:
            links = await h1_el.query_selector_all("a")
            for link in links:
                link_href = await link.get_attribute("href")
                if link_href and "/user/show/" in link_href:
                    if not href:
                        href = link_href
                    text = clean_text(await link.inner_text())
                    if text:
                        username = text
        if username or href:
            update_friend_info(list_id, username, href)
            print(f"    Updated metadata: {username} ({href})")

        if metadata_only:
            print(f"    Finished metadata-only scrape for {list_id}")
            return

        existing_rows = load_existing_rows(list_id)
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
            upsert_extracted(page_rows)

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

        mark_list_complete(list_id)
        print(f"  Processed list {list_id} ({total_rows} books)")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(user_agent=USER_AGENT)
        page = await context.new_page()
        for list_id, metadata_only in to_scrape:
            try:
                await process_list(page, list_id, metadata_only=metadata_only)
            except Exception as e:
                print(f"  Failed list {list_id}: {e}")
                mark_list_failed(list_id, e)
                with contextlib.suppress(Exception):
                    await page.goto("about:blank")
        await browser.close()


async def main():
    load_dotenv()
    db.init_db()

    parser = argparse.ArgumentParser(
        description="Seed database with user and friend library ratings."
    )
    parser.add_argument("--user", action="store_true", help="Download user library")
    parser.add_argument(
        "--friends", action="store_true", help="Scrape friend review lists"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force redownload or re-scraping"
    )
    args = parser.parse_args()

    # Default to running both if neither is specified
    run_user = args.user
    run_friends = args.friends
    if not run_user and not run_friends:
        run_user = True
        run_friends = True

    if run_user:
        email = os.getenv("GOODREADS_EMAIL")
        password = os.getenv("GOODREADS_PASSWORD")
        await download_user_library(email, password, force=args.force)

    if run_friends:
        await scrape_friend_ratings(force_all=args.force)


if __name__ == "__main__":
    asyncio.run(main())
