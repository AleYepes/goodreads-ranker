import argparse
import asyncio
import os
import re
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from playwright.async_api import async_playwright
import db

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
SIGNIN_URL = "https://www.goodreads.com/ap/signin?language=en_US&openid.assoc_handle=amzn_goodreads_web_na&openid.claimed_id=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.identity=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.mode=checkid_setup&openid.ns=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0&openid.pape.max_auth_age=0&openid.return_to=https%3A%2F%2Fwww.goodreads.com%2Fap-handler%2Fsign-in"

DEFAULT_FRIEND_LIST_IDS = [
    104343033, 104945614, 105258888, 11116469, 113185438, 115764833,
    118560285, 124720847, 129155685, 13448447, 13647498, 13737030,
    156484926, 160516894, 166997642, 174792571, 1834894, 18922126,
    18913667, 21397146, 22482559, 22726983, 23161382, 24885719,
    267189, 26052616, 27115955, 22978411, 31565140, 33074940,
    34518408, 40426330, 41797321, 42001957, 43400637, 46459461,
    51281420, 54115664, 5868084, 65139494, 70012245, 7043947,
    75706676, 76860332, 8136076, 90649237, 91998392
]

RATING_MAP = {
    "it was amazing": 5,
    "really liked it": 4,
    "liked it": 3,
    "it was ok": 2,
    "did not like it": 1,
}

CONCURRENCY = 2

def clean_text(text):
    return text.strip().replace("\n", "") if text else ""

def clean_isbn(val):
    if isinstance(val, str):
        if val.startswith('="') and val.endswith('"'):
            return val[2:-1]
        return val.strip()
    return val

async def download_user_library(email, password, db_path=None, force=False):
    """Download user's library export using Playwright and import into user_library table."""
    conn = db.get_connection(db_path)
    
    if not force:
        # Check if table already has data
        cursor = conn.execute("SELECT COUNT(*) FROM user_library")
        count = cursor.fetchone()[0]
        if count > 0:
            print(f"user_library table already has {count} rows. Skipping download. Use --force to override.")
            conn.close()
            return

    if not email or not password:
        raise RuntimeError("GOODREADS_EMAIL and GOODREADS_PASSWORD must be set to download user library.")

    print("Logging into Goodreads to download library export...")
    
    temp_dir = tempfile.mkdtemp()
    temp_csv_path = Path(temp_dir) / "library_export.csv"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=USER_AGENT,
            accept_downloads=True
        )
        page = await context.new_page()

        # Log in
        await page.goto(SIGNIN_URL)
        await page.fill('input[type="email"]', email)
        await page.fill('input[type="password"]', password)
        await page.click('input[type="submit"]')

        # Prep export
        await page.wait_for_selector(".homePrimaryColumn", timeout=60000)
        await page.goto("https://www.goodreads.com/review/import", wait_until="domcontentloaded")
        await page.wait_for_selector(".js-LibraryExport", timeout=10000)

        export_button = page.locator(".js-LibraryExport").first
        while True:
            if await export_button.is_visible():
                await export_button.click()
                break
            await page.wait_for_timeout(500)

        prepped_export_list = page.locator(".fileList")
        for _ in range(240):
            if await prepped_export_list.count() > 0 and await prepped_export_list.locator("a").count() > 0:
                break
            await page.wait_for_timeout(500)

        # Export library
        async with page.expect_download() as download_info:
            list_link = prepped_export_list.locator("a").first
            await list_link.click()

        download = await download_info.value
        await download.save_as(str(temp_csv_path))
        await browser.close()

    print("Export downloaded. Importing into database...")
    
    df = pd.read_csv(temp_csv_path)
    
    # Normalize columns
    df.columns = [col.lower().replace(' ', '_').replace('author_l-f', 'author_lf') for col in df.columns]
    
    # Clean ISBNs
    if 'isbn' in df.columns:
        df['isbn'] = df['isbn'].apply(clean_isbn)
    if 'isbn13' in df.columns:
        df['isbn13'] = df['isbn13'].apply(clean_isbn)

    # Convert NaN to None for SQL NULL compatibility
    df = df.replace({np.nan: None})

    columns = [
        "book_id", "title", "author", "author_lf", "additional_authors", "isbn", "isbn13",
        "my_rating", "publisher", "binding", "number_of_pages", "year_published",
        "original_publication_year", "date_read", "date_added", "bookshelves",
        "bookshelves_with_positions", "exclusive_shelf", "my_review", "spoiler",
        "private_notes", "read_count", "owned_copies"
    ]
    
    # Extract only matching columns and format rows
    rows = []
    for _, row in df.iterrows():
        row_tuple = tuple(row.get(col) for col in columns)
        rows.append(row_tuple)

    db.upsert_rows(conn, "user_library", rows, columns)
    
    print(f"Successfully imported {len(rows)} books from your library into the 'user_library' table.")
    
    # Clean up temp file
    try:
        os.remove(temp_csv_path)
        os.rmdir(temp_dir)
    except OSError:
        pass
        
    conn.close()

async def scrape_friend_ratings(db_path=None, friend_list_ids=None, force_all=False):
    """Scrape friend libraries/ratings lists and insert into friend_ratings table."""
    conn = db.get_connection(db_path)
    
    if friend_list_ids is None:
        friend_list_ids = DEFAULT_FRIEND_LIST_IDS

    # Ensure friend_lists table is initialized
    for lid in friend_list_ids:
        conn.execute("INSERT OR IGNORE INTO friend_lists (list_id, scrape_complete) VALUES (?, 0)", (lid,))
    conn.commit()

    if force_all:
        cursor = conn.execute("SELECT list_id FROM friend_lists")
    else:
        cursor = conn.execute("SELECT list_id FROM friend_lists WHERE scrape_complete != 1")
    
    to_scrape = [row["list_id"] for row in cursor.fetchall()]
    conn.close()

    if not to_scrape:
        print("All friend lists are already scraped. Use --force-all to re-scrape.")
        return

    print(f"Preparing to scrape {len(to_scrape)} friend lists...")

    page_pool = asyncio.Queue()

    async def process_list(list_id):
        page = await page_pool.get()
        try:
            print(f"Scraping list {list_id}...")
            url = f"https://www.goodreads.com/review/list/{list_id}?print=true&sort=date_added&order=d&view=reviews"
            await page.goto(url)
            
            extracted_data = []
            page_num = 1
            
            while True:
                try:
                    await page.wait_for_selector("#booksBody", timeout=10000)
                except Exception:
                    print(f"Could not find book table for list {list_id} (Page {page_num}).")
                    break

                rows = await page.query_selector_all("tr.bookalike.review")
                for row in rows:
                    try:
                        title_el = await row.query_selector(".field.title a")
                        title = clean_text(await title_el.inner_text()) if title_el else "Unknown"
                        
                        href = await title_el.get_attribute("href") if title_el else ""
                        bid_match = re.search(r'/book/show/(\d+)', href)
                        book_id = int(bid_match.group(1)) if bid_match else None
                        if not book_id:
                            continue

                        rating_el = await row.query_selector(".field.rating .staticStars")
                        rating_title = await rating_el.get_attribute("title") if rating_el else ""
                        rating = RATING_MAP.get(rating_title, 0)

                        pages_el = await row.query_selector(".field.num_pages .value")
                        pages_text = await pages_el.text_content() if pages_el else ""
                        num_pages_raw = re.sub(r"[^\d]", "", pages_text)
                        num_pages = int(num_pages_raw) if num_pages_raw else None

                        dr_el = await row.query_selector(".field.date_read .date_read_value")
                        date_read = clean_text(await dr_el.inner_text()) if dr_el else ""

                        da_el = await row.query_selector(".field.date_added span")
                        date_added = await da_el.get_attribute("title") if da_el else ""
                        if not date_added and da_el:
                            date_added = clean_text(await da_el.inner_text())

                        extracted_data.append((list_id, book_id, title, rating, num_pages, date_read, date_added))
                    except Exception as e:
                        print(f"Error parsing book in list {list_id}: {e}")
                        continue

                # Check next page
                next_btn = await page.query_selector("a.next_page")
                if next_btn:
                    cls = await next_btn.get_attribute("class")
                    if "disabled" not in cls:
                        async with page.expect_navigation():
                            await next_btn.click()
                        await asyncio.sleep(0.5)
                        page_num += 1
                        continue
                break

            if extracted_data:
                db_conn = db.get_connection(db_path)
                db.upsert_rows(
                    db_conn,
                    "friend_ratings",
                    extracted_data,
                    ["list_id", "book_id", "title", "rating", "num_pages", "date_read", "date_added"]
                )
                
                today = datetime.now().strftime("%Y-%m-%d")
                db_conn.execute(
                    "UPDATE friend_lists SET scrape_complete = 1, date_last_scraped = ? WHERE list_id = ?",
                    (today, list_id)
                )
                db_conn.commit()
                db_conn.close()
                print(f"Successfully processed list {list_id} ({len(extracted_data)} books)")
            else:
                db_conn = db.get_connection(db_path)
                today = datetime.now().strftime("%Y-%m-%d")
                db_conn.execute(
                    "UPDATE friend_lists SET scrape_complete = 1, date_last_scraped = ? WHERE list_id = ?",
                    (today, list_id)
                )
                db_conn.commit()
                db_conn.close()
                print(f"Processed list {list_id} (0 books found)")

        except Exception as e:
            print(f"Failed list {list_id}: {e}")
        finally:
            await page_pool.put(page)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent=USER_AGENT)
        
        for _ in range(CONCURRENCY):
            page = await context.new_page()
            await page_pool.put(page)

        tasks = [process_list(lid) for lid in to_scrape]
        await asyncio.gather(*tasks)
        await browser.close()

async def main():
    load_dotenv()
    db.init_db()

    parser = argparse.ArgumentParser(description="Seed database with user and friend library ratings.")
    parser.add_argument("--user", action="store_true", help="Download user library")
    parser.add_argument("--friends", action="store_true", help="Scrape friend review lists")
    parser.add_argument("--force", action="store_true", help="Force redownload or re-scraping")
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
