import argparse
import asyncio
import json
import os
import re
import glob
import heapq
import html
import csv
import random
import pandas as pd
import numpy as np
import traceback
from datetime import datetime
from bs4 import BeautifulSoup
from pathlib import Path
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
from playwright.async_api import async_playwright

RATING_MAP = {"it was amazing": 5, "really liked it": 4, "liked it": 3, "it was ok": 2, "did not like it": 1,}

def clean_text(text):
    return text.strip().replace("\n", "") if text else ""

async def download_library(email, password):

    def preprocess_library():
        df = pd.read_csv(MY_LIBRARY_PATH)
        df.columns = [col.lower().replace(' ','_') for col in df.columns]
        df.to_csv(MY_LIBRARY_PATH, index=False)
        return df

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(viewport={"width": 1280, "height": 800}, user_agent=USER_AGENT, accept_downloads=True)
        page = await context.new_page()
        
        # Log in
        await page.goto("https://www.goodreads.com/ap/signin?language=en_US&openid.assoc_handle=amzn_goodreads_web_na&openid.claimed_id=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.identity=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.mode=checkid_setup&openid.ns=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0&openid.pape.max_auth_age=0&openid.return_to=https%3A%2F%2Fwww.goodreads.com%2Fap-handler%2Fsign-in")
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
            list = prepped_export_list.locator("a").first
            await list.click()
        
        download = await download_info.value
        await download.save_as(MY_LIBRARY_PATH)
        await browser.close()

    return preprocess_library()


async def scrape_lists(lists_path, force_all=False):
    if not lists_path.exists():
        print(f"No list metadata found at {lists_path}")
        user_input = input("Enter comma-separated list IDs to track: ")
        if not user_input.strip():
            return pd.DataFrame()
        
        list_ids = [lid.strip() for lid in user_input.split(',') if lid.strip()]
        meta_df = pd.DataFrame({'list_id': list_ids})
        meta_df['scrape_complete'] = False
        meta_df['date_last_scraped'] = None
        meta_df.to_csv(lists_path, index=False)
    else:
        meta_df = pd.read_csv(lists_path)

    # Ensure columns exist and types are consistent
    for col in ['date_last_scraped', 'scrape_complete']:
        if col not in meta_df.columns: meta_df[col] = None
    meta_df['list_id'] = meta_df['list_id'].astype(int)

    if force_all:
        to_scrape_idxs = meta_df.index
    else:
        to_scrape_idxs = meta_df[meta_df['scrape_complete'] != True].index

    if len(to_scrape_idxs) == 0:
        if FRIENDS_LIBRARIES_PATH.exists():
            return pd.read_csv(FRIENDS_LIBRARIES_PATH)
        return pd.DataFrame()

    # Shared locks
    csv_lock = asyncio.Lock()
    meta_lock = asyncio.Lock()

    # Write header if file doesn't exist
    file_exists = FRIENDS_LIBRARIES_PATH.exists() and FRIENDS_LIBRARIES_PATH.stat().st_size > 0
    write_header = not file_exists

    if write_header:
        with open(FRIENDS_LIBRARIES_PATH, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["list_id", "book_id", "title", "rating", "num_pages", "date_read", "date_added"])

    async def process_list(page_pool, idx):
        page = await page_pool.get()
        try:
            # Re-read meta to get current state (though strictly we are just reading row data)
            # Accessing dataframe in thread-safe way is tricky if modifying, but here we just read row_data
            row_data = meta_df.loc[idx]
            list_id = str(row_data['list_id'])
            print(f"Scraping list {list_id}...")

            try:
                url = f"https://www.goodreads.com/review/list/{list_id}?print=true&sort=rating&view=reviews"
                await page.goto(url)
                
                while True:
                    try:
                        await page.wait_for_selector("#booksBody", timeout=10000)
                    except:
                        break

                    # Scrape rows
                    rows = await page.query_selector_all("tr.bookalike.review")
                    extracted_data = []
                    
                    for row in rows:
                        try:
                            title_el = await row.query_selector(".field.title a")
                            title = clean_text(await title_el.inner_text()) if title_el else "Unknown"
                            
                            href = await title_el.get_attribute("href") if title_el else ""
                            bid_match = re.search(r'/book/show/(\d+)', href)
                            book_id = bid_match.group(1) if bid_match else "Unknown"

                            rating_el = await row.query_selector(".field.rating .staticStars")
                            rating_title = await rating_el.get_attribute("title") if rating_el else ""
                            rating = RATING_MAP.get(rating_title, 0)

                            pages_el = await row.query_selector(".field.num_pages .value")
                            pages_text = await pages_el.text_content() if pages_el else ""
                            num_pages = re.sub(r"[^\d]", "", pages_text)

                            dr_el = await row.query_selector(".field.date_read .date_read_value")
                            date_read = clean_text(await dr_el.inner_text()) if dr_el else ""

                            da_el = await row.query_selector(".field.date_added span")
                            date_added = await da_el.get_attribute("title") if da_el else ""
                            if not date_added and da_el: 
                                date_added = clean_text(await da_el.inner_text())

                            extracted_data.append([list_id, book_id, title, rating, num_pages, date_read, date_added])
                        except Exception:
                            continue

                    # Write to CSV
                    if extracted_data:
                        async with csv_lock:
                            with open(FRIENDS_LIBRARIES_PATH, mode='a', newline='', encoding='utf-8') as f:
                                writer = csv.writer(f)
                                writer.writerows(extracted_data)

                    # Check next page
                    next_btn = await page.query_selector("a.next_page")
                    if next_btn:
                        cls = await next_btn.get_attribute("class")
                        if "disabled" not in cls:
                            async with page.expect_navigation():
                                await next_btn.click()
                            await asyncio.sleep(0.5)
                            continue
                    break
                
                # Update metadata on success
                async with meta_lock:
                    meta_df.at[idx, 'scrape_complete'] = True
                    meta_df.at[idx, 'date_last_scraped'] = datetime.now().strftime("%Y-%m-%d")
                    meta_df.to_csv(lists_path, index=False)
                
            except Exception as e:
                print(f"Failed list {list_id}: {e}")

        finally:
            page_pool.put_nowait(page)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent=USER_AGENT)
        page_pool = asyncio.Queue()
        for _ in range(CONCURRENCY):
            page_pool.put_nowait(await context.new_page())

        tasks = [process_list(page_pool, idx) for idx in to_scrape_idxs]
        await asyncio.gather(*tasks)
        await browser.close()
    
    if FRIENDS_LIBRARIES_PATH.exists():
        friends_df = pd.read_csv(FRIENDS_LIBRARIES_PATH)
        # Ensure consistency
        friends_df['book_id'] = pd.to_numeric(friends_df['book_id'], errors='coerce').fillna(0).astype(int)
        friends_df['list_id'] = pd.to_numeric(friends_df['list_id'], errors='coerce').fillna(0).astype(int)
        
        friends_df = friends_df.drop_duplicates(subset=['list_id', 'book_id'])
        friends_df.to_csv(FRIENDS_LIBRARIES_PATH, index=False)
        return friends_df
    return pd.DataFrame()


def parse_and_score_similar_books(encoded_str, scoring_func):
    if not isinstance(encoded_str, str) or not encoded_str:
        return []
    
    similar_books = []
    for item in encoded_str.split("|"):
        try:
            parts = item.split(":")
            book_id, avg, count = int(parts[0]), float(parts[1]), int(parts[2])
            similar_books.append((book_id, scoring_func(avg, count)))
        except ValueError:
            continue
    return similar_books


def filter_save_file():
    if OUTPUT_PATH.exists():
        try:
            book_df = pd.read_csv(OUTPUT_PATH, on_bad_lines='skip')
            star_cols = [col for col in book_df.columns if col.endswith('star')]
            int_cols = ['book_id', 'review_count', 'num_pages', 'author_followers', 
                        'want_to_read', 'author_num_books', 'currently_reading'] + star_cols
            for col in int_cols:
                book_df[col] = pd.to_numeric(book_df[col], errors='coerce').astype('Int64')
            book_df['year'] = pd.to_numeric(book_df['year'], errors='coerce').astype('Int16')

            book_df = book_df[~book_df['similar_books'].isna()]

            temp_path = OUTPUT_PATH.with_suffix('.tmp')
            book_df.to_csv(temp_path, index=False)
            temp_path.replace(OUTPUT_PATH)
        except Exception as e:
            print(f"Error cleaning file: {e}")
            traceback.print_exc()


def prep_crawl_heapq(library_df, friends_df, scoring_func):
    seed_ids = set(library_df['book_id'].dropna().astype(int))
    if friends_df is not None and not friends_df.empty:
        seed_ids.update(friends_df['book_id'].dropna().astype(int))

     # Prioritize library seed ids
    crawl_queue = {bid: 9e7 for bid in seed_ids}

    scraped_ids = set()
    if OUTPUT_PATH.exists():
        scraped_df = pd.read_csv(OUTPUT_PATH, usecols=['book_id', 'similar_books'], on_bad_lines='skip')
        scraped_ids.update(scraped_df['book_id'].dropna().astype(int))

        for similar_books_str in scraped_df['similar_books'].dropna():
            for book_id, score in parse_and_score_similar_books(similar_books_str, scoring_func):
                if book_id not in scraped_ids:
                    crawl_queue[book_id] = max(score, crawl_queue.get(book_id, 0))

        for book_id in scraped_ids:
            crawl_queue.pop(book_id, None)

    crawl_queue = [(-rating, book_id) for book_id, rating in crawl_queue.items()]
    heapq.heapify(crawl_queue)

    return crawl_queue, scraped_ids, {book_id for _, book_id in crawl_queue}, seed_ids


async def fetch_book(page, book_id, bad_book_ids):

    async def handle_response(response):       
        if collecting and "graphql" in response.url and response.request.method == "POST":
            try:
                json_body = await response.json()
                captured_payloads.append(json_body)
            except Exception:
                pass

    async def extract_linked_data_basics(page, book_id):
        script_locator = page.locator('script[type="application/ld+json"]').first
        await script_locator.wait_for(state="attached", timeout=PAGE_TIMEOUT_MS)
        content = await script_locator.text_content()
        ld = json.loads(content)

        agg_rating = ld.get("aggregateRating", {})
        title = html.unescape(ld.get("name", ""))
        authors = "|".join(a["name"] for a in ld.get("author", []) if "name" in a)
        langs = ld.get("inLanguage")
        if isinstance(langs, str):
            langs = "|".join(lang.strip() for lang in langs.split(";"))
        else:
            langs = ""
        return {
            "book_id": book_id,
            "title": title,
            "authors": authors,
            "avg_rating": agg_rating.get("ratingValue"),
            "review_count": agg_rating.get("reviewCount"),
            "num_pages": ld.get("numberOfPages"),
            "lang": langs,
        }


    async def extract_dom_data(page, book_data):
        html_content = await page.content()
        soup = BeautifulSoup(html_content, "html.parser")

        # Stars distribution
        for i in range(1, 6):   
            label = soup.find(attrs={"data-testid": f"labelTotal-{i}"})
            text = label.get_text().strip().split()[0].replace(",", "") if label else "0"
            book_data[f"{i}_star"] = int(text) if text.isdigit() else 0

        # Genres
        try:
            if await page.query_selector('button[aria-label="Show all items in the list"]'):
                await page.click('button[aria-label="Show all items in the list"]')
                await page.wait_for_timeout(100) 
                html_content = await page.content()
                soup = BeautifulSoup(html_content, "html.parser")
        except Exception:
            pass

        genre_nodes = soup.select(".BookPageMetadataSection__genreButton .Button__labelItem")
        genres = [node.get_text() for node in genre_nodes if node.get_text() != "...more"]
        book_data['genres'] = "|".join(genres)

        # Series id
        series_el = soup.select_one("h3.Text__italic a")
        book_data['series'] = series_el['href'].split('/')[-1] if series_el and series_el.get('href') else ""

        # Year
        pub_el = soup.find(attrs={"data-testid": "publicationInfo"})
        book_data['year'] = pub_el.get_text().split(", ")[-1].strip() if pub_el else ""

        # Description
        desc_el = soup.select_one("[data-testid='description'] span.Formatted") or \
                  soup.select_one(".DetailsLayoutRightParagraph__widthConstrained span.Formatted")
        if desc_el:
            for br in desc_el.find_all("br"):
                br.replace_with("\n")
            book_data['description'] = re.sub(r'\n{3,}', '\n\n', desc_el.get_text()).strip()
        else:
            book_data['description'] = ""

        # Currently reading 
        reading_el = soup.find(attrs={"data-testid": "currentlyReadingSignal"})
        if reading_el:
            text = reading_el.get_text()
            match = re.search(r'(\d+)', text.replace(",", ""))
            book_data['currently_reading'] = int(match.group(1)) if match else 0
        else:
            book_data['currently_reading'] = 0

        # Want to read
        wtr_el = soup.find(attrs={"data-testid": "toReadSignal"})
        if wtr_el:
            text = wtr_el.get_text()
            match = re.search(r'(\d+)', text.replace(",", ""))
            book_data['want_to_read'] = int(match.group(1)) if match else 0
        else:
            book_data['want_to_read'] = 0

        # Author name
        author_name_el = soup.find(attrs={"data-testid": "name"})
        book_data['primary_author'] = author_name_el.get_text().strip() if author_name_el else ""

        # Author stats
        author_stats_el = soup.select_one(".FeaturedPerson__infoPrimary .Text__subdued")
        book_data['author_num_books'] = 0
        book_data['author_followers'] = 0
        if author_stats_el:
            stats_text = author_stats_el.get_text(separator=" ", strip=True)
            
            books_match = re.search(r'([\d,]+)\s*books', stats_text)
            if books_match:
                book_data['author_num_books'] = int(books_match.group(1).replace(",", ""))

            # Author follower count
            followers_match = re.search(r'([\d,kKmM\.]+)\s*followers', stats_text)
            if followers_match:
                val = followers_match.group(1).lower().replace(",", "")
                if 'k' in val:
                    val = float(val.replace('k', '')) * 1e3
                elif 'm' in val:
                    val = float(val.replace('m', '')) * 1e6
                book_data['author_followers'] = int(val)

        return book_data

    async def extract_similar_books_json(page, book_data, captured_payloads, collecting):
        wait_attempts = 0
        while not any("getSimilarBooks" in str(p) for p in captured_payloads) and wait_attempts < PAYLOAD_WAIT_ATTEMPTS:
            await page.wait_for_timeout(500)
            wait_attempts += 1
        collecting = False
        
        similar_books = []
        for payload in captured_payloads:
            for book_edge in payload.get("data", {}).get("getSimilarBooks", {}).get("edges", []):
                book_node = book_edge.get("node", {})
                match = re.search(r'show/(\d+)', book_node.get("webUrl", ""))
                if match:
                    stats = book_node.get("work", {}).get("stats", {})
                    similar_books.append(f"{match.group(1)}:{stats.get('averageRating')}:{stats.get('ratingsCount')}")

        book_data["similar_books"] = "|".join(similar_books)
        return book_data
    
    async def close_modal(page, book_id):
        global MODAL_DISMISSED

        watch_seconds = MODAL_DISMISSED_WATCH_SECONDS if MODAL_DISMISSED else MODAL_WATCH_SECONDS
        deadline = datetime.now().timestamp() + watch_seconds
        overlay = page.locator(".Overlay").first
        close_btn = page.locator('.Overlay button[aria-label="Close"], .Overlay .Overlay__close button').first

        while datetime.now().timestamp() < deadline:
            try:
                if await overlay.count() == 0 or not await overlay.is_visible():
                    await page.wait_for_timeout(MODAL_POLL_MS)
                    continue

                for attempt in range(MODAL_CLOSE_ATTEMPTS):
                    try:
                        await close_btn.wait_for(state="visible", timeout=1000)
                        await close_btn.click(timeout=1000)
                    except Exception:
                        try:
                            await close_btn.evaluate("button => button.click()")
                        except Exception:
                            await page.keyboard.press("Escape")

                    try:
                        await overlay.wait_for(state="hidden", timeout=1000)
                        MODAL_DISMISSED = True
                        return True
                    except Exception:
                        if attempt == MODAL_CLOSE_ATTEMPTS - 1:
                            break

                await page.evaluate("""
                    document.querySelectorAll('.Overlay').forEach((overlay) => overlay.remove());
                    document.body.style.overflow = '';
                    document.documentElement.style.overflow = '';
                """)
                MODAL_DISMISSED = True
                tqdm.write(f"Removed stuck modal on {book_id}")
                return True

            except Exception as e:
                tqdm.write(f"Failed to close modal on {book_id}: {e}")
                return False

        return False

    async def check_if_404(page):
        if await page.locator(".ErrorPage__title").count() > 0:
            return True
        return False

    async def check_if_unavailable(page):
        return await page.evaluate("""
            () => document.body?.id === 'home' &&
                document.querySelector('h1')?.textContent?.trim().toLowerCase() === 'page unavailable'
        """)

    captured_payloads = []
    collecting = True
    page.on("response", handle_response)
    try:
        url = f"https://www.goodreads.com/book/show/{book_id}"
        response = None
        for attempt in range(MAX_RATE_LIMIT_RETRIES):
            response = await page.goto(url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS)
            status = response.status if response else None
            if status not in RATE_LIMIT_STATUSES:
                break

            wait_seconds = RATE_LIMIT_BACKOFF_SECONDS * (attempt + 1)
            tqdm.write(f"Rate limited on {book_id} ({status}). Sleeping {wait_seconds}s before retry.")
            await asyncio.sleep(wait_seconds)

        if response and response.status in RATE_LIMIT_STATUSES:
            raise ValueError(f"Book {book_id} blocked by rate limit ({response.status})")

        if await check_if_404(page):
            bad_book_ids.add(book_id)
            raise ValueError(f"Book {book_id} not found (404)")

        if await check_if_unavailable(page):
            bad_book_ids.add(book_id)
            raise ValueError(f"Book {book_id} unavailable")

        await close_modal(page, book_id)
        await page.evaluate(f"window.scrollBy(0, {random.randint(100,200)})")
        await close_modal(page, book_id)

        book_data = await extract_linked_data_basics(page, book_id)
        book_data = await extract_dom_data(page, book_data)
        book_data = await extract_similar_books_json(page, book_data, captured_payloads, collecting)

        return book_data

    except Exception as e:
        if "not found (404)" not in str(e) and "unavailable" not in str(e):
            tqdm.write(f"Failed {book_id} -- {e}")
        return None
    finally:
        collecting = False
        page.remove_listener("response", handle_response)
        try:
            await page.goto("about:blank")
        except Exception:
            pass


async def run_crawler(library_df, friends_df):

    async def block_media(route):
        if route.request.resource_type in ["image", "media", "font"]:
            await route.abort()
        else:
            await route.continue_()

    async def fetch_wrapper(page_pool, page, book_id, bad_book_ids):
        try:
            return await fetch_book(page, book_id, bad_book_ids)
        finally:
            page_pool.put_nowait(page)

    field_names = [
        "book_id", "title", "authors", "avg_rating", "review_count", 
        "num_pages", "lang", "1_star", "2_star", "3_star", "4_star", 
        "5_star", "genres", "series", "year", "description", "similar_books",
        "primary_author", "author_followers", "want_to_read", 
        "author_num_books", "currently_reading"
    ]

    bad_book_ids = set()
    cycle = 0
    scoring_algo_names = list(SCORING_FUNCTIONS.keys())
    while True:
        current_algo_name = scoring_algo_names[cycle % len(scoring_algo_names)]
        scoring_func = SCORING_FUNCTIONS[current_algo_name]

        # filter_save_file()
        crawl_queue, scraped_ids, queued_ids, seed_ids = prep_crawl_heapq(library_df, friends_df, scoring_func)
        if not crawl_queue:
            break

        file_exists = OUTPUT_PATH.exists()
        with open(OUTPUT_PATH, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            if not file_exists:
                writer.writeheader()

            remaining_seeds = seed_ids - scraped_ids - bad_book_ids
            pbar = tqdm(
                total=len(scraped_ids) + len(crawl_queue), 
                initial=len(scraped_ids), 
                unit='book',
                desc=f"{current_algo_name} | {f'{len(remaining_seeds)} Seeds remaining' if remaining_seeds else 'Seeds done'}" 
            )

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=False) # Running headed is required for GraphQL triggers to fire
                context = await browser.new_context(user_agent=USER_AGENT)
                await context.route("**/*", block_media)

                page_pool = asyncio.Queue()
                for _ in range(CONCURRENCY):
                    page_pool.put_nowait(await context.new_page())

                active_tasks = set()
                processed = 0
                try:
                    while (crawl_queue or active_tasks) and processed < RESTART_THRESHOLD:
                        while crawl_queue and not page_pool.empty() and processed < RESTART_THRESHOLD:
                            _, book_id = heapq.heappop(crawl_queue)
                            if book_id in scraped_ids or book_id in bad_book_ids:
                                continue

                            page = page_pool.get_nowait()
                            task = asyncio.create_task(fetch_wrapper(page_pool, page, book_id, bad_book_ids))
                            active_tasks.add(task)

                        if not active_tasks:
                            break

                        done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
                        for task in done:
                            try:
                                processed += 1
                                book_data = task.result()
                                if book_data:
                                    writer.writerow(book_data)
                                    f.flush()
                                    
                                    bid = book_data['book_id']
                                    scraped_ids.add(bid)

                                    if bid in remaining_seeds:
                                        remaining_seeds.remove(bid)
                                        if remaining_seeds:
                                            pbar.set_description(f"{len(remaining_seeds)} Seeds remaining")
                                        else:
                                            pbar.set_description(f"{current_algo_name} | Seeds done")

                                    pbar.update(1)
                                
                                    added = 0
                                    for similar_id, score in parse_and_score_similar_books(book_data.get('similar_books', ''), scoring_func):
                                        if similar_id not in scraped_ids and similar_id not in queued_ids:
                                            heapq.heappush(crawl_queue, (-score, similar_id))
                                            queued_ids.add(similar_id)
                                            added += 1
                                    pbar.total += added
                            except Exception as e:
                                tqdm.write(f"\nError post-processing task: {e}")
                                traceback.print_exc()

                finally:
                    pbar.close()
                    for task in active_tasks:
                        task.cancel()
                    if active_tasks:
                        await asyncio.gather(*active_tasks, return_exceptions=True)
                    await browser.close()
                    await asyncio.sleep(1)

        cycle += 1


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-fd", "--fd", action="store_true", help="Force download of my library")
    parser.add_argument("-fs", "--fs", action="store_true", help="Force scrape of friend lists")
    args = parser.parse_args()

    if args.fd or not MY_LIBRARY_PATH.exists():
        load_dotenv()
        library_df = await download_library(os.getenv("GOODREADS_EMAIL"), os.getenv("GOODREADS_PASSWORD"))
    else:
        library_df = pd.read_csv(MY_LIBRARY_PATH)

    force_lists = args.fs or not FRIENDS_LIBRARIES_PATH.exists()
    friends_df = await scrape_lists(FRIEND_LISTS_PATH, force_all=force_lists)

    try:
        await run_crawler(library_df, friends_df)
    except KeyboardInterrupt:
        pass


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
MY_LIBRARY_PATH = DATA_DIR / "goodreads_library_export.csv"
OUTPUT_PATH = DATA_DIR / "books.csv"
FRIENDS_LIBRARIES_PATH = DATA_DIR / "friend_ratings.csv"
FRIEND_LISTS_PATH = DATA_DIR / "friend_lists.csv"

CONCURRENCY = 2
PAYLOAD_WAIT_ATTEMPTS = 20
PAGE_TIMEOUT_MS = 20000
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
RESTART_THRESHOLD = 100
RATE_LIMIT_STATUSES = {403, 429}
MAX_RATE_LIMIT_RETRIES = 3
RATE_LIMIT_BACKOFF_SECONDS = 15
MODAL_WATCH_SECONDS = 4
MODAL_DISMISSED_WATCH_SECONDS = 1
MODAL_POLL_MS = 250
MODAL_CLOSE_ATTEMPTS = 3
MODAL_DISMISSED = False

SCORING_FUNCTIONS = {
    "Rating": lambda avg_rating, rating_count: avg_rating - avg_rating / np.log10(rating_count + 10),
    "Count": lambda avg_rating, rating_count: rating_count,
}

if __name__ == "__main__":
    asyncio.run(main())
