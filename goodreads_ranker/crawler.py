import asyncio
import contextlib
import heapq
import html
import json
import random
import re
import traceback
from datetime import datetime, timedelta

import numpy as np
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from tqdm.asyncio import tqdm

from . import db

PAYLOAD_WAIT_ATTEMPTS = 20
PAGE_TIMEOUT_MS = 20000
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
RESTART_THRESHOLD = 100
RATE_LIMIT_STATUSES = {403, 429}
MAX_RATE_LIMIT_RETRIES = 3
RATE_LIMIT_BACKOFF_SECONDS = 10
MODAL_WATCH_SECONDS = 4
MODAL_DISMISSED_WATCH_SECONDS = 1
MODAL_POLL_MS = 250
MODAL_CLOSE_ATTEMPTS = 3
MODAL_DISMISSED = (
    False  # The modal appears once per session. Once dismissed, we can use a shorter watch window for subsequent books.
)

SCORING_FUNCTIONS = {
    "Rating": lambda avg_rating, rating_count: avg_rating - avg_rating / np.log10(rating_count + 10),
    "Count": lambda avg_rating, rating_count: rating_count,
}


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


def parse_scraped_at(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        try:
            return datetime.strptime(str(value), "%Y-%m-%d")
        except ValueError:
            return None


def is_stale_scrape(value, now=None):
    scraped_at = parse_scraped_at(value)
    if scraped_at is None:
        return True
    return scraped_at < (now or datetime.now()) - timedelta(days=30)


def prep_crawl_heapq(scoring_func, limit=None, force_recrawl=False, db_path=None):
    with db.get_connection(db_path) as db_conn:
        cursor = db_conn.execute("SELECT DISTINCT book_id FROM reader_libraries WHERE book_id IS NOT NULL")
        seed_ids = {int(row["book_id"]) for row in cursor.fetchall()}

        include_expansion = limit is not None

        cursor = db_conn.execute(
            """
            SELECT book_id, similar_books, date_last_scraped
            FROM books
            WHERE book_id IS NOT NULL
            """
        )
        scraped_rows = cursor.fetchall()
        scraped_by_id = {int(row["book_id"]): row for row in scraped_rows}
        scraped_ids = set(scraped_by_id)
        recrawl_ids = {
            book_id
            for book_id, row in scraped_by_id.items()
            if force_recrawl and is_stale_scrape(row["date_last_scraped"])
        }

        crawl_queue = {}
        for book_id in seed_ids:
            if book_id not in scraped_ids or book_id in recrawl_ids:
                crawl_queue[book_id] = (0, 0)

        if include_expansion:
            for row in scraped_rows:
                similar_books_str = row["similar_books"]
                if similar_books_str:
                    for book_id, score in parse_and_score_similar_books(similar_books_str, scoring_func):
                        should_crawl = book_id not in scraped_ids or book_id in recrawl_ids
                        if should_crawl and book_id not in seed_ids:
                            priority = (1, -score)
                            existing = crawl_queue.get(book_id)
                            if existing is None or priority < existing:
                                crawl_queue[book_id] = priority

    crawl_queue = [(group, sort_score, book_id) for book_id, (group, sort_score) in crawl_queue.items()]
    heapq.heapify(crawl_queue)

    return (
        crawl_queue,
        scraped_ids,
        {book_id for _, _, book_id in crawl_queue},
        seed_ids,
        recrawl_ids,
    )


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
        langs = "|".join(lang.strip() for lang in langs.split(";")) if isinstance(langs, str) else ""
        return {
            "book_id": book_id,
            "title": title,
            "authors": authors,
            "avg_rating": agg_rating.get("ratingValue"),
            "review_count": agg_rating.get("reviewCount"),
            "num_pages": ld.get("numberOfPages"),
            "style": langs,
        }

    async def extract_dom_data(page, book_data):
        html_content = await page.content()
        soup = BeautifulSoup(html_content, "html.parser")

        for i in range(1, 6):
            label = soup.select_one(f'[data-testid="labelTotal-{i}"]')
            text = label.get_text().strip().split()[0].replace(",", "") if label else "0"
            book_data[f"star_{i}"] = int(text) if text.isdigit() else 0

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
        book_data["genres"] = "|".join(genres)

        series_el = soup.select_one("h3.Text__italic a")
        href = series_el.get("href") if series_el else None
        book_data["series"] = str(href).split("/")[-1] if series_el and isinstance(href, str) else ""

        pub_el = soup.select_one('[data-testid="publicationInfo"]')
        book_data["year"] = pub_el.get_text().split(", ")[-1].strip() if pub_el else ""

        desc_el = soup.select_one("[data-testid='description'] span.Formatted") or soup.select_one(
            ".DetailsLayoutRightParagraph__widthConstrained span.Formatted"
        )
        if desc_el:
            for br in desc_el.find_all("br"):
                br.replace_with("\n")
            book_data["description"] = re.sub(r"\n{3,}", "\n\n", desc_el.get_text()).strip()
        else:
            book_data["description"] = ""

        reading_el = soup.select_one('[data-testid="currentlyReadingSignal"]')
        if reading_el:
            text = reading_el.get_text()
            match = re.search(r"(\d+)", text.replace(",", ""))
            book_data["currently_reading"] = int(match.group(1)) if match else 0
        else:
            book_data["currently_reading"] = 0

        wtr_el = soup.select_one('[data-testid="toReadSignal"]')
        if wtr_el:
            text = wtr_el.get_text()
            match = re.search(r"(\d+)", text.replace(",", ""))
            book_data["want_to_read"] = int(match.group(1)) if match else 0
        else:
            book_data["want_to_read"] = 0

        author_name_el = soup.select_one('[data-testid="name"]')
        book_data["primary_author"] = author_name_el.get_text().strip() if author_name_el else ""

        author_stats_el = soup.select_one(".FeaturedPerson__infoPrimary .Text__subdued")
        book_data["author_num_books"] = 0
        book_data["author_followers"] = 0
        if author_stats_el:
            stats_text = author_stats_el.get_text(separator=" ", strip=True)

            books_match = re.search(r"([\d,]+)\s*books", stats_text)
            if books_match:
                book_data["author_num_books"] = int(books_match.group(1).replace(",", ""))

            followers_match = re.search(r"([\d,kKmM\.]+)\s*followers", stats_text)
            if followers_match:
                val = followers_match.group(1).lower().replace(",", "")
                if "k" in val:
                    val = float(val.replace("k", "")) * 1e3
                elif "m" in val:
                    val = float(val.replace("m", "")) * 1e6
                book_data["author_followers"] = int(val)

        return book_data

    async def extract_similar_books_json(page, book_data, captured_payloads):
        wait_attempts = 0
        while not any("getSimilarBooks" in str(p) for p in captured_payloads) and wait_attempts < PAYLOAD_WAIT_ATTEMPTS:
            await page.wait_for_timeout(500)
            wait_attempts += 1

        similar_books = []
        for payload in captured_payloads:
            for book_edge in payload.get("data", {}).get("getSimilarBooks", {}).get("edges", []):
                book_node = book_edge.get("node", {})
                match = re.search(r"show/(\d+)", book_node.get("webUrl", ""))
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
        return await page.locator(".ErrorPage__title").count() > 0

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
        await page.evaluate(f"window.scrollBy(0, {random.randint(100, 200)})")
        await close_modal(page, book_id)

        book_data = await extract_linked_data_basics(page, book_id)
        book_data = await extract_dom_data(page, book_data)
        book_data = await extract_similar_books_json(page, book_data, captured_payloads)

        return book_data

    except Exception as e:
        if "not found (404)" not in str(e) and "unavailable" not in str(e):
            tqdm.write(f"Failed {book_id} -- {e}")
        return None
    finally:
        collecting = False
        page.remove_listener("response", handle_response)
        with contextlib.suppress(Exception):
            await page.goto("about:blank")


async def run_crawler(limit=None, concurrency=2, force_recrawl=False, db_path=None):
    db.init_db(db_path)

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
        "book_id",
        "title",
        "authors",
        "avg_rating",
        "review_count",
        "num_pages",
        "lang",
        "star_1",
        "star_2",
        "star_3",
        "star_4",
        "star_5",
        "genres",
        "series",
        "year",
        "description",
        "similar_books",
        "primary_author",
        "author_followers",
        "want_to_read",
        "author_num_books",
        "currently_reading",
        "date_last_scraped",
    ]

    bad_book_ids = set()
    cycle = 0
    scoring_algo_names = list(SCORING_FUNCTIONS.keys())
    with db.get_connection(db_path) as db_conn:
        enforce_limit = limit is not None and limit > 0
        include_expansion = limit is not None

        if enforce_limit:
            already_scraped = db_conn.execute(
                "SELECT COUNT(*) FROM books WHERE date_last_scraped IS NOT NULL"
            ).fetchone()[0]
            if already_scraped >= limit:
                print(f"  Already have {already_scraped} scraped books (>= limit of {limit}). Skipping crawler.")
                return
            remaining_budget = limit - already_scraped
        else:
            already_scraped = 0
            remaining_budget = None

        total_processed = 0

        while True:
            current_algo_name = scoring_algo_names[cycle % len(scoring_algo_names)]
            scoring_func = SCORING_FUNCTIONS[current_algo_name]

            crawl_queue, scraped_ids, queued_ids, seed_ids, recrawl_ids = prep_crawl_heapq(
                scoring_func,
                limit=limit,
                force_recrawl=force_recrawl,
                db_path=db_path,
            )
            if not crawl_queue:
                print("  No more books to crawl.")
                break

            remaining_seeds = {
                book_id
                for book_id in seed_ids
                if (book_id not in scraped_ids or book_id in recrawl_ids) and book_id not in bad_book_ids
            }
            pbar = tqdm(
                total=len(scraped_ids) + len(crawl_queue),
                initial=len(scraped_ids),
                unit="book",
                desc=f"{current_algo_name} | {f'{len(remaining_seeds)} Seeds remaining' if remaining_seeds else 'Seeds done'}",
            )

            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=False
                )  # Goodreads only triggers the similar-books GraphQL request in headed Chromium.
                context = await browser.new_context(user_agent=USER_AGENT)
                await context.route("**/*", block_media)

                page_pool = asyncio.Queue()
                for _ in range(concurrency):
                    page_pool.put_nowait(await context.new_page())

                active_tasks = set()
                processed_in_cycle = 0
                try:
                    while (crawl_queue or active_tasks) and processed_in_cycle < RESTART_THRESHOLD:
                        if enforce_limit and remaining_budget is not None and total_processed >= remaining_budget:
                            break

                        while crawl_queue and not page_pool.empty() and processed_in_cycle < RESTART_THRESHOLD:
                            if (
                                enforce_limit
                                and remaining_budget is not None
                                and total_processed + len(active_tasks) >= remaining_budget
                            ):
                                break

                            _, _, book_id = heapq.heappop(crawl_queue)
                            if (book_id in scraped_ids and book_id not in recrawl_ids) or book_id in bad_book_ids:
                                continue

                            page = page_pool.get_nowait()
                            task = asyncio.create_task(fetch_wrapper(page_pool, page, book_id, bad_book_ids))
                            active_tasks.add(task)

                        if not active_tasks:
                            break

                        done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
                        for task in done:
                            try:
                                processed_in_cycle += 1
                                total_processed += 1
                                book_data = task.result()
                                if book_data:
                                    book_data["date_last_scraped"] = datetime.now().isoformat()
                                    row_tuple = tuple(book_data.get(field) for field in field_names)
                                    db.upsert_rows(db_conn, "books", [row_tuple], field_names)

                                    bid = book_data["book_id"]
                                    scraped_ids.add(bid)
                                    recrawl_ids.discard(bid)

                                    if bid in remaining_seeds:
                                        remaining_seeds.remove(bid)
                                        if remaining_seeds:
                                            pbar.set_description(f"{len(remaining_seeds)} Seeds remaining")
                                        else:
                                            pbar.set_description(f"{current_algo_name} | Seeds done")

                                    pbar.update(1)

                                    added = 0
                                    if include_expansion:
                                        for (
                                            similar_id,
                                            score,
                                        ) in parse_and_score_similar_books(
                                            book_data.get("similar_books", ""),
                                            scoring_func,
                                        ):
                                            if (
                                                similar_id not in scraped_ids or similar_id in recrawl_ids
                                            ) and similar_id not in queued_ids:
                                                heapq.heappush(crawl_queue, (1, -score, similar_id))
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

            if enforce_limit and remaining_budget is not None and total_processed >= remaining_budget:
                total_scraped_now = already_scraped + total_processed
                print(
                    f"  Reached crawl target: {total_scraped_now} total scraped books (target was {limit}). Stopping."
                )
                break

            cycle += 1
