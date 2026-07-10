import asyncio
import base64
import json
import math
import os
from datetime import datetime
import httpx
from tqdm import tqdm

from . import db

API_URL = "https://kxbwmqov6jgg3daaamb744ycu4.appsync-api.us-east-1.amazonaws.com/graphql"

# GraphQL Query Templates
PROBE_QUERY = """
query getBookByLegacyId($legacyBookId: Int!) {
  getBookByLegacyId(legacyId: $legacyBookId) {
    legacyId
    work { bestBook { legacyId } }
  }
}
"""

BOOK_QUERY = """
query getBookByLegacyId($legacyBookId: Int!) {
  getBookByLegacyId(legacyId: $legacyBookId) {
    id
    legacyId
    title
    titleComplete
    description
    webUrl
    imageUrl
    primaryContributorEdge {
      node { id legacyId name isGrAuthor webUrl followers { totalCount } works { totalCount } }
      role
    }
    secondaryContributorEdges {
      node { id legacyId name isGrAuthor webUrl followers { totalCount } works { totalCount } }
      role
    }
    bookSeries {
      userPosition
      series { id title webUrl }
    }
    bookGenres {
      genre { name webUrl }
    }
    details {
      asin isbn isbn13 format numPages publisher publicationTime
      language { name }
    }
    work {
      id
      legacyId
      stats {
        averageRating
        ratingsCount
        ratingsCountDist
        textReviewsCount
        textReviewsLanguageCounts { count isoLanguageCode }
      }
      details {
        webUrl
        shelvesUrl
        publicationTime
        originalTitle
        awardsWon { name webUrl awardedAt category designation }
      }
      editions(pagination: {limit: 20}) {
        totalCount
        edges { node { legacyId title } }
      }
    }
  }
}
"""

EDITIONS_PAGE_QUERY = """
query getBookByLegacyId($legacyBookId: Int!, $pagination: PaginationInput!) {
  getBookByLegacyId(legacyId: $legacyBookId) {
    work {
      editions(pagination: $pagination) {
        edges { node { legacyId title } }
      }
    }
  }
}
"""

SIMILAR_QUERY = """
query GetSimilarBooks($id: ID!, $limit: Int!) {
  getSimilarBooks(id: $id, pagination: {limit: $limit}) {
    edges {
      node {
        legacyId
        title
        work { stats { averageRating ratingsCount } }
      }
    }
  }
}
"""

SOCIAL_QUERY = """
query GetSocialSignals($bookId: ID!) {
  getSocialSignals(bookId: $bookId, shelfStatus: [CURRENTLY_READING, TO_READ]) {
    name
    count
  }
}
"""


class InvalidLegacyIdError(Exception):
    pass


def make_after_token(page_number: int) -> str:
    raw = json.dumps({"next_page": page_number}, separators=(",", ":"))
    return base64.b64encode(raw.encode()).decode()


async def gql(client: httpx.AsyncClient, headers: dict, operation_name: str, query: str, variables: dict) -> dict:
    for attempt in range(3):
        try:
            resp = await client.post(
                API_URL,
                json={"operationName": operation_name, "variables": variables, "query": query},
                headers=headers,
                timeout=15.0,
            )
            if resp.status_code in {403, 429}:
                await asyncio.sleep((attempt + 1) * 5.0)
                continue
            resp.raise_for_status()
            data = resp.json()
            errors = data.get("errors")
            if errors:
                first_msg = errors[0].get("message", "")
                if "Variable 'legacyBookId' has an invalid value." in first_msg:
                    raise InvalidLegacyIdError(first_msg)
                raise RuntimeError(f"GraphQL error: {errors}")
            return data["data"]
        except (httpx.HTTPError, RuntimeError) as e:
            if isinstance(e, InvalidLegacyIdError):
                raise
            if attempt == 2:
                raise
            await asyncio.sleep((attempt + 1) * 2.0)
    raise RuntimeError(f"Operation {operation_name} failed after 3 attempts")


async def fetch_remaining_editions(
    client: httpx.AsyncClient, headers: dict, canonical_legacy_id: int, total_count: int
) -> list[dict]:
    if not total_count:
        return []

    page_count = min(math.ceil(total_count / 20), 10)
    if page_count <= 1:
        return []

    async def fetch_page(page_number: int):
        variables = {
            "legacyBookId": canonical_legacy_id,
            "pagination": {"limit": 20, "after": make_after_token(page_number)},
        }
        try:
            data = await gql(client, headers, "getBookByLegacyId", EDITIONS_PAGE_QUERY, variables)
            book_data = data.get("getBookByLegacyId") or {}
            work_data = book_data.get("work") or {}
            editions_data = work_data.get("editions") or {}
            edges = editions_data.get("edges") or []
            return [edge["node"] for edge in edges if edge and edge.get("node")]
        except Exception:
            return []

    pages = await asyncio.gather(*(fetch_page(p) for p in range(2, page_count + 1)))
    return [node for page in pages for node in page]


async def fetch_similar_books(client: httpx.AsyncClient, headers: dict, book_kca_id: str, limit: int = 20) -> list[dict]:
    if not book_kca_id:
        return []
    try:
        data = await gql(client, headers, "GetSimilarBooks", SIMILAR_QUERY, {"id": book_kca_id, "limit": limit})
        similar = data.get("getSimilarBooks") or {}
        edges = similar.get("edges") or []
        return [edge["node"] for edge in edges if edge and edge.get("node")]
    except Exception:
        return []


async def fetch_social_signals(client: httpx.AsyncClient, headers: dict, book_kca_id: str) -> list[dict]:
    if not book_kca_id:
        return []
    try:
        data = await gql(client, headers, "GetSocialSignals", SOCIAL_QUERY, {"bookId": book_kca_id})
        return data.get("getSocialSignals") or []
    except Exception:
        return []


async def resolve_and_save_book(client: httpx.AsyncClient, headers: dict, db_conn, legacy_id: int, allowed_sources: list) -> bool:
    now = datetime.utcnow().isoformat() + "Z"

    # 1. Probe fetch
    try:
        probe_data = await gql(client, headers, "getBookByLegacyId", PROBE_QUERY, {"legacyBookId": legacy_id})
    except InvalidLegacyIdError as e:
        db_conn.execute(
            """
            INSERT OR REPLACE INTO crawl_queue (legacy_id, status, error_count, last_error_message, processed_at)
            VALUES (?, 'error', 1, ?, ?)
            """,
            (legacy_id, str(e), now),
        )
        db_conn.commit()
        return False
    except Exception as e:
        row = db_conn.execute("SELECT error_count FROM crawl_queue WHERE legacy_id = ?", (legacy_id,)).fetchone()
        error_count = (row["error_count"] or 0) + 1 if row else 1
        status = "error" if error_count >= 3 else "pending"
        db_conn.execute(
            """
            INSERT OR REPLACE INTO crawl_queue (legacy_id, status, error_count, last_error_message, processed_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (legacy_id, status, error_count, str(e), now),
        )
        db_conn.commit()
        return False

    book_node = probe_data.get("getBookByLegacyId")
    if not book_node:
        db_conn.execute(
            """
            INSERT OR REPLACE INTO crawl_queue (legacy_id, status, error_count, last_error_message, processed_at)
            VALUES (?, 'error', 1, 'Book not found (data is null)', ?)
            """,
            (legacy_id, now),
        )
        db_conn.commit()
        return False

    work_node = book_node.get("work") or {}
    best_book = work_node.get("bestBook") or {}
    best_book_legacy_id = best_book.get("legacyId") or book_node.get("legacyId")

    if not best_book_legacy_id:
        db_conn.execute(
            """
            INSERT OR REPLACE INTO crawl_queue (legacy_id, status, error_count, last_error_message, processed_at)
            VALUES (?, 'error', 1, 'No best book legacy ID resolved', ?)
            """,
            (legacy_id, now),
        )
        db_conn.commit()
        return False

    # 2. Branch on canonicality
    if legacy_id != best_book_legacy_id:
        db_conn.execute(
            """
            INSERT OR REPLACE INTO crawl_queue (legacy_id, status, error_count, processed_at)
            VALUES (?, 'mapped_to_canonical', 0, ?)
            """,
            (legacy_id, now),
        )
        db_conn.commit()

        canonical_row = db_conn.execute("SELECT 1 FROM books WHERE legacy_id = ?", (best_book_legacy_id,)).fetchone()
        if canonical_row:
            return True

        db_conn.execute(
            """
            INSERT OR IGNORE INTO crawl_queue (legacy_id, status, priority, discovered_via)
            VALUES (?, 'pending', 0.0, 'seed')
            """,
            (best_book_legacy_id,),
        )
        db_conn.commit()

        return await resolve_and_save_book(client, headers, db_conn, best_book_legacy_id, allowed_sources)

    # 3. Full fetch on canonical ID
    try:
        full_data = await gql(client, headers, "getBookByLegacyId", BOOK_QUERY, {"legacyBookId": legacy_id})
    except InvalidLegacyIdError as e:
        db_conn.execute(
            """
            INSERT OR REPLACE INTO crawl_queue (legacy_id, status, error_count, last_error_message, processed_at)
            VALUES (?, 'error', 1, ?, ?)
            """,
            (legacy_id, str(e), now),
        )
        db_conn.commit()
        return False
    except Exception as e:
        row = db_conn.execute("SELECT error_count FROM crawl_queue WHERE legacy_id = ?", (legacy_id,)).fetchone()
        error_count = (row["error_count"] or 0) + 1 if row else 1
        status = "error" if error_count >= 3 else "pending"
        db_conn.execute(
            """
            INSERT OR REPLACE INTO crawl_queue (legacy_id, status, error_count, last_error_message, processed_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (legacy_id, status, error_count, str(e), now),
        )
        db_conn.commit()
        return False

    book_node = full_data.get("getBookByLegacyId")
    if not book_node:
        db_conn.execute(
            """
            INSERT OR REPLACE INTO crawl_queue (legacy_id, status, error_count, last_error_message, processed_at)
            VALUES (?, 'error', 1, 'Book not found on full fetch', ?)
            """,
            (legacy_id, now),
        )
        db_conn.commit()
        return False

    # 4. Concurrent auxiliary fetches
    work_node = book_node.get("work") or {}
    book_kca_id = book_node.get("id")

    editions_conn = work_node.get("editions") or {}
    total_editions = editions_conn.get("totalCount") or 0
    page1_edges = editions_conn.get("edges") or []
    page1_editions = [edge["node"] for edge in page1_edges if edge and edge.get("node")]

    remaining_editions_task = fetch_remaining_editions(client, headers, legacy_id, total_editions)
    similar_task = fetch_similar_books(client, headers, book_kca_id)
    social_task = fetch_social_signals(client, headers, book_kca_id)

    remaining_editions, similar_list, social_list = await asyncio.gather(
        remaining_editions_task, similar_task, social_task, return_exceptions=True
    )

    if isinstance(remaining_editions, Exception):
        remaining_editions = []
    if isinstance(similar_list, Exception):
        similar_list = []
    if isinstance(social_list, Exception):
        social_list = []

    all_editions = page1_editions + remaining_editions

    # 5. Write to database
    try:
        with db_conn:
            work_stats = work_node.get("stats") or {}
            work_details = work_node.get("details") or {}
            dist = work_stats.get("ratingsCountDist") or []
            star_1 = dist[0] if len(dist) > 0 else 0
            star_2 = dist[1] if len(dist) > 1 else 0
            star_3 = dist[2] if len(dist) > 2 else 0
            star_4 = dist[3] if len(dist) > 3 else 0
            star_5 = dist[4] if len(dist) > 4 else 0

            text_reviews_lang_counts = work_stats.get("textReviewsLanguageCounts")
            text_reviews_lang_counts_json = json.dumps(text_reviews_lang_counts) if text_reviews_lang_counts else None

            db_conn.execute(
                """
                INSERT OR REPLACE INTO works (
                    legacy_id, original_title, publication_time, web_url, shelves_url,
                    average_rating, ratings_count, star_1, star_2, star_3, star_4, star_5,
                    text_reviews_count, text_reviews_language_counts, editions_total_count,
                    editions_coverage_complete, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    work_node.get("legacyId"),
                    work_details.get("originalTitle"),
                    work_details.get("publicationTime"),
                    work_details.get("webUrl"),
                    work_details.get("shelvesUrl"),
                    work_stats.get("averageRating"),
                    work_stats.get("ratingsCount"),
                    star_1,
                    star_2,
                    star_3,
                    star_4,
                    star_5,
                    work_stats.get("textReviewsCount"),
                    text_reviews_lang_counts_json,
                    total_editions,
                    1 if total_editions <= 200 else 0,
                    now,
                ),
            )

            details = book_node.get("details") or {}
            lang = details.get("language") or {}
            language_name = lang.get("name")
            db_conn.execute(
                """
                INSERT OR REPLACE INTO books (
                    legacy_id, work_id, title, title_complete, description, image_url, web_url,
                    asin, isbn, isbn13, format, num_pages, publisher, publication_time, language_name, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    book_node.get("legacyId"),
                    work_node.get("legacyId"),
                    book_node.get("title"),
                    book_node.get("titleComplete"),
                    book_node.get("description"),
                    book_node.get("imageUrl"),
                    book_node.get("webUrl"),
                    details.get("asin"),
                    details.get("isbn"),
                    details.get("isbn13"),
                    details.get("format"),
                    details.get("numPages"),
                    details.get("publisher"),
                    details.get("publicationTime"),
                    language_name,
                    now,
                ),
            )

            primary_edge = book_node.get("primaryContributorEdge")
            secondary_edges = book_node.get("secondaryContributorEdges") or []

            contributors_to_save = []
            if primary_edge and primary_edge.get("node"):
                contributors_to_save.append((primary_edge["node"], primary_edge.get("role")))
            for edge in secondary_edges:
                if edge and edge.get("node"):
                    contributors_to_save.append((edge["node"], edge.get("role")))

            for node, role in contributors_to_save:
                followers = node.get("followers") or {}
                works = node.get("works") or {}
                db_conn.execute(
                    """
                    INSERT OR REPLACE INTO contributors (
                        legacy_id, name, web_url, is_gr_author, works_count, followers_count
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        node.get("legacyId"),
                        node.get("name"),
                        node.get("webUrl"),
                        1 if node.get("isGrAuthor") else 0,
                        works.get("totalCount") or 0,
                        followers.get("totalCount") or 0,
                    ),
                )
                db_conn.execute(
                    """
                    INSERT OR REPLACE INTO book_contributors (
                        legacy_id, contributor_id, role
                    ) VALUES (?, ?, ?)
                    """,
                    (book_node.get("legacyId"), node.get("legacyId"), role),
                )

            book_series_list = book_node.get("bookSeries") or []
            for bs in book_series_list:
                series_node = bs.get("series")
                if series_node:
                    series_title = series_node.get("title")
                    series_web_url = series_node.get("webUrl")

                    db_conn.execute(
                        "INSERT OR IGNORE INTO series (title, web_url) VALUES (?, ?)", (series_title, series_web_url)
                    )
                    series_row = db_conn.execute(
                        "SELECT id FROM series WHERE title = ? AND web_url = ?", (series_title, series_web_url)
                    ).fetchone()

                    if series_row:
                        series_id = series_row["id"]
                        db_conn.execute(
                            """
                            INSERT OR REPLACE INTO work_series (
                                work_id, series_id, user_position
                            ) VALUES (?, ?, ?)
                            """,
                            (work_node.get("legacyId"), series_id, bs.get("userPosition")),
                        )

            book_genres = book_node.get("bookGenres") or []
            for bg in book_genres:
                genre_node = bg.get("genre")
                if genre_node:
                    genre_name = genre_node.get("name")
                    db_conn.execute(
                        "INSERT OR REPLACE INTO genres (name, web_url) VALUES (?, ?)",
                        (genre_name, genre_node.get("webUrl")),
                    )
                    db_conn.execute(
                        """
                        INSERT OR REPLACE INTO work_genres (
                            work_id, genre_name
                        ) VALUES (?, ?)
                        """,
                        (work_node.get("legacyId"), genre_name),
                    )

            awards = work_details.get("awardsWon") or []
            for award in awards:
                db_conn.execute(
                    """
                    INSERT OR REPLACE INTO work_awards (
                        work_id, name, category, designation, awarded_at, web_url
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        work_node.get("legacyId"),
                        award.get("name"),
                        award.get("category"),
                        award.get("designation"),
                        award.get("awardedAt"),
                        award.get("webUrl"),
                    ),
                )

            for edition in all_editions:
                db_conn.execute(
                    """
                    INSERT OR REPLACE INTO known_editions (
                        work_id, legacy_id, title
                    ) VALUES (?, ?, ?)
                    """,
                    (work_node.get("legacyId"), edition.get("legacyId"), edition.get("title")),
                )

            for sig in social_list:
                db_conn.execute(
                    """
                    INSERT OR REPLACE INTO social_signals (
                        work_id, signal_name, count, updated_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (work_node.get("legacyId"), sig.get("name"), sig.get("count"), now),
                )

            for rank_idx, sim in enumerate(similar_list):
                sim_work = sim.get("work") or {}
                sim_stats = sim_work.get("stats") or {}
                db_conn.execute(
                    """
                    INSERT OR REPLACE INTO similar_books (
                        work_id, similar_legacy_id, rank, title, average_rating, ratings_count, fetched_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        work_node.get("legacyId"),
                        sim.get("legacyId"),
                        rank_idx + 1,
                        sim.get("title"),
                        sim_stats.get("averageRating"),
                        sim_stats.get("ratingsCount"),
                        now,
                    ),
                )

                sim_legacy_id = sim.get("legacyId")
                avg_rating = sim_stats.get("averageRating")
                ratings_count = sim_stats.get("ratingsCount")
                if sim_legacy_id and avg_rating is not None and ratings_count is not None:
                    priority = avg_rating - avg_rating / math.log10(ratings_count + 10)
                    db_conn.execute(
                        """
                        INSERT INTO crawl_queue (legacy_id, priority, status, discovered_via)
                        VALUES (?, ?, 'pending', 'similar')
                        ON CONFLICT(legacy_id) DO UPDATE SET
                            priority = MAX(priority, excluded.priority)
                        WHERE status = 'pending'
                        """,
                        (sim_legacy_id, priority),
                    )

            db_conn.execute(
                """
                INSERT OR REPLACE INTO crawl_queue (legacy_id, status, error_count, processed_at)
                VALUES (?, 'done', 0, ?)
                """,
                (legacy_id, now),
            )

            # 6. Sibling pruning
            db_conn.execute(
                """
                UPDATE crawl_queue
                SET status = 'skipped_known_edition', processed_at = ?
                WHERE legacy_id IN (SELECT legacy_id FROM known_editions WHERE work_id = ?)
                  AND status = 'pending'
                """,
                (now, work_node.get("legacyId")),
            )

    except Exception as e:
        row = db_conn.execute("SELECT error_count FROM crawl_queue WHERE legacy_id = ?", (legacy_id,)).fetchone()
        error_count = (row["error_count"] or 0) + 1 if row else 1
        status = "error" if error_count >= 3 else "pending"
        db_conn.execute(
            """
            INSERT OR REPLACE INTO crawl_queue (legacy_id, status, error_count, last_error_message, processed_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (legacy_id, status, error_count, str(e), now),
        )
        db_conn.commit()
        return False

    return True


def populate_seeds(db_conn):
    cursor = db_conn.execute("SELECT DISTINCT legacy_id FROM reader_libraries WHERE legacy_id IS NOT NULL")
    seeds = [row["legacy_id"] for row in cursor.fetchall()]
    for seed in seeds:
        db_conn.execute(
            """
            INSERT OR IGNORE INTO crawl_queue (legacy_id, status, priority, discovered_via)
            VALUES (?, 'pending', 0.0, 'seed')
            """,
            (seed,),
        )
    db_conn.commit()


def handle_force_recrawl(db_conn):
    db_conn.execute(
        """
        UPDATE crawl_queue
        SET status = 'pending'
        WHERE legacy_id IN (
            SELECT legacy_id FROM books WHERE datetime(fetched_at) < datetime('now', '-30 days')
        )
        AND status = 'done'
        """
    )
    db_conn.commit()


async def run_crawler(limit=None, concurrency=2, force_recrawl=False, db_path=None):
    from dotenv import load_dotenv

    load_dotenv()

    db.init_db(db_path)

    api_key = os.getenv("X_API_KEY", "da2-xpgsdydkbregjhpr6ejzqdhuwy")
    headers = {
        "content-type": "application/json",
        "x-api-key": api_key,
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    with db.get_connection(db_path) as db_conn:
        populate_seeds(db_conn)

        if force_recrawl:
            handle_force_recrawl(db_conn)

        allowed_sources = ["seed"]
        if limit is not None:
            allowed_sources.append("similar")

        if limit is not None and limit > 0:
            already_scraped = db_conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
            if already_scraped >= limit:
                print(f"  Already have {already_scraped} scraped books (>= limit of {limit}). Skipping crawler.")
                return

        in_flight = set()

        async def fetch_task(client, legacy_id):
            try:
                await resolve_and_save_book(client, headers, db_conn, legacy_id, allowed_sources)
            finally:
                in_flight.remove(legacy_id)

        completed_count = db_conn.execute(
            "SELECT COUNT(*) FROM crawl_queue WHERE status IN ('done', 'error', 'mapped_to_canonical', 'skipped_known_edition')"
        ).fetchone()[0]
        pending_count = db_conn.execute("SELECT COUNT(*) FROM crawl_queue WHERE status = 'pending'").fetchone()[0]

        pbar = tqdm(
            total=completed_count + pending_count,
            initial=completed_count,
            unit="book",
            desc="Crawling books",
        )

        async with httpx.AsyncClient() as client:
            active_tasks = set()

            while True:
                if limit is not None and limit > 0:
                    total_scraped = db_conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
                    if total_scraped >= limit:
                        print(f"  Reached crawl target: {total_scraped} total scraped books (limit was {limit}). Stopping.")
                        break
                    remaining_budget = limit - total_scraped
                else:
                    remaining_budget = None

                needed = concurrency - len(in_flight)
                if needed > 0:
                    placeholders = ",".join("?" for _ in allowed_sources)
                    query = f"""
                        SELECT legacy_id FROM crawl_queue
                        WHERE status = 'pending'
                          AND discovered_via IN ({placeholders})
                        ORDER BY (discovered_via = 'seed') DESC, priority DESC
                    """
                    pending_rows = db_conn.execute(query, allowed_sources).fetchall()
                    pending_ids = [row["legacy_id"] for row in pending_rows if row["legacy_id"] not in in_flight]

                    if remaining_budget is not None:
                        pending_ids = pending_ids[: min(needed, remaining_budget)]
                    else:
                        pending_ids = pending_ids[:needed]

                    for legacy_id in pending_ids:
                        in_flight.add(legacy_id)
                        task = asyncio.create_task(fetch_task(client, legacy_id))
                        active_tasks.add(task)

                if active_tasks:
                    done, active_tasks = await asyncio.wait(active_tasks, timeout=0.1)

                placeholders = ",".join("?" for _ in allowed_sources)
                total_pending = db_conn.execute(
                    f"SELECT COUNT(*) FROM crawl_queue WHERE status = 'pending' AND discovered_via IN ({placeholders})",
                    allowed_sources,
                ).fetchone()[0]

                completed_now = db_conn.execute(
                    "SELECT COUNT(*) FROM crawl_queue WHERE status IN ('done', 'error', 'mapped_to_canonical', 'skipped_known_edition')"
                ).fetchone()[0]
                pending_now = db_conn.execute(
                    f"SELECT COUNT(*) FROM crawl_queue WHERE status = 'pending' AND discovered_via IN ({placeholders})",
                    allowed_sources,
                ).fetchone()[0]

                pbar.total = completed_now + pending_now
                pbar.n = completed_now
                pbar.refresh()

                if total_pending == 0 and not in_flight:
                    break

                if len(in_flight) >= concurrency or (total_pending == 0 and in_flight):
                    if active_tasks:
                        done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
                    else:
                        await asyncio.sleep(0.1)

        pbar.close()
