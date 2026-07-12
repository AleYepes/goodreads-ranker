import asyncio
import base64
import contextlib
import json
import math
import os
import re
import time
from contextlib import nullcontext
from datetime import date

import httpx
from tqdm import tqdm

from . import db
from .utils import USER_AGENT, parse_id_from_slug, parse_slug

API_URL = "https://kxbwmqov6jgg3daaamb744ycu4.appsync-api.us-east-1.amazonaws.com/graphql"

BOOK_QUERY = """
query getBookByLegacyId($legacyBookId: Int!, $pagination: PaginationInput!) {
  getBookByLegacyId(legacyId: $legacyBookId) {
    id
    legacyId
    title
    titleComplete
    description
    webUrl
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
      genre { id name webUrl }
    }
    details {
      asin isbn isbn13 format numPages publisher publicationTime
      language { name }
    }
    work {
      bestBook { legacyId }
      stats {
        ratingsCountDist
      }
      details {
        publicationTime
        awardsWon { name webUrl awardedAt category designation }
      }
      editions(pagination: $pagination) {
        totalCount
        edges { node { id legacyId title } }
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


class InvalidLegacyIdError(Exception):
    pass


_gql_sem: asyncio.Semaphore | None = None
_cooldown_until: float = 0.0


def make_after_token(page_number: int) -> str:
    raw = json.dumps({"next_page": page_number}, separators=(",", ":"))
    return base64.b64encode(raw.encode()).decode()


async def gql(client: httpx.AsyncClient, headers: dict, operation_name: str, query: str, variables: dict) -> dict:
    global _cooldown_until

    for attempt in range(3):
        now = time.monotonic()
        if _cooldown_until > now:
            await asyncio.sleep(_cooldown_until - now)

        try:
            async with _gql_sem or nullcontext():
                resp = await client.post(
                    API_URL,
                    json={"operationName": operation_name, "variables": variables, "query": query},
                    headers=headers,
                    timeout=15.0,
                )

            if resp.status_code in {403, 429}:
                cooldown_delay = 30.0 * (attempt + 1)
                _cooldown_until = max(_cooldown_until, time.monotonic() + cooldown_delay)
                tqdm.write(
                    f"  Rate limited (status {resp.status_code}) on {operation_name}, attempt {attempt + 1}/3 — backing off {cooldown_delay:.1f}s"
                )
                await asyncio.sleep(cooldown_delay)
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


async def fetch_similar_books(
    client: httpx.AsyncClient, headers: dict, book_kca_id: str, limit: int = 20
) -> list[dict]:
    if not book_kca_id:
        return []
    try:
        data = await gql(client, headers, "GetSimilarBooks", SIMILAR_QUERY, {"id": book_kca_id, "limit": limit})
        similar = data.get("getSimilarBooks") or {}
        edges = similar.get("edges") or []
        return [edge["node"] for edge in edges if edge and edge.get("node")]
    except Exception:
        return []


async def resolve_and_save_book(
    client: httpx.AsyncClient,
    headers: dict,
    db_conn,
    legacy_id: int,
    allowed_sources: list,
    previous_editions: list[dict] | None = None,
    pagination_token: str | None = None,
) -> bool:
    now = date.today().strftime("%Y-%m-%d")

    try:
        pagination_variables: dict[str, int | str] = {"limit": 20}
        if pagination_token:
            pagination_variables["after"] = pagination_token

        full_data = await gql(
            client,
            headers,
            "getBookByLegacyId",
            BOOK_QUERY,
            {"legacyBookId": legacy_id, "pagination": pagination_variables},
        )

        book_node = full_data.get("getBookByLegacyId")
        if not book_node:
            db_conn.execute(
                """
                INSERT OR REPLACE INTO crawl_queue (book_id, status, error_count, last_error_message, date_processed)
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
                INSERT OR REPLACE INTO crawl_queue (book_id, status, error_count, last_error_message, date_processed)
                VALUES (?, 'error', 1, 'No best book legacy ID resolved', ?)
                """,
                (legacy_id, now),
            )
            db_conn.commit()
            return False

        editions_conn = work_node.get("editions") or {}
        total_editions = editions_conn.get("totalCount") or 0
        page1_edges = editions_conn.get("edges") or []
        current_page_editions = [edge["node"] for edge in page1_edges if edge and edge.get("node")]

        if legacy_id != best_book_legacy_id:
            db_conn.execute(
                """
                INSERT OR REPLACE INTO crawl_queue (book_id, status, error_count, date_processed)
                VALUES (?, 'mapped_to_canonical', 0, ?)
                """,
                (legacy_id, now),
            )
            db_conn.commit()

            canonical_row = db_conn.execute(
                "SELECT 1 FROM books WHERE legacy_id = ?", (best_book_legacy_id,)
            ).fetchone()
            if canonical_row:
                db_conn.execute(
                    "INSERT OR IGNORE INTO book_editions (book_id, edition_legacy_id, edition_kca_id, title) "
                    "VALUES (?, ?, ?, ?)",
                    (best_book_legacy_id, legacy_id, book_node.get("id"), book_node.get("title")),
                )

                for edition in current_page_editions:
                    db_conn.execute(
                        """
                        INSERT OR REPLACE INTO book_editions (
                            book_id, edition_legacy_id, edition_kca_id, title
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (best_book_legacy_id, edition.get("legacyId"), edition.get("id"), edition.get("title")),
                    )

                db_conn.commit()
                return True

            db_conn.execute(
                """
                INSERT OR IGNORE INTO crawl_queue (book_id, status, priority, discovered_via)
                VALUES (?, 'pending', 0.0, 'seed')
                """,
                (best_book_legacy_id,),
            )
            db_conn.commit()

            result = await resolve_and_save_book(
                client,
                headers,
                db_conn,
                best_book_legacy_id,
                allowed_sources,
                previous_editions=current_page_editions,
                pagination_token=make_after_token(2),
            )
            if result:
                db_conn.execute(
                    "INSERT OR IGNORE INTO book_editions (book_id, edition_legacy_id, edition_kca_id, title) "
                    "VALUES (?, ?, ?, ?)",
                    (best_book_legacy_id, legacy_id, book_node.get("id"), book_node.get("title")),
                )
                db_conn.commit()
            return result

        book_kca_id = book_node.get("id")
        all_editions = current_page_editions + (previous_editions or [])

        try:
            similar_list = await fetch_similar_books(client, headers, book_kca_id)
        except Exception:
            similar_list = []

        pending_before = (
            db_conn.execute(
                """
            SELECT COUNT(*)
            FROM crawl_queue
            WHERE status='pending'
            AND book_id IN ({})
            """.format(",".join("?" * len(all_editions))),
                [e["legacyId"] for e in all_editions],
            ).fetchone()[0]
            if all_editions
            else 0
        )

        tqdm.write(
            f"[Editions] {book_node.get('title')[:50]!r} "
            f"total={total_editions} "
            f"page1_editions={len(current_page_editions)} "
            f"pending_skipped={pending_before}"
        )

        with db_conn:
            work_stats = work_node.get("stats") or {}
            work_details = work_node.get("details") or {}
            dist = work_stats.get("ratingsCountDist") or []
            stars = [dist[i] if len(dist) > i else 0 for i in range(5)]

            details = book_node.get("details") or {}
            lang = details.get("language") or {}
            language_name = lang.get("name")

            db_conn.execute(
                """
                INSERT OR REPLACE INTO books (
                    legacy_id, kca_id, title, title_complete, description, web_url,
                    asin, isbn, isbn13, format, num_pages, language_name, publisher, publication_time,
                    original_publication_time, star_1, star_2, star_3, star_4, star_5,
                    date_fetched
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    book_node.get("legacyId"),
                    book_kca_id,
                    book_node.get("title"),
                    book_node.get("titleComplete"),
                    book_node.get("description"),
                    book_node.get("webUrl"),
                    details.get("asin"),
                    details.get("isbn"),
                    details.get("isbn13"),
                    details.get("format"),
                    details.get("numPages"),
                    language_name,
                    details.get("publisher"),
                    details.get("publicationTime"),
                    work_details.get("publicationTime"),
                    stars[0],
                    stars[1],
                    stars[2],
                    stars[3],
                    stars[4],
                    now,
                ),
            )

            primary_edge = book_node.get("primaryContributorEdge")
            if primary_edge and primary_edge.get("node"):
                primary_node = primary_edge["node"]
                followers = primary_node.get("followers") or {}
                works = primary_node.get("works") or {}
                db_conn.execute(
                    """
                    INSERT OR REPLACE INTO contributors (
                        legacy_id, kca_id, name, web_url, is_gr_author, works_count, followers_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        primary_node.get("legacyId"),
                        primary_node.get("id"),
                        primary_node.get("name"),
                        primary_node.get("webUrl"),
                        1 if primary_node.get("isGrAuthor") else 0,
                        works.get("totalCount") or 0,
                        followers.get("totalCount") or 0,
                    ),
                )
                db_conn.execute(
                    """
                    INSERT OR REPLACE INTO book_contributors (book_id, contributor_id, role, is_primary)
                    VALUES (?, ?, ?, 1)
                    """,
                    (
                        book_node.get("legacyId"),
                        primary_node.get("legacyId"),
                        primary_edge.get("role"),
                    ),
                )

            secondary_edges = book_node.get("secondaryContributorEdges") or []
            for s_edge in secondary_edges:
                if s_edge and s_edge.get("node"):
                    s_node = s_edge["node"]
                    followers = s_node.get("followers") or {}
                    works = s_node.get("works") or {}
                    db_conn.execute(
                        """
                        INSERT OR REPLACE INTO contributors (
                            legacy_id, kca_id, name, web_url, is_gr_author, works_count, followers_count
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            s_node.get("legacyId"),
                            s_node.get("id"),
                            s_node.get("name"),
                            s_node.get("webUrl"),
                            1 if s_node.get("isGrAuthor") else 0,
                            works.get("totalCount") or 0,
                            followers.get("totalCount") or 0,
                        ),
                    )
                    db_conn.execute(
                        """
                        INSERT OR REPLACE INTO book_contributors (book_id, contributor_id, role, is_primary)
                        VALUES (?, ?, ?, 0)
                        """,
                        (
                            book_node.get("legacyId"),
                            s_node.get("legacyId"),
                            s_edge.get("role"),
                        ),
                    )

            book_series_list = book_node.get("bookSeries") or []
            for bs in book_series_list:
                series_node = bs.get("series")
                if series_node:
                    series_title = series_node.get("title")
                    series_web_url = series_node.get("webUrl")
                    if series_web_url:
                        try:
                            series_legacy_id = parse_id_from_slug(series_web_url)
                            db_conn.execute(
                                """
                                INSERT OR IGNORE INTO series (legacy_id, kca_id, title, web_url)
                                VALUES (?, ?, ?, ?)
                                """,
                                (series_legacy_id, series_node.get("id"), series_title, series_web_url),
                            )

                            pos = bs.get("userPosition")
                            pos_int = None
                            if pos is not None:
                                with contextlib.suppress(ValueError, TypeError):
                                    pos_int = int(float(pos))

                            db_conn.execute(
                                """
                                INSERT OR REPLACE INTO book_series (book_id, series_id, position)
                                VALUES (?, ?, ?)
                                """,
                                (book_node.get("legacyId"), series_legacy_id, pos_int),
                            )
                        except Exception as e:
                            tqdm.write(f"    Failed parsing series {series_web_url}: {e}")

            book_genres = book_node.get("bookGenres") or []
            for bg in book_genres:
                genre_node = bg.get("genre")
                if genre_node:
                    genre_name = genre_node.get("name")
                    genre_web_url = genre_node.get("webUrl")
                    if genre_web_url:
                        try:
                            genre_legacy_id = parse_slug(genre_web_url)
                            db_conn.execute(
                                """
                                INSERT OR IGNORE INTO genres (legacy_id, kca_id, name, web_url)
                                VALUES (?, ?, ?, ?)
                                """,
                                (genre_legacy_id, genre_node.get("id"), genre_name, genre_web_url),
                            )
                            db_conn.execute(
                                """
                                INSERT OR REPLACE INTO book_genres (book_id, genre_id)
                                VALUES (?, ?)
                                """,
                                (book_node.get("legacyId"), genre_legacy_id),
                            )
                        except Exception as e:
                            tqdm.write(f"    Failed parsing genre {genre_web_url}: {e}")

            awards = work_details.get("awardsWon") or []
            for award in awards:
                award_web_url = award.get("webUrl")
                if award_web_url:
                    try:
                        award_legacy_id = parse_id_from_slug(award_web_url)
                        db_conn.execute(
                            """
                            INSERT OR IGNORE INTO awards (legacy_id, name, web_url)
                            VALUES (?, ?, ?)
                            """,
                            (award_legacy_id, award.get("name"), award_web_url),
                        )

                        awarded_at = award.get("awardedAt")
                        date_awarded = None
                        if awarded_at:
                            try:
                                from dateutil import parser as date_parser

                                date_awarded = date_parser.parse(str(awarded_at)).strftime("%Y-%m-%d")
                            except Exception:
                                if re.match(r"^\d{4}$", str(awarded_at)):
                                    date_awarded = f"{awarded_at}-01-01"
                                else:
                                    date_awarded = str(awarded_at)

                        db_conn.execute(
                            """
                            INSERT OR REPLACE INTO book_awards (book_id, award_id, category, designation, date_awarded)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                book_node.get("legacyId"),
                                award_legacy_id,
                                award.get("category"),
                                award.get("designation"),
                                date_awarded,
                            ),
                        )
                    except Exception as e:
                        tqdm.write(f"    Failed parsing award {award_web_url}: {e}")

            for edition in all_editions:
                db_conn.execute(
                    """
                    INSERT OR REPLACE INTO book_editions (
                        book_id, edition_legacy_id, edition_kca_id, title
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (book_node.get("legacyId"), edition.get("legacyId"), edition.get("id"), edition.get("title")),
                )

            for sim in similar_list:
                sim_work = sim.get("work") or {}
                sim_stats = sim_work.get("stats") or {}
                db_conn.execute(
                    """
                    INSERT OR REPLACE INTO book_similar_books (
                        book_id, similar_legacy_id, title, average_rating, ratings_count, date_fetched
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        book_node.get("legacyId"),
                        sim.get("legacyId"),
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
                        INSERT INTO crawl_queue (book_id, priority, status, discovered_via)
                        VALUES (?, ?, 'pending', 'similar')
                        ON CONFLICT(book_id) DO UPDATE SET
                            priority = MAX(priority, excluded.priority)
                        WHERE status = 'pending'
                        """,
                        (sim_legacy_id, priority),
                    )

            db_conn.execute(
                """
                INSERT OR REPLACE INTO crawl_queue (book_id, status, error_count, date_processed)
                VALUES (?, 'done', 0, ?)
                """,
                (legacy_id, now),
            )

            db_conn.execute(
                """
                UPDATE crawl_queue
                SET status = 'skipped_known_edition', date_processed = ?
                WHERE book_id IN (SELECT edition_legacy_id FROM book_editions WHERE book_id = ?)
                  AND status = 'pending'
                """,
                (now, book_node.get("legacyId")),
            )

    except InvalidLegacyIdError as e:
        db_conn.execute(
            """
            INSERT OR REPLACE INTO crawl_queue (book_id, status, error_count, last_error_message, date_processed)
            VALUES (?, 'error', 1, ?, ?)
            """,
            (legacy_id, str(e), now),
        )
        db_conn.commit()
        return False
    except Exception as e:
        row = db_conn.execute("SELECT error_count FROM crawl_queue WHERE book_id = ?", (legacy_id,)).fetchone()
        error_count = (row["error_count"] or 0) + 1 if row else 1
        status = "error" if error_count >= 3 else "pending"
        db_conn.execute(
            """
            INSERT OR REPLACE INTO crawl_queue (book_id, status, error_count, last_error_message, date_processed)
            VALUES (?, ?, ?, ?, ?)
            """,
            (legacy_id, status, error_count, str(e), now),
        )
        db_conn.commit()
        return False

    return True


def populate_seeds(db_conn):
    cursor = db_conn.execute("SELECT DISTINCT book_id FROM reader_libraries WHERE book_id IS NOT NULL")
    seeds = [row["book_id"] for row in cursor.fetchall()]
    for seed in seeds:
        db_conn.execute(
            """
            INSERT OR IGNORE INTO crawl_queue (book_id, status, priority, discovered_via)
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
        WHERE book_id IN (
            SELECT legacy_id FROM books WHERE date_fetched < date('now', '-30 days')
        )
        AND status = 'done'
        """
    )
    db_conn.commit()


async def run_crawler(limit=None, force_recrawl=False, db_path=None):
    global _gql_sem
    _gql_sem = asyncio.Semaphore(3)

    from dotenv import load_dotenv

    load_dotenv()

    db.init_db(db_path)

    headers = {
        "content-type": "application/json",
        "x-api-key": os.getenv("X_API_KEY", "da2-xpgsdydkbregjhpr6ejzqdhuwy"),
        "user-agent": USER_AGENT,
    }

    with db.get_connection(db_path) as db_conn:
        populate_seeds(db_conn)

        if force_recrawl:
            handle_force_recrawl(db_conn)

        expand_similar = limit is not None
        effective_limit = limit if (limit is not None and limit > 0) else None

        allowed_sources = ["seed"]
        if expand_similar:
            allowed_sources.append("similar")

        if effective_limit is not None:
            already_scraped = db_conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
            if already_scraped >= effective_limit:
                print(
                    f"  Already have {already_scraped} scraped books (>= limit of {effective_limit}). Skipping crawler."
                )
                return

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
            while True:
                if effective_limit is not None:
                    total_scraped = db_conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
                    if total_scraped >= effective_limit:
                        print(
                            f"  Reached crawl target: {total_scraped} total scraped books (limit was {effective_limit}). Stopping."
                        )
                        break

                placeholders = ",".join("?" for _ in allowed_sources)
                query = f"""
                    SELECT book_id FROM crawl_queue
                    WHERE status = 'pending'
                      AND discovered_via IN ({placeholders})
                    ORDER BY (discovered_via = 'seed') DESC, priority DESC
                    LIMIT 1
                """
                row = db_conn.execute(query, allowed_sources).fetchone()
                if not row:
                    break

                legacy_id = row["book_id"]

                await resolve_and_save_book(client, headers, db_conn, legacy_id, allowed_sources)

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

        pbar.close()
