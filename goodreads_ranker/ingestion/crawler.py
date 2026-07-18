import asyncio
import contextlib
import re
from datetime import date

import httpx
from tqdm import tqdm

from goodreads_ranker.core import db
from goodreads_ranker.core.utils import parse_date_str, parse_id_from_slug, parse_slug
from goodreads_ranker.ingestion import api_client


def flatten_book(book_node: dict, work_node: dict, book_kca_id: str | None, now: str) -> dict:
    work_stats = work_node.get("stats") or {}
    work_details = work_node.get("details") or {}
    dist = work_stats.get("ratingsCountDist") or []
    stars = [dist[i] if len(dist) > i else 0 for i in range(5)]
    details = book_node.get("details") or {}
    lang = details.get("language") or {}
    language_name = lang.get("name")

    return {
        "legacy_id": book_node.get("legacyId"),
        "kca_id": book_kca_id,
        "title": book_node.get("title"),
        "title_complete": book_node.get("titleComplete"),
        "description": book_node.get("description"),
        "web_url": book_node.get("webUrl"),
        "asin": details.get("asin"),
        "isbn": details.get("isbn"),
        "isbn13": details.get("isbn13"),
        "format": details.get("format"),
        "num_pages": details.get("numPages"),
        "language_name": language_name,
        "publisher": details.get("publisher"),
        "publication_time": details.get("publicationTime"),
        "original_publication_time": work_details.get("publicationTime"),
        "star_1": stars[0],
        "star_2": stars[1],
        "star_3": stars[2],
        "star_4": stars[3],
        "star_5": stars[4],
        "date_fetched": now,
    }


def flatten_contributors(book_node: dict) -> list[dict]:
    res = []
    primary_edge = book_node.get("primaryContributorEdge")
    if primary_edge:
        res.append(flatten_contributor_node(primary_edge, is_primary=1))
    secondary_edges = book_node.get("secondaryContributorEdges") or []
    for s_edge in secondary_edges:
        res.append(flatten_contributor_node(s_edge, is_primary=0))
    return [c for c in res if c is not None]


def flatten_contributor_node(edge: dict, is_primary: int) -> dict | None:
    if not edge or not edge.get("node"):
        return None
    node = edge["node"]
    followers = node.get("followers") or {}
    works = node.get("works") or {}
    return {
        "legacy_id": node.get("legacyId"),
        "kca_id": node.get("id"),
        "name": node.get("name"),
        "web_url": node.get("webUrl"),
        "is_gr_author": 1 if node.get("isGrAuthor") else 0,
        "works_count": works.get("totalCount") or 0,
        "followers_count": followers.get("totalCount") or 0,
        "role": edge.get("role"),
        "is_primary": is_primary,
    }


def flatten_series(book_node: dict) -> list[dict]:
    res = []
    book_series_list = book_node.get("bookSeries") or []
    for bs in book_series_list:
        series_node = bs.get("series")
        if series_node:
            series_title = series_node.get("title")
            series_web_url = series_node.get("webUrl")
            if series_web_url:
                try:
                    series_legacy_id = parse_id_from_slug(series_web_url)
                    pos = bs.get("userPosition")
                    pos_int = None
                    if pos is not None:
                        with contextlib.suppress(ValueError, TypeError):
                            pos_int = int(float(pos))
                    res.append(
                        {
                            "legacy_id": series_legacy_id,
                            "kca_id": series_node.get("id"),
                            "title": series_title,
                            "web_url": series_web_url,
                            "position": pos_int,
                        }
                    )
                except Exception as e:
                    tqdm.write(f"Failed parsing series {series_web_url}: {e}")
    return res


def flatten_genres(book_node: dict) -> list[dict]:
    res = []
    book_genres = book_node.get("bookGenres") or []
    for bg in book_genres:
        genre_node = bg.get("genre")
        if genre_node:
            genre_name = genre_node.get("name")
            genre_web_url = genre_node.get("webUrl")
            if genre_web_url:
                try:
                    genre_legacy_id = parse_slug(genre_web_url)
                    res.append(
                        {
                            "legacy_id": genre_legacy_id,
                            "kca_id": genre_node.get("id"),
                            "name": genre_name,
                            "web_url": genre_web_url,
                        }
                    )
                except Exception as e:
                    tqdm.write(f"Failed parsing genre {genre_web_url}: {e}")
    return res


def flatten_awards(work_details: dict) -> list[dict]:
    res = []
    awards = work_details.get("awardsWon") or []
    for award in awards:
        award_web_url = award.get("webUrl")
        if award_web_url:
            try:
                award_legacy_id = parse_id_from_slug(award_web_url)
                awarded_at = award.get("awardedAt")
                date_awarded = None
                if awarded_at:
                    parsed = parse_date_str(str(awarded_at))
                    if parsed:
                        date_awarded = parsed
                    elif re.match(r"^\d{4}$", str(awarded_at)):
                        date_awarded = f"{awarded_at}-01-01"
                    else:
                        date_awarded = str(awarded_at)
                res.append(
                    {
                        "legacy_id": award_legacy_id,
                        "name": award.get("name"),
                        "web_url": award_web_url,
                        "category": award.get("category"),
                        "designation": award.get("designation"),
                        "date_awarded": date_awarded,
                    }
                )
            except Exception as e:
                tqdm.write(f"Failed parsing award {award_web_url}: {e}")
    return res


def flatten_editions(all_editions: list[dict]) -> list[dict]:
    return [{"edition_legacy_id": ed.get("legacyId"), "edition_kca_id": ed.get("id")} for ed in all_editions if ed]


def flatten_similar_books(similar_list: list[dict]) -> list[dict]:
    res = []
    for sim in similar_list:
        sim_work = sim.get("work") or {}
        sim_stats = sim_work.get("stats") or {}
        res.append(
            {
                "legacy_id": sim.get("legacyId"),
                "average_rating": sim_stats.get("averageRating"),
                "ratings_count": sim_stats.get("ratingsCount"),
            }
        )
    return res


async def _resolve_canonical_edition(
    client: httpx.AsyncClient,
    headers: dict,
    db_conn,
    legacy_id: int,
    best_book_legacy_id: int,
    allowed_sources: list,
    sem: asyncio.Semaphore | None,
    cooldown: list[float],
    current_page_editions: list[dict],
    book_node: dict,
    now: str,
    db_path=None,
) -> bool:
    if db.book_exists(db_conn, best_book_legacy_id):
        db.link_editions_to_canonical(
            db_conn, best_book_legacy_id, legacy_id, book_node.get("id"), current_page_editions
        )
        db.mark_known_editions_skipped(db_conn, best_book_legacy_id, now)
        db.set_crawl_status(db_conn, legacy_id, "mapped_to_canonical", 0, None, now)
        return True

    result = await resolve_and_save_book(
        client,
        headers,
        db_conn,
        best_book_legacy_id,
        allowed_sources,
        sem,
        cooldown,
        previous_editions=current_page_editions,
        pagination_token=api_client.make_after_token(2),
        db_path=db_path,
    )
    if result:
        db.link_editions_to_canonical(db_conn, best_book_legacy_id, legacy_id, book_node.get("id"), [])
    db.set_crawl_status(db_conn, legacy_id, "mapped_to_canonical", 0, None, now)
    return result


async def resolve_and_save_book(
    client: httpx.AsyncClient,
    headers: dict,
    db_conn,
    legacy_id: int,
    allowed_sources: list,
    sem: asyncio.Semaphore | None,
    cooldown: list[float],
    previous_editions: list[dict] | None = None,
    pagination_token: str | None = None,
    db_path=None,
) -> bool:
    now = date.today().strftime("%Y-%m-%d")

    try:
        book_node = await api_client._fetch_book_node(
            client,
            headers,
            legacy_id,
            sem,
            cooldown,
            pagination_token=pagination_token,
        )
        if not book_node:
            db.set_crawl_status(db_conn, legacy_id, "error", 1, "Book not found (data is null)", now)
            return False

        work_node = book_node.get("work") or {}
        best_book = work_node.get("bestBook") or {}
        best_book_legacy_id = best_book.get("legacyId") or book_node.get("legacyId")

        if not best_book_legacy_id:
            db.set_crawl_status(db_conn, legacy_id, "error", 1, "No best book legacy ID resolved", now)
            return False

        editions_conn = work_node.get("editions") or {}
        page1_edges = editions_conn.get("edges") or []
        current_page_editions = [edge["node"] for edge in page1_edges if edge and edge.get("node")]

        if legacy_id != best_book_legacy_id:
            return await _resolve_canonical_edition(
                client,
                headers,
                db_conn,
                legacy_id,
                best_book_legacy_id,
                allowed_sources,
                sem,
                cooldown,
                current_page_editions,
                book_node,
                now,
                db_path=db_path,
            )

        book_kca_id = book_node.get("id")
        all_editions = current_page_editions + (previous_editions or [])

        try:
            similar_list = await api_client.fetch_similar_books(client, headers, book_kca_id, sem, cooldown)
        except Exception:
            similar_list = []

        # Flatten raw nodes to flat structures
        flat_book_data = flatten_book(book_node, work_node, book_kca_id, now)
        flat_contributors = flatten_contributors(book_node)
        flat_series = flatten_series(book_node)
        flat_genres = flatten_genres(book_node)
        flat_awards = flatten_awards(work_node.get("details") or {})
        flat_editions = flatten_editions(all_editions)
        flat_similar = flatten_similar_books(similar_list)

        # Save to database
        db.save_book_core(db_conn, flat_book_data)
        db.save_contributors(db_conn, book_node.get("legacyId"), flat_contributors)
        db.save_series(db_conn, book_node.get("legacyId"), flat_series)
        db.save_genres(db_conn, book_node.get("legacyId"), flat_genres)
        db.save_awards(db_conn, book_node.get("legacyId"), flat_awards)
        db.save_editions(db_conn, book_node.get("legacyId"), flat_editions)
        db.save_similar_books_and_enqueue(db_conn, book_node.get("legacyId"), flat_similar, now)

        db.mark_known_editions_skipped(db_conn, book_node.get("legacyId"), now)
        db.set_crawl_status(db_conn, legacy_id, "done", 0, None, now)

    except api_client.InvalidLegacyIdError as e:
        db.set_crawl_status(db_conn, legacy_id, "error", 1, str(e), now)
        return False
    except Exception as e:
        error_count = db.get_crawl_error_count(db_conn, legacy_id) + 1
        status = "error" if error_count >= 3 else "pending"
        db.set_crawl_status(db_conn, legacy_id, status, error_count, str(e), now)
        return False

    return True


async def run_crawler(limit=None, force_crawl=False, db_path=None):
    sem = asyncio.Semaphore(3)
    cooldown = [0.0]

    db.init_db(db_path)
    headers = api_client.build_headers()

    with db.get_connection(db_path) as db_conn:
        db.populate_seeds(db_conn)

        if force_crawl:
            db.handle_force_crawl(db_conn)

        expand_similar = limit is not None
        effective_limit = limit if (limit is not None and limit > 0) else None

        allowed_sources = ["seed"]
        if expand_similar:
            allowed_sources.append("similar")

        if effective_limit is not None:
            already_scraped = db.get_total_books_count(db_conn)
            if already_scraped >= effective_limit:
                print(
                    f"Already have {already_scraped} scraped books (>= limit of {effective_limit}). Skipping crawler."
                )
                return

        completed_count = db.count_crawl_queue(
            db_conn, ["done", "error", "mapped_to_canonical", "skipped_known_edition"], allowed_sources
        )
        pending_count = db.count_crawl_queue(db_conn, ["pending"], allowed_sources)

        pbar = tqdm(
            total=completed_count + pending_count,
            initial=completed_count,
            unit="book",
            desc="Crawling books",
        )

        async with httpx.AsyncClient() as client:
            while True:
                if effective_limit is not None:
                    total_scraped = db.get_total_books_count(db_conn)
                    if total_scraped >= effective_limit:
                        print(
                            f"Reached crawl target: {total_scraped} total scraped books (limit was {effective_limit}). Stopping."
                        )
                        break

                batch_ids = db.get_pending_crawl_batch(db_conn, allowed_sources, limit=5)
                if not batch_ids:
                    break

                async def process_one(legacy_id):
                    with db.get_connection(db_path) as task_conn:
                        await resolve_and_save_book(
                            client, headers, task_conn, legacy_id, allowed_sources, sem, cooldown, db_path=db_path
                        )

                await asyncio.gather(*(process_one(lid) for lid in batch_ids))

                completed_now = db.count_crawl_queue(
                    db_conn, ["done", "error", "mapped_to_canonical", "skipped_known_edition"], allowed_sources
                )
                pending_now = db.count_crawl_queue(db_conn, ["pending"], allowed_sources)

                pbar.total = completed_now + pending_now
                pbar.n = completed_now
                pbar.refresh()

        pbar.close()
