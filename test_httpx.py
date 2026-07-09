import asyncio
import base64
import json
import math

import httpx

API_URL = "https://kxbwmqov6jgg3daaamb744ycu4.appsync-api.us-east-1.amazonaws.com/graphql"
HEADERS = {
    "content-type": "application/json",
    "x-api-key": "da2-xpgsdydkbregjhpr6ejzqdhuwy",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}

MAX_EDITION_PAGES = 10   # hard cap discovered during exploration (200 editions max)
EDITIONS_PAGE_SIZE = 20  # hard cap, cannot be raised

BOOK_QUERY = """
query getBookByLegacyId($legacyBookId: Int!) {
  getBookByLegacyId(legacyId: $legacyBookId) {
    id
    legacyId
    title
    titleComplete
    description
    descriptionStripped: description(stripped: true)
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
      bestBook { id legacyId }
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


def make_after_token(page_number: int) -> str:
    raw = json.dumps({"next_page": page_number}, separators=(",", ":"))
    return base64.b64encode(raw.encode()).decode()


async def gql(client: httpx.AsyncClient, operation_name: str, query: str, variables: dict) -> dict:
    resp = await client.post(
        API_URL,
        json={"operationName": operation_name, "variables": variables, "query": query},
        headers=HEADERS,
        timeout=15.0,
    )
    data = resp.json()
    if data.get("errors"):
        raise RuntimeError(f"{operation_name} errors: {data['errors']}")
    return data["data"]


async def fetch_book(client: httpx.AsyncClient, legacy_book_id: int) -> dict:
    data = await gql(client, "getBookByLegacyId", BOOK_QUERY, {"legacyBookId": legacy_book_id})
    return data["getBookByLegacyId"]


async def fetch_remaining_edition_pages(
    client: httpx.AsyncClient, canonical_legacy_id: int, total_count: int
) -> list[dict]:
    """Fetch pages 2..N (page 1 already came back with the main book fetch),
    capped at MAX_EDITION_PAGES. Forged tokens are valid and can be
    requested concurrently."""
    if not total_count:
        return []

    page_count = min(math.ceil(total_count / EDITIONS_PAGE_SIZE), MAX_EDITION_PAGES)
    if page_count <= 1:
        return []

    async def fetch_page(page_number: int):
        variables = {
            "legacyBookId": canonical_legacy_id,
            "pagination": {"limit": EDITIONS_PAGE_SIZE, "after": make_after_token(page_number)},
        }
        data = await gql(client, "getBookByLegacyId", EDITIONS_PAGE_QUERY, variables)
        book_data = data.get("getBookByLegacyId") or {}
        work_data = book_data.get("work") or {}
        editions_data = work_data.get("editions") or {}
        edges = editions_data.get("edges") or []
        return [edge["node"] for edge in edges if edge and edge.get("node")]

    pages = await asyncio.gather(*(fetch_page(p) for p in range(2, page_count + 1)))
    return [node for page in pages for node in page]


async def fetch_similar_books(client: httpx.AsyncClient, book_kca_id: str, limit: int = 20) -> list[dict]:
    data = await gql(client, "GetSimilarBooks", SIMILAR_QUERY, {"id": book_kca_id, "limit": limit})
    similar = data.get("getSimilarBooks") or {}
    edges = similar.get("edges") or []
    return [edge["node"] for edge in edges if edge and edge.get("node")]


async def fetch_social_signals(client: httpx.AsyncClient, book_kca_id: str) -> list[dict]:
    data = await gql(client, "GetSocialSignals", SOCIAL_QUERY, {"bookId": book_kca_id})
    return data.get("getSocialSignals") or []


async def resolve_book(client: httpx.AsyncClient, seed_legacy_id: int) -> dict:
    """Orchestrates the full resolution flow for one seed legacy_book_id,
    returning a dict shaped roughly like the schema.sql tables it feeds."""

    seed_book = await fetch_book(client, seed_legacy_id)
    if not seed_book:
        raise ValueError(f"Book with legacy ID {seed_legacy_id} not found.")

    work = seed_book.get("work") or {}
    best_book = work.get("bestBook") or {}
    best_book_legacy_id = best_book.get("legacyId") or seed_book.get("legacyId")

    if best_book_legacy_id != seed_book.get("legacyId"):
        canonical_book = await fetch_book(client, best_book_legacy_id)
    else:
        canonical_book = seed_book

    if not canonical_book:
        raise ValueError(f"Canonical book with legacy ID {best_book_legacy_id} not found.")

    canonical_work = canonical_book.get("work") or {}
    editions_conn = canonical_work.get("editions") or {}
    total_editions = editions_conn.get("totalCount") or 0

    page1_edges = editions_conn.get("edges") or []
    page1_editions = [edge["node"] for edge in page1_edges if edge and edge.get("node")]
    remaining_editions = await fetch_remaining_edition_pages(
        client, canonical_book.get("legacyId"), total_editions
    )
    all_known_editions = page1_editions + remaining_editions

    similar, social = await asyncio.gather(
        fetch_similar_books(client, canonical_book.get("id")),
        fetch_social_signals(client, canonical_book.get("id")),
    )

    work_details = canonical_work.get("details") or {}
    work_stats = canonical_work.get("stats") or {}
    book_details = canonical_book.get("details") or {}

    flat_book_details = {k: v for k, v in book_details.items() if k != "language"}
    language_name = (book_details.get("language") or {}).get("name")

    primary_edge = canonical_book.get("primaryContributorEdge")
    secondary_edges = canonical_book.get("secondaryContributorEdges") or []
    
    book_contributors = []
    if primary_edge and primary_edge.get("node"):
        book_contributors.append({
            "contributor": primary_edge["node"],
            "role": primary_edge.get("role")
        })
    for edge in secondary_edges:
        if edge and edge.get("node"):
            book_contributors.append({
                "contributor": edge["node"],
                "role": edge.get("role")
            })

    book_series = [
        {"series": bs["series"], "user_position": bs.get("userPosition")}
        for bs in (canonical_book.get("bookSeries") or [])
        if bs and bs.get("series")
    ]

    book_genres = [
        bg["genre"]["name"]
        for bg in (canonical_book.get("bookGenres") or [])
        if bg and bg.get("genre") and bg["genre"].get("name")
    ]

    return {
        "works": {
            "id": canonical_work.get("id"),
            "legacy_id": canonical_work.get("legacyId"),
            "best_book_legacy_id": best_book_legacy_id,
            "original_title": work_details.get("originalTitle"),
            "publication_time": work_details.get("publicationTime"),
            "web_url": work_details.get("webUrl"),
            "shelves_url": work_details.get("shelvesUrl"),
            "average_rating": work_stats.get("averageRating"),
            "ratings_count": work_stats.get("ratingsCount"),
            "ratings_count_dist": work_stats.get("ratingsCountDist"),
            "text_reviews_count": work_stats.get("textReviewsCount"),
            "text_reviews_language_counts": work_stats.get("textReviewsLanguageCounts"),
            "editions_total_count": total_editions,
            "editions_coverage_complete": total_editions <= MAX_EDITION_PAGES * EDITIONS_PAGE_SIZE,
        },
        "work_awards": work_details.get("awardsWon") or [],
        "books": {
            "id": canonical_book.get("id"),
            "legacy_id": canonical_book.get("legacyId"),
            "work_id": canonical_work.get("id"),
            "is_canonical": True,
            "title": canonical_book.get("title"),
            "title_complete": canonical_book.get("titleComplete"),
            "description_stripped": canonical_book.get("descriptionStripped"),
            "image_url": canonical_book.get("imageUrl"),
            "web_url": canonical_book.get("webUrl"),
            **flat_book_details,
            "language_name": language_name,
        },
        "book_contributors": book_contributors,
        "book_series": book_series,
        "book_genres": book_genres,
        "known_editions": all_known_editions,
        "social_signals": social,
        "similar_books": similar,
    }


async def main():
    seed_ids = [28139880, 1]
    async with httpx.AsyncClient() as client:
        for seed_id in seed_ids:
            print(f"\n{'=' * 60}\nResolving seed legacy_book_id={seed_id}\n{'=' * 60}")
            result = await resolve_book(client, seed_id)
            print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())