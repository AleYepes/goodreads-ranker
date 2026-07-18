import asyncio
import base64
import json
import time
from contextlib import nullcontext

import httpx
from tqdm import tqdm

from goodreads_ranker.core import config
from goodreads_ranker.core.utils import USER_AGENT

API_URL = config.get_api_url()

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
        edges { node { id legacyId } }
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


def make_after_token(page_number: int) -> str:
    raw = json.dumps({"next_page": page_number}, separators=(",", ":"))
    return base64.b64encode(raw.encode()).decode()


def build_headers() -> dict:
    return {
        "content-type": "application/json",
        "x-api-key": config.get_api_key(),
        "user-agent": USER_AGENT,
    }


async def gql(
    client: httpx.AsyncClient,
    headers: dict,
    operation_name: str,
    query: str,
    variables: dict,
    sem: asyncio.Semaphore | None,
    cooldown: list[float],
) -> dict:
    for attempt in range(3):
        now = time.monotonic()
        if cooldown[0] > now:
            await asyncio.sleep(cooldown[0] - now)

        try:
            async with sem or nullcontext():
                resp = await client.post(
                    API_URL,
                    json={"operationName": operation_name, "variables": variables, "query": query},
                    headers=headers,
                    timeout=15.0,
                )

            if resp.status_code in {403, 429}:
                cooldown_delay = 30.0 * (attempt + 1)
                cooldown[0] = max(cooldown[0], time.monotonic() + cooldown_delay)
                tqdm.write(
                    f"Rate limited (status {resp.status_code}) on {operation_name}, attempt {attempt + 1}/3 — backing off {cooldown_delay:.1f}s"
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
    client: httpx.AsyncClient,
    headers: dict,
    book_kca_id: str | None,
    sem: asyncio.Semaphore | None,
    cooldown: list[float],
    limit: int = 20,
) -> list[dict]:
    if not book_kca_id:
        return []
    try:
        data = await gql(
            client, headers, "GetSimilarBooks", SIMILAR_QUERY, {"id": book_kca_id, "limit": limit}, sem, cooldown
        )
        similar = data.get("getSimilarBooks") or {}
        edges = similar.get("edges") or []
        return [edge["node"] for edge in edges if edge and edge.get("node")]
    except Exception:
        return []


async def _fetch_book_node(
    client: httpx.AsyncClient,
    headers: dict,
    legacy_id: int,
    sem: asyncio.Semaphore | None,
    cooldown: list[float],
    pagination_token: str | None = None,
) -> dict | None:
    pagination_variables: dict[str, int | str] = {"limit": 20}
    if pagination_token:
        pagination_variables["after"] = pagination_token

    full_data = await gql(
        client,
        headers,
        "getBookByLegacyId",
        BOOK_QUERY,
        {"legacyBookId": legacy_id, "pagination": pagination_variables},
        sem,
        cooldown,
    )
    return full_data.get("getBookByLegacyId")
