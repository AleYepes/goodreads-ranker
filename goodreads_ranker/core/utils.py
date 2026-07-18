import logging
import re
from datetime import UTC, datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def clean_text(text: str | None) -> str:
    return text.strip().replace("\n", "") if text else ""


def parse_id_from_slug(slug: str) -> int:
    match = re.search(r"(\d+)", slug.strip("/").split("/")[-1])
    if not match:
        raise ValueError(f"Could not parse numeric ID from slug: {slug}")
    return int(match.group(1))


def parse_slug(slug: str) -> str:
    if not slug:
        raise ValueError("Slug path cannot be empty")
    return slug.strip("/").split("/")[-1]


def parse_date_str(dt_str: str) -> str:
    """Normalise an arbitrary date string to YYYY-MM-DD.

    Falls back to the stripped input if parsing fails.
    """
    if not dt_str:
        return ""

    cleaned = dt_str.strip()

    try:
        as_float = float(cleaned)
        if as_float.is_integer():
            return datetime.fromtimestamp(as_float / 1000, tz=UTC).strftime("%Y-%m-%d")
    except ValueError, OverflowError, OSError:
        pass

    try:
        from dateutil import parser

        return parser.parse(cleaned).strftime("%Y-%m-%d")
    except Exception as e:
        logger.warning("Failed to parse date string '%s': %s", dt_str, e)
        return cleaned


def compute_continuous(elo_df: pd.DataFrame) -> pd.Series:
    results = pd.Series(np.nan, index=elo_df.index, dtype=float)
    for stars in elo_df["original_rating"].dropna().unique():
        mask = elo_df["original_rating"] == stars
        subset = elo_df.loc[mask, "elo_score"]

        if len(subset) > 1 and subset.max() > subset.min():
            norm = (subset - subset.min()) / (subset.max() - subset.min())
            results.loc[mask] = (stars + (norm * 0.99) - 0.5).to_numpy(copy=True)
        else:
            results.loc[mask] = float(stars)
    results.index = elo_df["legacy_id"]
    return results


def merge_elo_state(existing_rows: list[dict], target_ratings: dict[int, float]) -> pd.DataFrame:
    if existing_rows:
        elo_df = pd.DataFrame(existing_rows)
    else:
        elo_df = pd.DataFrame(columns=["legacy_id", "original_rating", "elo_score", "matches_played"])

    elo_df = elo_df.set_index("legacy_id")
    elo_df = elo_df[elo_df.index.isin(target_ratings.keys())]

    common_books = elo_df.index.intersection(target_ratings.keys())
    for bid in common_books:
        elo_df.at[bid, "original_rating"] = target_ratings[bid]

    new_books_ids = [bid for bid in target_ratings if bid not in elo_df.index]
    if new_books_ids:
        new_entries = pd.DataFrame(
            {
                "original_rating": [target_ratings[bid] for bid in new_books_ids],
                "elo_score": 1200.0,
                "matches_played": 0,
            },
            index=new_books_ids,
        )
        elo_df = pd.concat([elo_df, new_entries])

    elo_df = elo_df.reset_index().rename(columns={"index": "legacy_id"})
    return elo_df


def assemble_embedding_matrix(legacy_ids: list[int], vectors_by_id: dict[int, bytes]) -> tuple[np.ndarray, np.ndarray]:
    valid_vectors = []
    valid_mask = []
    expected_dim = None

    for legacy_id in legacy_ids:
        bid = int(legacy_id)
        vector_blob = vectors_by_id.get(bid)
        if vector_blob is None or len(vector_blob) % np.dtype(np.float32).itemsize != 0:
            valid_mask.append(False)
            continue

        vector = np.frombuffer(vector_blob, dtype=np.float32).copy()
        dim = len(vector)
        if expected_dim is None:
            expected_dim = dim
        elif dim != expected_dim:
            valid_mask.append(False)
            continue

        valid_vectors.append(vector)
        valid_mask.append(True)

    matrix = (
        np.vstack(valid_vectors).astype(np.float32, copy=False) if valid_vectors else np.empty((0, 0), dtype=np.float32)
    )
    return np.array(valid_mask, dtype=bool), matrix
