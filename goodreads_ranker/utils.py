import logging
import re

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
            from datetime import datetime, timezone

            return datetime.fromtimestamp(as_float / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    except (ValueError, OverflowError, OSError):
        pass

    try:
        from dateutil import parser

        return parser.parse(cleaned).strftime("%Y-%m-%d")
    except (parser.ParserError, ValueError, OverflowError) as e:
        logger.warning("Failed to parse date string '%s': %s", dt_str, e)
        return cleaned


def format_string_for_embedding(items: list, kind: str | None = None) -> str:
    if not isinstance(items, list) or len(items) == 0:
        return ""

    n = len(items)
    res = items[0] if n == 1 else f"{', '.join(items[:-1])}{',' if n > 2 else ''} and {items[-1]}"

    prefix = f"{kind.capitalize()}{'s' if n > 1 else ''}: " if kind else ""
    return f"{prefix}{res}"


def join_embedding_parts(title: str, authors: str, genres: str, desc: str) -> str:
    text = f"Book: {title}\n"
    if authors:
        text += f"Written by: {authors}\n"
    if genres:
        text += f"{genres}\n"
    if desc:
        text += f"{desc}"
    return text
