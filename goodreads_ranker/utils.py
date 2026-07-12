import re

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def clean_text(text):
    return text.strip().replace("\n", "") if text else ""


def parse_id_from_slug(slug: str) -> int:
    match = re.search(r"(\d+)", slug.strip("/").split("/")[-1])
    if not match:
        raise ValueError(f"Could not parse numeric ID from slug: {slug}")
    return int(match.group(1))


def parse_slug(slug: str) -> str:
    """Extracts the last non-empty segment of a URL path as the slug."""
    if not slug:
        raise ValueError("Slug path cannot be empty")
    return slug.strip("/").split("/")[-1]
