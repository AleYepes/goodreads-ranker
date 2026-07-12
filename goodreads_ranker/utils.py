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
    if not slug:
        raise ValueError("Slug path cannot be empty")
    return slug.strip("/").split("/")[-1]


def as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


def parse_optional_int(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return int(value)
