import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_EMBEDDING_MODEL = "leoipulsar/qwen3-embedding:8b"


def get_goodreads_email() -> str | None:
    return os.environ.get("GOODREADS_EMAIL")


def get_goodreads_password() -> str | None:
    return os.environ.get("GOODREADS_PASSWORD")


def get_embedding_model() -> str:
    return os.environ.get("OLLAMA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def get_api_url() -> str:
    return os.environ.get(
        "GOODREADS_API_URL", "https://kxbwmqov6jgg3daaamb744ycu4.appsync-api.us-east-1.amazonaws.com/graphql"
    )


def get_api_key() -> str:
    return os.environ.get("GOODREADS_API_KEY", "da2-xpgsdydkbregjhpr6ejzqdhuwy")
