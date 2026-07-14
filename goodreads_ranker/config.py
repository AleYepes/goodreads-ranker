import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:8b"


def get_goodreads_email() -> str | None:
    return os.environ.get("GOODREADS_EMAIL")


def get_goodreads_password() -> str | None:
    return os.environ.get("GOODREADS_PASSWORD")


def get_embedding_model() -> str:
    return os.environ.get("OLLAMA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
