import contextlib
import os
import re
import subprocess
import time

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from . import db


@contextlib.contextmanager
def _ensure_ollama(model: str):
    import ollama

    server_started_here = False
    proc = None

    try:
        ollama.list()
    except Exception:
        print("  Ollama server not detected — starting 'ollama serve'...")
        proc = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        server_started_here = True
        for _ in range(30):
            time.sleep(1)
            try:
                ollama.list()
                break
            except Exception:
                pass
        else:
            proc.terminate()
            raise RuntimeError(
                "Ollama server did not become ready in time. "
                "Check that 'ollama' is on your PATH and is a valid installation."
            )

    available = {m.model for m in ollama.list().models if m.model is not None}

    def _normalise(name: str) -> str:
        return name if ":" in name else f"{name}:latest"

    if _normalise(model) not in {_normalise(m) for m in available}:
        print(f"  Model '{model}' not found locally — pulling (this may take a while)...")
        ollama.pull(model)
        print(f"  Model '{model}' ready.")

    try:
        yield
    finally:
        if server_started_here and proc is not None:
            proc.terminate()
            proc.wait(timeout=10)


def format_string_for_embedding(items, kind=None, truncate=0):
    if not isinstance(items, list) or len(items) == 0:
        return ""

    n = len(items)
    if n == 1:
        res = items[0]
    elif n > truncate > 1:
        res = f"{', '.join(items[:truncate])}, and {items[truncate]}"
    else:
        res = f"{', '.join(items[:-1])}{',' if n > 2 else ''} and {items[-1]}"

    prefix = f"{kind.capitalize()}{'s' if n > 1 else ''}: " if kind else ""
    return f"{prefix}{res}"


def join_embedding_parts(title, authors, genres, desc):
    text = f"Book: {title}\n"
    if authors:
        text += f"Written by: {authors}\n"
    if genres:
        text += f"{genres}\n"
    if desc:
        text += f"{desc}"
    return text


def build_embedding_inputs(db_conn):
    cursor = db_conn.execute("SELECT book_id, title, authors, genres, description FROM books ORDER BY book_id")
    rows = cursor.fetchall()

    inputs = {}
    for row in rows:
        book_id = int(row["book_id"])
        title = row["title"] or ""

        authors_raw = row["authors"] or ""
        authors_list = [a.strip() for a in authors_raw.split("|") if a.strip()]
        authors_post = format_string_for_embedding(authors_list, truncate=4)

        genres_raw = row["genres"] or ""
        genres_list = [g.strip() for g in genres_raw.split("|") if g.strip()]
        genres_post = format_string_for_embedding(genres_list, kind="genre")

        desc_raw = row["description"] or ""
        desc_clean = re.sub(r"\s+", " ", desc_raw).strip()
        desc_list = [desc_clean] if desc_clean else []
        desc_post = format_string_for_embedding(desc_list, kind="description")

        embedding_input = join_embedding_parts(title, authors_post, genres_post, desc_post)
        inputs[book_id] = embedding_input

    return inputs


def find_books_needing_embeddings(db_conn, all_inputs, model):
    if not all_inputs:
        return []

    import hashlib

    input_hashes = {book_id: hashlib.md5(text.encode("utf-8")).hexdigest() for book_id, text in all_inputs.items()}

    cursor = db_conn.execute(
        """
        SELECT b.book_id,
               e.dim,
               e.vector,
               e.text_hash
        FROM books b
        LEFT JOIN embeddings e ON e.book_id = b.book_id AND e.embedding_model = ?
        ORDER BY b.book_id
        """,
        (model,),
    )

    queued = []
    expected_dim = None
    for row in cursor.fetchall():
        book_id = int(row["book_id"])
        if book_id not in all_inputs:
            continue
        if row["vector"] is None:
            queued.append(book_id)
            continue
        if not db.is_valid_embedding_blob(row["vector"], row["dim"]):
            queued.append(book_id)
            continue
        dim = int(row["dim"])
        if expected_dim is None:
            expected_dim = dim
        elif dim != expected_dim:
            queued.append(book_id)
            continue

        current_hash = input_hashes.get(book_id, "")
        if not current_hash or row["text_hash"] != current_hash:
            queued.append(book_id)

    return queued


def generate_embeddings(batch_size=128, model=None, db_path=None):
    load_dotenv()
    db.init_db(db_path)

    if not model:
        model = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:8b")

    with db.get_connection(db_path) as db_conn:
        all_inputs = build_embedding_inputs(db_conn)
        if not all_inputs:
            print("  No books found. Run crawler first.")
            db_conn.close()
            return

        missing_ids = find_books_needing_embeddings(db_conn, all_inputs, model)

        if not missing_ids:
            print(f"  Nothing to embed: all books have valid embeddings for model '{model}'.")
            db_conn.close()
            return

        import hashlib

        import ollama

        with _ensure_ollama(model):
            for i in tqdm(range(0, len(missing_ids), batch_size)):
                batch_ids = missing_ids[i : i + batch_size]
                batch_strings = [all_inputs[bid] for bid in batch_ids]

                try:
                    response = ollama.embed(model=model, input=batch_strings)
                    embeddings_list = response["embeddings"]

                    vectors = np.array(embeddings_list, dtype=np.float32)

                    batch_hashes = {bid: hashlib.md5(all_inputs[bid].encode("utf-8")).hexdigest() for bid in batch_ids}
                    db.save_embeddings(db_conn, batch_ids, vectors, model, batch_hashes)
                except Exception as e:
                    print(f"\n  Error generating embeddings for batch starting with book_id {batch_ids[0]}: {e}")
                    continue
