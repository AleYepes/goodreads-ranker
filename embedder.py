import argparse
import os
import re

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

import db


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


def build_embedding_inputs(conn):
    """Query all books and build their formatted embedding input strings."""
    cursor = conn.execute(
        "SELECT book_id, title, authors, genres, description FROM books"
        " ORDER BY book_id"
    )
    rows = cursor.fetchall()

    inputs = {}
    for row in rows:
        book_id = int(row["book_id"])
        title = row["title"] or ""

        # Authors
        authors_raw = row["authors"] or ""
        authors_list = [a.strip() for a in authors_raw.split("|") if a.strip()]
        authors_post = format_string_for_embedding(authors_list, truncate=4)

        # Genres
        genres_raw = row["genres"] or ""
        genres_list = [g.strip() for g in genres_raw.split("|") if g.strip()]
        genres_post = format_string_for_embedding(genres_list, kind="genre")

        # Description
        desc_raw = row["description"] or ""
        desc_clean = re.sub(r"\s+", " ", desc_raw).strip()
        desc_list = [desc_clean] if desc_clean else []
        desc_post = format_string_for_embedding(desc_list, kind="description")

        # Combined string
        embedding_input = join_embedding_parts(
            title, authors_post, genres_post, desc_post
        )
        inputs[book_id] = embedding_input

    return inputs


def find_books_needing_embeddings(conn, all_inputs):
    if not all_inputs:
        return []

    cursor = conn.execute(
        """
        SELECT b.book_id,
               COALESCE(b.verified_embedding, 0) AS verified_embedding,
               e.dim,
               e.vector
        FROM books b
        LEFT JOIN embeddings e ON e.book_id = b.book_id
        ORDER BY b.book_id
        """
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
        if int(row["verified_embedding"] or 0) == 0:
            queued.append(book_id)

    return queued


def generate_embeddings(batch_size=128, model=None, db_path=None):
    """Identify books missing embeddings and generate them using Ollama."""
    load_dotenv()
    db.init_db(db_path)

    conn = db.get_connection(db_path)

    # Get all book IDs and input strings
    all_inputs = build_embedding_inputs(conn)
    if not all_inputs:
        print("No books found in database. Please crawl books first.")
        conn.close()
        return

    missing_ids = find_books_needing_embeddings(conn, all_inputs)

    if not missing_ids:
        print("Nothing to embed: all scraped books have valid verified embeddings.")
        conn.close()
        return

    if not model:
        model = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:8b")

    import ollama

    print(
        f"Generating embeddings for {len(missing_ids)} books using Ollama model '{model}'..."
    )

    for i in tqdm(range(0, len(missing_ids), batch_size)):
        batch_ids = missing_ids[i : i + batch_size]
        batch_strings = [all_inputs[bid] for bid in batch_ids]

        try:
            response = ollama.embed(model=model, input=batch_strings)
            embeddings_list = response["embeddings"]

            vectors = np.array(embeddings_list, dtype=np.float32)
            db.save_embeddings(conn, batch_ids, vectors)
            conn.executemany(
                "UPDATE books SET verified_embedding = 1 WHERE book_id = ?",
                [(int(book_id),) for book_id in batch_ids],
            )
            conn.commit()
        except Exception as e:
            print(
                f"\nError generating embeddings for batch starting with book_id {batch_ids[0]}: {e}"
            )
            # Continue with other batches
            continue

    print("Embedding generation process finished.")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Ollama embeddings for crawled books."
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for embedding calls"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Ollama embedding model name"
    )
    args = parser.parse_args()

    generate_embeddings(batch_size=args.batch_size, model=args.model)
