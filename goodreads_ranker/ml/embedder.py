import contextlib
import hashlib
import subprocess
import time

import numpy as np
from tqdm import tqdm

from goodreads_ranker.core import config, db, utils


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


@contextlib.contextmanager
def _ensure_ollama(embedding_model: str):
    import ollama

    server_started_here = False
    proc = None

    try:
        ollama.list()
    except Exception:
        print("Ollama server not detected — starting 'ollama serve'...")
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

    if _normalise(embedding_model) not in {_normalise(m) for m in available}:
        print(f"Model '{embedding_model}' not found locally — pulling...")
        ollama.pull(embedding_model)
        print(f"Model '{embedding_model}' ready.")

    try:
        yield
    finally:
        if server_started_here and proc is not None:
            proc.terminate()
            proc.wait(timeout=10)


def find_stale_or_missing_embeddings(existing_embeddings: dict[int, dict], all_inputs: dict[int, str]) -> list[int]:
    input_hashes = {legacy_id: hashlib.md5(text.encode("utf-8")).hexdigest() for legacy_id, text in all_inputs.items()}

    queued = []
    for legacy_id, _ in all_inputs.items():
        existing = existing_embeddings.get(legacy_id)
        if existing is None:
            queued.append(legacy_id)
            continue
        if existing.get("vector") is None:
            queued.append(legacy_id)
            continue
        if not db.is_valid_embedding_blob(existing.get("vector")):
            queued.append(legacy_id)
            continue
        current_hash = input_hashes.get(legacy_id, "")
        if not current_hash or existing.get("text_hash") != current_hash:
            queued.append(legacy_id)

    return queued


def generate_embeddings(batch_size=1, embedding_model=None, db_path=None):
    db.init_db(db_path)

    if not embedding_model:
        embedding_model = config.get_embedding_model()

    with db.get_connection(db_path) as db_conn:
        candidate_ids = db.get_candidate_book_legacy_ids(db_conn)
        metadata_list = db.get_book_metadata_for_embedding(db_conn, candidate_ids=candidate_ids)
        if not metadata_list:
            print("No books found matching candidate criteria. Run crawler/seeder first.")
            return

        all_inputs = {}
        for r in metadata_list:
            legacy_id = r["legacy_id"]
            title = r["title"]
            author_name = r["author_name"]
            authors_post = author_name.strip() if author_name and author_name.strip() else ""
            genres_post = format_string_for_embedding(r["genres"], kind="genre")

            desc_clean = utils.clean_description_text(r["description"])

            embedding_input = join_embedding_parts(title, authors_post, genres_post, desc_clean)
            all_inputs[legacy_id] = embedding_input

        existing_embeddings = db.get_existing_embeddings(db_conn, embedding_model)
        missing_ids = find_stale_or_missing_embeddings(existing_embeddings, all_inputs)

        if not missing_ids:
            print(f"Nothing to embed: all candidate books have valid embeddings for model '{embedding_model}'.")
            return

        import ollama

        with _ensure_ollama(embedding_model):
            unit = "book" if batch_size == 1 else f"batch ({batch_size} books)"
            for i in tqdm(range(0, len(missing_ids), batch_size), unit=unit, desc=embedding_model):
                batch_ids = missing_ids[i : i + batch_size]
                batch_strings = [all_inputs[bid] for bid in batch_ids]

                try:
                    response = ollama.embed(model=embedding_model, input=batch_strings)
                    embeddings_list = response["embeddings"]

                    vectors = np.array(embeddings_list, dtype=np.float32)

                    batch_hashes = {bid: hashlib.md5(all_inputs[bid].encode("utf-8")).hexdigest() for bid in batch_ids}
                    db.save_embeddings(db_conn, batch_ids, vectors, embedding_model, batch_hashes)
                except Exception as e:
                    print(f"\n  Error generating embeddings for batch starting with legacy_id {batch_ids[0]}: {e}")
                    continue
