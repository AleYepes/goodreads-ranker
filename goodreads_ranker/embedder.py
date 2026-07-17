import contextlib
import hashlib
import subprocess
import time

import numpy as np
from tqdm import tqdm

from . import config, db

# ---------------------------------------------------------------------------
# Ollama lifecycle
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------


def generate_embeddings(batch_size=64, embedding_model=None, db_path=None):
    db.init_db(db_path)

    if not embedding_model:
        embedding_model = config.get_embedding_model()

    with db.get_connection(db_path) as db_conn:
        all_inputs = db.build_embedding_inputs(db_conn)
        if not all_inputs:
            print("No books found. Run crawler first.")
            return

        missing_ids = db.find_stale_or_missing_embeddings(db_conn, all_inputs, embedding_model)

        if not missing_ids:
            print(f"Nothing to embed: all books have valid embeddings for model '{embedding_model}'.")
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
