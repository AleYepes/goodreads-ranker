import asyncio
import os
import sqlite3

import fire

from goodreads_ranker import db


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


class GoodreadsRankerCLI:
    def seed(self, user=None, friends=None, force=False):
        from dotenv import load_dotenv

        from goodreads_ranker import seeder

        load_dotenv()
        db.init_db()

        default_both = user is None and friends is None
        user = as_bool(user) if user is not None else False
        friends = as_bool(friends) if friends is not None else False
        force = as_bool(force)

        if default_both:
            user = True
            friends = True

        if user:
            email = os.getenv("GOODREADS_EMAIL")
            password = os.getenv("GOODREADS_PASSWORD")
            asyncio.run(seeder.download_user_library(email, password, force=force))

        if friends:
            asyncio.run(seeder.scrape_friend_ratings(force_all=force))

    def crawl(self, limit=None, concurrency=2, force_recrawl=False):
        from goodreads_ranker import crawler

        db.init_db()
        limit = parse_optional_int(limit)
        asyncio.run(
            crawler.run_crawler(
                limit=limit,
                concurrency=int(concurrency),
                force_recrawl=as_bool(force_recrawl),
            )
        )

    def embed(self, batch_size=128, model=None):
        from goodreads_ranker import embedder

        db.init_db()
        embedder.generate_embeddings(batch_size=int(batch_size), model=model)

    def rank(self, interactive=False, optimize=False, model=None):
        from goodreads_ranker import ranker

        db.init_db()
        ranker.run_ranking(
            interactive=as_bool(interactive), optimize=as_bool(optimize), model=model
        )

    def run_pipeline(
        self,
        seed=True,
        seed_user=True,
        seed_friends=True,
        limit=None,
        force_recrawl=False,
        optimize=False,
        model=None,
    ):
        db.init_db()
        seed = as_bool(seed)
        seed_user = as_bool(seed_user)
        seed_friends = as_bool(seed_friends)
        force_recrawl = as_bool(force_recrawl)
        optimize = as_bool(optimize)
        limit = parse_optional_int(limit)

        if seed:
            print("STEP 1: Seeding database")
            if seed_user or seed_friends:
                self.seed(user=seed_user, friends=seed_friends, force=False)
            else:
                print("No seed stages enabled")
        else:
            print("STEP 1: Seeding skipped")

        print("\nSTEP 2: Crawling book details")
        self.crawl(limit=limit, force_recrawl=force_recrawl)

        print("\nSTEP 3: Generating embeddings")
        self.embed(model=model)

        print("\nSTEP 4: Running models and predictions")
        self.rank(interactive=False, optimize=optimize, model=model)

        print("\nSTEP 5: Verifying pipeline state")
        self.verify()

        print("\nPipeline run finished successfully!")

    def verify(self, model=None):
        import hashlib
        import os

        if not model:
            model = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:8b")

        path = db.DB_PATH
        if not path.exists():
            print(f"Database not found at {path}.")
            return

        db_conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        db_conn.row_factory = sqlite3.Row
        try:
            tables = {
                row["name"]
                for row in db_conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }

            def table_count(table):
                if table not in tables:
                    return 0
                return db_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

            def table_columns(table):
                if table not in tables:
                    return set()
                return {
                    row["name"] for row in db_conn.execute(f"PRAGMA table_info({table})")
                }

            print("Row counts:")
            for table in (
                "user_library",
                "friend_lists",
                "friend_ratings",
                "books",
                "elo_ratings",
                "embeddings",
                "predictions",
                "model_params",
            ):
                print(f"  {table}: {table_count(table)}")

            incomplete_friends = 0
            friend_errors = 0
            if "friend_lists" in tables:
                incomplete_friends = db_conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM friend_lists
                    WHERE COALESCE(scrape_complete, 0) != 1
                    """
                ).fetchone()[0]
                if "scrape_error" in table_columns("friend_lists"):
                    friend_errors = db_conn.execute(
                        """
                        SELECT COUNT(*)
                        FROM friend_lists
                        WHERE scrape_error IS NOT NULL AND scrape_error != ''
                        """
                    ).fetchone()[0]

            seed_ids = set()
            if "user_library" in tables:
                seed_ids.update(
                    int(row["book_id"])
                    for row in db_conn.execute(
                        "SELECT book_id FROM user_library WHERE book_id IS NOT NULL"
                    )
                )
            if "friend_ratings" in tables:
                seed_ids.update(
                    int(row["book_id"])
                    for row in db_conn.execute(
                        "SELECT book_id FROM friend_ratings WHERE book_id IS NOT NULL"
                    )
                )
            scraped_ids = set()
            if "books" in tables:
                scraped_ids.update(
                    int(row["book_id"])
                    for row in db_conn.execute(
                        "SELECT book_id FROM books WHERE book_id IS NOT NULL"
                    )
                )
            seed_missing = len(seed_ids - scraped_ids)

            scraped_missing_embeddings = 0
            invalid_embeddings = 0
            outdated_embeddings = 0
            if {"books", "embeddings"}.issubset(tables):
                scraped_missing_embeddings = db_conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM books b
                    LEFT JOIN embeddings e
                        ON e.book_id = b.book_id AND e.embedding_model = ?
                    WHERE e.book_id IS NULL
                    """,
                    (model,),
                ).fetchone()[0]
                # Invalid blob embeddings for this model
                for row in db_conn.execute(
                    "SELECT dim, vector FROM embeddings WHERE embedding_model = ?",
                    (model,),
                ):
                    if not db.is_valid_embedding_blob(row["vector"], row["dim"]):
                        invalid_embeddings += 1
                from goodreads_ranker import embedder

                rw_conn = db.get_connection()
                all_inputs = embedder.build_embedding_inputs(rw_conn)
                rw_conn.close()
                current_hashes = {
                    bid: hashlib.md5(text.encode("utf-8")).hexdigest()
                    for bid, text in all_inputs.items()
                }
                for row in db_conn.execute(
                    "SELECT book_id, text_hash FROM embeddings WHERE embedding_model = ?",
                    (model,),
                ):
                    bid = int(row["book_id"])
                    if current_hashes.get(bid) != row["text_hash"]:
                        outdated_embeddings += 1

            prediction_count = table_count("predictions")
            null_prediction_fields = 0
            unread_scored = 0
            if "predictions" in tables:
                null_prediction_fields = db_conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM predictions
                    WHERE solo_pred_rating IS NULL
                       OR friend_pred_rating IS NULL
                       OR pred_rating IS NULL
                       OR final_rating IS NULL
                    """
                ).fetchone()[0]
                if "user_library" in tables:
                    unread_scored = db_conn.execute(
                        """
                        SELECT COUNT(*)
                        FROM predictions p
                        LEFT JOIN user_library u ON u.book_id = p.book_id
                        WHERE u.book_id IS NULL
                           OR COALESCE(u.exclusive_shelf, '') != 'read'
                        """
                    ).fetchone()[0]
                else:
                    unread_scored = prediction_count

            print(f"State checks (embedding model: '{model}'):")
            print(f"  friend_lists incomplete: {incomplete_friends}")
            print(f"  friend_lists with scrape_error: {friend_errors}")
            print(f"  seed books missing from books: {seed_missing}")
            print(f"  scraped books missing embeddings: {scraped_missing_embeddings}")
            print(f"  invalid embeddings: {invalid_embeddings}")
            print(f"  outdated embeddings (hash mismatch): {outdated_embeddings}")
            print(f"  prediction rows: {prediction_count}")
            print(f"  null prediction-field rows: {null_prediction_fields}")
            print(f"  unread scored count: {unread_scored}")
        finally:
            db_conn.close()


def main():
    fire.Fire(GoodreadsRankerCLI)


if __name__ == "__main__":
    main()
