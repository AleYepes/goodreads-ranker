import asyncio
import os
from pathlib import Path

import fire

import db


class GoodreadsRankerCLI:
    """CLI orchestrator for the Goodreads Ranker pipeline."""

    def seed(self, user=False, friends=False, force=False):
        """
        Download user library and/or scrape friend list reviews.

        Parameters:
        -----------
        user : bool
            Download user library export.
        friends : bool
            Scrape friend review list ratings.
        force : bool
            Force re-downloading/re-scraping.
        """
        from dotenv import load_dotenv

        import seeder

        load_dotenv()

        if not user and not friends:
            user = True
            friends = True

        if user:
            email = os.getenv("GOODREADS_EMAIL")
            password = os.getenv("GOODREADS_PASSWORD")
            asyncio.run(seeder.download_user_library(email, password, force=force))

        if friends:
            asyncio.run(seeder.scrape_friend_ratings(force_all=force))

    def crawl(self, limit=None, concurrency=2):
        """
        Run the Goodreads book detail crawler.

        Parameters:
        -----------
        limit : int, optional
            Hard cap on number of books to crawl.
        concurrency : int
            Number of concurrent Playwright pages.
        """
        import crawler

        # Parse limit to int if it's a string
        if limit is not None:
            limit = int(limit)
        asyncio.run(crawler.run_crawler(limit=limit, concurrency=int(concurrency)))

    def embed(self, batch_size=128, model=None):
        """
        Generate Ollama embeddings for crawled books.

        Parameters:
        -----------
        batch_size : int
            Batch size for embedding generation.
        model : str, optional
            Ollama embedding model name.
        """
        import embedder

        embedder.generate_embeddings(batch_size=int(batch_size), model=model)

    def rank(self, interactive=False, optimize=False):
        """
        Run ELO ratings refinement and ML recommendation model.

        Parameters:
        -----------
        interactive : bool
            Launch terminal interface for ELO pairwise ranking.
        optimize : bool
            Tune model hyperparameters using Nevergrad.
        """
        import ranker

        ranker.run_ranking(interactive=bool(interactive), optimize=bool(optimize))

    def run_pipeline(self, crawl_limit=100):
        """
        Run the entire seeding, crawling, embedding, and ranking pipeline end-to-end.

        Parameters:
        -----------
        crawl_limit : int
            Cap on crawled books for the pipeline run.
        """
        print("STEP 1: Seeding database")
        self.seed(user=True, friends=True, force=False)

        print("\nSTEP 2: Crawling book details")
        self.crawl(limit=int(crawl_limit))

        print("\nSTEP 3: Generating embeddings")
        self.embed()

        print("\nSTEP 4: Running models and predictions")
        self.rank(interactive=False, optimize=False)

        print("\nPipeline run finished successfully!")

    def migrate_csv(self):
        """Migrate all existing CSV files in the data/ directory into the SQLite database."""
        import numpy as np
        import pandas as pd

        db.init_db()
        conn = db.get_connection()

        # 1. Migrate user_library
        user_lib_path = Path("data/goodreads_library_export.csv")
        if user_lib_path.exists():
            print("Migrating user library...")
            df = pd.read_csv(user_lib_path)
            df = db.normalise_library_columns(df)

            # Clean Excel formulas from ISBNs
            from seeder import clean_isbn

            if "isbn" in df.columns:
                df["isbn"] = df["isbn"].apply(clean_isbn)
            if "isbn13" in df.columns:
                df["isbn13"] = df["isbn13"].apply(clean_isbn)

            df = df.replace({np.nan: None})

            columns = [
                "book_id",
                "title",
                "author",
                "author_lf",
                "additional_authors",
                "isbn",
                "isbn13",
                "my_rating",
                "publisher",
                "binding",
                "number_of_pages",
                "year_published",
                "original_publication_year",
                "date_read",
                "date_added",
                "bookshelves",
                "bookshelves_with_positions",
                "exclusive_shelf",
                "my_review",
                "spoiler",
                "private_notes",
                "read_count",
                "owned_copies",
            ]
            rows = [tuple(row.get(col) for col in columns) for _, row in df.iterrows()]
            db.upsert_rows(conn, "user_library", rows, columns)
            print(f"  Migrated {len(rows)} user library records.")
        else:
            print("User library CSV not found. Skipping.")

        # 2. Migrate friend_lists
        friend_lists_path = Path("data/friend_lists.csv")
        if friend_lists_path.exists():
            print("Migrating friend lists...")
            df = pd.read_csv(friend_lists_path)
            df = df.replace({np.nan: None})
            rows = [tuple(row) for row in df.values]
            columns = list(df.columns)
            db.upsert_rows(conn, "friend_lists", rows, columns)
            print(f"  Migrated {len(rows)} friend lists.")
        else:
            print("Friend lists CSV not found. Skipping.")

        # 3. Migrate friend_ratings
        friend_ratings_path = Path("data/friend_ratings.csv")
        if friend_ratings_path.exists():
            print("Migrating friend ratings...")
            df = pd.read_csv(friend_ratings_path)
            df = df.replace({np.nan: None})
            rows = [tuple(row) for row in df.values]
            columns = list(df.columns)
            db.upsert_rows(conn, "friend_ratings", rows, columns)
            print(f"  Migrated {len(rows)} friend ratings.")
        else:
            print("Friend ratings CSV not found. Skipping.")

        # 4. Migrate books
        books_path = Path("data/books.csv")
        if books_path.exists():
            print("Migrating crawled books...")
            df = pd.read_csv(books_path)
            df.columns = [
                col.replace("1_star", "star_1")
                .replace("2_star", "star_2")
                .replace("3_star", "star_3")
                .replace("4_star", "star_4")
                .replace("5_star", "star_5")
                for col in df.columns
            ]
            df = df.replace({np.nan: None})
            columns = [
                "book_id",
                "title",
                "authors",
                "avg_rating",
                "review_count",
                "num_pages",
                "lang",
                "star_1",
                "star_2",
                "star_3",
                "star_4",
                "star_5",
                "genres",
                "series",
                "year",
                "description",
                "similar_books",
                "primary_author",
                "author_followers",
                "want_to_read",
                "author_num_books",
                "currently_reading",
            ]
            rows = [tuple(row.get(col) for col in columns) for _, row in df.iterrows()]
            db.upsert_rows(conn, "books", rows, columns)
            print(f"  Migrated {len(rows)} crawled books.")
        else:
            print("Crawled books CSV not found. Skipping.")

        # 5. Migrate elo_ratings
        elo_path = Path("data/elo_ratings.csv")
        if elo_path.exists():
            print("Migrating ELO ratings...")
            df = pd.read_csv(elo_path)
            df = df.replace({np.nan: None})
            rows = [tuple(row) for row in df.values]
            columns = list(df.columns)
            db.upsert_rows(conn, "elo_ratings", rows, columns)
            print(f"  Migrated {len(rows)} ELO rating records.")
        else:
            print("ELO ratings CSV not found. Skipping.")

        # 6. Migrate embeddings (chunked)
        embeddings_path = Path("data/8b_embeddings.csv")
        if embeddings_path.exists():
            print("Migrating book embeddings (chunked, memory-safe)...")
            chunk_size = 5000
            for chunk in pd.read_csv(
                embeddings_path, chunksize=chunk_size, index_col="book_id"
            ):
                book_ids = chunk.index.values
                vectors = chunk.values.astype(np.float32)
                db.save_embeddings(conn, book_ids, vectors)
                print(f"  Migrated {len(book_ids)} embeddings...")
            print("Finished migrating embeddings.")
        else:
            print("Embeddings CSV not found. Skipping.")

        conn.close()
        print("CSV data migration complete!")


if __name__ == "__main__":
    fire.Fire(GoodreadsRankerCLI)
