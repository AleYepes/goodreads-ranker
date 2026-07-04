import asyncio
import os

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
    def seed(self, user=None, friends=None, force=False, list_ids=None):
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
            asyncio.run(seeder.download_my_library(email, password, force=force))

        if friends:
            parsed_ids = None
            if list_ids is not None:
                if isinstance(list_ids, str):
                    parsed_ids = [int(x.strip()) for x in list_ids.split(",") if x.strip()]
                else:
                    parsed_ids = [int(x) for x in list_ids]
            asyncio.run(seeder.scrape_reader_libraries(list_ids=parsed_ids, force_all=force))

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
        ranker.run_ranking(interactive=as_bool(interactive), optimize=as_bool(optimize), model=model)

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

        print("\nPipeline run finished successfully!")


def main():
    fire.Fire(GoodreadsRankerCLI)


if __name__ == "__main__":
    main()
