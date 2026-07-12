import asyncio

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
    def seed(self, force_seed=False, list_ids=None):
        from dotenv import load_dotenv

        from goodreads_ranker import seeder

        load_dotenv()
        db.init_db()

        parsed_ids = None
        if list_ids is not None:
            if isinstance(list_ids, str):
                parsed_ids = [int(x.strip()) for x in list_ids.split(",") if x.strip()]
            else:
                parsed_ids = [int(x) for x in list_ids]
        asyncio.run(seeder.scrape_reader_libraries(list_ids=parsed_ids, force_seed=as_bool(force_seed)))

    def crawl(self, limit=None, force_recrawl=False):
        from goodreads_ranker import crawler

        db.init_db()
        limit = parse_optional_int(limit)
        asyncio.run(
            crawler.run_crawler(
                limit=limit,
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
        limit=None,
        force_seed=False,
        force_recrawl=False,
        optimize=False,
        model=None,
    ):
        db.init_db()
        force_seed = as_bool(force_seed)
        force_recrawl = as_bool(force_recrawl)
        optimize = as_bool(optimize)
        limit = parse_optional_int(limit)

        print("\nSeeding database")
        self.seed(force_seed=force_seed)

        print("\nCrawling book details")
        self.crawl(limit=limit, force_recrawl=force_recrawl)

        print("\nGenerating embeddings")
        self.embed(model=model)

        print("\nRunning models and predictions")
        self.rank(interactive=False, optimize=optimize, model=model)

        print("\nPipeline run finished successfully!")


def main():
    fire.Fire(GoodreadsRankerCLI)


if __name__ == "__main__":
    main()
