import asyncio

import fire

from goodreads_ranker import db
from goodreads_ranker.utils import as_bool, parse_optional_int


class GoodreadsRankerCLI:
    def seed(self, force_seed=False, library_ids=None):
        from dotenv import load_dotenv

        from goodreads_ranker import seeder

        print("\nSeeding database")
        load_dotenv()
        db.init_db()
        if library_ids is not None:
            if isinstance(library_ids, str):
                library_ids = [int(x.strip()) for x in library_ids.split(",") if x.strip()]
            else:
                library_ids = [int(x) for x in library_ids]
        asyncio.run(seeder.scrape_libraries(library_ids=library_ids, force_seed=as_bool(force_seed)))

    def crawl(self, limit=None, force_recrawl=False):
        from goodreads_ranker import crawler

        print("\nCrawling book details")
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

        print("\nGenerating embeddings")
        db.init_db()
        embedder.generate_embeddings(batch_size=int(batch_size), model=model)

    def rank(self, interactive=False, optimize=False, model=None):
        from goodreads_ranker import ranker

        print("\nRunning models and predictions")
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
        limit = parse_optional_int(limit)
        self.seed(force_seed=force_seed)
        self.crawl(limit=limit, force_recrawl=force_recrawl)
        self.embed(model=model)
        self.rank(interactive=False, optimize=optimize, model=model)
        print("\nPipeline run finished successfully!")


def main():
    fire.Fire(GoodreadsRankerCLI)


if __name__ == "__main__":
    main()
