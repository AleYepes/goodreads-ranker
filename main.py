import asyncio
import os
import sys

import fire

from goodreads_ranker import config, db


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
    def init(self, force_init=False):
        """Interactively configure required environment variables and save them to .env.

        Args:
            force_init (bool): Force review and update of existing environment variables.
        """
        import questionary
        from dotenv import set_key

        dotenv_path = ".env"

        vars_to_configure = [
            {
                "key": "GOODREADS_EMAIL",
                "label": "Goodreads email",
                "kind": "text",
                "current": config.get_goodreads_email(),
            },
            {
                "key": "GOODREADS_PASSWORD",
                "label": "Goodreads password",
                "kind": "password",
                "current": config.get_goodreads_password(),
            },
            {
                "key": "OLLAMA_EMBEDDING_MODEL",
                "label": "Ollama embedding model",
                "kind": "model",
                "current": config.get_embedding_model(),
            },
        ]

        anything_prompted = False

        for var in vars_to_configure:
            key = var["key"]
            current = var["current"]
            already_in_env = bool(current) and key in os.environ

            if not force_init and already_in_env:
                continue

            if not sys.stdin.isatty():
                raise RuntimeError(
                    f"{key} is not set. Run 'python main.py init' to configure or set it directly in .env"
                )

            anything_prompted = True

            if var["kind"] == "model":
                value = _prompt_model(current or config.DEFAULT_EMBEDDING_MODEL, force_init)
            elif var["kind"] == "password":
                prompt_label = f"{var['label']}"
                if force_init and current:
                    prompt_label += " (leave blank to keep existing)"
                raw = questionary.password(prompt_label + ":").ask()
                if raw is None:
                    print("Setup cancelled.")
                    return
                value = raw if raw.strip() else (current or "")
            else:
                value = questionary.text(
                    f"{var['label']}:",
                    default=current or "",
                ).ask()
                if value is None:
                    print("Setup cancelled.")
                    return
                value = value.strip()

            if value:
                set_key(dotenv_path, key, value)
                os.environ[key] = value

        if anything_prompted:
            print("\n✓ Configuration saved to .env")

    def seed(self, force_seed=False, library_ids=None):
        """Scrape Goodreads library shelves to seed the database with books and ratings.

        Args:
            force_seed (bool): Force seeding even if shelves were already seeded.
            library_ids (str|list): Comma-separated list, python list, or path to a file of library IDs.
        """
        import re

        from goodreads_ranker import seeder

        print("\nSeeding database")
        db.init_db()
        if library_ids is not None:
            if isinstance(library_ids, str) and os.path.exists(library_ids):
                with open(library_ids) as f:
                    content = f.read()
                ids = []
                for line in content.splitlines():
                    clean_line = line.split("#")[0].strip()
                    for num in re.findall(r"\b\d+\b", clean_line):
                        ids.append(int(num))
                library_ids = list(dict.fromkeys(ids))
            elif isinstance(library_ids, str):
                library_ids = [int(x.strip()) for x in library_ids.split(",") if x.strip()]
            else:
                library_ids = [int(x) for x in library_ids]

        asyncio.run(seeder.scrape_libraries(library_ids=library_ids, force_seed=as_bool(force_seed)))

    def crawl(self, limit=None, force_crawl=False):
        """Crawl full book details (metadata, genres, descriptions) for all seeded books.

        Args:
            limit (int): Limit the number of books to crawl.
            force_crawl (bool): Force crawling even if details have already been crawled.
        """
        from goodreads_ranker import crawler

        print("\nCrawling book details")
        db.init_db()
        limit = parse_optional_int(limit)
        asyncio.run(
            crawler.run_crawler(
                limit=limit,
                force_crawl=as_bool(force_crawl),
            )
        )

    def embed(self, batch_size=1, embedding_model=None):
        """Generate Ollama embeddings for all books that are missing or outdated.

        Args:
            batch_size (int): Batch size for generating embeddings.
            embedding_model (str): Ollama embedding model name (overrides configured model).
        """
        from goodreads_ranker import embedder

        print("\nGenerating embeddings")
        db.init_db()
        embedder.generate_embeddings(batch_size=int(batch_size), embedding_model=embedding_model or None)

    def rank(self, interactive=False, optimize=False, embedding_model=None):
        """Run the ranking model and write predictions to the database.

        Args:
            interactive (bool): Run the ranking model in interactive mode.
            optimize (bool): Optimize model hyperparameters.
            embedding_model (str): Ollama embedding model name (overrides configured model).
        """
        from goodreads_ranker import ranker

        print("\nRunning models and predictions")
        db.init_db()
        ranker.run_ranking(
            interactive=as_bool(interactive),
            optimize=as_bool(optimize),
            embedding_model=embedding_model or None,
        )

    def run_pipeline(
        self,
        force_init=False,
        force_seed=False,
        library_ids=None,
        limit=None,
        force_crawl=False,
        batch_size=128,
        embedding_model=None,
        interactive=False,
        optimize=False,
    ):
        """Run the complete pipeline from initialization to ranking.

        Args:
            force_init (bool): Force review and update of existing environment variables.
            force_seed (bool): Force seeding even if shelves were already seeded.
            library_ids (str|list): Comma-separated list, python list, or path to a file of library IDs.
            limit (int): Limit the number of books to crawl.
            force_crawl (bool): Force crawling even if details have already been crawled.
            batch_size (int): Batch size for generating embeddings.
            embedding_model (str): Ollama embedding model name.
            interactive (bool): Run the ranking model in interactive mode.
            optimize (bool): Optimize model hyperparameters.
        """
        self.init(force_init=force_init)
        self.seed(force_seed=force_seed, library_ids=library_ids)
        self.crawl(limit=limit, force_crawl=force_crawl)
        self.embed(batch_size=batch_size, embedding_model=embedding_model)
        self.rank(interactive=interactive, optimize=optimize, embedding_model=embedding_model)

        print("\n✓ Pipeline run finished successfully!")


def _prompt_model(current: str, force_init: bool) -> str:
    """Prompt the user to select or type an Ollama embedding model."""
    import questionary

    manual_option = "Enter manually…"

    choices = []
    try:
        import ollama

        available = sorted(
            {m.model for m in ollama.list().models if m.model},
            key=lambda m: (m != current, m),
        )
        choices = available + [manual_option]
    except Exception:
        pass

    if choices:
        if force_init and current in choices:
            label = f"Ollama embedding model (current: {current}):"
        else:
            label = "Ollama embedding model:"

        selection = questionary.select(
            label, choices=choices, default=current if current in choices else choices[0]
        ).ask()
        if selection is None:
            print("Setup cancelled.")
            sys.exit(0)
        if selection != manual_option:
            return selection

    value = questionary.text(
        "Ollama embedding model (type the model name):",
        default=current,
    ).ask()
    if value is None:
        print("Setup cancelled.")
        sys.exit(0)
    return value.strip() or current


def main():
    os.environ["PAGER"] = f'"{sys.executable}" -c "import sys; sys.stdout.write(sys.stdin.read())"'

    fire.Fire(GoodreadsRankerCLI)


if __name__ == "__main__":
    main()
