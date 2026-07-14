import asyncio
import sys

import fire

from goodreads_ranker import config, db
from goodreads_ranker.utils import as_bool, parse_optional_int


class GoodreadsRankerCLI:
    def init(self, force=False):
        """Interactively configure required environment variables and save them to .env.

        Run without arguments to set up for the first time, or with --force to
        review and update existing values.
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
            already_in_env = bool(current) and key in __import__("os").environ

            # In guard mode (called from run_pipeline), skip vars that are already set.
            if not force and already_in_env:
                continue

            if not sys.stdin.isatty():
                raise RuntimeError(
                    f"{key} is not set. Run 'python main.py init' to configure or set it directly in .env"
                )

            anything_prompted = True

            if var["kind"] == "model":
                value = _prompt_model(current or config.DEFAULT_EMBEDDING_MODEL, force)
            elif var["kind"] == "password":
                prompt_label = f"{var['label']}"
                if force and current:
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

            if value:
                set_key(dotenv_path, key, value)
                __import__("os").environ[key] = value

        if anything_prompted:
            print("\n✓ Configuration saved to .env")

    def seed(self, force_seed=False, library_ids=None):
        """Scrape Goodreads library shelves to seed the database with books and ratings."""
        from goodreads_ranker import seeder

        print("\nSeeding database")
        db.init_db()
        if library_ids is not None:
            if isinstance(library_ids, str):
                library_ids = [int(x.strip()) for x in library_ids.split(",") if x.strip()]
            else:
                library_ids = [int(x) for x in library_ids]
        asyncio.run(seeder.scrape_libraries(library_ids=library_ids, force_seed=as_bool(force_seed)))

    def crawl(self, limit=None, force_recrawl=False):
        """Crawl full book details (metadata, genres, descriptions) for all seeded books."""
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
        """Generate Ollama embeddings for all books that are missing or outdated."""
        from goodreads_ranker import embedder

        print("\nGenerating embeddings")
        db.init_db()
        embedder.generate_embeddings(batch_size=int(batch_size), model=model or None)

    def rank(self, interactive=False, optimize=False, model=None):
        """Run the ranking model and write predictions to the database."""
        from goodreads_ranker import ranker

        print("\nRunning models and predictions")
        db.init_db()
        ranker.run_ranking(
            interactive=as_bool(interactive),
            optimize=as_bool(optimize),
            model=model or None,
        )

    def run_pipeline(
        self,
        limit=None,
        force_seed=False,
        force_recrawl=False,
        optimize=False,
        model=None,
    ):
        """Run the full pipeline: init (if needed) → seed → crawl → embed → rank."""
        self.init(force=False)
        limit = parse_optional_int(limit)
        self.seed(force_seed=force_seed)
        self.crawl(limit=limit, force_recrawl=force_recrawl)
        self.embed(model=model)
        self.rank(interactive=False, optimize=optimize, model=model)
        print("\nPipeline run finished successfully!")


def _prompt_model(current: str, force: bool) -> str:
    """Prompt the user to select or type an Ollama embedding model."""
    import questionary

    manual_option = "Enter manually…"

    choices = []
    try:
        import ollama

        available = sorted(
            {m.model for m in ollama.list().models if m.model},
            key=lambda m: (m != current, m),  # put current first
        )
        choices = available + [manual_option]
    except Exception:
        pass  # Ollama not running — fall through to free-text

    if choices:
        if force and current in choices:
            # Highlight current value at top of list
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

    # Free-text fallback (Ollama unreachable, or user chose "Enter manually…")
    value = questionary.text(
        "Ollama embedding model (type the model name):",
        default=current,
    ).ask()
    if value is None:
        print("Setup cancelled.")
        sys.exit(0)
    return value.strip() or current


def main():
    if len(sys.argv) == 1:
        GoodreadsRankerCLI().run_pipeline()
    else:
        fire.Fire(GoodreadsRankerCLI)


if __name__ == "__main__":
    main()
