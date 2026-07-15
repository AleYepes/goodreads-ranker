import asyncio
import sys

from goodreads_ranker import config, db
from goodreads_ranker.utils import as_bool, parse_optional_int

import fire

# Centralized CLI documentation mapping
COMMAND_HELPS = {
    "init": {
        "desc": "Interactively configure required environment variables in .env.",
        "flags": [
            ("--force", "Force update of existing environment variables.")
        ]
    },
    "seed": {
        "desc": "Scrape Goodreads library shelves to seed the database.",
        "flags": [
            ("--force", "Force seeding even if shelves were already seeded."),
            ("--library-ids", "Comma-separated list of Goodreads user/library IDs to seed.")
        ]
    },
    "crawl": {
        "desc": "Crawl detailed metadata for all seeded books.",
        "flags": [
            ("--limit", "Limit the number of books to crawl."),
            ("--force", "Force crawl even if details were already crawled.")
        ]
    },
    "embed": {
        "desc": "Generate Ollama vector embeddings for crawled books.",
        "flags": [
            ("--batch-size", "Batch size for generating embeddings (default: 128)."),
            ("--model", "Ollama model name (overrides configured default).")
        ]
    },
    "rank": {
        "desc": "Train ranking models and generate book recommendations.",
        "flags": [
            ("--interactive", "Train and evaluate models in interactive mode."),
            ("--optimize", "Optimize ranker model hyperparameters."),
            ("--model", "Specific algorithm or model to run for ranking.")
        ]
    },
    "run_pipeline": {
        "desc": "Run the complete pipeline from initialization to ranking.",
        "flags": [
            ("--limit", "Limit the number of books to crawl."),
            ("--force-init", "Force interactive configuration step."),
            ("--force-seed", "Force database seeding step."),
            ("--force-crawl", "Force book crawling step."),
            ("--optimize", "Optimize ranking model hyperparameters."),
            ("--interactive", "Run ranker in interactive mode."),
            ("--model", "Embedding/ranking model name to use.")
        ]
    }
}


class GoodreadsRankerCLI:
    def init(self, force=False):
        """Interactively configure required environment variables and save them to .env.

        Args:
            force (bool): Force review and update of existing environment variables.
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
                value = value.strip()

            if value:
                set_key(dotenv_path, key, value)
                __import__("os").environ[key] = value

        if anything_prompted:
            print("\n✓ Configuration saved to .env")

    def seed(self, force=False, library_ids=None):
        """Scrape Goodreads library shelves to seed the database with books and ratings.

        Args:
            force (bool): Force seeding even if shelves were already seeded.
            library_ids (str|list): Comma-separated list of Goodreads library/user IDs to seed.
        """
        from goodreads_ranker import seeder

        print("\nSeeding database")
        db.init_db()
        if library_ids is not None:
            if isinstance(library_ids, str):
                library_ids = [int(x.strip()) for x in library_ids.split(",") if x.strip()]
            else:
                library_ids = [int(x) for x in library_ids]
        asyncio.run(seeder.scrape_libraries(library_ids=library_ids, force_seed=as_bool(force)))

    def crawl(self, limit=None, force=False):
        """Crawl full book details (metadata, genres, descriptions) for all seeded books.

        Args:
            limit (int): Limit the number of books to crawl.
            force (bool): Force crawling even if details have already been crawled.
        """
        from goodreads_ranker import crawler

        print("\nCrawling book details")
        db.init_db()
        limit = parse_optional_int(limit)
        asyncio.run(
            crawler.run_crawler(
                limit=limit,
                force_crawl=as_bool(force),
            )
        )

    def embed(self, batch_size=128, model=None):
        """Generate Ollama embeddings for all books that are missing or outdated.

        Args:
            batch_size (int): Batch size for generating embeddings.
            model (str): Ollama embedding model name (overrides configured model).
        """
        from goodreads_ranker import embedder

        print("\nGenerating embeddings")
        db.init_db()
        embedder.generate_embeddings(batch_size=int(batch_size), model=model or None)

    def rank(self, interactive=False, optimize=False, model=None):
        """Run the ranking model and write predictions to the database.

        Args:
            interactive (bool): Run the ranking model in interactive mode.
            optimize (bool): Optimize model hyperparameters.
            model (str): Specific model/algorithm to run for ranking.
        """
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
        force_init=False,
        force_seed=False,
        force_crawl=False,
        optimize=False,
        model=None,
        interactive=False,
    ):
        """Run the full pipeline: init (if needed) → seed → crawl → embed → rank.

        Args:
            limit (int): Limit the number of books to crawl.
            force_init (bool): Force interactive configuration step.
            force_seed (bool): Force database seeding.
            force_crawl (bool): Force book crawling.
            optimize (bool): Optimize ranker hyperparameters.
            model (str): Embedding/ranking model name to use.
            interactive (bool): Run ranker in interactive mode.
        """
        self.init(force=force_init)
        limit = parse_optional_int(limit)
        self.seed(force=force_seed)
        self.crawl(limit=limit, force=force_crawl)  # Fixed NameError bug
        self.embed(model=model)
        self.rank(interactive=interactive, optimize=optimize, model=model)
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
            key=lambda m: (m != current, m),
        )
        choices = available + [manual_option]
    except Exception:
        pass

    if choices:
        if force and current in choices:
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


def print_help(command=None):
    """Outputs standardized, beautifully aligned CLI help screens."""
    if command is None:
        print("Goodreads Ranker CLI - Crawl, embed, and rank your Goodreads books.\n")
        print("Usage:")
        print("  python main.py <command> [<args>]\n")
        print("Available commands:")
        for cmd, info in COMMAND_HELPS.items():
            print(f"  {cmd:<22}{info['desc']}")
            for flag, flag_desc in info['flags']:
                print(f"    {flag:<20}{flag_desc}")
            print()
        print("Additional options:")
        print("  -h, --help            Show optional command flags.")
    else:
        info = COMMAND_HELPS.get(command)
        if info:
            # Format subcommand title gracefully
            title = command.replace("_", " ").title()
            print(f"Goodreads Ranker CLI - {title}\n")
            print("Usage:")
            print(f"  python main.py {command} [<args>]\n")
            print("Description:")
            print(f"  {info['desc']}\n")
            if info['flags']:
                print("Available options:")
                for flag, flag_desc in info['flags']:
                    print(f"  {flag:<22}{flag_desc}")
            else:
                print("This command does not accept any additional options.")


def main():
    # Identify if a help flag is requested, or if no parameters were supplied
    is_help = "-h" in sys.argv or "--help" in sys.argv or len(sys.argv) == 1

    if is_help:
        # Determine if help is requested for a specific subcommand
        subcommand = None
        for arg in sys.argv[1:]:
            if arg in COMMAND_HELPS:
                subcommand = arg
                break
        
        print_help(subcommand)
        sys.exit(0)

    # Disable Google Fire's default interactive console pager
    import os
    os.environ["PAGER"] = f'"{sys.executable}" -c "import sys; sys.stdout.write(sys.stdin.read())"'

    fire.Fire(GoodreadsRankerCLI)


if __name__ == "__main__":
    main()