#!/usr/bin/env python3
"""
Delete .epub files from directories that also contain a .mobi file.

Usage:
    python cleanup_epubs.py /path/to/directory [--dry-run]

Options:
    --dry-run   Preview which files would be deleted without actually deleting them.
"""

import argparse
import os
import sys


def find_and_remove_epubs(root_dir: str, dry_run: bool = False) -> None:
    deleted = 0
    errors = 0

    for dirpath, _, filenames in os.walk(root_dir):
        lower_names = [f.lower() for f in filenames]
        has_mobi = any(name.endswith(".mobi") for name in lower_names)

        if not has_mobi:
            continue

        for filename in filenames:
            if filename.lower().endswith(".epub"):
                filepath = os.path.join(dirpath, filename)
                if dry_run:
                    print(f"[DRY RUN] Would delete: {filepath}")
                else:
                    try:
                        os.remove(filepath)
                        print(f"Deleted: {filepath}")
                    except OSError as e:
                        print(f"Error deleting {filepath}: {e}", file=sys.stderr)
                        errors += 1
                        continue
                deleted += 1

    label = "Would delete" if dry_run else "Deleted"
    print(f"\n{label} {deleted} .epub file(s).", end="")
    if errors:
        print(f" {errors} error(s) encountered.", end="")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete .epub files from directories that also contain a .mobi file."
    )
    parser.add_argument("directory", help="Root directory to scan.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview deletions without removing any files.",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.directory)
    if not os.path.isdir(root):
        print(f"Error: '{root}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning: {root}")
    if args.dry_run:
        print("(dry-run mode — no files will be deleted)\n")
    else:
        print()

    find_and_remove_epubs(root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
