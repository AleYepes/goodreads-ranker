import csv
from pathlib import Path

# Resolve paths relative to this file so the script works regardless of CWD.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

import sys
sys.path.insert(0, str(_ROOT))

from goodreads_ranker.core.db import get_connection

CSV_PATH = _ROOT / "data" / "elo_ratings.csv"


def migrate_elo_ratings():
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found.")
        return

    print(f"Parsing {CSV_PATH}...")
    records = []

    with open(CSV_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                book_id = int(row["book_id"])
                original_rating = int(row["original_rating"]) if row["original_rating"] else None
                elo_score = float(row["elo_score"]) if row["elo_score"] else 1200.0
                matches_played = int(row["matches_played"]) if row["matches_played"] else 0

                records.append((book_id, original_rating, elo_score, matches_played))
            except ValueError as e:
                print(f"Skipping row due to parsing error: {row} ({e})")

    if not records:
        print("No valid records found to migrate.")
        return

    print(f"Inserting {len(records)} records into book_elo_ratings...")
    with get_connection() as conn:
        # Disable foreign key constraints temporarily, in case some book_ids
        # are not yet present in the 'books' table.
        conn.execute("PRAGMA foreign_keys=OFF")

        conn.executemany(
            """
            INSERT OR REPLACE INTO book_elo_ratings (book_id, original_rating, elo_score, matches_played)
            VALUES (?, ?, ?, ?)
            """,
            records,
        )
        conn.commit()

        conn.execute("PRAGMA foreign_keys=ON")

    print(f"Migration finished successfully. {len(records)} rows inserted.")


if __name__ == "__main__":
    migrate_elo_ratings()