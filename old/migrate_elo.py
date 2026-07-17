import csv
from pathlib import Path
from db import get_connection

CSV_PATH = Path("../data/elo_ratings.csv")

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
                original_rating = float(row["original_rating"]) if row["original_rating"] else None
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
        # Disable foreign key constraints temporarily, in case the 'books' table
        # does not yet contain these book IDs.
        conn.execute("PRAGMA foreign_keys=OFF")
        
        conn.executemany(
            """
            INSERT OR REPLACE INTO book_elo_ratings (book_id, original_rating, elo_score, matches_played)
            VALUES (?, ?, ?, ?)
            """,
            records,
        )
        conn.commit()
        
    print("Migration finished successfully.")

if __name__ == "__main__":
    migrate_elo_ratings()