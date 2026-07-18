import random

import pandas as pd

from goodreads_ranker.core import db, utils


def get_expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(rating_a, rating_b, result, k=32):
    expected_a = get_expected_score(rating_a, rating_b)
    return rating_a + k * (result - expected_a), rating_b + k * ((1 - result) - (1 - expected_a))


def run_interactive_ranking(elo_df, titles, star_rating):
    bucket_indices = elo_df[elo_df["original_rating"] == star_rating].index.tolist()
    if len(bucket_indices) < 2:
        return elo_df

    matches_stats = elo_df.loc[bucket_indices, "matches_played"]
    previous_min_matches = matches_stats.min()
    unranked_count = (matches_stats == 0).sum()

    print(f"\n--- Ranking {star_rating}-Star Books ({unranked_count} books with 0 matches) ---")
    while True:
        try:
            unranked_indices = [idx for idx in bucket_indices if elo_df.at[idx, "matches_played"] == 0]

            if len(unranked_indices) >= 2:
                index_a, index_b = random.sample(unranked_indices, 2)
            elif len(unranked_indices) == 1:
                index_a = unranked_indices[0]
                remaining_indices = [idx for idx in bucket_indices if idx != index_a]
                match_weights = 1 / (elo_df.loc[remaining_indices, "matches_played"] + 1)
                index_b = random.choices(remaining_indices, weights=match_weights, k=1)[0]
            else:
                match_weights = 1 / (elo_df.loc[bucket_indices, "matches_played"] + 1)
                index_a, index_b = random.choices(bucket_indices, weights=match_weights, k=2)
                if index_a == index_b:
                    continue

            title_a = titles.get(elo_df.at[index_a, "legacy_id"]) or f"Book {elo_df.at[index_a, 'legacy_id']}"
            title_b = titles.get(elo_df.at[index_b, "legacy_id"]) or f"Book {elo_df.at[index_b, 'legacy_id']}"

            user_choice = input(f"[1] {title_a}\n[2] {title_b}\nChoose (1 or 2, 'q' to quit): ").strip().lower()

            if user_choice in ["q", "quit"]:
                break
            if user_choice not in ["1", "2"]:
                continue

            elo_df.loc[[index_a, index_b], "matches_played"] += 1
            current_min_matches = elo_df.loc[bucket_indices, "matches_played"].min()
            if current_min_matches > previous_min_matches:
                print(f"All books have played at least {current_min_matches} match(es)")
                previous_min_matches = current_min_matches

            rating_a = elo_df.at[index_a, "elo_score"]
            rating_b = elo_df.at[index_b, "elo_score"]
            actual_result = 1 if user_choice == "1" else 0
            new_rating_a, new_rating_b = update_elo(rating_a, rating_b, actual_result)

            elo_df.at[index_a, "elo_score"] = new_rating_a
            elo_df.at[index_b, "elo_score"] = new_rating_b

        except KeyboardInterrupt:
            break

    return elo_df


def run_calibration(db_path=None):
    db.init_db(db_path)
    with db.get_connection(db_path) as db_conn:
        try:
            main_library_id = db.get_main_library_id(db_conn)
        except RuntimeError as err:
            raise RuntimeError("No main library found. Run seed first.") from err

        ratings_list = db.get_main_library_ratings(db_conn, main_library_id)
        if not ratings_list:
            raise RuntimeError("No book records found. Run crawler first.")

        target_df = pd.DataFrame(ratings_list)
        target_df["rating"] = pd.to_numeric(target_df["rating"], errors="coerce").replace(0, pd.NA)
        target_clean = target_df.dropna(subset=["rating"]).copy()

        if target_clean.empty:
            raise RuntimeError("No self rating records found. Run seed first.")

        existing_rows = db.get_elo_ratings(db_conn)
        target_ratings = dict(zip(target_clean["legacy_id"], target_clean["rating"], strict=True))
        elo_df = utils.merge_elo_state(existing_rows, target_ratings)

        titles = dict(zip(target_df["legacy_id"], target_df["title"], strict=True))

        for star in sorted(target_clean["rating"].unique(), reverse=True):
            elo_df = run_interactive_ranking(elo_df, titles, star)

        db_rows = []
        for row in elo_df.to_dict(orient="records"):
            db_rows.append(
                (
                    int(row["legacy_id"]),
                    int(row["original_rating"])
                    if row["original_rating"] is not None and pd.notna(row["original_rating"])
                    else None,
                    float(row["elo_score"]),
                    int(row["matches_played"]),
                )
            )
        db.save_elo_ratings(db_conn, db_rows)
        print("✓ Elo ratings calibration complete.")
