from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import KNeighborsRegressor
from tqdm import tqdm

from goodreads_ranker.core import db, utils


def safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or len(y) < 3 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return np.nan
    res = spearmanr(x, y)
    rho = cast(tuple[Any, ...], res)[0]
    val = float(rho)
    return val if not np.isnan(val) else np.nan


def calibrate_friend_ratings(overlap_df, shrink_after=10):
    x = overlap_df["rating"].astype(float).to_numpy(copy=True)
    y = overlap_df["my_refined"].astype(float).to_numpy(copy=True)

    if len(overlap_df) >= 3 and np.std(x) > 1e-9:
        slope, intercept = np.polyfit(x, y, 1)
    else:
        slope = 1.0
        intercept = float(y.mean() - x.mean())

    shrink = min(1.0, len(overlap_df) / shrink_after)
    slope = 1.0 + shrink * (slope - 1.0)
    intercept = shrink * intercept

    return slope, intercept


def run_friend_similarity(embedding_model=None, db_path=None):
    db.init_db(db_path)
    with db.get_connection(db_path) as db_conn:
        try:
            main_library_id = db.get_main_library_id(db_conn)
        except RuntimeError as err:
            raise RuntimeError("No main library found. Run seed first.") from err

        rows = db.get_friend_library_book_ratings(db_conn)
        friends = pd.DataFrame(rows)

        if friends.empty:
            print("No friend library data available for calibration.")
            return

        friends["rating"] = pd.to_numeric(friends["rating"], errors="coerce").replace(0, np.nan)
        friends = friends.dropna(subset=["rating"]).copy()
        friends["library_id"] = friends["library_id"].astype(int)

        book_ids = db.get_all_book_ids(db_conn)
        books_df = pd.DataFrame({"legacy_id": book_ids})
        if books_df.empty:
            print("No book records found in books table. Run crawler first.")
            return

        ratings_list = db.get_main_library_ratings(db_conn, main_library_id)
        target_df = pd.DataFrame(ratings_list)
        target_df["rating"] = pd.to_numeric(target_df["rating"], errors="coerce").replace(0, np.nan)

        existing_rows = db.get_elo_ratings(db_conn)
        target_ratings = target_df.dropna(subset=["rating"]).set_index("legacy_id")["rating"].to_dict()
        elo_df = utils.merge_elo_state(existing_rows, target_ratings)
        my_refined = utils.compute_continuous(elo_df)

        books_df = books_df.merge(my_refined.rename("my_refined"), on="legacy_id", how="left")

        if not embedding_model:
            from goodreads_ranker.core import config

            embedding_model = config.get_embedding_model()

        vectors_by_id = db.get_embeddings_by_model(db_conn, embedding_model)
        has_embedding, embedded_embeddings = utils.assemble_embedding_matrix(
            books_df["legacy_id"].tolist(), vectors_by_id
        )

        if embedded_embeddings.size == 0:
            raise RuntimeError("No valid embeddings found. Run embedder before friend similarity.")

        embedded_books_df = books_df[has_embedding].copy().reset_index(drop=True)
        embedding_matrix = np.asarray(embedded_embeddings)
        current_book_ids = embedded_books_df["legacy_id"].to_numpy()
        legacy_id_to_idx = {legacy_id: i for i, legacy_id in enumerate(current_book_ids)}

        my_books = embedded_books_df[["legacy_id", "my_refined"]].dropna().copy()
        my_legacy_id_set = set(my_books["legacy_id"].tolist())
        my_lookup = my_books.set_index("legacy_id")["my_refined"]

        if not my_lookup.empty:
            clip_min = float(my_lookup.min())
            clip_max = float(my_lookup.max())
        else:
            clip_min = 1.0
            clip_max = 5.0

        if len(my_books) >= 2:
            my_train_ids = my_books["legacy_id"].to_numpy()
            embeddings_me = embedding_matrix[[legacy_id_to_idx[bid] for bid in my_train_ids]]
            y_me = my_lookup.loc[my_train_ids].to_numpy(copy=True)
            knn_me_k = min(max(2, int(np.sqrt(len(my_books)))), len(my_books))
            knn_me = KNeighborsRegressor(n_neighbors=knn_me_k, metric="cosine", weights="distance", n_jobs=-1)
            knn_me.fit(embeddings_me, y_me)
        else:
            knn_me = None

        friend_scores = {}
        calibrated_friend_rows = []

        friend_groups = list(friends.groupby("library_id", sort=False))
        for library_id, friend_df in tqdm(
            friend_groups,
            desc="Calibrating friends",
            unit="friend",
        ):
            friend_df = friend_df.copy()
            friend_df = friend_df[friend_df["legacy_id"].isin(current_book_ids)].copy()
            if friend_df.empty:
                continue

            overlap = (
                friend_df.merge(my_books, on="legacy_id", how="inner").dropna(subset=["rating", "my_refined"]).copy()
            )
            overlap_count = len(overlap)

            if overlap_count >= 1:
                slope, intercept = calibrate_friend_ratings(overlap, clip_min)
            else:
                slope = 1.0
                intercept = 0.0

            friend_ratings = np.asarray(friend_df["rating"], dtype=float)
            friend_df["calibrated_rating"] = np.clip(
                slope * friend_ratings + intercept,
                clip_min,
                clip_max,
            )

            for _, row in friend_df.iterrows():
                calibrated_friend_rows.append((int(library_id), int(row["legacy_id"]), float(row["calibrated_rating"])))

            if overlap_count < 3 or knn_me is None:
                friend_scores[int(library_id)] = None
                continue

            overlap_ids = set(overlap["legacy_id"].tolist())
            friend_legacy_id_set = set(friend_df["legacy_id"].tolist())
            my_only_ids = sorted(my_legacy_id_set - overlap_ids)
            friend_only_ids = sorted(friend_legacy_id_set - overlap_ids)

            calibrated_overlap = np.clip(
                slope * overlap["rating"].astype(float).to_numpy(copy=True) + intercept,
                clip_min,
                clip_max,
            )
            my_union = [overlap["my_refined"].to_numpy(copy=True)]
            friend_union = [calibrated_overlap]

            raw_corr = safe_spearman(my_union[0], friend_union[0])

            if my_only_ids and len(friend_df) >= 2:
                friend_indexed = friend_df.set_index("legacy_id")
                embeddings_friend = embedding_matrix[[legacy_id_to_idx[bid] for bid in friend_df["legacy_id"]]]
                y_friend = friend_indexed.loc[friend_df["legacy_id"], "calibrated_rating"].to_numpy(copy=True)

                knn_friend_k = min(max(2, int(np.sqrt(len(friend_df)))), len(friend_df))
                knn_friend = KNeighborsRegressor(
                    n_neighbors=knn_friend_k, metric="cosine", weights="distance", n_jobs=-1
                )
                knn_friend.fit(embeddings_friend, y_friend)

                embeddings_my_only = embedding_matrix[[legacy_id_to_idx[bid] for bid in my_only_ids]]
                predicted_friend = knn_friend.predict(embeddings_my_only)

                my_union.append(my_lookup.loc[my_only_ids].to_numpy(copy=True))
                friend_union.append(predicted_friend)

            if friend_only_ids:
                embeddings_friend_only = embedding_matrix[[legacy_id_to_idx[bid] for bid in friend_only_ids]]
                predicted_me = knn_me.predict(embeddings_friend_only)

                friend_indexed = friend_df.set_index("legacy_id")
                real_friend = friend_indexed.loc[friend_only_ids, "calibrated_rating"].to_numpy(copy=True)

                my_union.append(predicted_me)
                friend_union.append(real_friend)

            my_union_arr = np.concatenate(my_union)
            friend_union_arr = np.concatenate(friend_union)
            synthetic_corr = safe_spearman(my_union_arr, friend_union_arr)

            if not np.isnan(raw_corr) and not np.isnan(synthetic_corr):
                score = 0.5 * raw_corr + 0.5 * synthetic_corr
            elif not np.isnan(raw_corr):
                score = raw_corr
            else:
                score = synthetic_corr

            if np.isnan(score):
                friend_scores[int(library_id)] = None
            else:
                friend_scores[int(library_id)] = float(score)

        db.save_friend_similarity_scores(db_conn, friend_scores)
        db.update_calibrated_ratings(db_conn, calibrated_friend_rows)
        print("✓ Friend similarity calibration and calibrated ratings successfully computed.")
