import random
from datetime import datetime
from typing import Any, cast

import nevergrad as ng
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, ndcg_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.svm import SVR
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops

from . import db

torch.sparse.check_sparse_tensor_invariants.disable()

DEFAULT_FRIEND_PARAMS = {
    "num_propagations": 0,
    "knn_neighbors": 18,
    "brr_alpha_1": 0.0007083523294476891,
    "brr_alpha_2": 0.0005052291699790206,
    "brr_lambda_1": 6.723461212511323e-06,
    "brr_lambda_2": 0.00065032051532833,
    "brr_uncertainty_penalty": 1.124317064784675,
    "svr_regularization": 0.045512095772648135,
    "svr_epsilon": 0.45635064943633097,
    "knn_weight": 0.6072006217417109,
    "brr_weight": 0.7469140779723126,
}

DEFAULT_SOLO_PARAMS = {
    "num_propagations": 1,
    "knn_neighbors": 39,
    "brr_alpha_1": 0.0007917501343982147,
    "brr_alpha_2": 0.0003673094481243756,
    "brr_lambda_1": 0.0009261898325379562,
    "brr_lambda_2": 0.0004845024834776367,
    "brr_uncertainty_penalty": 1.4875061034371395,
    "svr_regularization": 2.0283092484268437,
    "svr_epsilon": 0.2673464313322004,
    "knn_weight": 0.11589067396260068,
    "brr_weight": 0.1852941582795484,
}


def normalize_model_params(params):
    params = {key: value.item() if hasattr(value, "item") else value for key, value in dict(params).items()}
    params["num_propagations"] = int(params["num_propagations"])
    params["knn_neighbors"] = int(params["knn_neighbors"])
    return params


def get_or_create_model_params(db_conn, name, defaults):
    params = db.load_model_params(db_conn, name)
    if params is None:
        params = dict(defaults)
        db.save_model_params(db_conn, name, params)
        return params
    params = normalize_model_params(params)
    db.save_model_params(db_conn, name, params)
    return params


def load_valid_embeddings_for_books(db_conn, books_df, model=None):
    import hashlib
    import os

    from . import embedder

    if not model:
        model = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:8b")

    all_inputs = embedder.build_embedding_inputs(db_conn)
    input_hashes = {legacy_id: hashlib.md5(text.encode("utf-8")).hexdigest() for legacy_id, text in all_inputs.items()}

    rows = db_conn.execute(
        """
        SELECT book_id AS legacy_id, vector, text_hash
        FROM embeddings
        WHERE embedding_model = ?
        """,
        (model,),
    ).fetchall()
    embedding_rows = {int(row["legacy_id"]): row for row in rows}

    counts = {
        "missing": 0,
        "invalid": 0,
        "unverified": 0,
        "dimension_mismatch": 0,
    }
    valid_vectors = []
    valid_mask = []
    expected_dim = None

    for legacy_id in books_df["legacy_id"]:
        bid = int(legacy_id)
        row = embedding_rows.get(bid)
        if not row or row["vector"] is None:
            counts["missing"] += 1
            valid_mask.append(False)
            continue

        vector_blob = row["vector"]
        if not db.is_valid_embedding_blob(vector_blob):
            counts["invalid"] += 1
            valid_mask.append(False)
            continue

        current_hash = input_hashes.get(bid, "")
        if not current_hash or row["text_hash"] != current_hash:
            counts["unverified"] += 1
            valid_mask.append(False)
            continue

        vector = np.frombuffer(vector_blob, dtype=np.float32).copy()
        dim = len(vector)
        if expected_dim is None:
            expected_dim = dim
        elif dim != expected_dim:
            counts["dimension_mismatch"] += 1
            valid_mask.append(False)
            continue

        valid_vectors.append(vector)
        valid_mask.append(True)

    matrix = (
        np.vstack(valid_vectors).astype(np.float32, copy=False) if valid_vectors else np.empty((0, 0), dtype=np.float32)
    )
    return np.array(valid_mask, dtype=bool), matrix, counts


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
                print(f"    All books have played at least {current_min_matches} match(es)")
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


def compute_continuous(elo_df):
    results = pd.Series(np.nan, index=elo_df.index, dtype=float)
    for stars in elo_df["original_rating"].dropna().unique():
        mask = elo_df["original_rating"] == stars
        subset = elo_df.loc[mask, "elo_score"]

        if len(subset) > 1 and subset.max() > subset.min():
            norm = (subset - subset.min()) / (subset.max() - subset.min())
            results.loc[mask] = (stars + (norm * 0.99) - 0.5).to_numpy(copy=True)
        else:
            results.loc[mask] = float(stars)
    results.index = elo_df["legacy_id"]
    return results


def refine_ratings(target_df, rating_col, db_conn, interactive=False):
    target_clean = target_df.dropna(subset=[rating_col]).copy()

    cursor = db_conn.execute("SELECT book_id AS legacy_id, original_rating, elo_score, matches_played FROM elo_ratings")
    rows = cursor.fetchall()

    if rows:
        elo_df = pd.DataFrame([dict(r) for r in rows])
    else:
        elo_df = pd.DataFrame(columns=["legacy_id", "original_rating", "elo_score", "matches_played"])

    elo_df = elo_df.set_index("legacy_id")

    common_books = elo_df.index.intersection(target_clean["legacy_id"])
    elo_df.loc[common_books, "original_rating"] = target_clean.set_index("legacy_id").loc[common_books, rating_col]

    new_books = target_clean[~target_clean["legacy_id"].isin(elo_df.index)]
    if not new_books.empty:
        new_entries = pd.DataFrame(
            {
                "original_rating": new_books[rating_col].values,
                "elo_score": 1200.0,
                "matches_played": 0,
            },
            index=new_books["legacy_id"],
        )
        elo_df = pd.concat([elo_df, new_entries])

    elo_df = elo_df.reset_index().rename(columns={"index": "legacy_id"})

    if interactive:
        titles = dict(zip(target_df["legacy_id"], target_df["title"], strict=True))
        for star in sorted(target_clean[rating_col].unique(), reverse=True):
            elo_df = run_interactive_ranking(elo_df, titles, star)

    db_rows = []
    for row in elo_df.to_dict(orient="records"):
        db_rows.append(
            (
                int(row["legacy_id"]),
                float(row["original_rating"])
                if row["original_rating"] is not None and pd.notna(row["original_rating"])
                else None,
                float(row["elo_score"]),
                int(row["matches_played"]),
            )
        )
    db.upsert_rows(
        db_conn,
        "elo_ratings",
        db_rows,
        ["book_id", "original_rating", "elo_score", "matches_played"],
    )

    return compute_continuous(elo_df)


def safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or len(y) < 3 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return np.nan
    res = spearmanr(x, y)
    rho = cast(tuple[Any, ...], res)[0]
    val = float(rho)
    return val if not np.isnan(val) else np.nan


def calibrate_friend_ratings(overlap_df, clip_min, shrink_after=10):
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


def get_similar_friend_ratings(
    books_df,
    db_conn,
    embeddings,
    min_overlap=5,
    min_correlation=0.3,
    min_similar_friends=2,
):
    cursor = db_conn.execute(
        """
        SELECT rl.list_id, rl.book_id AS legacy_id, rl.rating
        FROM reader_libraries rl
        JOIN readers r ON rl.list_id = r.list_id
        WHERE r.is_self = 0
        """
    )
    rows = cursor.fetchall()
    friends = pd.DataFrame([dict(r) for r in rows])

    if friends.empty:
        return pd.Series(dtype=float), [], pd.DataFrame()

    friends["rating"] = friends["rating"].astype("UInt8").replace(0, np.nan)
    friends["list_id"] = friends["list_id"].astype(str)

    current_book_ids = books_df["legacy_id"].to_numpy()
    current_book_id_set = set(current_book_ids.tolist())
    friends_df = cast(pd.DataFrame, friends[friends["legacy_id"].isin(list(current_book_id_set))])
    friends = cast(pd.DataFrame, friends_df.dropna(subset=["rating"]).copy())
    friends = cast(
        pd.DataFrame,
        friends.groupby(["list_id", "legacy_id"], as_index=False)["rating"].mean(),
    )

    my_books = books_df[["legacy_id", "my_refined"]].dropna().copy()
    my_legacy_id_set = set(my_books["legacy_id"].tolist())
    my_lookup = my_books.set_index("legacy_id")["my_refined"]
    clip_min = float(my_lookup.min())
    clip_max = float(my_lookup.max())

    embedding_matrix = np.asarray(embeddings)
    legacy_id_to_idx = {legacy_id: i for i, legacy_id in enumerate(current_book_ids)}

    my_train_ids = my_books["legacy_id"].to_numpy()
    embeddings_me = embedding_matrix[[legacy_id_to_idx[bid] for bid in my_train_ids]]
    y_me = my_lookup.loc[my_train_ids].to_numpy(copy=True)
    knn_me_k = min(max(2, int(np.sqrt(len(my_books)))), len(my_books))
    knn_me = KNeighborsRegressor(n_neighbors=knn_me_k, metric="cosine", weights="distance", n_jobs=-1)
    knn_me.fit(embeddings_me, y_me)

    friend_scores = []
    calibrated_friend_rows = []

    for list_id, friend_df in friends.groupby("list_id"):
        overlap = friend_df.merge(my_books, on="legacy_id", how="inner").dropna(subset=["rating", "my_refined"]).copy()
        overlap_count = len(overlap)
        if overlap_count < min_overlap:
            continue

        slope, intercept = calibrate_friend_ratings(overlap, clip_min)

        friend_df = friend_df.copy()
        friend_ratings = np.asarray(friend_df["rating"], dtype=float)
        friend_df["calibrated_rating"] = np.clip(
            slope * friend_ratings + intercept,
            clip_min,
            clip_max,
        )
        calibrated_friend_rows.append(friend_df)

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
            knn_friend = KNeighborsRegressor(n_neighbors=knn_friend_k, metric="cosine", weights="distance", n_jobs=-1)
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

        my_union = np.concatenate(my_union)
        friend_union = np.concatenate(friend_union)
        synthetic_corr = safe_spearman(my_union, friend_union)

        if not np.isnan(raw_corr) and not np.isnan(synthetic_corr):
            score = 0.5 * raw_corr + 0.5 * synthetic_corr
        elif not np.isnan(raw_corr):
            score = raw_corr
        else:
            score = synthetic_corr

        if np.isnan(score):
            continue

        friend_scores.append(
            {
                "list_id": list_id,
                "overlap_count": overlap_count,
                "union_count": len(my_union),
                "friend_books": len(friend_df),
                "slope": slope,
                "intercept": intercept,
                "raw_corr": raw_corr,
                "synthetic_corr": synthetic_corr,
                "score": score,
            }
        )

    if not friend_scores:
        return pd.Series(dtype=float), [], pd.DataFrame()

    friend_scores = pd.DataFrame(friend_scores).sort_values("score", ascending=False).reset_index(drop=True)
    if friend_scores.empty:
        return pd.Series(dtype=float), [], friend_scores

    selected = friend_scores[friend_scores["score"] >= min_correlation].copy()
    minimum_selected = min(min_similar_friends, len(friend_scores))
    if len(selected) < minimum_selected:
        selected = friend_scores.head(minimum_selected).copy()

    similar_friends = selected["list_id"].tolist()
    similar_friend_ratings = cast(pd.DataFrame, pd.concat(calibrated_friend_rows, ignore_index=True))
    selected_sub = cast(pd.DataFrame, selected[["list_id", "score"]])
    similar_friend_ratings = similar_friend_ratings.merge(
        selected_sub.rename(columns={"score": "friend_weight"}),
        on="list_id",
        how="inner",
    )
    similar_friend_ratings = similar_friend_ratings[
        ~similar_friend_ratings["legacy_id"].isin(list(my_legacy_id_set))
    ].copy()
    if similar_friend_ratings.empty:
        return pd.Series(dtype=float), similar_friends, friend_scores

    similar_friend_ratings["weighted_rating"] = (
        similar_friend_ratings["calibrated_rating"] * similar_friend_ratings["friend_weight"]
    )
    similar_friend_ratings = pd.DataFrame(
        similar_friend_ratings.groupby("legacy_id").agg(
            weighted_rating=("weighted_rating", "sum"),
            total_weight=("friend_weight", "sum"),
            supporting_friends=("list_id", "nunique"),
        )
    )
    similar_friend_ratings["rating"] = (
        similar_friend_ratings["weighted_rating"] / similar_friend_ratings["total_weight"]
    )

    rating_series = pd.Series(similar_friend_ratings["rating"])
    return rating_series, similar_friends, friend_scores


def build_adjacency_matrix(books_df, num_nodes, db_conn):
    id_to_idx = {int(bid): i for i, bid in enumerate(books_df["legacy_id"])}
    book_ids_set = set(id_to_idx.keys())

    cursor = db_conn.execute("SELECT book_id, similar_legacy_id FROM similar_books")
    edge_indices = []
    for row in cursor.fetchall():
        bid = int(row["book_id"])
        sim_id = int(row["similar_legacy_id"])
        if bid in book_ids_set and sim_id in book_ids_set:
            idx_a = id_to_idx[bid]
            idx_b = id_to_idx[sim_id]
            edge_indices.append([idx_a, idx_b])
            edge_indices.append([idx_b, idx_a])

    if not edge_indices:
        edge_index = torch.tensor([[], []], dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    edge_index_with_loops, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    edge_index_norm, edge_weight_norm = gcn_norm(edge_index_with_loops, num_nodes=num_nodes)

    assert isinstance(edge_index_norm, torch.Tensor), (
        "gcn_norm returned a SparseTensor; expected a dense edge_index Tensor"
    )
    assert edge_weight_norm is not None, "gcn_norm returned no edge weights"
    adj_matrix = torch.sparse_coo_tensor(
        edge_index_norm,
        edge_weight_norm,
        (num_nodes, num_nodes),
    )
    return adj_matrix


def prep_optimization(
    books_df,
    precomputed_embeddings,
    training_col,
    mrl_dimensions,
    max_propagations,
    budget=300,
):
    training_mask = ~books_df[training_col].isna()
    my_ratings = books_df.loc[training_mask, training_col].values
    if len(my_ratings) < 2:
        raise RuntimeError(f"Need at least 2 ratings to optimize {training_col}.")
    scaler = MinMaxScaler(feature_range=(0, 1))
    my_ratings_scaled = scaler.fit_transform(my_ratings.reshape(-1, 1)).flatten()

    def objective(
        num_propagations,
        knn_neighbors,
        brr_alpha_1,
        brr_alpha_2,
        brr_lambda_1,
        brr_lambda_2,
        brr_uncertainty_penalty,
        svr_regularization,
        svr_epsilon,
        knn_weight,
        brr_weight,
    ):

        all_embeddings = precomputed_embeddings[num_propagations]
        train_embeddings = all_embeddings[training_mask]
        y = my_ratings_scaled

        y_reals = []
        y_preds = []
        skf = KFold(n_splits=min(10, len(train_embeddings)), shuffle=True, random_state=42)
        for train_idx, test_idx in skf.split(train_embeddings):
            train_fold_embeddings, test_fold_embeddings = (
                train_embeddings[train_idx],
                train_embeddings[test_idx],
            )
            y_train, y_test = y[train_idx], y[test_idx]

            knn = KNeighborsRegressor(
                n_neighbors=min(knn_neighbors, len(train_fold_embeddings)),
                metric="cosine",
                weights="distance",
                n_jobs=-1,
            )
            knn.fit(train_fold_embeddings, y_train)
            knn_pred = knn.predict(test_fold_embeddings)

            brr = BayesianRidge(
                alpha_1=brr_alpha_1,
                alpha_2=brr_alpha_2,
                lambda_1=brr_lambda_1,
                lambda_2=brr_lambda_2,
                compute_score=True,
            )
            brr.fit(train_fold_embeddings, y_train)
            brr_mu, brr_std = cast(tuple[np.ndarray, np.ndarray], brr.predict(test_fold_embeddings, return_std=True))
            brr_pred = brr_mu - (brr_uncertainty_penalty * brr_std)

            svr = SVR(kernel="rbf", gamma="scale", C=svr_regularization, epsilon=svr_epsilon)
            svr.fit(train_fold_embeddings, y_train)
            svr_pred = svr.predict(test_fold_embeddings)

            remaining_weight = 1 - knn_weight
            actual_brr_weight = brr_weight * remaining_weight
            svr_weight = remaining_weight - actual_brr_weight
            final_pred = (knn_weight * knn_pred) + (actual_brr_weight * brr_pred) + (svr_weight * svr_pred)

            y_reals.append(y_test)
            y_preds.append(final_pred)

        y_reals = np.concatenate(y_reals)
        y_preds = np.concatenate(y_preds)

        mse = mean_squared_error(y_reals, y_preds)
        ndcg = ndcg_score(np.vstack([y_reals]), np.vstack([y_preds]))
        if np.std(y_preds) < 1e-9:
            spearman = 0
        else:
            spearman = safe_spearman(y_reals, y_preds)
            if np.isnan(spearman):
                spearman = 0.0

        return mse + (1 - ndcg) + (1 - spearman)

    parametrization = ng.p.Instrumentation(
        num_propagations=ng.p.Scalar(lower=0, upper=max_propagations).set_integer_casting(),
        knn_neighbors=ng.p.Scalar(lower=1, upper=mrl_dimensions // 3).set_integer_casting(),
        brr_alpha_1=ng.p.Scalar(lower=1e-7, upper=1e-3),
        brr_alpha_2=ng.p.Scalar(lower=1e-7, upper=1e-3),
        brr_lambda_1=ng.p.Log(lower=1e-6, upper=1e-1),
        brr_lambda_2=ng.p.Log(lower=1e-6, upper=1e-1),
        brr_uncertainty_penalty=ng.p.Scalar(lower=0, upper=2.0),
        svr_regularization=ng.p.Log(lower=1e-3, upper=1e2),
        svr_epsilon=ng.p.Scalar(lower=0.0, upper=1.0),
        knn_weight=ng.p.Scalar(lower=0, upper=1),
        brr_weight=ng.p.Scalar(lower=0, upper=1),
    )

    optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=budget)
    best_loss = float("inf")

    for _ in range(budget):
        x = optimizer.ask()
        loss = objective(*x.args, **x.kwargs)
        optimizer.tell(x, loss)
        if loss < best_loss:
            best_loss = loss

    best_params = optimizer.provide_recommendation().kwargs
    return best_params


def run_optimized(best_params, books_df, precomputed_embeddings, training_col):
    all_embeddings = precomputed_embeddings[best_params["num_propagations"]]

    training_mask = ~books_df[training_col].isna()
    my_ratings = books_df.loc[training_mask, training_col].values
    if len(my_ratings) < 2:
        raise RuntimeError(f"Need at least 2 ratings to model {training_col}.")
    scaler = MinMaxScaler(feature_range=(0, 1))
    my_ratings_scaled = scaler.fit_transform(my_ratings.reshape(-1, 1)).flatten()

    train_embeddings = all_embeddings[training_mask]
    y_train = my_ratings_scaled

    knn = KNeighborsRegressor(
        n_neighbors=min(int(best_params["knn_neighbors"]), len(train_embeddings)),
        metric="cosine",
        weights="distance",
        n_jobs=-1,
    )
    knn.fit(train_embeddings, y_train)
    knn_pred = knn.predict(all_embeddings)

    brr = BayesianRidge(
        alpha_1=best_params["brr_alpha_1"],
        alpha_2=best_params["brr_alpha_2"],
        lambda_1=best_params["brr_lambda_1"],
        lambda_2=best_params["brr_lambda_2"],
        compute_score=True,
    )
    brr.fit(train_embeddings, y_train)
    brr_mu, brr_std = cast(tuple[np.ndarray, np.ndarray], brr.predict(all_embeddings, return_std=True))
    brr_pred = brr_mu - (best_params["brr_uncertainty_penalty"] * brr_std)

    svr = SVR(
        kernel="rbf",
        gamma="scale",
        C=best_params["svr_regularization"],
        epsilon=best_params["svr_epsilon"],
    )
    svr.fit(train_embeddings, y_train)
    svr_pred = svr.predict(all_embeddings)

    knn_weight = best_params["knn_weight"]
    brr_weight = best_params["brr_weight"]

    remaining_weight = 1 - knn_weight
    brr_weight *= remaining_weight
    svr_weight = remaining_weight - brr_weight

    knn_pred = scaler.inverse_transform(knn_pred.reshape(-1, 1)).flatten()
    brr_pred = scaler.inverse_transform(brr_pred.reshape(-1, 1)).flatten()
    svr_pred = scaler.inverse_transform(svr_pred.reshape(-1, 1)).flatten()
    final_pred = (knn_weight * knn_pred) + (brr_weight * brr_pred) + (svr_weight * svr_pred)

    return final_pred


def run_ranking(interactive=False, optimize=False, model=None, db_path=None):
    db.init_db(db_path)
    with db.get_connection(db_path) as db_conn:
        db_conn.execute("DELETE FROM predictions")
        db_conn.commit()

        try:
            self_list_id = db.get_self_list_id(db_conn)
        except RuntimeError:
            print("No self reader found. Run seed first.")
            return

        cursor = db_conn.execute("SELECT * FROM books ORDER BY legacy_id")
        books_rows = cursor.fetchall()
        if not books_rows:
            print("No book records found in the books table. Run crawler first.")
            return

        cursor = db_conn.execute(
            """
            SELECT b.*, l.rating AS my_rating
            FROM books b
            LEFT JOIN reader_libraries l
                ON b.legacy_id = l.book_id AND l.list_id = ?
            ORDER BY b.legacy_id
            """,
            (self_list_id,),
        )
        books_df = pd.DataFrame([dict(r) for r in cursor.fetchall()])
        books_df["my_rating"] = pd.Series(
            pd.to_numeric(books_df["my_rating"], errors="coerce"),
            index=books_df.index,
            name="my_rating",
        ).replace(0, np.nan)
        books_df["description"] = (
            books_df["description"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        )

        if books_df["my_rating"].notna().sum() == 0:
            print("No self rating records found. Run seed first.")
            return

        print("  Running ELO ratings refinement...")
        my_refined = refine_ratings(books_df, "my_rating", db_conn, interactive=interactive)

        books_df = books_df.merge(my_refined.rename("my_refined"), on="legacy_id", how="left")

        print("  Loading valid embeddings...")
        has_embedding, embedded_embeddings, invalid_counts = load_valid_embeddings_for_books(
            db_conn, books_df, model=model
        )
        excluded_count = int((~has_embedding).sum())
        if excluded_count:
            print(
                "  Excluding "
                f"{excluded_count} books from model inputs "
                f"(missing={invalid_counts['missing']}, "
                f"invalid={invalid_counts['invalid']}, "
                f"unverified={invalid_counts['unverified']}, "
                f"dimension_mismatch={invalid_counts['dimension_mismatch']})."
            )
        if embedded_embeddings.size == 0:
            print("  No valid embeddings found. Run embedder before ranking.")
            return

        print("  Calibrating friend ratings...")

        embedded_books_df = books_df[has_embedding].copy().reset_index(drop=True)

        similar_friend_ratings, similar_friends, friend_scores = get_similar_friend_ratings(
            embedded_books_df, db_conn, embedded_embeddings, min_correlation=0.3
        )
        friend_ratings_series = pd.Series(similar_friend_ratings)
        friend_ratings_dict = friend_ratings_series.to_dict()
        books_df["training_ratings"] = pd.Series(books_df["my_refined"]).fillna(
            pd.Series(books_df["legacy_id"]).map(friend_ratings_dict)
        )
        embedded_books_df["training_ratings"] = pd.Series(embedded_books_df["my_refined"]).fillna(
            pd.Series(embedded_books_df["legacy_id"]).map(friend_ratings_dict)
        )

        my_rating_count = len(books_df[books_df["my_rating"].notna()])
        print("  Taste calibration")
        print(f"    My ratings ({my_rating_count})")

        friend_meta = db_conn.execute("SELECT list_id, username FROM readers WHERE is_self != 1").fetchall()
        list_to_user = {int(row["list_id"]): row["username"] or str(row["list_id"]) for row in friend_meta}

        if not friend_scores.empty:
            selected_scores = cast(
                pd.DataFrame,
                friend_scores[friend_scores["list_id"].astype(int).isin([int(x) for x in similar_friends])],
            )
            records = cast(list[dict[str, Any]], selected_scores.to_dict(orient="records"))
            for row in records:
                lid = int(row["list_id"])
                username = list_to_user.get(lid, str(lid))
                overlap_c = int(row["overlap_count"])
                print(f"    {username} - {lid} ({overlap_c} books)")

        print("  Running graph GCN propagation...")
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        train_size = (~pd.isna(embedded_books_df["training_ratings"])).sum()
        solo_train_size = (~pd.isna(embedded_books_df["my_refined"])).sum()
        if train_size < 2 or solo_train_size < 2:
            print(
                "  Not enough rated books with valid embeddings to train ranking models "
                f"(training={train_size}, personal={solo_train_size})."
            )
            return
        mrl_dimensions = 32
        while mrl_dimensions * 2 < train_size:
            mrl_dimensions *= 2

        embeddings_tensor = torch.tensor(embedded_embeddings, dtype=torch.float32, device=device)
        adj_matrix = build_adjacency_matrix(embedded_books_df, len(embedded_books_df), db_conn).to(device)

        propagated = embeddings_tensor.clone()
        norm_l2 = Normalizer(norm="l2")
        precomputed_embeddings = [norm_l2.transform(propagated.cpu().numpy())]

        max_propagations = 2
        for _ in range(max_propagations):
            propagated = torch.sparse.mm(adj_matrix, propagated)
            precomputed_embeddings.append(norm_l2.transform(propagated.cpu().numpy()))

        del propagated, adj_matrix, embeddings_tensor

        if optimize:
            print("  Tuning hyperparameters via Nevergrad (budget=200)...")
            friend_params = prep_optimization(
                embedded_books_df,
                precomputed_embeddings,
                "training_ratings",
                mrl_dimensions,
                max_propagations,
                budget=200,
            )
            solo_params = prep_optimization(
                embedded_books_df,
                precomputed_embeddings,
                "my_refined",
                mrl_dimensions,
                max_propagations,
                budget=200,
            )
            friend_params = normalize_model_params(friend_params)
            solo_params = normalize_model_params(solo_params)
            db.save_model_params(db_conn, "friend_params", friend_params)
            db.save_model_params(db_conn, "solo_params", solo_params)
        else:
            print("  Using stored/default hyperparameters...")
            friend_params = get_or_create_model_params(db_conn, "friend_params", DEFAULT_FRIEND_PARAMS)
            solo_params = get_or_create_model_params(db_conn, "solo_params", DEFAULT_SOLO_PARAMS)

        print("  Running ensemble models...")
        friend_final_pred_embedded = run_optimized(
            friend_params, embedded_books_df, precomputed_embeddings, "training_ratings"
        )
        solo_final_pred_embedded = run_optimized(solo_params, embedded_books_df, precomputed_embeddings, "my_refined")

        friend_final_pred = np.full(len(books_df), np.nan)
        solo_final_pred = np.full(len(books_df), np.nan)
        embedded_positions = np.where(has_embedding)[0]
        friend_final_pred[embedded_positions] = friend_final_pred_embedded
        solo_final_pred[embedded_positions] = solo_final_pred_embedded

        print("  Formulating final recommendations...")
        star_cols = ["star_1", "star_2", "star_3", "star_4", "star_5"]
        books_df["rating_count"] = books_df[star_cols].sum(axis=1)

        def get_avg_rating(row):
            total_stars = (
                row["star_1"] * 1 + row["star_2"] * 2 + row["star_3"] * 3 + row["star_4"] * 4 + row["star_5"] * 5
            )
            count = row["rating_count"]
            return total_stars / count if count > 0 else np.nan

        books_df["avg_rating"] = books_df.apply(get_avg_rating, axis=1)

        global_avg_rating = books_df["avg_rating"].mean()
        m = books_df["rating_count"].quantile(0.10)

        def weighted_rating(x, m=m, global_avg=global_avg_rating):
            v = float(x["rating_count"])
            book_avg_rating = float(x["avg_rating"]) if pd.notna(x["avg_rating"]) else global_avg
            if v == 0:
                return global_avg
            return (v / (v + m) * book_avg_rating) + (m / (v + m) * global_avg)

        count_adjusted = books_df.apply(weighted_rating, axis=1)

        scaler = MinMaxScaler()
        valid_mask = ~np.isnan(solo_final_pred)
        solo_pred = np.full(len(books_df), np.nan)
        friend_pred = np.full(len(books_df), np.nan)
        if valid_mask.any():
            solo_pred[valid_mask] = scaler.fit_transform(solo_final_pred[valid_mask].reshape(-1, 1)).flatten()
            friend_pred[valid_mask] = scaler.fit_transform(friend_final_pred[valid_mask].reshape(-1, 1)).flatten()
        count_adjusted_scaled = scaler.fit_transform(np.asarray(count_adjusted).reshape(-1, 1)).flatten()

        books_df["solo_pred_rating"] = solo_pred
        books_df["friend_pred_rating"] = friend_pred
        books_df["count_adjusted_rating"] = count_adjusted_scaled
        books_df["pred_rating"] = friend_pred * solo_pred
        books_df["final_rating"] = count_adjusted_scaled * friend_pred * solo_pred

        print("  Saving predictions to database...")
        now_str = datetime.now().strftime("%Y-%m-%d")
        predictions_data = []

        for row in books_df.to_dict(orient="records"):
            required = [
                row["solo_pred_rating"],
                row["friend_pred_rating"],
                row["pred_rating"],
                row["final_rating"],
            ]
            if any(pd.isna(value) for value in required):
                continue
            predictions_data.append(
                (
                    int(row["legacy_id"]),
                    float(row["solo_pred_rating"]),
                    float(row["friend_pred_rating"]),
                    float(row["count_adjusted_rating"]),
                    float(row["pred_rating"]),
                    float(row["final_rating"]),
                    now_str,
                )
            )

        db.upsert_rows(
            db_conn,
            "predictions",
            predictions_data,
            [
                "book_id",
                "solo_pred_rating",
                "friend_pred_rating",
                "count_adjusted_rating",
                "pred_rating",
                "final_rating",
                "date_updated",
            ],
        )

        print(f"  Saved predictions for {len(predictions_data)} books.")
