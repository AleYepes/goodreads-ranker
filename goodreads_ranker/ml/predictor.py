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
from tqdm import tqdm

from goodreads_ranker.core import config, db, utils

torch.sparse.check_sparse_tensor_invariants.disable()


def safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or len(y) < 3 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return np.nan
    res = spearmanr(x, y)
    rho = cast(tuple[Any, ...], res)[0]
    val = float(rho)
    return val if not np.isnan(val) else np.nan


def normalize_model_params(params):
    params = {key: value.item() if hasattr(value, "item") else value for key, value in dict(params).items()}
    params["num_propagations"] = int(params["num_propagations"])
    params["knn_neighbors"] = int(params["knn_neighbors"])
    return params


def build_adjacency_matrix(books_df, num_nodes, db_conn):
    id_to_idx = {int(bid): i for i, bid in enumerate(books_df["legacy_id"])}
    book_ids_set = set(id_to_idx.keys())

    edges = db.get_similar_books_edges(db_conn)
    edge_indices = []
    for bid, sim_id in edges:
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


def _build_ensemble_predictions(
    params: dict,
    train_embeddings: np.ndarray,
    y_train: np.ndarray,
    predict_embeddings: np.ndarray,
) -> np.ndarray:
    knn = KNeighborsRegressor(
        n_neighbors=min(int(params["knn_neighbors"]), len(train_embeddings)),
        metric="cosine",
        weights="distance",
        n_jobs=-1,
    )
    knn.fit(train_embeddings, y_train)
    knn_pred = knn.predict(predict_embeddings)

    brr = BayesianRidge(
        alpha_1=params["brr_alpha_1"],
        alpha_2=params["brr_alpha_2"],
        lambda_1=params["brr_lambda_1"],
        lambda_2=params["brr_lambda_2"],
        compute_score=True,
    )
    brr.fit(train_embeddings, y_train)
    brr_mu, brr_std = cast(tuple[np.ndarray, np.ndarray], brr.predict(predict_embeddings, return_std=True))
    brr_pred = brr_mu - (params["brr_uncertainty_penalty"] * brr_std)

    svr = SVR(kernel="rbf", gamma="scale", C=params["svr_regularization"], epsilon=params["svr_epsilon"])
    svr.fit(train_embeddings, y_train)
    svr_pred = svr.predict(predict_embeddings)

    knn_weight = params["knn_weight"]
    brr_weight = params["brr_weight"]
    remaining_weight = 1 - knn_weight
    actual_brr_weight = brr_weight * remaining_weight
    svr_weight = remaining_weight - actual_brr_weight

    return (knn_weight * knn_pred) + (actual_brr_weight * brr_pred) + (svr_weight * svr_pred)


def prep_optimization(
    books_df,
    precomputed_embeddings,
    training_col,
    mrl_dimensions,
    max_propagations,
    budget=100,
    desc="Optimizing model",
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
        params = {
            "knn_neighbors": knn_neighbors,
            "brr_alpha_1": brr_alpha_1,
            "brr_alpha_2": brr_alpha_2,
            "brr_lambda_1": brr_lambda_1,
            "brr_lambda_2": brr_lambda_2,
            "brr_uncertainty_penalty": brr_uncertainty_penalty,
            "svr_regularization": svr_regularization,
            "svr_epsilon": svr_epsilon,
            "knn_weight": knn_weight,
            "brr_weight": brr_weight,
        }
        all_embeddings = precomputed_embeddings[num_propagations]
        train_embeddings = all_embeddings[training_mask]
        y = my_ratings_scaled

        y_reals = []
        y_preds = []
        skf = KFold(n_splits=min(10, len(train_embeddings)), shuffle=True, random_state=42)
        for train_idx, test_idx in skf.split(train_embeddings):
            train_fold_embeddings = train_embeddings[train_idx]
            test_fold_embeddings = train_embeddings[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            fold_pred = _build_ensemble_predictions(params, train_fold_embeddings, y_train, test_fold_embeddings)
            y_reals.append(y_test)
            y_preds.append(fold_pred)

        y_reals_arr = np.concatenate(y_reals)
        y_preds_arr = np.concatenate(y_preds)

        mse = mean_squared_error(y_reals_arr, y_preds_arr)
        ndcg = ndcg_score(np.vstack([y_reals_arr]), np.vstack([y_preds_arr]))
        if np.std(y_preds_arr) < 1e-9:
            spearman = 0
        else:
            spearman = safe_spearman(y_reals_arr, y_preds_arr)
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

    for _ in tqdm(range(budget), desc=desc, unit="trial"):
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
    final_pred_scaled = _build_ensemble_predictions(best_params, train_embeddings, my_ratings_scaled, all_embeddings)
    return scaler.inverse_transform(final_pred_scaled.reshape(-1, 1)).flatten()


def _get_or_optimize_hyperparams(
    db_conn,
    name,
    embedding_model,
    training_col,
    embedded_books_df,
    precomputed_embeddings,
    mrl_dimensions,
    max_propagations,
    train_set_size,
    force_optimize,
    optimization_budget,
):
    existing = None if force_optimize else db.get_prediction_hyperparams(db_conn, name, embedding_model)
    if existing is not None:
        return normalize_model_params(existing)

    desc = "Optimizing friend-taste model" if name == "friend_params" else "Optimizing solo-taste model"
    params = prep_optimization(
        embedded_books_df,
        precomputed_embeddings,
        training_col,
        mrl_dimensions,
        max_propagations,
        budget=optimization_budget,
        desc=desc,
    )
    params = normalize_model_params(params)
    db.save_prediction_hyperparams(db_conn, name, embedding_model, params, int(train_set_size))
    return params


def run_prediction(
    optimize=None,
    force_optimize=False,
    embedding_model=None,
    min_friend_similarity=0.3,
    optimization_budget=100,
    db_path=None,
):
    # Support backward compatibility if someone passes optimize instead of force_optimize
    if optimize is not None:
        force_optimize = optimize

    db.init_db(db_path)
    with db.get_connection(db_path) as db_conn:
        try:
            main_library_id = db.get_main_library_id(db_conn)
        except RuntimeError as err:
            raise RuntimeError("No main library found. Run seed first.") from err

        books_rows = db.get_books_for_prediction(db_conn, main_library_id)
        if not books_rows:
            raise RuntimeError("No book records found in the books table. Run crawler first.")

        books_df = pd.DataFrame(books_rows)
        books_df["my_rating"] = pd.Series(
            pd.to_numeric(books_df["my_rating"], errors="coerce"),
            index=books_df.index,
        ).replace(0, np.nan)

        if books_df["my_rating"].notna().sum() == 0:
            raise RuntimeError("No self rating records found. Run seed first.")

        # Compute my_refined
        existing_rows = db.get_elo_ratings(db_conn)
        target_ratings = books_df.dropna(subset=["my_rating"]).set_index("legacy_id")["my_rating"].to_dict()
        elo_df = utils.merge_elo_state(existing_rows, target_ratings)
        my_refined = utils.compute_continuous(elo_df)
        books_df = books_df.merge(my_refined.rename("my_refined"), on="legacy_id", how="left")

        # Load embeddings
        if not embedding_model:
            embedding_model = config.get_embedding_model()

        vectors_by_id = db.get_embeddings_by_model(db_conn, embedding_model)
        has_embedding, embedded_embeddings = utils.assemble_embedding_matrix(
            books_df["legacy_id"].tolist(), vectors_by_id
        )

        if embedded_embeddings.size == 0:
            raise RuntimeError("No valid embeddings found. Run embedder before ranking.")

        embedded_books_df = books_df[has_embedding].copy().reset_index(drop=True)

        # Get friend calibrated ratings
        friend_calibrated = db.get_friend_calibrated_ratings(db_conn, min_friend_similarity)

        friend_ratings_dict = {}
        if friend_calibrated:
            fc_df = pd.DataFrame(friend_calibrated)
            fc_df["weighted_val"] = fc_df["calibrated_rating"] * fc_df["similarity_score"]
            my_legacy_id_set = set(books_df.dropna(subset=["my_refined"])["legacy_id"].tolist())
            fc_df = fc_df[~fc_df["book_legacy_id"].isin(my_legacy_id_set)]

            if not fc_df.empty:
                grouped = fc_df.groupby("book_legacy_id").agg(
                    weighted_sum=("weighted_val", "sum"), total_weight=("similarity_score", "sum")
                )
                friend_ratings_dict = (grouped["weighted_sum"] / grouped["total_weight"]).to_dict()

        has_friend_data = bool(friend_ratings_dict)

        legacy_id_series = pd.Series(books_df["legacy_id"], index=books_df.index)
        embedded_legacy_id_series = pd.Series(embedded_books_df["legacy_id"], index=embedded_books_df.index)
        books_my_refined = cast(pd.Series, books_df["my_refined"])
        embedded_my_refined = cast(pd.Series, embedded_books_df["my_refined"])

        books_df["training_ratings"] = books_my_refined.fillna(legacy_id_series.map(friend_ratings_dict))
        embedded_books_df["training_ratings"] = embedded_my_refined.fillna(
            embedded_legacy_id_series.map(friend_ratings_dict)
        )

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        train_size = (~pd.isna(embedded_books_df["training_ratings"])).sum()
        solo_train_size = (~pd.isna(embedded_books_df["my_refined"])).sum()
        if (has_friend_data and train_size < 2) or solo_train_size < 2:
            raise RuntimeError(
                "Not enough rated books with valid embeddings to train ranking models "
                f"(training={train_size}, personal={solo_train_size})."
            )

        mrl_dimensions = 32
        while mrl_dimensions * 2 < max(train_size, solo_train_size):
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

        # Solo model (always run)
        solo_params = _get_or_optimize_hyperparams(
            db_conn,
            "solo_params",
            embedding_model,
            "my_refined",
            embedded_books_df,
            precomputed_embeddings,
            mrl_dimensions,
            max_propagations,
            solo_train_size,
            force_optimize,
            optimization_budget,
        )
        solo_final_pred_embedded = run_optimized(solo_params, embedded_books_df, precomputed_embeddings, "my_refined")

        solo_final_pred = np.full(len(books_df), np.nan)
        embedded_positions = np.where(has_embedding)[0]
        solo_final_pred[embedded_positions] = solo_final_pred_embedded

        # Friend model (conditional)
        friend_final_pred = np.full(len(books_df), np.nan)
        if has_friend_data:
            friend_params = _get_or_optimize_hyperparams(
                db_conn,
                "friend_params",
                embedding_model,
                "training_ratings",
                embedded_books_df,
                precomputed_embeddings,
                mrl_dimensions,
                max_propagations,
                train_size,
                force_optimize,
                optimization_budget,
            )
            friend_final_pred_embedded = run_optimized(
                friend_params, embedded_books_df, precomputed_embeddings, "training_ratings"
            )
            friend_final_pred[embedded_positions] = friend_final_pred_embedded

        # Metadata processing & rating count adjustments
        star_cols = ["star_1", "star_2", "star_3", "star_4", "star_5"]
        books_df["rating_count"] = books_df[star_cols].sum(axis=1)

        total_stars = (
            books_df["star_1"] * 1
            + books_df["star_2"] * 2
            + books_df["star_3"] * 3
            + books_df["star_4"] * 4
            + books_df["star_5"] * 5
        )
        books_df["avg_rating"] = np.where(books_df["rating_count"] > 0, total_stars / books_df["rating_count"], np.nan)

        v = books_df["rating_count"].astype(float)
        book_avg = books_df["avg_rating"].fillna(0.0).astype(float)
        count_adjusted = np.where(
            v > 0,
            book_avg - (book_avg / np.log10(v + 10)),
            0.0,
        )

        scaler = MinMaxScaler()
        valid_mask = ~np.isnan(solo_final_pred)
        solo_pred = np.full(len(books_df), np.nan)
        friend_pred = np.full(len(books_df), np.nan)

        if valid_mask.any():
            solo_pred[valid_mask] = scaler.fit_transform(solo_final_pred[valid_mask].reshape(-1, 1)).flatten()

        if has_friend_data:
            friend_valid_mask = ~np.isnan(friend_final_pred)
            if friend_valid_mask.any():
                friend_pred[friend_valid_mask] = scaler.fit_transform(
                    friend_final_pred[friend_valid_mask].reshape(-1, 1)
                ).flatten()

        count_adjusted_scaled = scaler.fit_transform(np.asarray(count_adjusted).reshape(-1, 1)).flatten()

        books_df["solo_pred_rating"] = solo_pred
        books_df["friend_pred_rating"] = friend_pred if has_friend_data else None
        books_df["count_adjusted_rating"] = count_adjusted_scaled

        if has_friend_data:
            books_df["final_rating"] = count_adjusted_scaled * friend_pred * solo_pred
        else:
            books_df["final_rating"] = count_adjusted_scaled * solo_pred

        now_str = datetime.now().strftime("%Y-%m-%d")
        predictions_data = []

        for row in books_df.to_dict(orient="records"):
            required = [
                row["solo_pred_rating"],
                row["final_rating"],
            ]
            if any(pd.isna(value) for value in required):
                continue
            predictions_data.append(
                (
                    int(row["legacy_id"]),
                    float(row["solo_pred_rating"]),
                    float(row["friend_pred_rating"])
                    if row["friend_pred_rating"] is not None and pd.notna(row["friend_pred_rating"])
                    else None,
                    float(row["count_adjusted_rating"]),
                    float(row["final_rating"]),
                    now_str,
                )
            )

        if predictions_data:
            db.save_book_predictions(db_conn, predictions_data, embedding_model)
            db.prune_book_predictions(db_conn, [x[0] for x in predictions_data], embedding_model)
            print("✓ Ranking predictions complete and saved.")
