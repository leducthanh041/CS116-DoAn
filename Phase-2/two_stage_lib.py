# two_stage_lib.py
# Clean, terminal-friendly 2-stage recommender training.
# Stage 1: Item-Item CF (TF-IDF + Cosine kNN) + random search
# Stage 2: LightGBM ranker (binary) trained on Stage-1 candidates
#
# Design goals:
# - Avoid converting full raw tables to pandas.
# - Use Polars LazyFrame for IO/filtering and only collect needed columns.
# - Stage 1 precomputes UI aggregates ONCE; random search only recomputes weights.

from __future__ import annotations

import os
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

import joblib
import lightgbm as lgb


# ============================================================
# 0) IO: scan parquet chunks (Polars LazyFrame)
# ============================================================

def _glob_parquets(data_dir: str, contains: str) -> List[str]:
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".parquet")]
    files = [f for f in files if contains in os.path.basename(f)]
    if not files:
        raise FileNotFoundError(f"Không tìm thấy parquet chứa '{contains}' trong: {data_dir}")
    return sorted(files)


def scan_user(data_dir: str) -> pl.LazyFrame:
    return pl.scan_parquet(_glob_parquets(data_dir, "user_chunk"))


def scan_item(data_dir: str) -> pl.LazyFrame:
    return pl.scan_parquet(_glob_parquets(data_dir, "item_chunk"))


def scan_transaction(data_dir: str) -> pl.LazyFrame:
    # notebook của bạn dùng purchase_history_daily_chunk
    return pl.scan_parquet(_glob_parquets(data_dir, "purchase_history_daily_chunk"))


def ensure_created_datetime_lf(
    lf: pl.LazyFrame,
    candidates: Tuple[str, ...] = ("created_datetime", "created_date", "timestamp"),
) -> pl.LazyFrame:
    """
    Ensure there is a `created_datetime` Datetime column.
    """
    cols = lf.columns
    if "created_datetime" in cols:
        return lf.with_columns(pl.col("created_datetime").cast(pl.Datetime, strict=False))
    for c in candidates:
        if c in cols:
            return lf.with_columns(pl.col(c).cast(pl.Datetime, strict=False).alias("created_datetime"))
    raise ValueError(f"Không tìm thấy cột thời gian trong {candidates}. Cột hiện có: {cols}")


def split_train_valid_lf(
    lf_trx: pl.LazyFrame,
    year: int,
    train_month_start: int,
    train_month_end: int,
    valid_month: int,
) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
    lf_trx = ensure_created_datetime_lf(lf_trx)
    lf_year = lf_trx.filter(pl.col("created_datetime").dt.year() == year)

    lf_train = lf_year.filter(
        pl.col("created_datetime").dt.month().is_between(train_month_start, train_month_end, closed="both")
    )
    lf_valid = lf_year.filter(pl.col("created_datetime").dt.month() == valid_month)
    return lf_train, lf_valid


# ============================================================
# 1) Stage 1: Precompute UI aggregates once
# ============================================================

@dataclass
class Stage1Precomputed:
    # CSR structure
    n_users: int
    n_items: int
    indptr: np.ndarray
    indices: np.ndarray

    # ID maps
    user_index_to_id: List[Any]
    item_index_to_id: List[Any]
    user_id_to_index: Dict[Any, int]
    item_id_to_index: Dict[Any, int]

    # UI-level columns aligned with CSR data order
    freq_cnt: np.ndarray
    total_spent_ui: np.ndarray
    days_since_ui: np.ndarray

    # User totals (aligned with CSR data order by row-user mapping)
    u_total_cnt: np.ndarray
    u_total_spent: np.ndarray

    # Precomputed terms for preferences (aligned with CSR data order)
    l1_cnt_norm: np.ndarray
    l1_spent_norm: np.ndarray
    l1_rec_term: np.ndarray

    l2_cnt_norm: np.ndarray
    l2_spent_norm: np.ndarray
    l2_rec_term: np.ndarray

    # For evaluation
    valid_pairs_user: np.ndarray
    valid_pairs_item: np.ndarray


def _factorize_two_cols_to_codes(user_ids: np.ndarray, item_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[Any], List[Any]]:
    # stable factorization
    u_uniques, u_codes = np.unique(user_ids, return_inverse=True)
    i_uniques, i_codes = np.unique(item_ids, return_inverse=True)
    return u_codes.astype(np.int32), i_codes.astype(np.int32), list(u_uniques), list(i_uniques)


def _build_csr_structure(u_codes: np.ndarray, i_codes: np.ndarray, n_users: int, n_items: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build CSR structure for unique (u,i) pairs.
    Returns: order (permutation), indptr, indices
    """
    order = np.lexsort((i_codes, u_codes))  # sort by u then i
    u_sorted = u_codes[order]
    i_sorted = i_codes[order]

    # indptr
    counts = np.bincount(u_sorted, minlength=n_users)
    indptr = np.zeros(n_users + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(counts, dtype=np.int64)
    indices = i_sorted.astype(np.int32)
    return order, indptr, indices


def precompute_stage1(
    lf_train: pl.LazyFrame,
    lf_valid: pl.LazyFrame,
    lf_item: pl.LazyFrame,
    user_col: str = "customer_id",
    item_col: str = "item_id",
    price_col: str = "price",
    qty_col: str = "quantity",
    cat_l1_col: str = "category_l1",
    cat_l2_col: str = "category_l2",
    quick_limit_train: Optional[int] = None,
    quick_limit_valid: Optional[int] = None,
) -> Stage1Precomputed:
    """
    Precompute all heavy aggregations ONCE in Polars.
    Random search trials only recompute weights and fit TF-IDF + kNN.
    """

    # 1) Select minimal trx cols early
    need_cols = [user_col, item_col, "created_datetime"]
    if price_col in lf_train.columns:
        need_cols.append(price_col)
    if qty_col in lf_train.columns:
        need_cols.append(qty_col)

    lf_train2 = ensure_created_datetime_lf(lf_train).select([c for c in need_cols if c in lf_train.columns])
    lf_valid2 = ensure_created_datetime_lf(lf_valid).select([user_col, item_col, "created_datetime"])

    if quick_limit_train is not None:
        lf_train2 = lf_train2.limit(int(quick_limit_train))
    if quick_limit_valid is not None:
        lf_valid2 = lf_valid2.limit(int(quick_limit_valid))

    # 2) Item meta (only needed cols) + join categories
    lf_item_small = lf_item.select([c for c in [item_col, cat_l1_col, cat_l2_col] if c in lf_item.columns]).unique(subset=[item_col])

    train = lf_train2.join(lf_item_small, on=item_col, how="left")

    # 3) Monetary spent_row
    if price_col in train.columns:
        train = train.with_columns(pl.col(price_col).cast(pl.Float64, strict=False).fill_null(0.0).alias(price_col))
    else:
        train = train.with_columns(pl.lit(0.0).alias(price_col))

    if qty_col in train.columns:
        train = train.with_columns(pl.col(qty_col).cast(pl.Float64, strict=False).fill_null(1.0).alias(qty_col))
    else:
        train = train.with_columns(pl.lit(1.0).alias(qty_col))

    train = train.with_columns((pl.col(price_col) * pl.col(qty_col)).alias("spent_row"))

    # 4) Fill categories
    if cat_l1_col in train.columns:
        train = train.with_columns(pl.col(cat_l1_col).fill_null("__UNK_CAT_L1__"))
    else:
        train = train.with_columns(pl.lit("__NO_CAT_L1__").alias(cat_l1_col))

    if cat_l2_col in train.columns:
        train = train.with_columns(pl.col(cat_l2_col).fill_null("__UNK_CAT_L2__"))
    else:
        train = train.with_columns(pl.lit("__NO_CAT_L2__").alias(cat_l2_col))

    # 5) cutoff_dt
    cutoff_dt = train.select(pl.col("created_datetime").max()).collect(streaming=True).item()
    if cutoff_dt is None:
        raise RuntimeError("Train split rỗng sau khi lọc. Kiểm tra lại split hoặc dữ liệu.")

    # 6) UI aggregate
    ui = (
        train.group_by([user_col, item_col])
        .agg([
            pl.len().alias("freq_cnt"),
            pl.col("spent_row").sum().alias("total_spent_ui"),
            pl.col("created_datetime").max().alias("last_purchase_dt"),
            pl.first(cat_l1_col).alias(cat_l1_col),
            pl.first(cat_l2_col).alias(cat_l2_col),
        ])
    ).with_columns([
        (
        pl.when((pl.lit(cutoff_dt) - pl.col("last_purchase_dt")).dt.total_days().cast(pl.Int32) < 0)
        .then(0)
        .otherwise((pl.lit(cutoff_dt) - pl.col("last_purchase_dt")).dt.total_days().cast(pl.Int32))
        .fill_null(9999)
        .alias("days_since_ui")
    ),
    ])

    # 7) User totals
    user_tot = (
        train.group_by(user_col)
        .agg([
            pl.len().alias("u_total_cnt"),
            pl.col("spent_row").sum().alias("u_total_spent"),
        ])
    ).with_columns([
        pl.when(pl.col("u_total_cnt").cast(pl.Float64) < 1.0)
        .then(1.0)
        .otherwise(pl.col("u_total_cnt").cast(pl.Float64)),
        pl.col("u_total_spent").cast(pl.Float64),
    ])

    ui = ui.join(user_tot, on=user_col, how="left").with_columns([
        pl.col("u_total_cnt").fill_null(1.0),
        pl.col("u_total_spent").fill_null(0.0),
    ])

    # 8) L1 aggregates
    u_l1 = (
        train.group_by([user_col, cat_l1_col])
        .agg([
            pl.len().alias("l1_cnt"),
            pl.col("spent_row").sum().alias("l1_spent"),
            pl.col("created_datetime").max().alias("l1_last_dt"),
        ])
    )

    ui = ui.join(u_l1, on=[user_col, cat_l1_col], how="left").with_columns([
        pl.col("l1_cnt").fill_null(0).cast(pl.Float64),
        pl.col("l1_spent").fill_null(0.0).cast(pl.Float64),
        (
        pl.when((pl.lit(cutoff_dt) - pl.col("l1_last_dt")).dt.total_days().cast(pl.Int32) < 0)
        .then(0)
        .otherwise((pl.lit(cutoff_dt) - pl.col("l1_last_dt")).dt.total_days().cast(pl.Int32))
        .fill_null(9999)
        .alias("l1_days")
    ),
    ])

    # 9) L2 aggregates
    u_l2 = (
        train.group_by([user_col, cat_l2_col])
        .agg([
            pl.len().alias("l2_cnt"),
            pl.col("spent_row").sum().alias("l2_spent"),
            pl.col("created_datetime").max().alias("l2_last_dt"),
        ])
    )

    ui = ui.join(u_l2, on=[user_col, cat_l2_col], how="left").with_columns([
        pl.col("l2_cnt").fill_null(0).cast(pl.Float64),
        pl.col("l2_spent").fill_null(0.0).cast(pl.Float64),
        (
        pl.when((pl.lit(cutoff_dt) - pl.col("l2_last_dt")).dt.total_days().cast(pl.Int32) < 0)
        .then(0)
        .otherwise((pl.lit(cutoff_dt) - pl.col("l2_last_dt")).dt.total_days().cast(pl.Int32))
        .fill_null(9999)
        .alias("l2_days")
    ),
    ])

    # Collect UI to numpy
    ui_pd = ui.select([
        user_col, item_col,
        "freq_cnt", "total_spent_ui", "days_since_ui",
        "u_total_cnt", "u_total_spent",
        "l1_cnt", "l1_spent", "l1_days",
        "l2_cnt", "l2_spent", "l2_days",
    ]).collect(streaming=True).to_pandas()

    # Factorize users/items once
    u_ids = ui_pd[user_col].to_numpy()
    i_ids = ui_pd[item_col].to_numpy()

    u_codes, i_codes, u_uniques, i_uniques = _factorize_two_cols_to_codes(u_ids, i_ids)
    n_users, n_items = len(u_uniques), len(i_uniques)

    user_id_to_index = {uid: idx for idx, uid in enumerate(u_uniques)}
    item_id_to_index = {iid: idx for idx, iid in enumerate(i_uniques)}

    order, indptr, indices = _build_csr_structure(u_codes, i_codes, n_users, n_items)

    # Align arrays with CSR ordering
    ui_pd = ui_pd.iloc[order].reset_index(drop=True)

    freq_cnt = ui_pd["freq_cnt"].astype(np.float32).to_numpy()
    total_spent_ui = ui_pd["total_spent_ui"].astype(np.float32).to_numpy()
    days_since_ui = ui_pd["days_since_ui"].astype(np.float32).to_numpy()

    u_total_cnt = ui_pd["u_total_cnt"].astype(np.float32).to_numpy()
    u_total_spent = ui_pd["u_total_spent"].astype(np.float32).to_numpy()

    # Precompute normalized terms
    # NOTE: log1p(0)=0 okay. Add eps for spent denom.
    l1_cnt = ui_pd["l1_cnt"].astype(np.float32).to_numpy()
    l1_spent = ui_pd["l1_spent"].astype(np.float32).to_numpy()
    l1_days = ui_pd["l1_days"].astype(np.float32).to_numpy()

    l2_cnt = ui_pd["l2_cnt"].astype(np.float32).to_numpy()
    l2_spent = ui_pd["l2_spent"].astype(np.float32).to_numpy()
    l2_days = ui_pd["l2_days"].astype(np.float32).to_numpy()

    l1_cnt_norm = (np.log1p(l1_cnt) / np.log1p(np.clip(u_total_cnt, 1e-6, None))).astype(np.float32)
    l1_spent_norm = (np.log1p(l1_spent) / (np.log1p(np.clip(u_total_spent, 0.0, None)) + 1e-9)).astype(np.float32)

    l2_cnt_norm = (np.log1p(l2_cnt) / np.log1p(np.clip(u_total_cnt, 1e-6, None))).astype(np.float32)
    l2_spent_norm = (np.log1p(l2_spent) / (np.log1p(np.clip(u_total_spent, 0.0, None)) + 1e-9)).astype(np.float32)

    # valid pairs to evaluate (filter unknown items/users later)
    valid_pd = lf_valid2.collect(streaming=True).to_pandas()
    v_u = valid_pd[user_col].to_numpy()
    v_i = valid_pd[item_col].to_numpy()

    pre_obj = Stage1Precomputed(
        n_users=n_users,
        n_items=n_items,
        indptr=indptr,
        indices=indices,
        user_index_to_id=u_uniques,
        item_index_to_id=i_uniques,
        user_id_to_index=user_id_to_index,
        item_id_to_index=item_id_to_index,
        freq_cnt=freq_cnt,
        total_spent_ui=total_spent_ui,
        days_since_ui=days_since_ui,
        u_total_cnt=u_total_cnt,
        u_total_spent=u_total_spent,
        l1_cnt_norm=l1_cnt_norm,
        l1_spent_norm=l1_spent_norm,
        l1_rec_term=None,  # filled in per-trial (lambda)
        l2_cnt_norm=l2_cnt_norm,
        l2_spent_norm=l2_spent_norm,
        l2_rec_term=None,  # filled in per-trial (lambda)
        valid_pairs_user=v_u,
        valid_pairs_item=v_i,
    )

    # attach day arrays for recency preference terms
    pre_obj._l1_days = l1_days.astype(np.float32)
    pre_obj._l2_days = l2_days.astype(np.float32)

    return pre_obj



# ============================================================
# 2) Stage 1 model: TF-IDF + Cosine kNN (fit from precompute)
# ============================================================

class ItemItemCFStage1:
    """
    Fit from Stage1Precomputed to avoid heavy recomputation during random search.
    """

    def __init__(
        self,
        pre: Stage1Precomputed,
        k_eval: int = 500,
        # base
        weight_type: str = "log_count",  # binary|count|log_count|rel_freq
        # UI recency
        use_ui_recency: bool = True,
        ui_recency_lambda: float = 0.01,
        # UI monetary
        use_ui_monetary: bool = True,
        # category pref
        use_cat_l1_pref: bool = True,
        alpha_l1_cnt: float = 0.10,
        alpha_l1_spent: float = 0.10,
        alpha_l1_rec: float = 0.10,
        l1_recency_lambda: float = 0.01,
        use_cat_l2_pref: bool = True,
        alpha_l2_cnt: float = 0.15,
        alpha_l2_spent: float = 0.15,
        alpha_l2_rec: float = 0.15,
        l2_recency_lambda: float = 0.01,
        # knn
        n_neighbors: int = 100,
    ):
        self.pre = pre
        self.k_eval = int(k_eval)

        self.weight_type = weight_type
        self.use_ui_recency = use_ui_recency
        self.ui_recency_lambda = float(ui_recency_lambda)
        self.use_ui_monetary = use_ui_monetary

        self.use_cat_l1_pref = use_cat_l1_pref
        self.alpha_l1_cnt = float(alpha_l1_cnt)
        self.alpha_l1_spent = float(alpha_l1_spent)
        self.alpha_l1_rec = float(alpha_l1_rec)
        self.l1_recency_lambda = float(l1_recency_lambda)

        self.use_cat_l2_pref = use_cat_l2_pref
        self.alpha_l2_cnt = float(alpha_l2_cnt)
        self.alpha_l2_spent = float(alpha_l2_spent)
        self.alpha_l2_rec = float(alpha_l2_rec)
        self.l2_recency_lambda = float(l2_recency_lambda)

        self.n_neighbors = int(n_neighbors)

    # ---------- build ui matrix data vector ----------
    def _compute_w_ui(self) -> np.ndarray:
        pre = self.pre

        # base weight
        if self.weight_type == "binary":
            base = np.ones_like(pre.freq_cnt, dtype=np.float32)
        elif self.weight_type == "count":
            base = pre.freq_cnt.astype(np.float32)
        elif self.weight_type == "log_count":
            base = np.log1p(pre.freq_cnt).astype(np.float32)
        elif self.weight_type == "rel_freq":
            # rel_freq = freq / user_total_freq (per row already has user total aligned)
            base = (pre.freq_cnt / np.clip(pre.u_total_cnt, 1e-6, None)).astype(np.float32)
        else:
            raise ValueError(f"Unknown weight_type: {self.weight_type}")

        # recency factor (days_since_ui already aligned)
        if self.use_ui_recency:
            ui_rec = np.exp(-self.ui_recency_lambda * pre.days_since_ui).astype(np.float32)
        else:
            ui_rec = np.ones_like(base, dtype=np.float32)

        # monetary factor
        if self.use_ui_monetary:
            ui_mon = np.log1p(np.clip(pre.total_spent_ui, 0.0, None)).astype(np.float32)
        else:
            ui_mon = np.ones_like(base, dtype=np.float32)

        # preference terms
        if self.use_cat_l1_pref:
            l1_rec_term = np.exp(-self.l1_recency_lambda * pre.l1_days).astype(np.float32) if hasattr(pre, "l1_days") else None
            # we didn't store l1_days in pre -> so compute from valid? not possible.
            # Therefore: store l1_rec_term in precompute is required.
            # We'll compute from l1_days kept as hidden attribute on pre (set below).
        else:
            l1_rec_term = None

        return base, ui_rec, ui_mon

    def fit(self):
        # NOTE: We need l1_days / l2_days terms; keep them in hidden fields when precomputing.
        pre = self.pre
        if not hasattr(pre, "_l1_days") or not hasattr(pre, "_l2_days"):
            raise RuntimeError("Stage1Precomputed missing l1/l2 day arrays. Use `precompute_stage1()` from this lib.")

        # base
        if self.weight_type == "binary":
            base = np.ones_like(pre.freq_cnt, dtype=np.float32)
        elif self.weight_type == "count":
            base = pre.freq_cnt.astype(np.float32)
        elif self.weight_type == "log_count":
            base = np.log1p(pre.freq_cnt).astype(np.float32)
        elif self.weight_type == "rel_freq":
            base = (pre.freq_cnt / np.clip(pre.u_total_cnt, 1e-6, None)).astype(np.float32)
        else:
            raise ValueError(f"Unknown weight_type: {self.weight_type}")

        if self.use_ui_recency:
            ui_rec = np.exp(-self.ui_recency_lambda * pre.days_since_ui).astype(np.float32)
        else:
            ui_rec = np.ones_like(base, dtype=np.float32)

        if self.use_ui_monetary:
            ui_mon = np.log1p(np.clip(pre.total_spent_ui, 0.0, None)).astype(np.float32)
        else:
            ui_mon = np.ones_like(base, dtype=np.float32)

        if self.use_cat_l1_pref:
            l1_rec_term = np.exp(-self.l1_recency_lambda * pre._l1_days).astype(np.float32)
            pref_l1 = (
                self.alpha_l1_cnt * pre.l1_cnt_norm
                + self.alpha_l1_spent * pre.l1_spent_norm
                + self.alpha_l1_rec * l1_rec_term
            ).astype(np.float32)
        else:
            pref_l1 = np.zeros_like(base, dtype=np.float32)

        if self.use_cat_l2_pref:
            l2_rec_term = np.exp(-self.l2_recency_lambda * pre._l2_days).astype(np.float32)
            pref_l2 = (
                self.alpha_l2_cnt * pre.l2_cnt_norm
                + self.alpha_l2_spent * pre.l2_spent_norm
                + self.alpha_l2_rec * l2_rec_term
            ).astype(np.float32)
        else:
            pref_l2 = np.zeros_like(base, dtype=np.float32)

        w_ui = (base * ui_rec * ui_mon * (1.0 + pref_l1 + pref_l2)).astype(np.float32)

        # Build CSR (data aligned with indptr/indices)
        ui_matrix = csr_matrix((w_ui, pre.indices, pre.indptr), shape=(pre.n_users, pre.n_items), dtype=np.float32)
        self.ui_matrix_ = ui_matrix

        # popularity
        item_pop = np.asarray(ui_matrix.sum(axis=0)).ravel()
        self.popular_item_indices_ = np.argsort(-item_pop)

        # user history (indices per row)
        self.user_history_ = {}
        for u in range(pre.n_users):
            s, e = pre.indptr[u], pre.indptr[u + 1]
            self.user_history_[u] = set(pre.indices[s:e].tolist())

        # TF-IDF + kNN on item vectors
        self.tfidf_ = TfidfTransformer(norm="l2", use_idf=True, sublinear_tf=True)
        ui_tfidf = self.tfidf_.fit_transform(ui_matrix)
        X_items = ui_tfidf.T

        self.nn_model_ = NearestNeighbors(
            n_neighbors=self.n_neighbors + 1,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )
        self.nn_model_.fit(X_items)

        distances, indices = self.nn_model_.kneighbors(X_items, return_distance=True)
        sims = 1.0 - distances

        self.item_neighbors_ = indices[:, 1:]
        self.item_neighbor_sims_ = sims[:, 1:]

        return self

    # ---------- recommend ----------
    def recommend_for_user_id(self, user_id: Any, top_k: int, allow_repeat: bool = True) -> List[Any]:
        pre = self.pre
        if user_id not in pre.user_id_to_index:
            # cold user
            return [pre.item_index_to_id[i] for i in self.popular_item_indices_[:top_k]]

        uidx = pre.user_id_to_index[user_id]
        hist = self.user_history_.get(uidx, set())
        if not hist:
            return [pre.item_index_to_id[i] for i in self.popular_item_indices_[:top_k]]

        scores: Dict[int, float] = {}
        for item_i in hist:
            nbrs = self.item_neighbors_[item_i]
            sims = self.item_neighbor_sims_[item_i]
            for j, s in zip(nbrs, sims):
                if (not allow_repeat) and (j in hist):
                    continue
                scores[j] = scores.get(j, 0.0) + float(s)

        if not scores:
            rec_idx = self.popular_item_indices_[:top_k]
            return [pre.item_index_to_id[i] for i in rec_idx]

        rec_idx = [i for i, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]]
        return [pre.item_index_to_id[i] for i in rec_idx]

    # ---------- eval ----------
    def score_recall_hit(self, k: int, filter_train_history: bool = True) -> Dict[str, Any]:
        """
        Evaluate on VALID pairs included in pre.valid_pairs_*.
        """
        pre = self.pre
        k = int(k)

        # map valid to indices, drop unknown
        v_users = pre.valid_pairs_user
        v_items = pre.valid_pairs_item

        # build user->gt set
        user_to_gt: Dict[int, set] = {}
        for u_id, i_id in zip(v_users, v_items):
            if u_id not in pre.user_id_to_index:
                continue
            if i_id not in pre.item_id_to_index:
                continue
            uidx = pre.user_id_to_index[u_id]
            iidx = pre.item_id_to_index[i_id]
            user_to_gt.setdefault(uidx, set()).add(iidx)

        if not user_to_gt:
            return {"recall": 0.0, "hit": 0.0, "n_users_eval": 0}

        recalls = []
        hits = []

        for uidx, gt in user_to_gt.items():
            if filter_train_history:
                hist = self.user_history_.get(uidx, set())
                relevant = gt - hist
            else:
                relevant = gt

            if not relevant:
                continue

            user_id = pre.user_index_to_id[uidx]
            rec_items = self.recommend_for_user_id(user_id, top_k=k, allow_repeat=(not filter_train_history))
            rec_idx = set(pre.item_id_to_index[iid] for iid in rec_items if iid in pre.item_id_to_index)

            inter = rec_idx & relevant
            recalls.append(len(inter) / len(relevant))
            hits.append(1.0 if len(inter) > 0 else 0.0)

        if not recalls:
            return {"recall": 0.0, "hit": 0.0, "n_users_eval": 0}

        return {"recall": float(np.mean(recalls)), "hit": float(np.mean(hits)), "n_users_eval": len(recalls)}


# ============================================================
# 3) Stage 1 random search (overwrite best artifacts)
# ============================================================

DEFAULT_STAGE1_PARAM_SPACE = {
    "weight_type": ["log_count", "rel_freq"],
    "n_neighbors": [50, 100, 200],
    "use_ui_recency": [True],
    "ui_recency_lambda": [0.005, 0.01, 0.02],
    "use_cat_l1_pref": [True],
    "alpha_l1_cnt": [0.05, 0.1, 0.2],
    "alpha_l1_spent": [0.05, 0.1],
    "alpha_l1_rec": [0.05, 0.1],
    "use_cat_l2_pref": [False, True],
    "alpha_l2_cnt": [0.1, 0.15],
    "alpha_l2_spent": [0.1],
    "alpha_l2_rec": [0.1],
}


def _sample_params(param_space: Dict[str, List[Any]], n_trials: int, seed: int) -> List[Dict[str, Any]]:
    rnd = random.Random(seed)
    keys = list(param_space.keys())
    tried = set()
    out = []
    while len(out) < n_trials:
        cfg = {k: rnd.choice(param_space[k]) for k in keys}
        key = json.dumps(cfg, sort_keys=True)
        if key in tried:
            continue
        tried.add(key)
        out.append(cfg)
    return out


@dataclass
class Stage1SearchResult:
    best_cfg: Dict[str, Any]
    best_metrics: Dict[str, Any]
    results_df: pd.DataFrame


def train_stage1_random_search(
    pre: Stage1Precomputed,
    k_eval: int,
    n_trials: int,
    param_space: Optional[Dict[str, List[Any]]] = None,
    seed: int = 42,
    artifacts_dir: str = "./artifacts",
    model_path: str = "./artifacts/stage1_item_item_cf.pkl",
    params_path: str = "./artifacts/stage1_best_params.json",
    results_csv: str = "./artifacts/stage1_random_search_results.csv",
) -> Stage1SearchResult:
    os.makedirs(artifacts_dir, exist_ok=True)
    if param_space is None:
        param_space = DEFAULT_STAGE1_PARAM_SPACE

    configs = _sample_params(param_space, n_trials, seed)

    best_recall = -1.0
    best_cfg = None
    best_metrics = None

    rows = []

    for t, cfg in enumerate(configs, 1):
        print(f"\n[Stage1] Trial {t}/{n_trials} cfg={cfg}")

        model = ItemItemCFStage1(pre=pre, k_eval=k_eval, **cfg).fit()
        metrics = model.score_recall_hit(k=k_eval, filter_train_history=True)

        row = {"trial": t, **cfg, **metrics}
        rows.append(row)

        print(f"  Recall@{k_eval}={metrics['recall']:.6f} | Hit@{k_eval}={metrics['hit']:.6f} | n_users={metrics['n_users_eval']}")

        if metrics["recall"] > best_recall:
            best_recall = metrics["recall"]
            best_cfg = cfg
            best_metrics = metrics

            # save best (overwrite)
            to_save = {
                "pre": pre,
                "cfg": cfg,
                "k_eval": k_eval,
            }
            joblib.dump(to_save, model_path)

            with open(params_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)

            print(f"  NEW BEST -> overwrite: {model_path}")

        pd.DataFrame(rows).sort_values("recall", ascending=False).to_csv(results_csv, index=False)

    if best_cfg is None or best_metrics is None:
        raise RuntimeError("Stage1 random search failed.")

    return Stage1SearchResult(
        best_cfg=best_cfg,
        best_metrics=best_metrics,
        results_df=pd.DataFrame(rows).sort_values("recall", ascending=False),
    )


def load_best_stage1(model_path: str) -> ItemItemCFStage1:
    obj = joblib.load(model_path)
    pre = obj["pre"]
    cfg = obj["cfg"]
    k_eval = obj.get("k_eval", 500)
    return ItemItemCFStage1(pre=pre, k_eval=k_eval, **cfg).fit()


# ============================================================
# 4) Stage 2: build ranking dataset from Stage1 candidates
# ============================================================

def build_ground_truth(df_valid_pairs: pd.DataFrame, user_col: str, item_col: str) -> Dict[Any, List[Any]]:
    dfv = df_valid_pairs[[user_col, item_col]].dropna().drop_duplicates()
    return dfv.groupby(user_col)[item_col].apply(lambda s: list(set(s.tolist()))).to_dict()


def generate_candidates(
    stage1_model: ItemItemCFStage1,
    user_ids: Iterable[Any],
    k_cand: int,
    allow_repeat: bool,
) -> Dict[Any, List[Any]]:
    out = {}
    for u in user_ids:
        out[u] = stage1_model.recommend_for_user_id(u, top_k=k_cand, allow_repeat=allow_repeat)
    return out


def build_ranking_rows(
    candidates: Dict[Any, List[Any]],
    gt: Dict[Any, List[Any]],
) -> pd.DataFrame:
    rows = []
    for u, cand_items in candidates.items():
        gt_set = set(gt.get(u, []))
        for r, iid in enumerate(cand_items):
            rows.append({
                "customer_id": u,
                "item_id": iid,
                "label": 1 if iid in gt_set else 0,
                "stage1_rank": r,
            })
    return pd.DataFrame(rows)


# ============================================================
# 5) Stage 2: feature enrichment (incl. ./new-feature)
# ============================================================

def _scan_feature_parquet(new_feature_dir: str, filename: str) -> pl.LazyFrame:
    path = os.path.join(new_feature_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Thiếu file feature: {path}")
    return pl.scan_parquet(path)


def _auto_pick_numeric_col(df: pd.DataFrame, prefer: List[str], exclude: List[str]) -> Optional[str]:
    for c in prefer:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            return c
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None


def enrich_stage2_features(
    df_rank: pd.DataFrame,
    lf_user: pl.LazyFrame,
    lf_item: pl.LazyFrame,
    new_feature_dir: str,
    valid_month: int,
    top10_month_lag: int = 1,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Clean, deterministic feature merge:
      - Filter user/item to only those in df_rank (avoid full to_pandas)
      - Merge new-feature parquet files
    """

    # -------- filter user/item tables by rank ids (Polars JOIN, avoid is_in(huge_list)) --------
    rank_users_df = pl.DataFrame({"customer_id": df_rank["customer_id"].unique()})
    rank_items_df = pl.DataFrame({"item_id": df_rank["item_id"].unique()})

    user_schema = set(lf_user.collect_schema().names())
    item_schema = set(lf_item.collect_schema().names())

    user_cols = ["customer_id", "gender", "province", "membership", "user_age_days", "days_since_install", "days_since_last_sync"]
    user_cols = [c for c in user_cols if c in user_schema] or ["customer_id"]

    item_cols = ["item_id", "price", "category_l1", "category_l2", "brand", "age_group_final"]
    item_cols = [c for c in item_cols if c in item_schema] or ["item_id"]

    df_user = (
        lf_user
        .join(rank_users_df.lazy(), on="customer_id", how="inner")
        .select(user_cols)
        .unique(subset=["customer_id"])
        .collect(streaming=True)
        .to_pandas()
    )

    df_item = (
        lf_item
        .join(rank_items_df.lazy(), on="item_id", how="inner")
        .select(item_cols)
        .unique(subset=["item_id"])
        .collect(streaming=True)
        .to_pandas()
    )

    df_out = df_rank.merge(df_user, on="customer_id", how="left")
    df_out = df_out.merge(df_item, on="item_id", how="left")

    # -------- load new-feature files --------
    # 1) price_segment (item)
    lf_price = (
        _scan_feature_parquet(new_feature_dir, "price_segment.parquet")
        .select(["item_id", "price_segment"])
        .join(rank_items_df.lazy(), on="item_id", how="inner")
        .unique(subset=["item_id"])
    )
    df_price = lf_price.collect(streaming=True).to_pandas()
    df_out = df_out.merge(df_price, on="item_id", how="left")


    # 2) buy_segment (user)
    lf_buy = (
        _scan_feature_parquet(new_feature_dir, "customer_behavior.parquet")
        .select(["customer_id", "buy_segment"])
        .join(rank_users_df.lazy(), on="customer_id", how="inner")
        .unique(subset=["customer_id"])
    )
    df_buy = lf_buy.collect(streaming=True).to_pandas()
    df_out = df_out.merge(df_buy, on="customer_id", how="left")



    # 3) luxury_level (user)
    lf_lux = (
        _scan_feature_parquet(new_feature_dir, "customer_luxury.parquet")
        .select(["customer_id", "luxury_level"])
        .join(rank_users_df.lazy(), on="customer_id", how="inner")
        .unique(subset=["customer_id"])
    )
    df_lux = lf_lux.collect(streaming=True).to_pandas()
    df_out = df_out.merge(df_lux, on="customer_id", how="left")


    # 4) age_final (user)
    lf_age = (
        _scan_feature_parquet(new_feature_dir, "customer_age_features.parquet")
        .select(["customer_id", "age_final"])
        .join(rank_users_df.lazy(), on="customer_id", how="inner")
        .unique(subset=["customer_id"])
    )
    df_age = lf_age.collect(streaming=True).to_pandas()
    df_out = df_out.merge(df_age, on="customer_id", how="left")


    # 5) brand_segment (category_l1-level): columns = [category_l1, brand_segment]
    lf_bseg = (
        _scan_feature_parquet(new_feature_dir, "brand_segment.parquet")
        .select(["category_l1", "brand_segment"])
        .unique(subset=["category_l1"])
    )

    df_bseg = lf_bseg.collect(streaming=True).to_pandas()

    # merge by category_l1
    if "category_l1" in df_out.columns:
        df_out = df_out.merge(df_bseg, on="category_l1", how="left")
    else:
        df_out["brand_segment"] = None


    # 6) top10_by_cat (item popularity)
    lf_top10 = (
        _scan_feature_parquet(new_feature_dir, "top10_by_cat_month.parquet")
        .join(rank_items_df.lazy(), on="item_id", how="inner")
    )
    df_top10 = lf_top10.collect(streaming=True).to_pandas()

    if "item_id" in df_top10.columns:
        num_col = _auto_pick_numeric_col(
            df_top10,
            prefer=["top10_by_cat", "score", "count_norm", "qty_norm", "sold_norm", "popularity"],
            exclude=["item_id", "category_l1", "category_l2", "category_l3"],
        )
        if num_col is not None:
            tmp = df_top10[["item_id", num_col]].drop_duplicates("item_id").rename(columns={num_col: "top10_by_cat_score"})
            df_out = df_out.merge(tmp, on="item_id", how="left")
            df_out["is_top10_by_cat"] = df_out["top10_by_cat_score"].notna().astype(np.int8)
        else:
            df_out["top10_by_cat_score"] = np.nan
            df_out["is_top10_by_cat"] = 0
    else:
        df_out["top10_by_cat_score"] = np.nan
        df_out["is_top10_by_cat"] = 0

    # 7) top10_by_cat_month
    ref_month = ((int(valid_month) - int(top10_month_lag) - 1) % 12) + 1  # 1..12

    lf_top10m0 = _scan_feature_parquet(new_feature_dir, "top10_by_cat_month.parquet")

    # month_int: ưu tiên cast trực tiếp; nếu month dạng string khác (vd "2024-11") thì lấy 1-2 chữ số cuối
    lf_top10m0 = lf_top10m0.with_columns(
        pl.coalesce([
            pl.col("month").cast(pl.Int32, strict=False),
            pl.col("month").cast(pl.Utf8).str.extract(r"(\d{1,2})$").cast(pl.Int32, strict=False),
        ]).alias("month_int")
    )

    lf_top10m = (
        lf_top10m0
        .filter(pl.col("month_int") == ref_month)
        .join(rank_items_df.lazy(), on="item_id", how="inner")
    )

    df_top10m = lf_top10m.collect(streaming=True).to_pandas()


    if "item_id" in df_top10m.columns and "month" in df_top10m.columns:
        rank_col = "rank" if "rank" in df_top10m.columns else _auto_pick_numeric_col(
            df_top10m,
            prefer=["rank", "rnk", "order", "pop_rank"],
            exclude=["item_id", "month", "category_l1", "category_l2", "category_l3"],
        )
        if rank_col is not None:
            tmp = df_top10m[df_top10m["month"] == ref_month][["item_id", rank_col]].drop_duplicates("item_id").rename(columns={rank_col: "top10_rank_month"})
            df_out = df_out.merge(tmp, on="item_id", how="left")
            df_out["is_top10_by_cat_month"] = df_out["top10_rank_month"].notna().astype(np.int8)
            df_out["top10_rank_month"] = df_out["top10_rank_month"].fillna(999).astype(np.int16)
        else:
            df_out["is_top10_by_cat_month"] = 0
            df_out["top10_rank_month"] = 999
    else:
        df_out["is_top10_by_cat_month"] = 0
        df_out["top10_rank_month"] = 999

    # -------- dtypes & feature list --------
    if "price" in df_out.columns:
        df_out["price"] = pd.to_numeric(df_out["price"], errors="coerce").fillna(0.0)

    if "age_final" in df_out.columns:
        df_out["age_final"] = pd.to_numeric(df_out["age_final"], errors="coerce")

    df_out["stage1_rank"] = df_out["stage1_rank"].astype(np.int32)

    cat_cols = [
        "gender", "province", "membership",
        "category_l1", "category_l2", "brand", "age_group_final",
        "price_segment", "buy_segment", "luxury_level", "brand_segment",
    ]
    cat_cols = [c for c in cat_cols if c in df_out.columns]
    for c in cat_cols:
        df_out[c] = df_out[c].astype("category")

    feature_cols = [c for c in df_out.columns if c not in ["label", "customer_id", "item_id"]]
    return df_out, feature_cols, cat_cols


# ============================================================
# 6) Stage 2: LightGBM training
# ============================================================

@dataclass
class Stage2TrainResult:
    metrics: Dict[str, Any]


def train_stage2_lightgbm(
    df_rank: pd.DataFrame,
    feature_cols: List[str],
    cat_cols: List[str],
    num_boost_round: int = 200,
    valid_size: float = 0.2,
    random_state: int = 42,
    device_type: str = "cpu",  # cpu|gpu
    model_path: str = "./artifacts/lgb_stage2_ranking.txt",
    feat_path: str = "./artifacts/stage2_feature_cols.json",
) -> Stage2TrainResult:
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

    users = df_rank["customer_id"].unique()
    tr_users, va_users = train_test_split(users, test_size=valid_size, random_state=random_state)

    df_tr = df_rank[df_rank["customer_id"].isin(tr_users)].reset_index(drop=True)
    df_va = df_rank[df_rank["customer_id"].isin(va_users)].reset_index(drop=True)

    X_tr = df_tr[feature_cols]
    y_tr = df_tr["label"].astype(int)

    X_va = df_va[feature_cols]
    y_va = df_va["label"].astype(int)

    cat_names = [c for c in cat_cols if c in feature_cols]
    cat_idx = [feature_cols.index(c) for c in cat_names]

    dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_idx, free_raw_data=False)
    dvalid = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_idx, free_raw_data=False)

    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 50,
        "verbosity": -1,
        "device_type": device_type,
        "seed": random_state,
    }

    bst = lgb.train(
        params,
        dtrain,
        num_boost_round=int(num_boost_round),
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=[lgb.log_evaluation(period=50)],
    )

    bst.save_model(model_path, num_iteration=getattr(bst, "best_iteration", int(num_boost_round)))

    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump({"feature_cols": feature_cols, "cat_cols": cat_names}, f, ensure_ascii=False, indent=2)

    metrics = {
        "n_train_rows": int(len(df_tr)),
        "n_valid_rows": int(len(df_va)),
        "n_train_users": int(len(tr_users)),
        "n_valid_users": int(len(va_users)),
        "best_iteration": int(getattr(bst, "best_iteration", int(num_boost_round))),
    }
    return Stage2TrainResult(metrics=metrics)
