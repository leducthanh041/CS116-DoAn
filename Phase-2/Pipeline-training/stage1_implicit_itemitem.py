
"""
stage1_implicit_itemitem.py

Stage-1 candidate generator using implicit's item-item recommenders:
- implicit.nearest_neighbours.TFIDFRecommender
- implicit.nearest_neighbours.CosineRecommender

This module deliberately does NOT use sklearn NearestNeighbors.

Key configs:
- len_hist: number of days (history window) used to build user-item matrix
- len_val: number of days (validation window) used to compute Recall/Hit
- N_trend: number of trending items appended as backfill
- N_cand: number of candidates per implicit model (TFIDF + Cosine)
- n_iter: kept for pipeline compatibility but not used by these two models
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

from implicit.nearest_neighbours import TFIDFRecommender, CosineRecommender
from tqdm.auto import tqdm


# ---------------------------
# Utilities
# ---------------------------

def _ensure_datetime(lf: pl.LazyFrame, col: str = "created_date") -> pl.LazyFrame:
    """Ensure `col` exists and is castable to Polars Datetime.

    We intentionally keep this strict: Stage-1 needs a time column to build
    rolling train/valid windows. If the column is missing, we raise a helpful
    error that lists available columns.
    """
    if col not in lf.columns:
        cols_preview = ", ".join(lf.columns[:50])
        raise ValueError(
            f"LazyFrame must contain `{col}`. "
            f"Available columns (first 50): [{cols_preview}]. "
            f"Fix: pass the correct time column via `--created_col <colname>`."
        )
    return lf.with_columns(pl.col(col).cast(pl.Datetime, strict=False))


def split_train_valid_by_days(
    lf_all: pl.LazyFrame,
    created_col: str = "created_date",
    len_hist: int = 180,
    len_val: int = 30,
) -> Tuple[pl.LazyFrame, pl.LazyFrame, pl.Datetime, pl.Datetime, pl.Datetime]:
    """
    Split by day windows:
      ref_dt = max(created_datetime)
      valid: [ref_dt - len_val days, ref_dt]
      train: [valid_start - len_hist days, valid_start)

    Returns:
      lf_train, lf_valid, train_start, valid_start, ref_dt
    """
    lf_all = _ensure_datetime(lf_all, created_col)

    # compute ref_dt (max)
    ref_dt = lf_all.select(pl.col(created_col).max()).collect()[0, 0]
    if ref_dt is None:
        raise ValueError(f"`{created_col}` is empty / cannot compute max timestamp")

    # Polars datetime arithmetic uses duration
    valid_start = ref_dt - pl.duration(days=len_val)
    train_start = valid_start - pl.duration(days=len_hist)

    lf_train = lf_all.filter((pl.col(created_col) >= train_start) & (pl.col(created_col) < valid_start))
    lf_valid = lf_all.filter((pl.col(created_col) >= valid_start) & (pl.col(created_col) <= ref_dt))

    return lf_train, lf_valid, train_start, valid_start, ref_dt


def build_user_items_csr(
    lf: pl.LazyFrame,
    user_col: str = "customer_id",
    item_col: str = "item_id",
    qty_col: str = "quantity",
    price_col: str = "price",
    weight_type: str = "log_count",  # "binary" | "count" | "log_count" | "log_qty" | "log_spent"
) -> Tuple[csr_matrix, List, List, Dict, Dict]:
    """
    Build CSR matrix user_items of shape (n_users, n_items).

    weight_type meanings:
      - binary: 1 per (u,i)
      - count: transaction count per (u,i)
      - log_count: log1p(count)
      - log_qty: log1p(sum(quantity)) per (u,i) (quantity missing -> 1)
      - log_spent: log1p(sum(price*quantity)) (price missing -> 0)
    """
    cols = [user_col, item_col]
    if qty_col in lf.columns:
        cols.append(qty_col)
    if price_col in lf.columns:
        cols.append(price_col)

    df = lf.select(cols).collect()

    # enforce types
    df = df.with_columns(
        [
            pl.col(user_col).cast(pl.Utf8),
            pl.col(item_col).cast(pl.Utf8),
        ]
    )

    if qty_col in df.columns:
        df = df.with_columns(pl.col(qty_col).cast(pl.Float64, strict=False).fill_null(1.0).fill_nan(1.0))
    else:
        df = df.with_columns(pl.lit(1.0).alias(qty_col))

    if price_col in df.columns:
        df = df.with_columns(pl.col(price_col).cast(pl.Float64, strict=False).fill_null(0.0).fill_nan(0.0))
    else:
        df = df.with_columns(pl.lit(0.0).alias(price_col))

    df = df.with_columns((pl.col(price_col) * pl.col(qty_col)).alias("spent_row"))

    # aggregate (u,i)
    ui = (
        df.group_by([user_col, item_col])
          .agg(
              pl.len().alias("cnt"),
              pl.col(qty_col).sum().alias("sum_qty"),
              pl.col("spent_row").sum().alias("sum_spent"),
          )
    )

    # choose weight
    if weight_type == "binary":
        ui = ui.with_columns(pl.lit(1.0).alias("value"))
    elif weight_type == "count":
        ui = ui.with_columns(pl.col("cnt").cast(pl.Float64).alias("value"))
    elif weight_type == "log_count":
        ui = ui.with_columns(pl.col("cnt").cast(pl.Float64).log1p().alias("value"))
    elif weight_type == "log_qty":
        ui = ui.with_columns(pl.col("sum_qty").cast(pl.Float64).log1p().alias("value"))
    elif weight_type == "log_spent":
        ui = ui.with_columns(pl.col("sum_spent").cast(pl.Float64).log1p().alias("value"))
    else:
        raise ValueError(f"Unknown weight_type: {weight_type}")

    # encode to contiguous ids
    users = ui.select(user_col).unique().sort(user_col).to_series().to_list()
    items = ui.select(item_col).unique().sort(item_col).to_series().to_list()

    user_id_to_index = {u: i for i, u in enumerate(users)}
    item_id_to_index = {it: j for j, it in enumerate(items)}

    u_idx = ui.select(pl.col(user_col)).to_series().to_list()
    i_idx = ui.select(pl.col(item_col)).to_series().to_list()
    data = ui.select(pl.col("value")).to_series().to_numpy().astype(np.float32)

    row = np.fromiter((user_id_to_index[u] for u in u_idx), dtype=np.int32, count=len(u_idx))
    col = np.fromiter((item_id_to_index[it] for it in i_idx), dtype=np.int32, count=len(i_idx))

    mat = csr_matrix((data, (row, col)), shape=(len(users), len(items)), dtype=np.float32)

    return mat, users, items, user_id_to_index, item_id_to_index


def top_trending_items(user_items: csr_matrix, top_n: int) -> np.ndarray:
    if top_n <= 0:
        return np.array([], dtype=np.int32)
    pop = np.asarray(user_items.sum(axis=0)).ravel()
    return np.argsort(-pop)[:top_n].astype(np.int32)


# ---------------------------
# Stage-1 Model
# ---------------------------

@dataclass
class Stage1Metrics:
    recall: float
    hit: float
    n_users_eval: int


class Stage1ImplicitItemItem:
    """
    Stage-1 candidate generator (implicit TFIDF + implicit Cosine + trending backfill)

    Notes:
      - This uses implicit's RecommenderBase API: fit(user_items) and recommend(userid, user_items_row, ...).
      - 'n_iter' is ignored (kept only for config compatibility).
    """

    def __init__(
        self,
        N_cand: int = 300,
        N_trend: int = 100,
        num_threads: int = 0,
        K_model: Optional[int] = None,
        n_iter: int = 1,  # ignored
        filter_already_liked_items: bool = True,
        use_tqdm: bool = True,
    ):
        self.N_cand = int(N_cand)
        self.N_trend = int(N_trend)
        self.num_threads = int(num_threads)
        self.K_model = int(K_model) if K_model is not None else None
        self.n_iter = int(n_iter)
        self.filter_already_liked_items = bool(filter_already_liked_items)
        self.use_tqdm = bool(use_tqdm)

        # learned
        self.user_items_: Optional[csr_matrix] = None
        self.users_: Optional[List[str]] = None
        self.items_: Optional[List[str]] = None
        self.user_id_to_index_: Optional[Dict[str, int]] = None
        self.item_id_to_index_: Optional[Dict[str, int]] = None
        self.trending_item_idx_: Optional[np.ndarray] = None

        self.model_tfidf_: Optional[TFIDFRecommender] = None
        self.model_cosine_: Optional[CosineRecommender] = None

    def fit_from_csr(self, user_items: csr_matrix, users: List[str], items: List[str]):
        self.user_items_ = user_items
        self.users_ = users
        self.items_ = items
        self.user_id_to_index_ = {u: i for i, u in enumerate(users)}
        self.item_id_to_index_ = {it: j for j, it in enumerate(items)}

        # choose K_model automatically if not set
        # - implicit models use K (number of neighbours stored in item-item similarity matrix)
        # - we set it >= N_cand to avoid truncation in practice
        K = self.K_model if self.K_model is not None else max(20, min(500, 3 * self.N_cand))

        self.model_tfidf_ = TFIDFRecommender(K=K, num_threads=self.num_threads)
        self.model_cosine_ = CosineRecommender(K=K, num_threads=self.num_threads)

        # fit expects CSR (users x items)
        self.model_tfidf_.fit(user_items, show_progress=True)
        self.model_cosine_.fit(user_items, show_progress=True)

        self.trending_item_idx_ = top_trending_items(user_items, self.N_trend)
        return self

    def recommend_for_user_index(
        self,
        uidx: int,
        top_k: Optional[int] = None,
        filter_already_liked_items: Optional[bool] = None,
    ) -> List[int]:
        """Recommend item indices for a known user index.

        Parameters
        ----------
        uidx:
            Internal user index.
        top_k:
            Number of items to return (after union + dedup).
        filter_already_liked_items:
            Override for whether to filter items already present in the user's
            history (training interactions). If None, defaults to
            `self.filter_already_liked_items`.

        Important
        ---------
        During evaluation, `filter_train_history` and recommendation-time
        filtering must be consistent:
          - If evaluating UNFILTERED recall (includes repeats), do NOT filter
            already-liked items in `recommend()`.
          - If evaluating FILTERED recall (new-item recall), DO filter already-
            liked items to avoid displacing novel candidates.
        """

        if top_k is None:
            top_k = 2 * self.N_cand + self.N_trend

        if filter_already_liked_items is None:
            filter_already_liked_items = self.filter_already_liked_items

        # cold-start: no interactions row
        if self.user_items_ is None:
            raise RuntimeError("Model not fit.")
        row = self.user_items_[uidx]
        if row.nnz == 0:
            return list(self.trending_item_idx_[:top_k])

        # collect from both models
        cand = []
        # TFIDF
        ids_t, _ = self.model_tfidf_.recommend(
            uidx, row, N=self.N_cand, filter_already_liked_items=filter_already_liked_items
        )
        cand.extend(ids_t.tolist() if hasattr(ids_t, "tolist") else list(ids_t))
        # Cosine
        ids_c, _ = self.model_cosine_.recommend(
            uidx, row, N=self.N_cand, filter_already_liked_items=filter_already_liked_items
        )
        cand.extend(ids_c.tolist() if hasattr(ids_c, "tolist") else list(ids_c))

        # trending backfill
        if self.trending_item_idx_ is not None and self.N_trend > 0:
            cand.extend(self.trending_item_idx_.tolist())

        # dedup while preserving order
        seen = set()
        out = []
        for x in cand:
            if x not in seen:
                out.append(int(x))
                seen.add(int(x))
            if len(out) >= top_k:
                break

        return out

    def recommend_for_user_id(self, user_id: str, top_k: Optional[int] = None) -> List[str]:
        if self.user_id_to_index_ is None:
            raise RuntimeError("Model not fit.")
        if user_id not in self.user_id_to_index_:
            # unknown user -> trending
            idxs = list(self.trending_item_idx_[: (top_k or (2 * self.N_cand + self.N_trend))])
            return [self.items_[i] for i in idxs]
        uidx = self.user_id_to_index_[user_id]
        rec_idx = self.recommend_for_user_index(uidx, top_k=top_k)
        return [self.items_[i] for i in rec_idx]

    def eval_recall_hit(
        self,
        lf_valid: pl.LazyFrame,
        user_col: str = "customer_id",
        item_col: str = "item_id",
        filter_train_history: bool = True,
        k: Optional[int] = None,
    ) -> Stage1Metrics:
        if self.user_items_ is None or self.user_id_to_index_ is None or self.item_id_to_index_ is None:
            raise RuntimeError("Model not fit.")

        if k is None:
            k = 2 * self.N_cand + self.N_trend

        dfv = lf_valid.select([user_col, item_col]).collect()
        if dfv.height == 0:
            return Stage1Metrics(recall=0.0, hit=0.0, n_users_eval=0)

        # map items to indices; keep only known items
        dfv = dfv.with_columns(
            [
                pl.col(user_col).cast(pl.Utf8),
                pl.col(item_col).cast(pl.Utf8),
            ]
        )

        # build gt dict in python (OK for eval size; for huge, can optimize later)
        user_to_items: Dict[str, set] = {}
        for u, it in zip(dfv[user_col].to_list(), dfv[item_col].to_list()):
            if it not in self.item_id_to_index_:
                continue
            user_to_items.setdefault(u, set()).add(self.item_id_to_index_[it])

        recalls = []
        hits = []

        iterator = user_to_items.items()
        if self.use_tqdm:
            iterator = tqdm(
                iterator,
                total=len(user_to_items),
                desc=f"Eval Stage1 Recall/Hit @K={k} (filter_train_history={filter_train_history})",
                leave=False,
            )

        for u, gt in iterator:
            if u not in self.user_id_to_index_:
                continue
            uidx = self.user_id_to_index_[u]

            if filter_train_history:
                train_hist = set(self.user_items_[uidx].indices)
                relevant = gt - train_hist
                # For FILTERED (new-item) eval, enforce recommendation-time
                # filtering so repeats do not displace novel candidates.
                rec_filter = True
            else:
                relevant = gt
                # For UNFILTERED eval, do NOT filter already-liked items,
                # otherwise repeats in `relevant` are impossible to retrieve.
                rec_filter = False

            if not relevant:
                continue

            rec = self.recommend_for_user_index(uidx, top_k=k, filter_already_liked_items=rec_filter)
            inter = set(rec) & relevant
            recalls.append(len(inter) / len(relevant))
            hits.append(1.0 if inter else 0.0)

        if not recalls:
            return Stage1Metrics(recall=0.0, hit=0.0, n_users_eval=0)

        return Stage1Metrics(recall=float(np.mean(recalls)), hit=float(np.mean(hits)), n_users_eval=len(recalls))


def load_stage1_from_artifacts(
    meta_npz_path: str,
    user_items_npz_path: str,
    tfidf_npz_path: str,
    cosine_npz_path: str,
) -> "Stage1ImplicitItemItem":
    """
    Load a trained Stage-1 model from artifacts created by train_stage1_implicit_itemitem.py
    """
    import json
    from scipy.sparse import load_npz
    from implicit.nearest_neighbours import TFIDFRecommender, CosineRecommender

    meta = np.load(meta_npz_path, allow_pickle=True)
    users = meta["user_ids"].tolist()
    items = meta["item_ids"].tolist()
    trending_idx = meta["trending_item_idx"].astype(np.int32)
    config = json.loads(str(meta["config"]))

    model = Stage1ImplicitItemItem(
        N_cand=int(config.get("N_cand", 300)),
        N_trend=int(config.get("N_trend", 100)),
        num_threads=int(config.get("num_threads", 0)),
        K_model=config.get("K_model", None),
        n_iter=int(config.get("n_iter", 1)),
        filter_already_liked_items=(not bool(config.get("allow_repeat", False))),
    )

    model.user_items_ = load_npz(user_items_npz_path).tocsr()
    model.users_ = users
    model.items_ = items
    model.user_id_to_index_ = {u: i for i, u in enumerate(users)}
    model.item_id_to_index_ = {it: j for j, it in enumerate(items)}
    model.trending_item_idx_ = trending_idx

    model.model_tfidf_ = TFIDFRecommender.load(tfidf_npz_path)
    model.model_cosine_ = CosineRecommender.load(cosine_npz_path)

    return model
