# -*- coding: utf-8 -*-
"""
stage2_train_and_testflow_v3.py

Stage-2 Ranking (LightGBM LambdaRank) + Testflow (Stage1 -> Stage2 -> Metrics)

Key changes per user request:
  - build_feature_label() no longer loads/merges Stage-1 candidates parquet.
  - build_feature_label() builds (customer_id, item_id, features, Y) directly from transactions:
      base pairs = unique (customer_id,item_id) in HIST union RECENT
      label Y=1 if pair appears in RECENT else 0
  - train_pl is built by negative sampling from fl
  - after training, persist feature_columns.json (used later by predict_stage2)
  - ranking/testflow uses Stage-1 model_final to generate candidates for users in groundtruth.pkl,
    then rebuilds Stage-2 features for those (user,item) pairs using the SAME feature pipeline,
    then predict_stage2 -> calculate_metrics_at_k -> save submission.

Run:
  python stage2_train_and_testflow_v3.py --config stage2_full_config_v1.json
"""

from __future__ import annotations

import os
import json
import time
import pickle
from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from tqdm.auto import tqdm
from scipy.sparse import csr_matrix

from implicit.nearest_neighbours import TFIDFRecommender, CosineRecommender


# -------------------------
# Logging
# -------------------------
def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _glob_scan(glob_path: str) -> pl.LazyFrame:
    return pl.scan_parquet(glob_path)


# -------------------------
# Config
# -------------------------
@dataclass
class Stage2Config:
    # paths
    transactions_path_glob: str
    items_path_glob: str
    artifacts_dir: str = "./artifacts_stage2"
    run_name: str = "run_stage2"
    max_rows_feature_label: int = 0
    popular_pool_size: int = 50000
    stage1_max_rows_tx: int = 0  # cap rows when rebuilding stage1 CSR (smoke)
    pred_batch_rows: int = 200000  # batch rows when predicting stage2 (avoid to_pandas on huge df)
    # windows for feature/label building
    begin_hist: str = "2024-08-01"
    end_hist: str = "2024-11-30"
    begin_recent: str = "2024-12-01"
    end_recent: str = "2024-12-31"

    smoke: bool = False
    max_pos_pairs: int = 0
    restrict_hist_to_pairs_users: bool = False   # optional, ưu tiên nếu muốn bật/tắt độc lập với smoke
    max_pairs_for_features: int = 0 

    # feature sources (optional)
    brand_segment_path: Optional[str] = None
    customer_age_features_path: Optional[str] = None
    customer_behavior_path: Optional[str] = None
    customer_luxury_path: Optional[str] = None
    price_segment_path: Optional[str] = None
    top10_by_cat_month_path: Optional[str] = None

    # feature knobs
    enable_cooc: bool = False
    cooc_same_day: bool = True
    cooc_min_count: int = 600

    max_rows_lgbm_train: int = 0
    max_rows_lgbm_valid: int = 0


    # negative sampling
    N_neg: int = 10
    keep_users_without_pos: bool = False
    random_state: int = 42

    max_rows_tx_hist: int = 2000
    max_rows_tx_recent: int = 1000
    max_users_train: int = 500
    # LGBM params
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_data_in_leaf: int = 200
    n_estimators: int = 200

    # Stage1 -> Stage2 testflow
    stage1_model_dir: Optional[str] = None  # path to Stage1 model_final/
    gt_path: Optional[str] = None           # groundtruth.pkl (Jan/2025)
    stage1_top_k: int = 200
    allow_repeat: bool = False
    metric_k: int = 50
    filter_bought_eval: bool = True
    batch_users: int = 2000
    submission_out: str = "./artifacts_stage2/submission_jan2025.csv"

    # IO / speed
    max_rows_tx: int = 0   # 0 => no cap
    quick: bool = False


def load_config(path: str) -> Stage2Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Filter unknown keys to avoid TypeError when configs contain legacy fields.
    allowed = {f.name for f in fields(Stage2Config)}
    unknown = sorted([k for k in raw.keys() if k not in allowed])
    if unknown:
        log(f"[WARN] Ignoring unknown config keys: {unknown}")
        raw = {k: v for k, v in raw.items() if k in allowed}

    return Stage2Config(**raw)


# -------------------------
# Groundtruth loader (pickle/json)
# -------------------------
def load_groundtruth_pkl(path: str) -> Dict[str, Set[str]]:
    # supports pickle dict[user->items] OR list of dict rows OR dict-of-lists OR parquet not needed here
    with open(path, "rb") as f:
        obj = pickle.load(f)

    gt: Dict[str, Set[str]] = {}

    if isinstance(obj, dict) and ("customer_id" not in obj) and ("item_id" not in obj):
        # dict[user -> list/set]
        for u, items in obj.items():
            if items is None:
                continue
            if isinstance(items, (list, set, tuple)):
                gt[str(u)] = set(str(x) for x in items)
            else:
                gt.setdefault(str(u), set()).add(str(items))
        return gt

    if isinstance(obj, list):
        # list of rows
        for row in obj:
            if not isinstance(row, dict):
                continue
            u = row.get("customer_id")
            it = row.get("item_id")
            if u is None or it is None:
                continue
            gt.setdefault(str(u), set()).add(str(it))
        return gt

    if isinstance(obj, dict) and ("customer_id" in obj) and ("item_id" in obj):
        cu = obj["customer_id"]
        ci = obj["item_id"]
        for u, it in zip(cu, ci):
            if u is None or it is None:
                continue
            gt.setdefault(str(u), set()).add(str(it))
        return gt

    raise ValueError("Unsupported groundtruth.pkl structure.")


# -------------------------
# Stage-1 loader from model_final (TFIDF + Cosine)
# -------------------------
class Stage1FromModelFinal:
    """
    Minimal Stage1 retriever that:
      - loads TFIDF/Cosine item-item models and mappings from model_final/
      - rebuilds user-item CSR + user_history_ from transactions in Stage1 bundle_meta train window
      - recommends candidates for given users
      - exposes get_user_history_dict() per your function (string IDs)
    """

    def __init__(self, model_dir: str, transactions_path_glob: str, max_rows_tx : int = 0,num_threads: int = 0):
        self.model_dir = model_dir
        self.transactions_path_glob = transactions_path_glob
        self.max_rows_tx = int(max_rows_tx)
        self.num_threads = int(num_threads)

        meta_path = os.path.join(model_dir, "bundle_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing bundle_meta.json in {model_dir}")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.train_begin = _parse_date(meta["train_begin"])
        self.train_end = _parse_date(meta["train_end"])

        # load mappings
        with open(os.path.join(model_dir, "users_item.json"), "r", encoding="utf-8") as f:
            self.user_ids_ = [str(x) for x in json.load(f)]
        with open(os.path.join(model_dir, "items.json"), "r", encoding="utf-8") as f:
            self.item_ids_ = [str(x) for x in json.load(f)]
        self.user_id_to_index_ = {u: i for i, u in enumerate(self.user_ids_)}
        self.user_index_to_id_ = {i: u for u, i in self.user_id_to_index_.items()}
        self.index_to_item_id_ = {i: it for i, it in enumerate(self.item_ids_)}

        # load models
        self.tfidf = TFIDFRecommender.load(os.path.join(model_dir, "item_tfidf.npz"))
        self.cosine = CosineRecommender.load(os.path.join(model_dir, "item_cosine.npz"))
        self.trending_idx = np.load(os.path.join(model_dir, "trending_idx.npy"))

        self.user_items_: Optional[csr_matrix] = None
        self.user_history_: Dict[int, List[int]] = {}
        self.top_k: int = 200

        self._build_user_items_and_history()

    def _build_user_items_and_history(self) -> None:
        log("[Stage1] Rebuild user-item CSR + history from transactions ...")
        tx = _glob_scan(self.transactions_path_glob)

        cols = tx.collect_schema().names()
        need = {"customer_id", "item_id", "created_date"}
        missing = [c for c in need if c not in cols]
        if missing:
            raise ValueError(f"Transactions missing required columns: {missing}")

        has_qty = "quantity" in cols
        tx = tx.with_columns([
            pl.col("customer_id").cast(pl.Utf8),
            pl.col("item_id").cast(pl.Utf8),
            pl.col("created_date").cast(pl.Date, strict=False),
            (pl.col("quantity").cast(pl.Float64, strict=False).fill_null(1.0).fill_nan(1.0) if has_qty else pl.lit(1.0).alias("quantity")),
        ]).select(["customer_id", "item_id", "created_date", "quantity"])

        tx = tx.filter(
            pl.col("created_date").is_between(
                pl.lit(self.train_begin.date(), dtype=pl.Date),
                pl.lit(self.train_end.date(), dtype=pl.Date),
                closed="both",
            )
        )

        df = tx.collect()
        if self.max_rows_tx and self.max_rows_tx > 0:
            tx = tx.head(int(self.max_rows_tx))

        df = tx.collect()

        # keep only known users/items (consistent with Stage1 mappings)
        df = df.filter(pl.col("customer_id").is_in(self.user_ids_))
        df = df.filter(pl.col("item_id").is_in(self.item_ids_))

        if df.height == 0:
            # empty CSR
            self.user_items_ = csr_matrix((len(self.user_ids_), len(self.item_ids_)), dtype=np.float32)
            self.user_history_ = {}
            return

        # aggregate counts
        ui = df.group_by(["customer_id", "item_id"]).agg(pl.col("quantity").sum().alias("v"))
        u_idx = ui["customer_id"].to_list()
        it_idx = ui["item_id"].to_list()
        data = ui["v"].to_numpy().astype(np.float32)

        row = np.fromiter((self.user_id_to_index_[str(u)] for u in u_idx), dtype=np.int32, count=len(u_idx))
        col = np.fromiter((self.item_ids_.index(str(it)) for it in it_idx), dtype=np.int32, count=len(it_idx))
        # NOTE: item_ids_.index is O(n). Build map:
        it_map = {it: j for j, it in enumerate(self.item_ids_)}
        col = np.fromiter((it_map[str(it)] for it in it_idx), dtype=np.int32, count=len(it_idx))

        self.user_items_ = csr_matrix((data, (row, col)), shape=(len(self.user_ids_), len(self.item_ids_)), dtype=np.float32)

        # history
        hist = {}
        for r, c in zip(row.tolist(), col.tolist()):
            hist.setdefault(int(r), []).append(int(c))
        self.user_history_ = hist

    def recommend_candidates(self, users: List[str], allow_repeat: bool = False) -> pd.DataFrame:
        if self.user_items_ is None:
            raise RuntimeError("Stage1 user_items_ not built.")

        rows = []
        top_k = int(self.top_k)

        for u in users:
            u = str(u)
            if u not in self.user_id_to_index_:
                # cold user -> trending
                cand_idx = self.trending_idx[:top_k].tolist()
                for it_idx in cand_idx:
                    rows.append((u, self.index_to_item_id_[int(it_idx)]))
                continue

            uidx = self.user_id_to_index_[u]
            urow = self.user_items_[uidx]
            if urow.nnz == 0:
                cand_idx = self.trending_idx[:top_k].tolist()
                for it_idx in cand_idx:
                    rows.append((u, self.index_to_item_id_[int(it_idx)]))
                continue

            ids_t, _ = self.tfidf.recommend(uidx, urow, N=top_k, filter_already_liked_items=not allow_repeat)
            ids_c, _ = self.cosine.recommend(uidx, urow, N=top_k, filter_already_liked_items=not allow_repeat)

            # merge + dedup, keep order
            seen = set()
            merged = []
            for arr in (ids_t, ids_c):
                for ix in (arr.tolist() if hasattr(arr, "tolist") else list(arr)):
                    ix = int(ix)
                    if ix not in seen:
                        seen.add(ix)
                        merged.append(ix)
                    if len(merged) >= top_k:
                        break
                if len(merged) >= top_k:
                    break

            # backfill with trending
            if len(merged) < top_k:
                for ix in self.trending_idx.tolist():
                    ix = int(ix)
                    if ix not in seen:
                        seen.add(ix)
                        merged.append(ix)
                    if len(merged) >= top_k:
                        break

            for ix in merged[:top_k]:
                rows.append((u, self.index_to_item_id_[int(ix)]))

        return pd.DataFrame(rows, columns=["customer_id", "item_id"])

    def get_user_history_dict(self) -> Dict[str, Set[str]]:
        history_dict: Dict[str, Set[str]] = {}
        for u_idx, item_indices in self.user_history_.items():
            user_str = str(self.user_index_to_id_[u_idx])
            items_str: Set[str] = set()
            for i_idx in item_indices:
                if i_idx in self.index_to_item_id_:
                    items_str.add(str(self.index_to_item_id_[i_idx]))
            history_dict[user_str] = items_str
        return history_dict


# -------------------------
# Stage-2 feature engineering (NO candidates parquet)
# -------------------------
def _load_tx_min(cfg: Stage2Config, begin_all: datetime, end_all: datetime) -> pl.DataFrame:
    tx = _glob_scan(cfg.transactions_path_glob)
    cols = tx.collect_schema().names()

    need = {"customer_id", "item_id", "created_date"}
    missing = [c for c in need if c not in cols]
    if missing:
        raise ValueError(f"Transactions missing required columns: {missing}")

    has_qty = "quantity" in cols

    tx = tx.with_columns([
        pl.col("customer_id").cast(pl.Int64, strict=False),
        pl.col("item_id").cast(pl.Utf8),
        pl.col("created_date").cast(pl.Date, strict=False),
        (pl.col("quantity").cast(pl.Float64, strict=False).fill_null(1.0).fill_nan(1.0) if has_qty else pl.lit(1.0).alias("quantity")),
    ]).select(["customer_id", "item_id", "created_date", "quantity"])

    tx = tx.filter(
        pl.col("created_date").is_between(
            pl.lit(begin_all.date(), dtype=pl.Date),
            pl.lit(end_all.date(), dtype=pl.Date),
            closed="both",
        )
    )

    df = tx.collect()
    if cfg.max_rows_tx and int(cfg.max_rows_tx) > 0:
        df = df.head(int(cfg.max_rows_tx))
    return df


def _load_items_min(cfg: Stage2Config) -> pl.DataFrame:
    it = _glob_scan(cfg.items_path_glob)
    cols = it.collect_schema().names()
    if "item_id" not in cols:
        raise ValueError("items table missing item_id")

    # normalize category column
    has_cat = "category" in cols
    has_cat_l2 = "category_l2" in cols
    category_expr = pl.col("category").cast(pl.Utf8) if has_cat else (pl.col("category_l2").cast(pl.Utf8) if has_cat_l2 else pl.lit(None, dtype=pl.Utf8).alias("category"))

    it2 = it.select([
        pl.col("item_id").cast(pl.Utf8),
        (pl.col("brand_final").cast(pl.Utf8) if "brand_final" in cols else pl.lit(None, dtype=pl.Utf8).alias("brand_final")),
        (pl.col("age_group_final").cast(pl.Utf8) if "age_group_final" in cols else pl.lit(None, dtype=pl.Utf8).alias("age_group_final")),
        category_expr.alias("category"),
        (pl.col("category_l1").cast(pl.Utf8) if "category_l1" in cols else pl.lit(None, dtype=pl.Utf8).alias("category_l1")),
    ]).unique()

    return it2.collect()


def _load_optional_lookup(path: str | None, required_cols: list[str]) -> pl.DataFrame | None:
    if not path:
        return None
    if not os.path.exists(path):
        return None

    lf = pl.scan_parquet(path).select(required_cols)
    # collect streaming để giảm peak memory
    return lf.collect(engine="streaming")


def build_pairs_and_label(tx_min: pl.DataFrame, begin_hist: datetime, end_hist: datetime, begin_recent: datetime, end_recent: datetime) -> pl.DataFrame:
    # hist pairs
    hist_pairs = tx_min.filter(
        pl.col("created_date").is_between(pl.lit(begin_hist.date(), dtype=pl.Date), pl.lit(end_hist.date(), dtype=pl.Date), closed="both")
    ).select(["customer_id", "item_id"]).unique()

    # recent pairs (positives)
    recent_pairs = tx_min.filter(
        pl.col("created_date").is_between(pl.lit(begin_recent.date(), dtype=pl.Date), pl.lit(end_recent.date(), dtype=pl.Date), closed="both")
    ).select(["customer_id", "item_id"]).unique()

    base = pl.concat([hist_pairs, recent_pairs]).unique()

    # label Y
    recent_key = recent_pairs.with_columns(pl.lit(1).alias("Y"))
    fl = base.join(recent_key, on=["customer_id", "item_id"], how="left").with_columns(
        pl.col("Y").fill_null(0).cast(pl.Int8)
    )
    return fl


def build_features_for_pairs(
    cfg: Stage2Config,
    pairs: pl.DataFrame,            # must have customer_id + item_id (+ optional Y)
    tx_min: pl.DataFrame,
    items_df: pl.DataFrame,
    begin_hist: datetime,
    end_hist: datetime,
) -> pl.DataFrame:
    """
    Build Stage-2 features for (customer_id,item_id) pairs.
    Optimizations:
      - Optional restrict HIST aggregation to users in pairs (SMOKE only by config)
      - Select only required columns from items/lookups
      - Minimize columns early, avoid wide joins, avoid accidental rank overwrite
    """

    # -------------------------
    # 0) Normalize keys + keep minimal cols
    # -------------------------
    pairs2 = pairs.with_columns([
        pl.col("customer_id").cast(pl.Utf8),
        pl.col("item_id").cast(pl.Utf8),
    ])

    keep_cols = ["customer_id", "item_id"] + (["Y"] if "Y" in pairs2.columns else [])
    pairs2 = pairs2.select(keep_cols)

    # Optional: cap pairs for smoke speed
    max_pairs = int(getattr(cfg, "max_pairs_for_features", 0) or 0)
    if max_pairs > 0 and pairs2.height > max_pairs:
        # keep distribution across users
        n_users = pairs2.select("customer_id").n_unique()
        head_per_user = max(1, max_pairs // max(1, n_users))
        pairs2 = (
            pairs2.sort("customer_id")
                  .group_by("customer_id", maintain_order=True)
                  .head(head_per_user)
                  .head(max_pairs)
        )

    # -------------------------
    # 1) items_min: only needed columns
    # -------------------------
    item_cols_needed = ["item_id", "brand_final", "age_group_final", "category", "category_l1"]
    item_cols_exist = [c for c in item_cols_needed if c in items_df.columns]

    # Always require item_id
    if "item_id" not in item_cols_exist:
        item_cols_exist = ["item_id"]

    items_min = (
        items_df.select(item_cols_exist)
                .with_columns(pl.col("item_id").cast(pl.Utf8))
                .unique("item_id")
    )

    # Join item attrs to pairs
    fl = pairs2.join(items_min, on="item_id", how="left")

    # -------------------------
    # 2) HIST window
    # -------------------------
    hist = (
        tx_min.filter(
            pl.col("created_date").is_between(
                pl.lit(begin_hist.date(), dtype=pl.Date),
                pl.lit(end_hist.date(), dtype=pl.Date),
                closed="both",
            )
        )
        .select(["customer_id", "item_id", "quantity", "created_date"])  # keep minimal
        .with_columns([
            pl.col("customer_id").cast(pl.Utf8),
            pl.col("item_id").cast(pl.Utf8),
            pl.col("quantity").cast(pl.Float64, strict=False).fill_null(0.0),
        ])
    )

    # Restrict HIST to users_in_pairs ONLY when smoke or explicit flag is True
    restrict_hist = bool(getattr(cfg, "restrict_hist_to_pairs_users", False)) or bool(getattr(cfg, "smoke", False))
    if restrict_hist:
        users_in_pairs = pairs2.select("customer_id").unique()
        hist = hist.join(users_in_pairs, on="customer_id", how="inner")

    # Join only needed item columns for count features
    join_cols = ["item_id"]
    for c in ["brand_final", "age_group_final", "category"]:
        if c in items_min.columns:
            join_cols.append(c)

    hist = hist.join(items_min.select(join_cols), on="item_id", how="left")

    # -------------------------
    # 3) Preference count features
    # -------------------------
    # Defaults to 0 if missing
    if "brand_final" in hist.columns and "brand_final" in fl.columns:
        brand_counts = (
            hist.filter(pl.col("brand_final").is_not_null())
                .group_by(["customer_id", "brand_final"])
                .agg(pl.col("quantity").sum().alias("brand_count"))
        )
        fl = fl.join(brand_counts, on=["customer_id", "brand_final"], how="left")
    else:
        fl = fl.with_columns(pl.lit(0.0).alias("brand_count"))

    if "age_group_final" in hist.columns and "age_group_final" in fl.columns:
        age_counts = (
            hist.filter(pl.col("age_group_final").is_not_null())
                .group_by(["customer_id", "age_group_final"])
                .agg(pl.col("quantity").sum().alias("age_count"))
        )
        fl = fl.join(age_counts, on=["customer_id", "age_group_final"], how="left")
    else:
        fl = fl.with_columns(pl.lit(0.0).alias("age_count"))

    if "category" in hist.columns and "category" in fl.columns:
        cat_counts = (
            hist.filter(pl.col("category").is_not_null())
                .group_by(["customer_id", "category"])
                .agg(pl.col("quantity").sum().alias("category_count"))
        )
        fl = fl.join(cat_counts, on=["customer_id", "category"], how="left")
    else:
        fl = fl.with_columns(pl.lit(0.0).alias("category_count"))

    fl = fl.with_columns([
        pl.col("brand_count").fill_null(0.0).cast(pl.Float64),
        pl.col("age_count").fill_null(0.0).cast(pl.Float64),
        pl.col("category_count").fill_null(0.0).cast(pl.Float64),
    ])

    # -------------------------
    # 4) Optional lookups (load ONLY required columns)
    # -------------------------

    # 1) brand_segment: category_l1, brand_segment (i64)
    if getattr(cfg, "brand_segment_path", None) and "category_l1" in fl.columns:
        brand_seg = _load_optional_lookup(cfg.brand_segment_path, required_cols=["category_l1", "brand_segment"])
        if brand_seg is not None:
            brand_seg = brand_seg.select(["category_l1", "brand_segment"]).with_columns(pl.col("category_l1").cast(pl.Utf8))
            fl = fl.join(brand_seg, on="category_l1", how="left")

    # 2) customer_age_features: customer_id, age_final (f64)
    if getattr(cfg, "customer_age_features_path", None):
        caf = _load_optional_lookup(cfg.customer_age_features_path, required_cols=["customer_id", "age_final"])
        if caf is not None:
            caf = caf.select(["customer_id", "age_final"]).with_columns(pl.col("customer_id").cast(pl.Utf8))
            fl = fl.join(caf, on="customer_id", how="left")

    # 3) customer_behavior: customer_id, buy_segment (i32)
    if getattr(cfg, "customer_behavior_path", None):
        cb = _load_optional_lookup(cfg.customer_behavior_path, required_cols=["customer_id", "buy_segment"])
        if cb is not None:
            cb = cb.select(["customer_id", "buy_segment"]).with_columns(pl.col("customer_id").cast(pl.Utf8))
            fl = fl.join(cb, on="customer_id", how="left")

    # 4) customer_luxury: customer_id, luxury_level (i32)
    if getattr(cfg, "customer_luxury_path", None):
        cl = _load_optional_lookup(cfg.customer_luxury_path, required_cols=["customer_id", "luxury_level"])
        if cl is not None:
            cl = cl.select(["customer_id", "luxury_level"]).with_columns(pl.col("customer_id").cast(pl.Utf8))
            fl = fl.join(cl, on="customer_id", how="left")

    # 5) price_segment: item_id, price_segment (i64)
    if getattr(cfg, "price_segment_path", None):
        ps = _load_optional_lookup(cfg.price_segment_path, required_cols=["item_id", "price_segment"])
        if ps is not None:
            ps = ps.select(["item_id", "price_segment"]).with_columns(pl.col("item_id").cast(pl.Utf8))
            fl = fl.join(ps, on="item_id", how="left")

    # 6) top10_by_cat_month: month (str), category_l1, rank
    if getattr(cfg, "top10_by_cat_month_path", None) and "category_l1" in fl.columns:
        t10 = _load_optional_lookup(cfg.top10_by_cat_month_path, required_cols=["month", "category_l1", "rank"])
        if t10 is not None:
            # IMPORTANT: ensure unique (month,category_l1) to avoid join explosion
            t10 = t10.group_by(["month","category_l1"]).agg(pl.col("rank").min().alias("rank"))
            month_key = _parse_date(cfg.begin_recent).strftime("%Y-%m")

            # Chuẩn hóa dtype + lọc đúng tháng để giảm size
            t10 = (
                t10.select(["month", "category_l1", "rank"])
                .with_columns([
                    pl.col("month").cast(pl.Utf8),
                    pl.col("category_l1").cast(pl.Utf8),
                    pl.col("rank").cast(pl.Int32, strict=False),
                ])
                .filter(pl.col("month") == pl.lit(month_key))
            )

            # QUAN TRỌNG: dedup về 1 dòng / key để tránh many-to-many join
            t10 = (
                t10.group_by(["month", "category_l1"])
                .agg(pl.col("rank").min().alias("top10_rank_in_cat_month"))
            )

            # Join
            fl = fl.with_columns(pl.lit(month_key).cast(pl.Utf8).alias("month"))
            fl = fl.join(t10, on=["month", "category_l1"], how="left")


    # -------------------------
    # 5) Optional cooc (only safe if hist already restricted)
    # -------------------------
    if bool(getattr(cfg, "enable_cooc", False)):
        # If not restricted, cooc can explode. Enforce restriction in smoke OR warn.
        if not restrict_hist and bool(getattr(cfg, "smoke", False)):
            # In smoke, force restrict to avoid kill
            users_in_pairs = pairs2.select("customer_id").unique()
            hist = hist.join(users_in_pairs, on="customer_id", how="inner")
        fl = add_cooc_features(cfg, fl, hist)

    return fl

def add_cooc_features(cfg: Stage2Config, fl: pl.DataFrame, hist: pl.DataFrame) -> pl.DataFrame:
    # Build basket key
    if cfg.cooc_same_day:
        basket = hist.select(["customer_id", "created_date", "item_id"]).unique()
        basket = basket.with_columns((pl.col("customer_id").cast(pl.Utf8) + "_" + pl.col("created_date").cast(pl.Utf8)).alias("basket_id"))
    else:
        basket = hist.select(["customer_id", "item_id"]).unique().with_columns(pl.col("customer_id").cast(pl.Utf8).alias("basket_id"))

    # explode pairs within basket: O(n^2) per basket, expensive. This is kept consistent with your original design,
    # but may be slow on full data.
    # Strategy: collect per basket items list, then generate pairs in python.
    b = basket.group_by("basket_id").agg(pl.col("item_id").unique().alias("items"))
    rows = []
    for items in b["items"].to_list():
        items = [str(x) for x in items if x is not None]
        if len(items) < 2:
            continue
        items = sorted(set(items))
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                rows.append((items[i], items[j]))
    if not rows:
        fl = fl.with_columns([pl.lit(0.0).alias("cooc_max_with_hist"), pl.lit(0.0).alias("cooc_sum_with_hist")])
        return fl

    pair_df = pl.DataFrame(rows, schema=[("i1", pl.Utf8), ("i2", pl.Utf8)])
    cooc = pair_df.group_by(["i1", "i2"]).len().rename({"len": "cooc"}).filter(pl.col("cooc") >= int(cfg.cooc_min_count))

    # hist items per user
    hist_items = hist.select(["customer_id", "item_id"]).unique().group_by("customer_id").agg(pl.col("item_id").alias("hist_items"))

    # For each (user,cand_item), compute max/sum cooc with hist_items by scanning pairs.
    cooc_map = {}
    for i1, i2, c in zip(cooc["i1"].to_list(), cooc["i2"].to_list(), cooc["cooc"].to_list()):
        cooc_map[(str(i1), str(i2))] = float(c)
        cooc_map[(str(i2), str(i1))] = float(c)

    fl2 = fl.join(hist_items, on="customer_id", how="left")
    maxs, sums = [], []
    for cand_it, hlist in zip(fl2["item_id"].to_list(), fl2["hist_items"].to_list()):
        cand_it = str(cand_it)
        if hlist is None:
            maxs.append(0.0); sums.append(0.0); continue
        hlist = [str(x) for x in hlist]
        vals = [cooc_map.get((cand_it, hi), 0.0) for hi in hlist]
        if not vals:
            maxs.append(0.0); sums.append(0.0)
        else:
            maxs.append(float(max(vals)))
            sums.append(float(sum(vals)))
    fl2 = fl2.with_columns([
        pl.Series("cooc_max_with_hist", maxs),
        pl.Series("cooc_sum_with_hist", sums),
    ]).drop(["hist_items"])
    return fl2


def build_feature_label(cfg: Stage2Config, run_dir: str) -> pl.DataFrame:
    """
    Build a feature-label *base* table for Stage-2:
      - POS = purchases in RECENT window (Y=1)
      - NEG_POOL = sampled from popular items (HIST), excluding user's history (Y=0)
      - Build features for POS + NEG_POOL (NO final negative sampling here)

    Final negative sampling (pick N_neg per user) must be done later in step [3/8].
    """
    log("[2/8] Build feature-label table (POS + NEG_POOL; NO final negative sampling) ...")

    begin_hist = _parse_date(cfg.begin_hist)
    end_hist = _parse_date(cfg.end_hist)
    begin_recent = _parse_date(cfg.begin_recent)
    end_recent = _parse_date(cfg.end_recent)

    # ---- load minimal tx for hist / recent separately (row caps for smoke) ----
    tx_hist = _load_tx_min_window(cfg, begin_hist, end_hist, max_rows=int(getattr(cfg, "max_rows_tx_hist", 0)))
    tx_recent = _load_tx_min_window(cfg, begin_recent, end_recent, max_rows=int(getattr(cfg, "max_rows_tx_recent", 0)))
    items_df = _load_items_min(cfg)

    # ---- POS pairs (recent purchases) ----
    pos_pairs = (
        tx_recent.select(["customer_id", "item_id"])
        .drop_nulls()
        .unique()
        .with_columns([
            pl.col("customer_id").cast(pl.Utf8),
            pl.col("item_id").cast(pl.Utf8),
            pl.lit(1, dtype=pl.Int8).alias("Y"),
        ])
    )

    # smoke: cap users first
    max_users_train = int(getattr(cfg, "max_users_train", 0) or 0)
    if max_users_train > 0:
        keep_users = (
            pos_pairs.select("customer_id")
            .unique()
            .sort("customer_id")
            .head(max_users_train)
        )
        pos_pairs = pos_pairs.join(keep_users, on="customer_id", how="inner")

    # smoke: cap POS pairs (optional)
    max_pos_pairs = int(getattr(cfg, "max_pos_pairs", 0) or 0)
    if max_pos_pairs > 0 and pos_pairs.height > max_pos_pairs:
        n_users = pos_pairs.select("customer_id").n_unique()
        head_per_user = max(1, max_pos_pairs // max(1, n_users))
        pos_pairs = (
            pos_pairs.sort("customer_id")
                    .group_by("customer_id", maintain_order=True)
                    .head(head_per_user)
                    .head(max_pos_pairs)
        )
        log(f"[SMOKE] POS pairs limited to {pos_pairs.height:,} (max_pos_pairs={max_pos_pairs})")

    if pos_pairs.height == 0:
        raise ValueError("No POS pairs found in RECENT window after limits.")

    log(f"[2/8] POS pairs: {pos_pairs.height:,} (users={pos_pairs.select('customer_id').n_unique():,})")

    # ---- Popular pool from HIST (for NEG_POOL) ----
    pool_size = int(getattr(cfg, "popular_pool_size", 50000) or 50000)
    pop_items_df = (
        tx_hist.select(["item_id", "quantity"])
        .drop_nulls()
        .with_columns(pl.col("item_id").cast(pl.Utf8))
        .group_by("item_id")
        .agg(pl.col("quantity").sum().alias("q"))
        .sort("q", descending=True)
        .head(pool_size)
        .select("item_id")
    )
    pop_items = pop_items_df["item_id"].to_list()
    if not pop_items:
        raise ValueError("popular pool is empty; cannot build NEG pool.")

    # ---- User history pairs to exclude in NEG_POOL ----
    # Use HIST + RECENT to avoid leaking positives into negatives.
    hist_pairs = (
        pl.concat(
            [tx_hist.select(["customer_id", "item_id"]), tx_recent.select(["customer_id", "item_id"])],
            how="vertical",
        )
        .drop_nulls()
        .with_columns([
            pl.col("customer_id").cast(pl.Utf8),
            pl.col("item_id").cast(pl.Utf8),
        ])
        .unique(["customer_id", "item_id"])
    )

    # ---- NEG_POOL sampling controls ----
    neg_pool_per_user = int(getattr(cfg, "neg_pool_per_user", 50) or 50)   # pool size per user (NOT final k)
    oversample = int(getattr(cfg, "neg_pool_oversample", 5) or 5)
    seed = int(getattr(cfg, "neg_seed", getattr(cfg, "random_state", 42)))
    rng = np.random.default_rng(seed)

    user_list = pos_pairs.select("customer_id").unique()["customer_id"].to_list()
    batch_users = int(getattr(cfg, "batch_users", 5000) or 5000)

    neg_batches: List[pl.DataFrame] = []
    for i in tqdm(range(0, len(user_list), batch_users), desc=f"NEG_POOL sampling (pool_per_user={neg_pool_per_user})"):
        u_batch = user_list[i:i + batch_users]
        if not u_batch:
            continue

        # We generate more then filter by hist, then keep up to neg_pool_per_user per user.
        m = len(u_batch) * neg_pool_per_user * oversample

        u_rep = np.repeat(np.array(u_batch, dtype=object), neg_pool_per_user * oversample)
        it_rep = np.array([pop_items[j] for j in rng.integers(0, len(pop_items), size=m)], dtype=object)

        neg0 = pl.DataFrame(
            {"customer_id": u_rep, "item_id": it_rep},
            schema={"customer_id": pl.Utf8, "item_id": pl.Utf8},
        ).unique(["customer_id", "item_id"])

        # anti-join to remove existing history pairs for these users
        hist_b = hist_pairs.join(
            pl.DataFrame({"customer_id": u_batch}, schema={"customer_id": pl.Utf8}),
            on="customer_id",
            how="inner",
        )
        neg1 = neg0.join(hist_b, on=["customer_id", "item_id"], how="anti")

        # Stable pseudo-random order (so reruns reproducible)
        neg1 = neg1.with_columns(
            (pl.col("customer_id") + pl.lit("|") + pl.col("item_id") + pl.lit(f"|{seed}"))
            .hash(seed=seed)
            .alias("_rk")
        )

        # Keep a POOL (NOT final k_neg)
        neg1 = (
            neg1.sort(["customer_id", "_rk"])
                .group_by("customer_id", maintain_order=True)
                .head(neg_pool_per_user)
                .drop("_rk")
                .with_columns(pl.lit(0, dtype=pl.Int8).alias("Y"))
        )

        neg_batches.append(neg1)

    neg_pool = (
        pl.concat(neg_batches, how="vertical")
        if neg_batches
        else pl.DataFrame(schema={"customer_id": pl.Utf8, "item_id": pl.Utf8, "Y": pl.Int8})
    )

    log(f"[2/8] NEG_POOL pairs: {neg_pool.height:,} (pool_per_user={neg_pool_per_user}, users={neg_pool.select('customer_id').n_unique():,})")

    # ---- Combine POS + NEG_POOL (still manageable) ----
    pairs_lbl = pl.concat([pos_pairs, neg_pool], how="vertical")

    # (optional) final cap for smoke ONLY
    max_rows_fl = int(getattr(cfg, "max_rows_feature_label", 0) or 0)
    if max_rows_fl > 0 and pairs_lbl.height > max_rows_fl:
        pairs_lbl = pairs_lbl.head(max_rows_fl)
        log(f"[SMOKE] pairs_lbl capped to {pairs_lbl.height:,} (max_rows_feature_label={max_rows_fl})")

    # ---- Build features for POS+NEG_POOL using HIST window ----
    # Important: we intentionally DO NOT do final negative sampling here.
    fl = build_features_for_pairs(
        cfg=cfg,
        pairs=pairs_lbl,
        tx_min=tx_hist,       # only HIST needed for preference features
        items_df=items_df,
        begin_hist=begin_hist,
        end_hist=end_hist,
    )

    return fl

def build_pairs_and_label_sampled(
    cfg: Stage2Config,
    tx_min: pl.DataFrame,
    begin_hist: datetime,
    end_hist: datetime,
    begin_recent: datetime,
    end_recent: datetime,
) -> pl.DataFrame:
    """
    Memory-safe pair builder:
      - POS pairs: unique (customer_id,item_id) in RECENT window
      - NEG pairs: sampled per-user from a popular-item pool computed on HIST
    Output schema: customer_id, item_id, Y (Int8 0/1)
    """

    # -------------------------
    # Knobs (provide defaults if config thiếu)
    # -------------------------
    neg_per_pos = int(getattr(cfg, "neg_per_pos", 5))               # number of negatives per positive
    popular_pool_size = int(getattr(cfg, "popular_pool_size", 50000))  # size of popular item pool
    max_train_users = int(getattr(cfg, "max_train_users", 0))       # 0 => no cap
    max_pos_per_user = int(getattr(cfg, "max_pos_per_user", 0))     # 0 => no cap
    random_state = int(getattr(cfg, "random_state", 42))

    # -------------------------
    # 1) POS pairs from RECENT
    # -------------------------
    tx_recent = tx_min.filter(
        pl.col("created_date").is_between(
            pl.lit(begin_recent.date(), dtype=pl.Date),
            pl.lit(end_recent.date(), dtype=pl.Date),
            closed="both",
        )
    ).select(["customer_id", "item_id"]).unique()

    if tx_recent.height == 0:
        raise ValueError("RECENT window has 0 positive pairs; cannot train stage2.")

    # Optional cap number of users to control memory (useful for smoke)
    if max_train_users and max_train_users > 0:
        users = tx_recent.select("customer_id").unique().head(max_train_users)
        tx_recent = tx_recent.join(users, on="customer_id", how="inner")

    # Optional cap positives per user (avoid heavy buyers exploding pairs)
    if max_pos_per_user and max_pos_per_user > 0:
        tx_recent = (
            tx_recent.with_columns(pl.int_range(0, pl.len()).over("customer_id").alias("_r"))
            .filter(pl.col("_r") < max_pos_per_user)
            .drop("_r")
        )

    pos = tx_recent.with_columns(pl.lit(1, dtype=pl.Int8).alias("Y"))

    # -------------------------
    # 2) Build per-user HIST item set (for exclusion)
    # -------------------------
    tx_hist_pairs = tx_min.filter(
        pl.col("created_date").is_between(
            pl.lit(begin_hist.date(), dtype=pl.Date),
            pl.lit(end_hist.date(), dtype=pl.Date),
            closed="both",
        )
    ).select(["customer_id", "item_id"]).unique()

    # Reduce hist to users in pos to avoid huge group_by
    pos_users = pos.select("customer_id").unique()
    tx_hist_pairs = tx_hist_pairs.join(pos_users, on="customer_id", how="inner")

    # hist_items list per user
    hist_list = tx_hist_pairs.group_by("customer_id").agg(
        pl.col("item_id").alias("hist_items")
    )

    # pos_items list per user (to exclude positives from negatives)
    pos_list = pos.select(["customer_id", "item_id"]).group_by("customer_id").agg(
        pl.col("item_id").alias("pos_items")
    )

    user_excl = hist_list.join(pos_list, on="customer_id", how="left").with_columns(
        [
            pl.col("pos_items").fill_null([]),
        ]
    )

    # -------------------------
    # 3) Popular item pool from HIST (global)
    # -------------------------
    pop = (
        tx_min.filter(
            pl.col("created_date").is_between(
                pl.lit(begin_hist.date(), dtype=pl.Date),
                pl.lit(end_hist.date(), dtype=pl.Date),
                closed="both",
            )
        )
        .group_by("item_id")
        .len()
        .sort("len", descending=True)
        .select("item_id")
        .head(popular_pool_size)
    )
    pop_items = pop["item_id"].to_list()
    if len(pop_items) == 0:
        raise ValueError("Popular pool is empty (HIST window has no items).")

    # -------------------------
    # 4) Sample negatives per user (Python loop, memory-safe)
    # -------------------------
    import numpy as np
    rng = np.random.default_rng(random_state)

    # Materialize user_excl into Python dicts (size = #pos_users, not huge)
    users = user_excl["customer_id"].to_list()
    hist_items_col = user_excl["hist_items"].to_list()
    pos_items_col = user_excl["pos_items"].to_list()

    neg_rows = []
    for u, hist_items_u, pos_items_u in zip(users, hist_items_col, pos_items_col):
        excl = set(hist_items_u) | set(pos_items_u)
        n_pos_u = len(pos_items_u)
        if n_pos_u == 0:
            continue
        need = neg_per_pos * n_pos_u

        # sample with retries; allow repeats in sampling but dedup later
        chosen = []
        # quick exit if pool too small
        if need > 0:
            tries = 0
            max_tries = max(10_000, need * 20)
            while len(chosen) < need and tries < max_tries:
                it = pop_items[int(rng.integers(0, len(pop_items)))]
                tries += 1
                if it in excl:
                    continue
                chosen.append(it)

        # dedup
        chosen = list(dict.fromkeys(chosen))
        for it in chosen[:need]:
            neg_rows.append((u, it, 0))

    neg = pl.DataFrame(
        neg_rows,
        schema=[("customer_id", pl.Int64), ("item_id", pl.Utf8), ("Y", pl.Int8)],
    )

    # -------------------------
    # 5) Union POS + NEG
    # -------------------------
    pairs_lbl = pl.concat(
        [
            pos.select(["customer_id", "item_id", "Y"]),
            neg.select(["customer_id", "item_id", "Y"]),
        ],
        how="vertical",
    ).unique(["customer_id", "item_id"])

    log(
        f"[2/8] pairs_lbl built: pos={pos.height:,} neg={neg.height:,} total={pairs_lbl.height:,} "
        f"(neg_per_pos={neg_per_pos}, popular_pool={len(pop_items):,})"
    )
    return pairs_lbl


# -------------------------
# Negative sampling
# -------------------------
import polars as pl
from tqdm.auto import tqdm

def negative_sample(
    fl: pl.DataFrame,
    k_neg_per_user: int = 10,
    keep_users_without_pos: bool = False,
    seed: int = 42,
    batch_users: int = 600000,
) -> pl.DataFrame:
    if "Y" not in fl.columns:
        raise ValueError("fl must contain label column Y")

    pos = fl.filter(pl.col("Y") == 1)
    neg = fl.filter(pl.col("Y") == 0)

    if not keep_users_without_pos:
        pos_users_df = pos.select("customer_id").unique()
        neg = neg.join(pos_users_df, on="customer_id", how="inner")
        users = pos_users_df["customer_id"].to_list()
    else:
        users = neg.select("customer_id").unique()["customer_id"].to_list()

    out_negs = []
    for i in tqdm(range(0, len(users), batch_users), desc=f"Negative sampling (k={k_neg_per_user})"):
        u_batch = users[i:i+batch_users]
        u_df = pl.DataFrame({"customer_id": u_batch})

        neg_b = neg.join(u_df, on="customer_id", how="inner")

        neg_b = neg_b.with_columns(
            (
                pl.col("customer_id").cast(pl.Utf8)
                + pl.lit("|")
                + pl.col("item_id").cast(pl.Utf8)
                + pl.lit("|")
                + pl.lit(str(seed))
            ).hash(seed=seed).alias("_rk")
        )

        neg_b = (
            neg_b.sort(["customer_id", "_rk"])
                 .group_by("customer_id", maintain_order=True)
                 .head(k_neg_per_user)
                 .drop("_rk")
        )

        out_negs.append(neg_b)

    neg_s = pl.concat(out_negs, how="vertical") if out_negs else neg.head(0)

    train = pl.concat([pos, neg_s], how="vertical")
    train = train.sample(fraction=1.0, shuffle=True, seed=seed)
    return train

import json, hashlib, os

def _hash_cfg(cfg: Stage2Config, keys: list[str]) -> str:
    d = {k: getattr(cfg, k, None) for k in keys}
    s = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]

def _cache_ok(meta_path: str, want_hash: str) -> bool:
    if not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("cfg_hash") == want_hash
    except Exception:
        return False

# -------------------------
# Stage-2 training + predict + metrics (per user functions)
# -------------------------
def train_lgbm_ranker(train_df: pd.DataFrame, cfg: Stage2Config, valid_df: Optional[pd.DataFrame] = None):
    ignore_cols = {"customer_id", "item_id", "Y", "label", "created_date", "created_datetime"}
    feature_cols = [c for c in train_df.columns if c not in ignore_cols]

    log(f"[Stage2] Training with {len(feature_cols)} features.")

    train_df = train_df.sort_values("customer_id")
    q_train = train_df.groupby("customer_id").size().values
    X_train = train_df[feature_cols]
    y_train = train_df["Y"] if "Y" in train_df.columns else train_df["label"]

    lgb_train = lgb.Dataset(X_train, y_train, group=q_train)

    valid_sets = [lgb_train]
    valid_names = ["train"]
    callbacks = [lgb.log_evaluation(period=50)]

    if valid_df is not None:
        valid_df = valid_df.sort_values("customer_id")
        q_valid = valid_df.groupby("customer_id").size().values
        X_valid = valid_df[feature_cols]
        y_valid = valid_df["Y"] if "Y" in valid_df.columns else valid_df["label"]
        lgb_eval = lgb.Dataset(X_valid, y_valid, group=q_valid, reference=lgb_train)
        valid_sets.append(lgb_eval)
        valid_names.append("valid")
        callbacks.append(lgb.early_stopping(stopping_rounds=50))

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10, 20],
        "boosting_type": "gbdt",
        "learning_rate": float(cfg.learning_rate),
        "num_leaves": int(cfg.num_leaves),
        "min_data_in_leaf": int(cfg.min_data_in_leaf),
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "random_state": int(cfg.random_state),
    }

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=int(cfg.n_estimators),
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    return model, feature_cols


def predict_stage2(
    model,
    fl_pred_pd: pd.DataFrame,
    feature_cols: list[str],
    top_k: int,
):
    """
    Predict và rank Stage-2.
    Input: pandas DataFrame (NOT polars).
    Must contain: customer_id, item_id, feature_cols.
    """

    df = fl_pred_pd.copy()

    # ---- Predict ----
    df["pred_score"] = model.predict(df[feature_cols])

    # ---- Rank per user ----
    df = (
        df.sort_values(["customer_id", "pred_score"], ascending=[True, False])
          .groupby("customer_id", as_index=False)
          .head(top_k)
    )

    return df


def calculate_metrics_at_k(
    pred_df: pd.DataFrame,
    gt_dict: Dict[str, Set[str]],
    train_history: Dict[str, Set[str]],
    k: int = 10,
    filter_bought_items: bool = True,
) -> Dict[str, float]:
    pred_map = (
        pred_df.sort_values(["customer_id", "pred_score"], ascending=[True, False])
        .groupby("customer_id")["item_id"]
        .apply(lambda x: list(x)[:k])
        .to_dict()
    )

    metrics = {"all": {"p": [], "r": [], "ndcg": []}, "warm": {"p": [], "r": [], "ndcg": []}, "cold": {"p": [], "r": [], "ndcg": []}}

    for user, truth_items in gt_dict.items():
        user = str(user)
        truth_items = set(str(x) for x in truth_items)

        hist_items = set(str(x) for x in train_history.get(user, []))
        is_cold = len(hist_items) == 0

        relevant_items = truth_items.copy()
        if filter_bought_items:
            relevant_items = relevant_items - hist_items
            if len(relevant_items) == 0:
                continue

        recs = [str(x) for x in pred_map.get(user, [])][:k]
        hits = len(set(recs) & relevant_items)

        precision = hits / k
        recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0.0

        dcg = 0.0
        for i, item in enumerate(recs):
            if item in relevant_items:
                dcg += 1.0 / np.log2(i + 2)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_items), k)))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        group = "cold" if is_cold else "warm"
        for g in ("all", group):
            metrics[g]["p"].append(precision)
            metrics[g]["r"].append(recall)
            metrics[g]["ndcg"].append(ndcg)

    final_res = {}
    mode_str = "NEW" if filter_bought_items else "ALL"
    for group in ["all", "warm", "cold"]:
        if metrics[group]["p"]:
            final_res[f"{group}_{mode_str}_P@{k}"] = float(np.mean(metrics[group]["p"]))
            final_res[f"{group}_{mode_str}_R@{k}"] = float(np.mean(metrics[group]["r"]))
            final_res[f"{group}_{mode_str}_NDCG@{k}"] = float(np.mean(metrics[group]["ndcg"]))
        else:
            final_res[f"{group}_{mode_str}_P@{k}"] = 0.0
            final_res[f"{group}_{mode_str}_R@{k}"] = 0.0
            final_res[f"{group}_{mode_str}_NDCG@{k}"] = 0.0
    final_res["n_eval_users"] = int(len(metrics["all"]["p"]))
    return final_res

def _load_tx_min_window(cfg: Stage2Config, begin_dt: datetime, end_dt: datetime, max_rows: int = 0) -> pl.DataFrame:
    tx = pl.scan_parquet(cfg.transactions_path_glob)

    cols = set(tx.collect_schema().names())  # tránh warning .columns

    need = {"customer_id", "item_id", "created_date"}
    missing = [c for c in need if c not in cols]
    if missing:
        raise ValueError(f"Transactions missing required columns: {missing}. Available: {sorted(cols)}")

    has_qty = "quantity" in cols
    has_price = "price" in cols
    has_disc = "discount_rate" in cols

    tx = tx.with_columns([
        pl.col("customer_id").cast(pl.Utf8),
        pl.col("item_id").cast(pl.Utf8),
        pl.col("created_date").cast(pl.Date, strict=False),
        (pl.col("quantity").cast(pl.Float64, strict=False).fill_null(1.0).fill_nan(1.0) if has_qty else pl.lit(1.0).alias("quantity")),
        (pl.col("price").cast(pl.Float64, strict=False).fill_null(0.0).fill_nan(0.0) if has_price else pl.lit(0.0).alias("price")),
        (pl.col("discount_rate").cast(pl.Float64, strict=False).fill_null(0.0).fill_nan(0.0) if has_disc else pl.lit(0.0).alias("discount_rate")),
    ]).select(["customer_id", "item_id", "created_date", "quantity", "price", "discount_rate"])

    tx = tx.filter(
        pl.col("created_date").is_between(
            pl.lit(begin_dt.date(), dtype=pl.Date),
            pl.lit(end_dt.date(), dtype=pl.Date),
            closed="both",
        )
    )

    if max_rows and max_rows > 0:
        tx = tx.head(max_rows)

    # collect (Polars >=1.25 khuyến nghị engine)
    df = tx.collect(engine="streaming")
    return df


# -------------------------
# Main
# -------------------------
def main(cfg_path: str) -> None:
    cfg = load_config(cfg_path)

    run_dir = os.path.join(cfg.artifacts_dir, "runs", cfg.run_name)
    _ensure_dir(run_dir)

    log("==== Stage2 Train + Testflow v3 ====")
    log(f"Config: {cfg_path}")
    log(f"Run dir: {run_dir}")

    # [2/8] feature-label
    log("[2/8] Build feature-label ...")
    fl_path = os.path.join(run_dir, "fl_pos.parquet")
    fl_meta = os.path.join(run_dir, "fl_pos.meta.json")

    cfg_hash_fl = _hash_cfg(cfg, keys=[
        "begin_hist","end_hist","begin_recent","end_recent",
        "transactions_path_glob","items_path_glob",
        "enable_cooc",  # nếu có
        # + thêm các path lookup nếu dùng:
        "brand_segment_path","customer_age_features_path","customer_behavior_path",
        "customer_luxury_path","price_segment_path","top10_by_cat_month_path",
    ])
    if os.path.exists(fl_path) and _cache_ok(fl_meta, cfg_hash_fl):
        log(f"[CACHE] load fl -> {fl_path}")
        fl = pl.read_parquet(fl_path)
    else:
        fl = build_feature_label(cfg, run_dir)
        # -------------------------
        # SMOKE LIMIT: feature-label rows
        # -------------------------
        max_fl = int(getattr(cfg, "max_rows_feature_label", 0))
        if max_fl and max_fl > 0 and fl.height > max_fl:
            # ưu tiên giữ phân bố theo user (tránh 1 user chiếm hết)
            fl = (
                fl.sort("customer_id")
                .group_by("customer_id", maintain_order=True)
                .head(max(1, max_fl // max(1, fl.select("customer_id").n_unique())))
                .head(max_fl)
            )
            log(f"[SMOKE] fl limited to {fl.height:,} rows (max_rows_feature_label={max_fl})")
        fl.write_parquet(fl_path)
        with open(fl_meta, "w", encoding="utf-8") as f:
            json.dump({"cfg_hash": cfg_hash_fl, "rows": fl.height, "cols": fl.columns}, f, ensure_ascii=False, indent=2)
        log(f"[SAVE] fl -> {fl_path}")

    log(f"[INFO] fl shape: {fl.height:,} x {len(fl.columns)}")


    # [3/8] negative sampling -> train_pl
    log("[3/8] Negative sampling ...")
    train_path = os.path.join(run_dir, "train_sampled.parquet")
    train_meta = os.path.join(run_dir, "train_sampled.meta.json")

    cfg_hash_train = _hash_cfg(cfg, keys=[
        "N_neg","keep_users_without_pos","random_state",
        # thêm gì ảnh hưởng sampling:
    ])
    if os.path.exists(train_path) and _cache_ok(train_meta, cfg_hash_train):
        log(f"[CACHE] load train_sampled -> {train_path}")
        train_pl = pl.read_parquet(train_path)
    else:
        # IMPORTANT: bạn nên dùng phiên bản fast polars (k_neg_per_user=10)
        train_pl = negative_sample(
            fl,
            k_neg_per_user=int(cfg.N_neg),
            keep_users_without_pos=bool(cfg.keep_users_without_pos),
            seed=int(cfg.random_state),
            batch_users=int(getattr(cfg, "batch_users", 50000)),
        )
        train_pl.write_parquet(train_path)
        with open(train_meta, "w", encoding="utf-8") as f:
            json.dump({"cfg_hash": cfg_hash_train, "rows": train_pl.height}, f, ensure_ascii=False, indent=2)
        log(f"[SAVE] train_sampled -> {train_path} (rows={train_pl.height:,})")


    log("[4/8] Train/Load LightGBM ...")
    model_path = os.path.join(run_dir, "lgbm_ranker.txt")
    feat_path = os.path.join(run_dir, "feature_columns.json")
    model_meta = os.path.join(run_dir, "lgbm_ranker.meta.json")

    cfg_hash_model = _hash_cfg(cfg, keys=[
        "learning_rate","num_leaves","min_data_in_leaf","n_estimators",
        # + các hyperparam bạn dùng thực sự trong train_lgbm_ranker
    ])

    if os.path.exists(model_path) and os.path.exists(feat_path) and _cache_ok(model_meta, cfg_hash_model):
        log(f"[CACHE] model -> {model_path}")
        model = lgb.Booster(model_file=model_path)
        with open(feat_path, "r", encoding="utf-8") as f:
            feature_cols = json.load(f)
    else:
        train_pd = train_pl.to_pandas()

        # đảm bảo id là string, không null
        train_pd = train_pd[train_pd["customer_id"].notna()]
        train_pd["customer_id"] = train_pd["customer_id"].astype(str)
        train_pd["item_id"] = train_pd["item_id"].astype(str)

        # encode chỉ feature, không encode id/label
        skip_cols = {"customer_id", "item_id", "Y"}
        for c in train_pd.columns:
            if c in skip_cols:
                continue
            if train_pd[c].dtype == "object":
                train_pd[c] = pd.Categorical(train_pd[c]).codes

        # debug trước train
        print("train_pd rows:", len(train_pd))
        print("n unique users:", train_pd["customer_id"].nunique(dropna=False))
        print(train_pd["customer_id"].value_counts(dropna=False).head(5))

        # -------------------------
        # SMOKE LIMIT for LightGBM train
        # -------------------------
        max_train = int(getattr(cfg, "max_rows_lgbm_train", 0))
        if max_train and max_train > 0 and len(train_pd) > max_train:
            # 1) giữ số user (query) vừa đủ để tổng dòng xấp xỉ max_train
            # (lambdarank yêu cầu group theo customer_id)
            user_sizes = train_pd.groupby("customer_id").size().sort_values(ascending=False)

            # lấy users cho tới khi đủ max_train rows
            total = 0
            keep_users = []
            for u, cnt in user_sizes.items():
                keep_users.append(u)
                total += int(cnt)
                if total >= max_train:
                    break

            train_pd = train_pd[train_pd["customer_id"].isin(keep_users)]

            # 2) nếu vẫn vượt quá max_train, cắt thêm "head per user" để không phá group
            if len(train_pd) > max_train:
                # tính head_per_user xấp xỉ
                n_users = train_pd["customer_id"].nunique()
                head_per_user = max(1, max_train // max(1, n_users))

                train_pd = (
                    train_pd.sort_values(["customer_id"])
                            .groupby("customer_id", group_keys=False)
                            .head(head_per_user)
                )

            print(f"[SMOKE] train_pd limited to {len(train_pd):,} rows, users={train_pd['customer_id'].nunique():,}")


        model, feature_cols = train_lgbm_ranker(train_pd, cfg, valid_df=None)

        model.save_model(model_path)
        with open(feat_path, "w", encoding="utf-8") as f:
            json.dump(feature_cols, f, ensure_ascii=False, indent=2)
        with open(model_meta, "w", encoding="utf-8") as f:
            json.dump({"cfg_hash": cfg_hash_model, "n_features": len(feature_cols)}, f, ensure_ascii=False, indent=2)

        log(f"[SAVE] model -> {model_path}")
        log(f"[SAVE] feature_columns -> {feat_path} (n={len(feature_cols)})")


    # [7/8] Stage2 ranking (testflow)
    if not cfg.stage1_model_dir or not cfg.gt_path:
        log("[INFO] stage1_model_dir or gt_path not provided -> skip testflow ranking.")
        return

    # ---- ALWAYS load GT + test_users first (even when candidates cache exists) ----
    log(">>> 1. Load Groundtruth (Test users) ...")
    gt = load_groundtruth_pkl(cfg.gt_path)
    test_users = list(gt.keys())

    max_test_users = int(getattr(cfg, "max_test_users", 0) or 0)
    if max_test_users > 0:
        test_users = test_users[:max_test_users]
        gt = {u: gt[u] for u in test_users}

    log(f">>> Found {len(test_users):,} users in Test Set (capped={max_test_users})")

    # ---- ALWAYS init Stage1 model (needed by batched ranking loop) ----
    stage1 = Stage1FromModelFinal(
        cfg.stage1_model_dir,
        cfg.transactions_path_glob,
        max_rows_tx=int(getattr(cfg, "stage1_max_rows_tx", 0) or 0),
    )
    stage1.top_k = int(cfg.stage1_top_k)

    # ---- OPTIONAL: cache candidates for debugging only ----
    cand_path = os.path.join(run_dir, "candidates_test.parquet")
    cand_meta = os.path.join(run_dir, "candidates_test.meta.json")

    cfg_hash_cand = _hash_cfg(cfg, keys=[
        "stage1_model_dir", "stage1_top_k", "allow_repeat", "batch_users", "gt_path", "max_test_users", "stage1_max_rows_tx"
    ])

    if os.path.exists(cand_path) and _cache_ok(cand_meta, cfg_hash_cand):
        log(f"[CACHE] candidates -> {cand_path}")
    else:
        log(">>> 1b. (Optional) Caching candidates_test.parquet ...")
        all_cands = []
        bs = max(1, int(getattr(cfg, "batch_users", 200) or 200))
        for i in tqdm(range(0, len(test_users), bs), desc="Stage1 candidates (batched; cache)"):
            batch = test_users[i:i + bs]
            cand_batch = stage1.recommend_candidates(batch, allow_repeat=bool(cfg.allow_repeat))
            all_cands.append(cand_batch)

        candidates = pd.concat(all_cands, ignore_index=True) if all_cands else pd.DataFrame(columns=["customer_id", "item_id"])
        pl.from_pandas(candidates).write_parquet(cand_path)
        with open(cand_meta, "w", encoding="utf-8") as f:
            json.dump({"cfg_hash": cfg_hash_cand, "rows": int(candidates.shape[0])}, f, ensure_ascii=False, indent=2)
        log(f"[SAVE] candidates -> {cand_path} (rows={candidates.shape[0]:,})")

    # ---- Stage2 ranking (BATCHED) ----
    log(">>> 2. Ranking (Stage 2)...")

    # Prepare windows + shared tables once
    begin_hist = _parse_date(cfg.begin_hist)
    end_hist = _parse_date(cfg.end_hist)
    begin_recent = _parse_date(cfg.begin_recent)
    end_recent = _parse_date(cfg.end_recent)
    begin_all = min(begin_hist, begin_recent)
    end_all = max(end_hist, end_recent)

    tx_min = _load_tx_min(cfg, begin_all, end_all)
    max_rows_tx_pred = int(getattr(cfg, "max_rows_tx_pred", 0) or 0)
    if max_rows_tx_pred > 0 and tx_min.height > max_rows_tx_pred:
        tx_min = tx_min.head(max_rows_tx_pred)
    items_df = _load_items_min(cfg)

    df_pred_parts = []
    bs_users = max(1, int(getattr(cfg, "batch_users", 200) or 200))

    cfg_hash_pred = _hash_cfg(cfg, keys=[
        "begin_hist", "end_hist",
        "transactions_path_glob", "items_path_glob",
        "enable_cooc",
        "brand_segment_path", "customer_age_features_path", "customer_behavior_path",
        "customer_luxury_path", "price_segment_path", "top10_by_cat_month_path",
        "stage1_model_dir", "stage1_top_k", "allow_repeat",
        "gt_path", "max_test_users", "max_rows_tx_pred", "stage1_max_rows_tx",
    ])

    pred_cache_dir = os.path.join(run_dir, "pred_batches")
    _ensure_dir(pred_cache_dir)

    for i in tqdm(range(0, len(test_users), bs_users), desc="Testflow batched ranking"):
        u_batch = test_users[i:i + bs_users]

        batch_key = f"{i:09d}_{i+len(u_batch)-1:09d}_{cfg_hash_pred}"
        batch_path = os.path.join(pred_cache_dir, f"df_pred_{batch_key}.parquet")

        if os.path.exists(batch_path):
            df_pred_parts.append(pl.read_parquet(batch_path).to_pandas())
            continue

        # Stage1 candidates for this batch
        cand_batch = stage1.recommend_candidates(u_batch, allow_repeat=bool(cfg.allow_repeat))

        pairs = pl.from_pandas(cand_batch).with_columns([
            pl.col("customer_id").cast(pl.Utf8, strict=False),
            pl.col("item_id").cast(pl.Utf8, strict=False),
        ]).select(["customer_id", "item_id"]).unique()

        fl_pred_batch = build_features_for_pairs(cfg, pairs, tx_min, items_df, begin_hist, end_hist)

        # Convert only needed cols to pandas
        fl_pred_pd = fl_pred_batch.select(["customer_id", "item_id"] + feature_cols).to_pandas()

        # Ensure all feature cols exist
        for col in feature_cols:
            if col not in fl_pred_pd.columns:
                fl_pred_pd[col] = 0

        # Categorical -> codes
        for c in feature_cols:
            if fl_pred_pd[c].dtype == "object":
                fl_pred_pd[c] = pd.Categorical(fl_pred_pd[c]).codes

        df_pred_batch = predict_stage2(
            model,
            fl_pred_pd,
            feature_cols=feature_cols,
            top_k=int(cfg.metric_k),
        )

        out_batch = df_pred_batch[["customer_id", "item_id", "pred_score"]]
        pl.from_pandas(out_batch).write_parquet(batch_path)
        df_pred_parts.append(out_batch)

    df_pred = pd.concat(df_pred_parts, ignore_index=True) if df_pred_parts else pd.DataFrame(columns=["customer_id", "item_id", "pred_score"])
    log(f"[INFO] df_pred shape (batched): {df_pred.shape}")



    # [8/8] Metrics + save submission
    log(">>> 3. Calculating Metrics...")
    train_history = stage1.get_user_history_dict()
    metrics = calculate_metrics_at_k(
        df_pred,
        gt,
        train_history,
        k=int(cfg.metric_k),
        filter_bought_items=bool(cfg.filter_bought_eval),
    )
    log("--- TEST RESULTS ---")
    for k, v in metrics.items():
        log(f"  {k}: {v}")

    metrics_path = os.path.join(run_dir, "jan2025_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    log(f"[SAVE] metrics -> {metrics_path}")

    out_path = cfg.submission_out
    _ensure_dir(os.path.dirname(out_path) or ".")
    df_pred[["customer_id", "item_id", "pred_score"]].to_csv(out_path, index=False)
    log(f"[SAVE] submission -> {out_path}")

    log("DONE.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    main(args.config)
