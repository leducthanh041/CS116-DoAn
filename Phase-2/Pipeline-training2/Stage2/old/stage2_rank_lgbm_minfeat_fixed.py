# -*- coding: utf-8 -*-
"""
stage2_rank_lgbm.py

Stage-2 Ranking for a 2-stage recommender:
  Stage-1: implicit TFIDFRecommender + CosineRecommender + Trending backfill (already trained)
  Stage-2: Feature augmentation + Pointwise ranking model:
           LightGBM ranker (LGBMClassifier)

This script:
  1) Loads Stage-1 artifacts (best_stage1_meta/tfidf/cosine + best_params.json)
  2) Rebuilds the Stage-1 CSR (history window per best params) for scoring candidates with scores/ranks
  3) Generates per-user candidates with retrieval features (source flags, ranks, scores)
  4) Builds a feature-label training table:
       - Features from user/item/transactions + engineered parquet features
       - Label Y from RECENT purchases
       - Negative sampling per user via N_neg
  5) Trains the ranking model (SVM + L1 selection)
  6) Evaluates offline on RECENT (FILTERED + UNFILTERED)
  7) Evaluates private test using groundtruth.pkl (Precision@K, NDCG@K), if provided
  8) Saves all artifacts and caches for faster reruns

Assumed schemas:
  user:
    customer_id (i32), gender (str), province (str), membership (str),
    created_date (date), install_app (str), install_datetime (date)
  transaction:
    item_id (str), price (decimal), quantity (i32), customer_id (i32),
    created_date (date), channel (str), payment (str), location (i32),
    discount (decimal), list_price (decimal), category_l2 (str), discount_rate (decimal)
  item:
    item_id (str), price (decimal), category_l1 (str), category_l2 (str),
    brand (str), item_type (str), gender_target_final (str), age_group_final (str)

Engineered features (optional):
  brand_segment.parquet: category_l1, brand_segment (i64)
  customer_age_features.parquet: customer_id, age_final (f64)
  customer_behavior.parquet: customer_id, buy_segment (i32)
  customer_luxury.parquet: customer_id, luxury_level
  price_segment.parquet: item_id, price_segment
  top10_by_cat_month.parquet: month(str YYYY-MM), category_l1, item_id, rank(u32)

Design choices (per your request):
  - Feature window HIST ends at train_end, length len_hist from Stage-1 best_params.
  - Label window RECENT starts at recent_begin, length len_recent from Stage-1 best_params.
  - Negative sampling: keep all positives, sample N_neg negatives per user from that user's candidates.
  - Output top-k: 10 (configurable).
  - Evaluation: report BOTH FILTERED and UNFILTERED; FILTERED is primary.

Run:
  python stage2_rank_lgbm.py --config stage2_full_config.json
"""

from __future__ import annotations

import os
import json
import math
import pickle
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix, save_npz, load_npz
from tqdm.auto import tqdm

from implicit.nearest_neighbours import TFIDFRecommender, CosineRecommender

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib

# LightGBM
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

# --- sklearn compatibility: OneHotEncoder sparse vs sparse_output ---
def make_onehot_encoder():
    """Return OneHotEncoder compatible across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        # older sklearn
        return make_onehot_encoder()



def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


@dataclass
class Stage2Config:
    transactions_path_glob: str
    items_path_glob: str
    users_path_glob: str

    brand_segment_path: Optional[str] = ".././new-feature/brand_segment.parquet"
    customer_age_features_path: Optional[str] = ".././new-feature/customer_age_features.parquet"
    customer_behavior_path: Optional[str] = ".././new-feature/customer_behavior.parquet"
    customer_luxury_path: Optional[str] = ".././new-feature/customer_luxury.parquet"
    price_segment_path: Optional[str] = ".././new-feature/price_segment.parquet"
    top10_by_cat_month_path: Optional[str] = ".././new-feature/top10_by_cat_month.parquet"

    stage1_run_dir: str = "./artifacts_stage1/runs/full_randomsearch"
    stage1_best_params_json: str = "best_params.json"
    stage1_meta_npz: str = "best_stage1_meta.npz"
    stage1_tfidf_npz: str = "best_stage1_tfidf.npz"
    stage1_cosine_npz: str = "best_stage1_cosine.npz"

    train_end: str = "2024-11-30"
    recent_begin: str = "2024-12-01"

    # For FILTERED evaluation: filter out items the user bought in this full range (inclusive)
    filter_hist_begin: str = "2024-01-01"
    filter_hist_end: str = "2024-12-31"

    N_cand: Optional[int] = None
    N_trend: Optional[int] = None

    N_neg: int = 10
    random_state: int = 42

    top10_month_lag: int = 1

    out_dir: str = "./artifacts_stage2"
    run_name: str = "run"
    cache_dir: str = "./artifacts_stage2/cache"
    save_intermediate: bool = True

    batch_users: int = 5000
    max_users_train: int = 0
    max_users_eval: int = 0
    max_rows_tx_cache: int = 0

    # LightGBM params
    lgbm_objective: str = "binary"
    lgbm_learning_rate: float = 0.05
    lgbm_n_estimators: int = 500
    lgbm_num_leaves: int = 63
    lgbm_max_depth: int = -1
    lgbm_min_child_samples: int = 50
    lgbm_subsample: float = 0.8
    lgbm_colsample_bytree: float = 0.8
    lgbm_reg_alpha: float = 0.0
    lgbm_reg_lambda: float = 0.0
    lgbm_random_state: int = 42
    lgbm_early_stopping_rounds: int = 50
    lgbm_test_size: float = 0.1

    groundtruth_pkl: Optional[str] = None
    evaluate_private_test: bool = True

    topk: int = 10


def load_config(path: str) -> Stage2Config:
    with open(path, "r", encoding="utf-8") as f:
        return Stage2Config(**json.load(f))


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def safe_join(df: pl.DataFrame,
              other: pl.DataFrame,
              *,
              on: Optional[List[str]] = None,
              left_on: Optional[List[str]] = None,
              right_on: Optional[List[str]] = None,
              how: str = "left",
              suffix: str = "_r") -> pl.DataFrame:
    """Polars join that avoids DuplicateError by dropping overlapping non-key columns from `other`.

    This is necessary when multiple joins would otherwise create repeated '*_right' columns.
    """
    if on is not None:
        left_keys = list(on)
        right_keys = list(on)
    else:
        if left_on is None or right_on is None:
            raise ValueError("Provide `on` or both `left_on` and `right_on`.")
        left_keys = list(left_on)
        right_keys = list(right_on)

    # Columns from right that will be kept (non-join keys, plus any right keys that don't match left keys)
    left_cols = set(df.columns)
    right_cols = set(other.columns)

    # Keys on right that share the same name as left join keys are not duplicated by Polars.
    shared_key_names = set(right_keys) & set(left_keys)

    # Potential collisions are right columns that already exist in left, excluding shared key names.
    collisions = (right_cols & left_cols) - shared_key_names
    if collisions:
        other = other.drop(list(collisions))

    if on is not None:
        return df.join(other, on=on, how=how, suffix=suffix)
    return df.join(other, left_on=left_on, right_on=right_on, how=how, suffix=suffix)


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _hash_dict(d: dict) -> str:
    s = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]




def encode_categoricals_with_maps(df: "pd.DataFrame",
                                 cat_cols: List[str],
                                 cat_maps: Optional[Dict[str, Dict[str, int]]] = None
                                 ) -> Tuple["pd.DataFrame", Dict[str, Dict[str, int]]]:
    """Encode categorical columns into integer codes with per-column mapping.
    Unknowns map to -1. Returns (df_encoded, maps)."""
    import pandas as pd
    maps_out: Dict[str, Dict[str, int]] = {} if cat_maps is None else {k: dict(v) for k, v in cat_maps.items()}

    for c in cat_cols:
        if c not in df.columns:
            continue
        s = df[c].astype("string").fillna("__MISSING__")
        if cat_maps is None or c not in maps_out:
            uniq = pd.unique(s)
            m = {str(v): i for i, v in enumerate(uniq)}
            maps_out[c] = m
        m = maps_out[c]
        df[c] = s.map(lambda x: m.get(str(x), -1)).astype("int32")
    return df, maps_out



def _month_key(dt: datetime) -> str:
    return dt.strftime("%Y-%m")


def _shift_months(dt: datetime, months: int) -> datetime:
    y, m = dt.year, dt.month + int(months)
    while m <= 0:
        y -= 1
        m += 12
    while m > 12:
        y += 1
        m -= 12
    import calendar
    d = min(dt.day, calendar.monthrange(y, m)[1])
    return dt.replace(year=y, month=m, day=d)


@dataclass

@dataclass
class LGBMBundle:
    model: Any
    feat_cols: List[str]
    numeric_cols: List[str]
    cat_cols: List[str]
    cat_maps: Dict[str, Dict[str, int]]  # value->code


@dataclass
class Stage1Artifacts:
    users: List[str]
    items: List[str]
    trending_idx: np.ndarray
    config: dict
    tfidf: TFIDFRecommender
    cosine: CosineRecommender


def load_stage1_artifacts(cfg: Stage2Config) -> Tuple[Stage1Artifacts, dict]:
    run_dir = cfg.stage1_run_dir
    best_params_path = os.path.join(run_dir, cfg.stage1_best_params_json)
    meta_path = os.path.join(run_dir, cfg.stage1_meta_npz)      # legacy name in config
    tfidf_path = os.path.join(run_dir, cfg.stage1_tfidf_npz)
    cosine_path = os.path.join(run_dir, cfg.stage1_cosine_npz)

    # 1) best params (always json)
    with open(best_params_path, "r", encoding="utf-8") as f:
        best_params = json.load(f)

    # 2) meta: prefer NPZ if exists, else JSON
    users: List[str]
    items: List[str]
    trending_idx: np.ndarray
    config: dict

    if os.path.exists(meta_path) and meta_path.lower().endswith(".npz"):
        meta = np.load(meta_path, allow_pickle=True)
        users = meta["users"].tolist()
        items = meta["items"].tolist()
        trending_idx = meta["trending_idx"].astype(np.int32)
        config = json.loads(str(meta["config_json"]))
    else:
        # Try JSON meta with same basename or known filenames
        # e.g. cfg.stage1_meta_npz == "best_stage1_meta.npz" -> "best_stage1_meta.json"
        cand_json = []
        base_no_ext, _ = os.path.splitext(meta_path)
        cand_json.append(base_no_ext + ".json")
        cand_json.append(os.path.join(run_dir, "best_stage1_meta.json"))
        cand_json.append(os.path.join(run_dir, "stage1_meta.json"))

        meta_json_path = next((p for p in cand_json if os.path.exists(p)), None)
        if meta_json_path is None:
            raise FileNotFoundError(
                f"Stage1 meta not found. Tried: {meta_path} and {cand_json}"
            )

        with open(meta_json_path, "r", encoding="utf-8") as f:
            meta_j = json.load(f)

        # Required keys in JSON meta
        # Expect: users, items, trending_idx, config (or config_json)
        if "users" not in meta_j or "items" not in meta_j or "trending_idx" not in meta_j:
            raise KeyError(
                f"Meta JSON missing keys. Need users/items/trending_idx. Got: {list(meta_j.keys())}"
            )

        users = list(meta_j["users"])
        items = list(meta_j["items"])
        trending_idx = np.asarray(meta_j["trending_idx"], dtype=np.int32)

        if "config" in meta_j and isinstance(meta_j["config"], dict):
            config = meta_j["config"]
        elif "config_json" in meta_j:
            # config_json might be dict or stringified json
            cj = meta_j["config_json"]
            config = cj if isinstance(cj, dict) else json.loads(str(cj))
        else:
            # if you didn't store config in json meta, fall back to best_params as config-ish
            config = {}

    # 3) load implicit models
    tfidf = TFIDFRecommender.load(tfidf_path)
    cosine = CosineRecommender.load(cosine_path)

    return Stage1Artifacts(users, items, trending_idx, config, tfidf, cosine), best_params

def load_min_transactions(cfg: Stage2Config, begin: datetime, end: datetime) -> pl.DataFrame:
    key = _hash_dict({
        "tx_glob": cfg.transactions_path_glob,
        "begin": begin.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "max_rows": int(cfg.max_rows_tx_cache),
    })
    _ensure_dir(cfg.cache_dir)
    cache_path = os.path.join(cfg.cache_dir, f"tx_min_{key}.parquet")
    if os.path.exists(cache_path):
        log(f"[CACHE] Load tx_min: {cache_path}")
        return pl.read_parquet(cache_path)

    log("[1/9] Scan transactions (lazy) ...")
    tx = pl.scan_parquet(cfg.transactions_path_glob)
    cols = set(tx.columns)

    def col_or_lit(name, litval, dtype):
        return pl.col(name).cast(dtype, strict=False) if name in cols else pl.lit(litval, dtype=dtype).alias(name)

    tx = tx.with_columns([
        pl.col("customer_id").cast(pl.Utf8),
        pl.col("item_id").cast(pl.Utf8),
        pl.col("created_date").cast(pl.Date, strict=False),
        col_or_lit("quantity", 1.0, pl.Float64).fill_null(1.0).fill_nan(1.0),
        col_or_lit("price", 0.0, pl.Float64).fill_null(0.0).fill_nan(0.0),
        col_or_lit("channel", "__MISSING__", pl.Utf8),
        col_or_lit("payment", "__MISSING__", pl.Utf8),
        col_or_lit("location", -1, pl.Int64),
        col_or_lit("discount_rate", 0.0, pl.Float64).fill_null(0.0).fill_nan(0.0),
        col_or_lit("category_l2", "__MISSING__", pl.Utf8),
    ]).filter(
        pl.col("created_date").is_between(pl.lit(begin.date(), pl.Date), pl.lit(end.date(), pl.Date), closed="both")
    ).select(["customer_id","item_id","created_date","quantity","price","channel","payment","location","discount_rate", "category_l2"])

    df = tx.collect(streaming=True)
    if cfg.max_rows_tx_cache and cfg.max_rows_tx_cache > 0:
        df = df.head(int(cfg.max_rows_tx_cache))
    log(f"[SAVE] Write tx_min -> {cache_path}")
    df.write_parquet(cache_path)
    return df


def load_items_users(cfg: Stage2Config) -> Tuple[pl.DataFrame, pl.DataFrame]:
    log("[2/9] Load items & users (minimal) ...")
    it = pl.scan_parquet(cfg.items_path_glob)
    us = pl.scan_parquet(cfg.users_path_glob)

    it_df = it.select([
        pl.col("item_id").cast(pl.Utf8),
        pl.col("price").cast(pl.Float64, strict=False),
        pl.col("category_l1").cast(pl.Utf8, strict=False),
        pl.col("category_l2").cast(pl.Utf8, strict=False),
        pl.col("brand").cast(pl.Utf8, strict=False),
        pl.col("item_type").cast(pl.Utf8, strict=False),
        pl.col("gender_target_final").cast(pl.Utf8, strict=False),
        pl.col("age_group_final").cast(pl.Utf8, strict=False),
    ]).collect(streaming=True)

    us_df = us.select([
        pl.col("customer_id").cast(pl.Utf8),
        pl.col("gender").cast(pl.Utf8, strict=False),
        pl.col("province").cast(pl.Utf8, strict=False),
        pl.col("membership").cast(pl.Utf8, strict=False),
        pl.col("install_app").cast(pl.Utf8, strict=False),
        pl.col("install_datetime").cast(pl.Date, strict=False),
    ]).collect(streaming=True)
    return it_df, us_df


def build_stage1_csr(cfg: Stage2Config, df_tx: pl.DataFrame, train_begin: datetime, train_end: datetime,
                     weight_type: str, users_ref: List[str], items_ref: List[str]) -> csr_matrix:
    key = _hash_dict({
        "train_begin": train_begin.strftime("%Y-%m-%d"),
        "train_end": train_end.strftime("%Y-%m-%d"),
        "weight_type": weight_type,
        "n_users": len(users_ref),
        "n_items": len(items_ref),
    })
    path = os.path.join(cfg.cache_dir, f"stage1_user_items_{key}.npz")
    _ensure_dir(cfg.cache_dir)
    if os.path.exists(path):
        log(f"[CACHE] Load stage1 CSR -> {path}")
        return load_npz(path).tocsr()

    log("[3/9] Rebuild Stage-1 CSR for candidate scoring ...")
    d = df_tx.filter(
        pl.col("created_date").is_between(pl.lit(train_begin.date(), pl.Date), pl.lit(train_end.date(), pl.Date), closed="both")
    ).select(["customer_id","item_id","quantity","price"]).with_columns(
        (pl.col("quantity") * pl.col("price")).alias("spent_row")
    )
    ui = d.group_by(["customer_id","item_id"]).agg([
        pl.len().alias("cnt"),
        pl.col("quantity").sum().alias("sum_qty"),
        pl.col("spent_row").sum().alias("sum_spent"),
    ])

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

    u2i = {u: i for i, u in enumerate(users_ref)}
    it2i = {it: j for j, it in enumerate(items_ref)}

    rows, cols, data = [], [], []
    for u, it, v in zip(ui["customer_id"].to_list(), ui["item_id"].to_list(), ui["value"].to_list()):
        u = str(u); it = str(it)
        if u in u2i and it in it2i:
            rows.append(u2i[u]); cols.append(it2i[it]); data.append(float(v))

    mat = csr_matrix((np.array(data, np.float32), (np.array(rows, np.int32), np.array(cols, np.int32))),
                     shape=(len(users_ref), len(items_ref)), dtype=np.float32)
    save_npz(path, mat)
    log(f"[SAVE] Stage-1 CSR -> {path}")
    return mat


def build_user_item_sets(df: pl.DataFrame, begin: datetime, end: datetime) -> Dict[str, set]:
    d = df.filter(
        pl.col("created_date").is_between(pl.lit(begin.date(), pl.Date), pl.lit(end.date(), pl.Date), closed="both")
    ).select(["customer_id","item_id"]).unique()
    out: Dict[str, set] = {}
    for u, it in zip(d["customer_id"].to_list(), d["item_id"].to_list()):
        out.setdefault(str(u), set()).add(str(it))
    return out


def generate_candidates(cfg: Stage2Config, art: Stage1Artifacts, user_ids: List[str], csr: csr_matrix,
                        N_cand: int, N_trend: int) -> pl.DataFrame:
    key = _hash_dict({
        "stage1_run_dir": cfg.stage1_run_dir,
        "n_users": len(user_ids),
        "N_cand": N_cand,
        "N_trend": N_trend,
        "train_end": cfg.train_end,
    })
    path = os.path.join(cfg.cache_dir, f"candidates_{key}.parquet")
    if os.path.exists(path):
        log(f"[CACHE] Load candidates -> {path}")
        return pl.read_parquet(path)

    log("[4/9] Generate candidates with retrieval features ...")
    u2i = {u: i for i, u in enumerate(art.users)}
    items = art.items
    trending = art.trending_idx[:int(N_trend)].tolist()
    trend_rank = {int(ix): r+1 for r, ix in enumerate(trending)}

    rows = []
    for u in tqdm(user_ids, desc="Candidates"):
        u = str(u)
        if u not in u2i or csr[u2i[u]].nnz == 0:
            for ix in trending:
                rows.append((u, items[int(ix)], 0.0, 0.0, 9999, 9999, trend_rank[int(ix)], 0, 0, 1))
            continue

        uid = u2i[u]
        row = csr[uid]

        ids_t, scores_t = art.tfidf.recommend(uid, row, N=int(N_cand), filter_already_liked_items=False)
        ids_c, scores_c = art.cosine.recommend(uid, row, N=int(N_cand), filter_already_liked_items=False)

        ids_t = ids_t.tolist() if hasattr(ids_t, "tolist") else list(ids_t)
        scores_t = scores_t.tolist() if hasattr(scores_t, "tolist") else list(scores_t)
        tf_rank = {int(ix): r+1 for r, ix in enumerate(ids_t)}
        tf_score = {int(ix): float(sc) for ix, sc in zip(ids_t, scores_t)}

        ids_c = ids_c.tolist() if hasattr(ids_c, "tolist") else list(ids_c)
        scores_c = scores_c.tolist() if hasattr(scores_c, "tolist") else list(scores_c)
        co_rank = {int(ix): r+1 for r, ix in enumerate(ids_c)}
        co_score = {int(ix): float(sc) for ix, sc in zip(ids_c, scores_c)}

        merged = ids_t + ids_c + trending
        seen = set()
        merged_uniq = []
        for ix in merged:
            ix = int(ix)
            if ix not in seen:
                seen.add(ix)
                merged_uniq.append(ix)
            if len(merged_uniq) >= (2*int(N_cand) + int(N_trend)):
                break

        for ix in merged_uniq:
            rows.append((
                u,
                items[ix],
                float(tf_score.get(ix, 0.0)),
                float(co_score.get(ix, 0.0)),
                int(tf_rank.get(ix, 9999)),
                int(co_rank.get(ix, 9999)),
                int(trend_rank.get(ix, 9999)),
                1 if ix in tf_rank else 0,
                1 if ix in co_rank else 0,
                1 if ix in trend_rank else 0
            ))

    df = pl.DataFrame(rows, schema=[
        ("customer_id", pl.Utf8),
        ("item_id", pl.Utf8),
        ("tfidf_score", pl.Float32),
        ("cosine_score", pl.Float32),
        ("rank_tfidf", pl.Int32),
        ("rank_cosine", pl.Int32),
        ("rank_trending", pl.Int32),
        ("is_from_tfidf", pl.Int8),
        ("is_from_cosine", pl.Int8),
        ("is_from_trending", pl.Int8),
    ])
    df.write_parquet(path)
    log(f"[SAVE] Candidates -> {path}")
    return df


def load_engineered_features(cfg: Stage2Config) -> Dict[str, pl.DataFrame]:
    log("[6/9] Load engineered feature tables ...")
    out: Dict[str, pl.DataFrame] = {}

    def maybe(path: Optional[str]) -> Optional[pl.DataFrame]:
        if not path or not os.path.exists(path):
            if path:
                log(f"[WARN] Missing: {path}")
            return None
        return pl.read_parquet(path)

    bs = maybe(cfg.brand_segment_path)
    if bs is not None:
        out["brand_segment"] = bs.select([pl.col("category_l1").cast(pl.Utf8), pl.col("brand_segment").cast(pl.Int64, strict=False)]).unique()

    age = maybe(cfg.customer_age_features_path)
    if age is not None:
        out["age_final"] = age.select([pl.col("customer_id").cast(pl.Utf8), pl.col("age_final").cast(pl.Float64, strict=False)]).unique()

    beh = maybe(cfg.customer_behavior_path)
    if beh is not None:
        out["buy_segment"] = beh.select([pl.col("customer_id").cast(pl.Utf8), pl.col("buy_segment").cast(pl.Int32, strict=False)]).unique()

    lux = maybe(cfg.customer_luxury_path)
    if lux is not None:
        out["luxury_level"] = lux.select([pl.col("customer_id").cast(pl.Utf8), pl.col("luxury_level").cast(pl.Utf8, strict=False)]).unique()

    ps = maybe(cfg.price_segment_path)
    if ps is not None:
        out["price_segment"] = ps.select([pl.col("item_id").cast(pl.Utf8), pl.col("price_segment").cast(pl.Utf8, strict=False)]).unique()

    topm = maybe(cfg.top10_by_cat_month_path)
    if topm is not None:
        out["top10_by_cat_month"] = topm.select([
            pl.col("month").cast(pl.Utf8),
            pl.col("category_l1").cast(pl.Utf8),
            pl.col("item_id").cast(pl.Utf8),
            pl.col("rank").cast(pl.Int32, strict=False),
        ])
    return out


def build_hist_aggs(cfg: Stage2Config, df_tx: pl.DataFrame, items_df: pl.DataFrame,
                    train_begin: datetime, train_end: datetime) -> pl.DataFrame:
    """Build minimal HIST aggregates from transactions, per (customer_id, item_id).

    Only keep transaction-derived features requested:
      - quantity (sum over HIST)
      - location (most recent location)
      - category_l2 (most recent tx category_l2)
    """
    key = _hash_dict({"train_begin": train_begin.strftime("%Y-%m-%d"), "train_end": train_end.strftime("%Y-%m-%d"), "v": "min_tx"})
    path = os.path.join(cfg.cache_dir, f"hist_ui_min_{key}.parquet")
    if os.path.exists(path):
        log(f"[CACHE] Load hist ui min -> {path}")
        return pl.read_parquet(path)

    log("[5/9] Build HIST minimal aggregates (ui) ...")
    hist = df_tx.filter(
        pl.col("created_date").is_between(pl.lit(train_begin.date(), pl.Date), pl.lit(train_end.date(), pl.Date), closed="both")
    )

    # Ensure required cols exist
    cols = set(hist.columns)
    if "location" not in cols:
        hist = hist.with_columns(pl.lit(-1).cast(pl.Int64).alias("location"))
    if "category_l2" not in cols:
        hist = hist.with_columns(pl.lit("__MISSING__").cast(pl.Utf8).alias("category_l2"))
    if "quantity" not in cols:
        hist = hist.with_columns(pl.lit(0.0).cast(pl.Float64).alias("quantity"))

    # last observation per (u,i)
    last = (
        hist.sort(["customer_id", "item_id", "created_date"])
            .group_by(["customer_id", "item_id"], maintain_order=True)
            .tail(1)
            .select(["customer_id", "item_id",
                     pl.col("location").alias("ui_last_location"),
                     pl.col("category_l2").alias("ui_last_tx_category_l2"),
                     pl.col("created_date").alias("ui_last_date")])
    )

    ui = hist.group_by(["customer_id","item_id"]).agg([
        pl.col("quantity").sum().alias("ui_sum_qty"),
    ]).join(last, on=["customer_id","item_id"], how="left")

    end_date = train_end.date()
    ui = ui.with_columns(
        ((pl.lit(end_date) - pl.col("ui_last_date")).dt.total_days().cast(pl.Int32)).alias("ui_recency_days")
    ).drop("ui_last_date")

    ui.write_parquet(path)
    log(f"[SAVE] Hist ui min -> {path}")
    return ui

def build_feature_label(cfg: Stage2Config, candidates: pl.DataFrame, df_tx: pl.DataFrame,
                        items_df: pl.DataFrame, users_df: pl.DataFrame,
                        ui_agg: pl.DataFrame,
                        feat_tables: Dict[str, pl.DataFrame],
                        train_begin: datetime, train_end: datetime,
                        recent_begin: datetime, recent_end: datetime) -> pl.DataFrame:
    key = _hash_dict({
        "train_end": train_end.strftime("%Y-%m-%d"),
        "recent_begin": recent_begin.strftime("%Y-%m-%d"),
        "recent_end": recent_end.strftime("%Y-%m-%d"),
        "N_neg": cfg.N_neg,
        "lag": cfg.top10_month_lag,
    })
    run_dir = os.path.join(cfg.out_dir, cfg.run_name)
    _ensure_dir(run_dir)
    sampled_path = os.path.join(run_dir, f"feature_label_sampled_{key}.parquet")
    if os.path.exists(sampled_path):
        log(f"[CACHE] Load feature_label_sampled -> {sampled_path}")
        return pl.read_parquet(sampled_path)

    log("[7/9] Build feature-label (joins + label + N_neg sampling) ...")
    # Reduce join payload: select only needed columns
    items_small = items_df.select(["item_id","category_l1","category_l2"])
    users_small = users_df.select(["customer_id","province","membership"])
    fl = safe_join(candidates, items_small, on=["item_id"], how="left", suffix="_item")
    fl = safe_join(fl, users_small, on=["customer_id"], how="left", suffix="_user")


    if "price_segment" in feat_tables:
        fl = fl.join(feat_tables["price_segment"], on="item_id", how="left")
    if "buy_segment" in feat_tables:
        fl = fl.join(feat_tables["buy_segment"], on="customer_id", how="left")
    if "luxury_level" in feat_tables:
        fl = fl.join(feat_tables["luxury_level"], on="customer_id", how="left")
    if "age_final" in feat_tables:
        fl = fl.join(feat_tables["age_final"], on="customer_id", how="left")
    if "brand_segment" in feat_tables:
        fl = fl.join(feat_tables["brand_segment"], on="category_l1", how="left")

    if "top10_by_cat_month" in feat_tables:
        mdt = _shift_months(train_end, -int(cfg.top10_month_lag))
        mkey = _month_key(mdt)
        topm = feat_tables["top10_by_cat_month"].with_columns(pl.col("month").alias("month_key"))
        fl = fl.with_columns(pl.lit(mkey).alias("trend_month"))
        fl = safe_join(
            fl,
            topm.select(["month_key","category_l1","item_id","rank"]),
            left_on=["trend_month","category_l1","item_id"],
            right_on=["month_key","category_l1","item_id"],
            how="left",
            suffix="_topm"
        ).rename({"rank":"rank_top10_by_cat_month"}).with_columns(pl.col("rank_top10_by_cat_month").fill_null(9999).cast(pl.Int32))
    else:
        fl = fl.with_columns(pl.lit(9999).alias("rank_top10_by_cat_month"))

    fl = safe_join(fl, ui_agg, on=["customer_id","item_id"], how="left", suffix="_ui")

    recent_pairs = df_tx.filter(
        pl.col("created_date").is_between(pl.lit(recent_begin.date(), pl.Date), pl.lit(recent_end.date(), pl.Date), closed="both")
    ).select(["customer_id","item_id"]).unique().with_columns(pl.lit(1, pl.Int8).alias("Y"))
    fl = fl.join(recent_pairs, on=["customer_id","item_id"], how="left").with_columns(pl.col("Y").fill_null(0).cast(pl.Int8))

    # negative sampling per user (FAST: Polars group sampling)
    #  - keep all positives
    #  - sample N_neg negatives per user from that user's candidate negatives
    log(f"[7/9] Negative sampling (polars) per user: N_neg={cfg.N_neg}")
    pos = fl.filter(pl.col("Y") == 1)

    if cfg.N_neg <= 0:
        fl_s = pos
    else:
        # shuffle within each user using random key, then take head(N_neg) per user
        neg = (
            fl.filter(pl.col("Y") == 0)
              .with_columns(pl.concat_str([pl.col("customer_id"), pl.col("item_id"), pl.lit(str(cfg.random_state))]).hash().alias("_r"))
              .sort(["customer_id", "_r"])
              .group_by("customer_id", maintain_order=True)
              .head(int(cfg.N_neg))
              .drop("_r")
        )
        fl_s = pl.concat([pos, neg], how="vertical")

    fl_s.write_parquet(sampled_path)
    log(f"[SAVE] Feature-label sampled -> {sampled_path}")
    return fl_s


def train_model(cfg: Stage2Config, fl: pl.DataFrame) -> Any:
    log("[8/9] Train ranking model (LightGBM) ...")
    if LGBMClassifier is None:
        raise ImportError("lightgbm is not available. Please install lightgbm in your environment.")

    key_cols = {"customer_id", "item_id", "Y", "install_datetime", "trend_month"}
    feat_cols = [c for c in fl.columns if c not in key_cols]

    numeric_cols, cat_cols = [], []
    for c in feat_cols:
        dt = fl.schema[c]
        if dt in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64, pl.Boolean):
            numeric_cols.append(c)
        else:
            cat_cols.append(c)

    pbar = tqdm(total=3, desc="Stage2 train (lgbm)", leave=True)
    dfp = fl.select(["Y"] + feat_cols).to_pandas()
    pbar.update(1)

    y = dfp["Y"].astype(int).values
    X = dfp.drop(columns=["Y"])

    for c in numeric_cols:
        if c in X.columns:
            X[c] = X[c].astype("float32")
            X[c] = X[c].fillna(0.0)

    X, cat_maps = encode_categoricals_with_maps(X, cat_cols, cat_maps=None)
    pbar.update(1)

    test_size = float(getattr(cfg, "lgbm_test_size", 0.1))
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=test_size,
        random_state=int(getattr(cfg, "lgbm_random_state", cfg.random_state)),
        stratify=y
    )

    n_pos = max(1, int((y_tr == 1).sum()))
    n_neg = max(1, int((y_tr == 0).sum()))
    scale_pos_weight = n_neg / n_pos

    model = LGBMClassifier(
        objective=str(getattr(cfg, "lgbm_objective", "binary")),
        learning_rate=float(getattr(cfg, "lgbm_learning_rate", 0.05)),
        n_estimators=int(getattr(cfg, "lgbm_n_estimators", 500)),
        num_leaves=int(getattr(cfg, "lgbm_num_leaves", 63)),
        max_depth=int(getattr(cfg, "lgbm_max_depth", -1)),
        min_child_samples=int(getattr(cfg, "lgbm_min_child_samples", 50)),
        subsample=float(getattr(cfg, "lgbm_subsample", 0.8)),
        colsample_bytree=float(getattr(cfg, "lgbm_colsample_bytree", 0.8)),
        reg_alpha=float(getattr(cfg, "lgbm_reg_alpha", 0.0)),
        reg_lambda=float(getattr(cfg, "lgbm_reg_lambda", 0.0)),
        random_state=int(getattr(cfg, "lgbm_random_state", cfg.random_state)),
        n_jobs=-1,
        scale_pos_weight=float(scale_pos_weight),
    )

    try:
        import lightgbm as lgb
        callbacks = [lgb.early_stopping(int(getattr(cfg, "lgbm_early_stopping_rounds", 50)), verbose=False)]
    except Exception:
        callbacks = []

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="binary_logloss",
        callbacks=callbacks,
    )
    pbar.update(1)
    pbar.close()

    return LGBMBundle(
        model=model,
        feat_cols=feat_cols,
        numeric_cols=numeric_cols,
        cat_cols=cat_cols,
        cat_maps=cat_maps,
    )

def dcg_at_k(rels: List[int], k: int) -> float:
    s = 0.0
    for i, r in enumerate(rels[:k], start=1):
        if r:
            s += 1.0 / math.log2(i + 1)
    return s


def ndcg_at_k(pred: List[str], gt: set, k: int) -> float:
    rels = [1 if it in gt else 0 for it in pred[:k]]
    dcg = dcg_at_k(rels, k)
    idcg = dcg_at_k([1]*min(len(gt), k), k)
    return (dcg/idcg) if idcg>0 else 0.0


def precision_at_k(pred: List[str], gt: set, k: int) -> float:
    return len(set(pred[:k]) & gt) / float(k) if k>0 else 0.0


def recall_at_k(pred: List[str], gt: set, k: int) -> float:
    return len(set(pred[:k]) & gt) / float(len(gt)) if gt else 0.0


def hit_at_k(pred: List[str], gt: set, k: int) -> float:
    return 1.0 if (set(pred[:k]) & gt) else 0.0

def precision_at_k_userwise_batched(pred: Dict[str, List[str]],
                                     gt: Dict[str, set],
                                     hist: Dict[str, set],
                                     *,
                                     filter_bought_items: bool = True,
                                     K: int = 10,
                                     batch_users: int = 5000,
                                     show_progress: bool = True) -> Tuple[float, int, List[str]]:
    """User-wise Precision@K with cold-start reporting, computed in batches with tqdm."""
    users = list(gt.keys())
    n = len(users)
    precisions: List[float] = []
    cold_start_users: List[str] = []

    step = max(1, int(batch_users))
    starts = range(0, n, step)
    if show_progress:
        starts = tqdm(list(starts), desc=f"Precision@{K} ({'FILTERED' if filter_bought_items else 'UNFILTERED'})", leave=False)

    for start in starts:
        end = min(start + step, n)
        for u in users[start:end]:
            if (u not in hist) or (u not in pred):
                cold_start_users.append(u)
                continue

            relevant = set(gt[u])
            if filter_bought_items:
                relevant -= set(hist[u])

            hits = len(set(pred[u][:K]) & relevant)
            precisions.append(hits / float(K))

    mean_prec = float(sum(precisions) / len(precisions)) if precisions else 0.0
    return mean_prec, len(cold_start_users), cold_start_users


def score_topk(cfg: Stage2Config, model: Any, feature_table: pl.DataFrame, users: List[str]) -> Dict[str, List[str]]:
    """Score all (user,item) rows in feature_table and return top-k items per user.
    Works with LGBMBundle (recommended) or a raw sklearn-like model.
    """
    topk = int(cfg.topk)

    # Bundle handling (ensures consistent feature schema and categorical encoding)
    bundle = model if isinstance(model, LGBMBundle) else None
    booster = bundle.model if bundle is not None else model

    key_cols = ["customer_id", "item_id"]
    drop_cols = {"Y", "install_datetime", "trend_month"}

    if bundle is not None:
        feat_cols = list(bundle.feat_cols)
    else:
        feat_cols = [c for c in feature_table.columns if c not in set(key_cols) and c not in drop_cols]

    # Ensure all feat_cols exist in feature_table (if missing, create with nulls)
    missing = [c for c in feat_cols if c not in feature_table.columns]
    if missing:
        feature_table = feature_table.with_columns([pl.lit(None).alias(c) for c in missing])

    dfp = feature_table.select(key_cols + feat_cols).to_pandas()
    X = dfp[feat_cols]

    if bundle is not None:
        # Encode categoricals using training maps (unknown -> -1)
        X, _ = encode_categoricals_with_maps(X, bundle.cat_cols, cat_maps=bundle.cat_maps)
        # Numeric casting/fill
        for c in bundle.numeric_cols:
            if c in X.columns:
                X[c] = X[c].astype("float32")
                X[c] = X[c].fillna(0.0)
        scores = booster.predict_proba(X)[:, 1]
    else:
        scores = (booster.predict_proba(X)[:, 1] if hasattr(booster, "predict_proba") else booster.predict(X))

    cust = dfp["customer_id"].astype(str).values
    item = dfp["item_id"].astype(str).values

    # Sort by (customer_id asc, score desc)
    order = np.lexsort((-scores, cust))

    out: Dict[str, List[str]] = {}
    cur_u: Optional[str] = None
    buf: List[str] = []
    for u, it in zip(cust[order], item[order]):
        if cur_u is None:
            cur_u = u
        if u != cur_u:
            out[cur_u] = buf[:topk]
            cur_u = u
            buf = []
        if len(buf) < topk:
            buf.append(it)
    if cur_u is not None:
        out[cur_u] = buf[:topk]

    # Ensure all requested users appear
    for u in users:
        out.setdefault(str(u), [])
    return out


def eval_metrics(cfg: Stage2Config, pred: Dict[str, List[str]], gt_recent: Dict[str, set],
                 hist: Dict[str, set], mode: str) -> dict:
    k = int(cfg.topk)
    users = list(gt_recent.keys())
    n_total = len(users)

    precs, recs, hits, ndcgs = [], [], [], []
    n = 0

    bs = max(1, int(getattr(cfg, "batch_users", 5000)))
    idxs = list(range(0, n_total, bs))
    for start in tqdm(idxs, desc=f"Eval metrics ({mode})", leave=False):
        end = min(start + bs, n_total)
        for u in users[start:end]:
            gt_set = set(gt_recent[u])
            p = pred.get(u, [])

            if mode == "filtered":
                h = hist.get(u, set())
                gt_set = gt_set - h
                if p:
                    p = [it for it in p if it not in h]

            if not gt_set:
                continue

            n += 1
            precs.append(precision_at_k(p, gt_set, k))
            recs.append(recall_at_k(p, gt_set, k))
            hits.append(hit_at_k(p, gt_set, k))
            ndcgs.append(ndcg_at_k(p, gt_set, k))

    if n == 0:
        return {"n_users": 0, "precision": 0.0, "recall": 0.0, "hit": 0.0, "ndcg": 0.0, "k": k}

    return {
        "n_users": int(n),
        "precision": float(sum(precs) / len(precs)),
        "recall": float(sum(recs) / len(recs)),
        "hit": float(sum(hits) / len(hits)),
        "ndcg": float(sum(ndcgs) / len(ndcgs)),
        "k": k,
    }


def eval_private(cfg: Stage2Config,
                 model: Any,
                 feature_table: pl.DataFrame,
                 hist_filter: Optional[Dict[str, set]] = None) -> Optional[dict]:
    """Evaluate on private test groundtruth.pkl.
    Reports BOTH FILTERED and UNFILTERED metrics (Precision@K, NDCG@K) and cold-start users.
    FILTERED means: remove items already bought in hist_filter from both GT and predictions.
    """
    if not cfg.groundtruth_pkl or not cfg.evaluate_private_test:
        return None
    if not os.path.exists(cfg.groundtruth_pkl):
        log(f"[WARN] groundtruth_pkl not found: {cfg.groundtruth_pkl}")
        return None

    gt_raw = pickle.load(open(cfg.groundtruth_pkl, "rb"))
    gt: Dict[str, set] = {}
    if isinstance(gt_raw, dict):
        for u, items in gt_raw.items():
            if isinstance(items, (list, set, tuple)):
                gt[str(u)] = set(map(str, items))
    elif isinstance(gt_raw, list):
        for row in gt_raw:
            u = str(row.get("customer_id"))
            items = row.get("item_id")
            if isinstance(items, (list, set, tuple)):
                gt[u] = set(map(str, items))
    else:
        log("[WARN] Unknown groundtruth.pkl format; skip.")
        return None

    users = list(gt.keys())
    if cfg.max_users_eval and cfg.max_users_eval > 0:
        users = users[:int(cfg.max_users_eval)]
        gt = {u: gt[u] for u in users}

    # Score for private users
    ft = feature_table.filter(pl.col("customer_id").is_in(users))
    # Preview feature table
    try:
        log("[PREVIEW] Feature table (first 5 rows):")
        print(ft.head(5), flush=True)
    except Exception as e:
        log(f"[WARN] Could not preview feature table: {e}")

    pred = score_topk(cfg, model, ft, users)

    k = int(cfg.topk)
    hist_filter = hist_filter or {}

    # UNFILTERED metrics
    prec_u, ndcg_u = [], []
    n_u = 0
    for u, gt_set in gt.items():
        p = pred.get(u, [])
        if not gt_set:
            continue
        n_u += 1
        prec_u.append(precision_at_k(p, gt_set, k))
        ndcg_u.append(ndcg_at_k(p, gt_set, k))
    unfiltered = {
        "n_users": int(n_u),
        "precision": float(np.mean(prec_u)) if prec_u else 0.0,
        "ndcg": float(np.mean(ndcg_u)) if ndcg_u else 0.0,
        "k": k,
    }

    # FILTERED metrics (remove already-bought items from GT and predictions)
    prec_f, ndcg_f = [], []
    n_f = 0
    for u, gt_set in gt.items():
        h = hist_filter.get(u, set())
        gt_f = set(gt_set) - set(h)
        if not gt_f:
            continue
        p = pred.get(u, [])
        if p:
            p = [it for it in p if it not in h]
        n_f += 1
        prec_f.append(precision_at_k(p, gt_f, k))
        ndcg_f.append(ndcg_at_k(p, gt_f, k))
    filtered = {
        "n_users": int(n_f),
        "precision": float(np.mean(prec_f)) if prec_f else 0.0,
        "ndcg": float(np.mean(ndcg_f)) if ndcg_f else 0.0,
        "k": k,
    }

    # Custom user-wise precision@k + cold-start counts
    prec_f_custom, ncs_f, cold_f = precision_at_k_userwise_batched(
        pred=pred, gt=gt, hist=hist_filter,
        filter_bought_items=True, K=k,
        batch_users=int(getattr(cfg, "batch_users", 5000)),
        show_progress=True
    )
    prec_u_custom, ncs_u, cold_u = precision_at_k_userwise_batched(
        pred=pred, gt=gt, hist=hist_filter,
        filter_bought_items=False, K=k,
        batch_users=int(getattr(cfg, "batch_users", 5000)),
        show_progress=True
    )

    return {
        "filtered": filtered,
        "unfiltered": unfiltered,
        "precision_custom_filtered": float(prec_f_custom),
        "precision_custom_unfiltered": float(prec_u_custom),
        "cold_start_users_filtered": int(ncs_f),
        "cold_start_users_unfiltered": int(ncs_u),
        "cold_start_user_ids_filtered": cold_f[:200],
        "cold_start_user_ids_unfiltered": cold_u[:200],
    }



def main(config_path: str) -> None:
    cfg = load_config(config_path)
    _ensure_dir(cfg.out_dir); _ensure_dir(cfg.cache_dir)
    run_dir = os.path.join(cfg.out_dir, cfg.run_name); _ensure_dir(run_dir)

    log("==== Stage2 Ranking (LightGBM) ====")
    log(f"Config: {config_path}")
    log(f"Run dir: {run_dir}")

    art, best = load_stage1_artifacts(cfg)

    len_hist = int(best.get("len_hist", art.config.get("len_hist", 120)))
    len_recent = int(best.get("len_recent", art.config.get("len_recent", 28)))
    N_cand = int(cfg.N_cand) if cfg.N_cand is not None else int(best.get("N_cand", art.config.get("N_cand", 100)))
    N_trend = int(cfg.N_trend) if cfg.N_trend is not None else int(best.get("N_trend", art.config.get("N_trend", 100)))
    weight_type = str(best.get("weight_type", art.config.get("weight_type", "log_count")))

    train_end = _parse_date(cfg.train_end)
    recent_begin = _parse_date(cfg.recent_begin)
    train_begin = train_end - timedelta(days=len_hist - 1)
    recent_end = recent_begin + timedelta(days=len_recent - 1)

    log(f"Anchors: train_begin={train_begin.date()} -> train_end={train_end.date()} | recent_begin={recent_begin.date()} -> recent_end={recent_end.date()}")
    log(f"Stage1 params: len_hist={len_hist}, len_recent={len_recent}, N_cand={N_cand}, N_trend={N_trend}, weight_type={weight_type}")
    log(f"Stage2: N_neg={cfg.N_neg}, topk={cfg.topk}")

    df_tx = load_min_transactions(cfg, train_begin, recent_end)
    items_df, users_df = load_items_users(cfg)

    gt_recent = build_user_item_sets(df_tx, recent_begin, recent_end)
    users = list(gt_recent.keys())
    if cfg.max_users_train and cfg.max_users_train > 0:
        users = users[:int(cfg.max_users_train)]
        gt_recent = {u: gt_recent[u] for u in users}
    log(f"[INFO] Users in RECENT (for training/eval): {len(users):,}")

    csr = build_stage1_csr(cfg, df_tx, train_begin, train_end, weight_type, art.users, art.items)
    candidates = generate_candidates(cfg, art, users, csr, N_cand, N_trend)

    feat_tables = load_engineered_features(cfg)
    ui_agg = build_hist_aggs(cfg, df_tx, items_df, train_begin, train_end)

    fl = build_feature_label(cfg, candidates, df_tx, items_df, users_df, ui_agg,
                             feat_tables, train_begin, train_end, recent_begin, recent_end)

    model = train_model(cfg, fl)
    model_path = os.path.join(run_dir, "stage2_model_lgbm.joblib")
    joblib.dump(model, model_path)
    log(f"[SAVE] Model -> {model_path}")

    # Build full feature table for scoring/eval (no sampling)
    full_key = _hash_dict({"train_end": cfg.train_end, "recent_begin": cfg.recent_begin, "len_recent": len_recent, "N_cand": N_cand, "N_trend": N_trend, "lag": cfg.top10_month_lag})
    full_path = os.path.join(run_dir, f"feature_table_full_{full_key}.parquet")
    if os.path.exists(full_path):
        log(f"[CACHE] Load full feature table -> {full_path}")
        ft = pl.read_parquet(full_path)
    else:
        log("[EVAL] Build full feature table ...")
        items_small = items_df.select(["item_id","category_l1","category_l2"])
        users_small = users_df.select(["customer_id","province","membership"])
        ft = safe_join(candidates, items_small, on=["item_id"], how="left", suffix="_item")
        ft = safe_join(ft, users_small, on=["customer_id"], how="left", suffix="_user")
        if "price_segment" in feat_tables: ft = ft.join(feat_tables["price_segment"], on="item_id", how="left")
        if "buy_segment" in feat_tables: ft = ft.join(feat_tables["buy_segment"], on="customer_id", how="left")
        if "luxury_level" in feat_tables: ft = ft.join(feat_tables["luxury_level"], on="customer_id", how="left")
        if "age_final" in feat_tables: ft = ft.join(feat_tables["age_final"], on="customer_id", how="left")
        if "brand_segment" in feat_tables: ft = ft.join(feat_tables["brand_segment"], on="category_l1", how="left")
        if "top10_by_cat_month" in feat_tables:
            mdt = _shift_months(train_end, -int(cfg.top10_month_lag))
            mkey = _month_key(mdt)
            topm = feat_tables["top10_by_cat_month"].with_columns(pl.col("month").alias("month_key"))
            ft = ft.with_columns(pl.lit(mkey).alias("trend_month"))
            ft = safe_join(
                ft,
                topm.select(["month_key","category_l1","item_id","rank"]),
                left_on=["trend_month","category_l1","item_id"],
                right_on=["month_key","category_l1","item_id"],
                how="left",
                suffix="_topm"
            ).rename({"rank":"rank_top10_by_cat_month"}).with_columns(pl.col("rank_top10_by_cat_month").fill_null(9999).cast(pl.Int32))
        else:
            ft = ft.with_columns(pl.lit(9999).alias("rank_top10_by_cat_month"))

        ft = safe_join(ft, ui_agg, on=["customer_id","item_id"], how="left", suffix="_ui")

        ft.write_parquet(full_path)
        log(f"[SAVE] Full feature table -> {full_path}")

    # Preview feature table
    try:
        log("[PREVIEW] Feature table (first 5 rows):")
        print(ft.head(5), flush=True)
    except Exception as e:
        log(f"[WARN] Could not preview feature table: {e}")

    pred = score_topk(cfg, model, ft, users)

    # Build FILTER history over full year (or configured range) for FILTERED metrics
    filter_begin = _parse_date(cfg.filter_hist_begin)
    filter_end = _parse_date(cfg.filter_hist_end)
    df_tx_filter = load_min_transactions(cfg, filter_begin, filter_end)
    hist_year = build_user_item_sets(df_tx_filter, filter_begin, filter_end)
    hist = hist_year  # backward-compat
    # NOTE: RECENT is used to create training labels; do not treat RECENT metrics as the primary objective.

    m_f = eval_metrics(cfg, pred, gt_recent, hist_year, "filtered")
    m_u = eval_metrics(cfg, pred, gt_recent, hist_year, "unfiltered")

    metrics = {"offline_filtered": m_f, "offline_unfiltered": m_u}
    log("===== OFFLINE METRICS (RECENT) =====")
    log(f"FILTERED   P@{cfg.topk}={m_f['precision']:.6f}  R@{cfg.topk}={m_f['recall']:.6f}  Hit@{cfg.topk}={m_f['hit']:.6f}  NDCG@{cfg.topk}={m_f['ndcg']:.6f}  n_users={m_f['n_users']:,}")
    log(f"UNFILTERED P@{cfg.topk}={m_u['precision']:.6f}  R@{cfg.topk}={m_u['recall']:.6f}  Hit@{cfg.topk}={m_u['hit']:.6f}  NDCG@{cfg.topk}={m_u['ndcg']:.6f}  n_users={m_u['n_users']:,}")

    priv = eval_private(cfg, model, ft, hist_year)
    if priv is not None:
        metrics["private_test"] = priv
        log("===== PRIVATE TEST (groundtruth.pkl) =====")
        log(f"FILTERED   P@{cfg.topk}={priv['filtered']['precision']:.6f}  NDCG@{cfg.topk}={priv['filtered']['ndcg']:.6f}  n_users={priv['filtered']['n_users']:,}  cold_start={priv['cold_start_users_filtered']}")
        log(f"UNFILTERED P@{cfg.topk}={priv['unfiltered']['precision']:.6f}  NDCG@{cfg.topk}={priv['unfiltered']['ndcg']:.6f}  n_users={priv['unfiltered']['n_users']:,}  cold_start={priv['cold_start_users_unfiltered']}")
        log(f"[CUSTOM PRECISION@{cfg.topk}] FILTERED={priv['precision_custom_filtered']:.6f} | cold-start users={priv['cold_start_users_filtered']}")
        log(f"[CUSTOM PRECISION@{cfg.topk}] UNFILTERED={priv['precision_custom_unfiltered']:.6f} | cold-start users={priv['cold_start_users_unfiltered']}")

    metrics_path = os.path.join(run_dir, "stage2_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    log(f"[SAVE] Metrics -> {metrics_path}")

    log("DONE: Stage2 ranking pipeline complete.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    main(args.config)
