# -*- coding: utf-8 -*-
"""
stage2_rank_svm_lasso.py

Stage-2 Ranking for a 2-stage recommender:
  Stage-1: implicit TFIDFRecommender + CosineRecommender + Trending backfill (already trained)
  Stage-2: Feature augmentation + Pointwise ranking model:
           L1 feature selection ("Lasso" via LogisticRegression L1) + Linear SVM ranker

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
  python stage2_rank_svm_lasso.py --config stage2_full_config.json
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
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
import joblib


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

    N_cand: Optional[int] = None
    N_trend: Optional[int] = None

    N_neg: int = 10
    random_state: int = 42

    top10_month_lag: int = 1

    out_dir: str = "./artifacts_stage2"
    run_name: str = "run"
    cache_dir: str = "./artifacts_stage2/cache"
    save_intermediate: bool = True

    batch_users: int = 100000
    max_users_train: int = 0
    max_users_eval: int = 0
    max_rows_tx_cache: int = 0

    l1_C: float = 0.3
    svm_C: float = 1.0
    class_weight: str = "balanced"
    max_iter_l1: int = 3000

    groundtruth_pkl: Optional[str] = None
    evaluate_private_test: bool = True

    topk: int = 10


def load_config(path: str) -> Stage2Config:
    with open(path, "r", encoding="utf-8") as f:
        return Stage2Config(**json.load(f))


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _hash_dict(d: dict) -> str:
    s = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]


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
    meta_path = os.path.join(run_dir, cfg.stage1_meta_npz)
    tfidf_path = os.path.join(run_dir, cfg.stage1_tfidf_npz)
    cosine_path = os.path.join(run_dir, cfg.stage1_cosine_npz)

    best_params = json.load(open(best_params_path, "r", encoding="utf-8"))
    meta = np.load(meta_path, allow_pickle=True)
    users = meta["users"].tolist()
    items = meta["items"].tolist()
    trending_idx = meta["trending_idx"].astype(np.int32)
    config = json.loads(str(meta["config_json"]))

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
    ]).filter(
        pl.col("created_date").is_between(pl.lit(begin.date(), pl.Date), pl.lit(end.date(), pl.Date), closed="both")
    ).select(["customer_id","item_id","created_date","quantity","price","channel","payment","location","discount_rate"])

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
                    train_begin: datetime, train_end: datetime) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    key = _hash_dict({"train_begin": train_begin.strftime("%Y-%m-%d"), "train_end": train_end.strftime("%Y-%m-%d")})
    path = os.path.join(cfg.cache_dir, f"hist_aggs_{key}.parquet")
    if os.path.exists(path):
        log(f"[CACHE] Load hist aggs -> {path}")
        aggs = pl.read_parquet(path)
        ui = aggs.filter(pl.col("_k")=="ui").drop("_k")
        u = aggs.filter(pl.col("_k")=="u").drop("_k")
        it = aggs.filter(pl.col("_k")=="i").drop("_k")
        uc = aggs.filter(pl.col("_k")=="uc").drop("_k")
        return ui, u, it, uc

    log("[5/9] Build HIST aggregates ...")
    hist = df_tx.filter(
        pl.col("created_date").is_between(pl.lit(train_begin.date(), pl.Date), pl.lit(train_end.date(), pl.Date), closed="both")
    ).with_columns((pl.col("quantity") * pl.col("price")).alias("spent_row")) \
     .join(items_df.select(["item_id","category_l1","brand"]), on="item_id", how="left")

    end_date = train_end.date()

    ui = hist.group_by(["customer_id","item_id"]).agg([
        pl.len().alias("ui_cnt"),
        pl.col("quantity").sum().alias("ui_sum_qty"),
        pl.col("spent_row").sum().alias("ui_sum_spent"),
        pl.col("created_date").max().alias("ui_last_date"),
    ]).with_columns(((pl.lit(end_date)-pl.col("ui_last_date")).dt.total_days().cast(pl.Int32)).alias("ui_recency_days")).drop("ui_last_date")

    u = hist.group_by(["customer_id"]).agg([
        pl.len().alias("u_cnt"),
        pl.col("quantity").sum().alias("u_sum_qty"),
        pl.col("spent_row").sum().alias("u_sum_spent"),
        pl.col("item_id").n_unique().alias("u_n_unique_items"),
        pl.col("created_date").max().alias("u_last_date"),
    ]).with_columns(((pl.lit(end_date)-pl.col("u_last_date")).dt.total_days().cast(pl.Int32)).alias("u_recency_days")).drop("u_last_date")

    it = hist.group_by(["item_id"]).agg([
        pl.len().alias("i_cnt"),
        pl.col("quantity").sum().alias("i_sum_qty"),
        pl.col("spent_row").sum().alias("i_sum_spent"),
        pl.col("customer_id").n_unique().alias("i_n_unique_users"),
        pl.col("created_date").max().alias("i_last_date"),
    ]).with_columns(((pl.lit(end_date)-pl.col("i_last_date")).dt.total_days().cast(pl.Int32)).alias("i_recency_days")).drop("i_last_date")

    uc = hist.group_by(["customer_id","category_l1"]).agg([
        pl.len().alias("uc_cnt"),
        pl.col("spent_row").sum().alias("uc_sum_spent"),
        pl.col("created_date").max().alias("uc_last_date"),
    ]).with_columns(((pl.lit(end_date)-pl.col("uc_last_date")).dt.total_days().cast(pl.Int32)).alias("uc_recency_days")).drop("uc_last_date")

    ag = pl.concat([
        ui.with_columns(pl.lit("ui").alias("_k")),
        u.with_columns(pl.lit("u").alias("_k")),
        it.with_columns(pl.lit("i").alias("_k")),
        uc.with_columns(pl.lit("uc").alias("_k")),
    ], how="diagonal_relaxed")
    ag.write_parquet(path)
    log(f"[SAVE] Hist aggs -> {path}")
    return ui, u, it, uc


def build_feature_label(cfg: Stage2Config, candidates: pl.DataFrame, df_tx: pl.DataFrame,
                        items_df: pl.DataFrame, users_df: pl.DataFrame,
                        ui_agg: pl.DataFrame, u_agg: pl.DataFrame, i_agg: pl.DataFrame, uc_agg: pl.DataFrame,
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
    fl = candidates.join(items_df, on="item_id", how="left").join(users_df, on="customer_id", how="left")

    fl = fl.with_columns(((pl.lit(train_end.date()) - pl.col("install_datetime")).dt.total_days().cast(pl.Int32)).alias("tenure_days")).with_columns(
        pl.col("tenure_days").fill_null(-1)
    )

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
        fl = fl.with_columns(pl.lit(mkey).alias("trend_month")).join(
            topm.select(["month_key","category_l1","item_id","rank"]),
            left_on=["trend_month","category_l1","item_id"],
            right_on=["month_key","category_l1","item_id"],
            how="left"
        ).rename({"rank":"rank_top10_by_cat_month"}).with_columns(pl.col("rank_top10_by_cat_month").fill_null(9999).cast(pl.Int32))
    else:
        fl = fl.with_columns(pl.lit(9999).alias("rank_top10_by_cat_month"))

    fl = fl.join(ui_agg, on=["customer_id","item_id"], how="left").join(u_agg, on="customer_id", how="left") \
           .join(i_agg, on="item_id", how="left").join(uc_agg, on=["customer_id","category_l1"], how="left")

    for c in ["ui_cnt","ui_sum_qty","ui_sum_spent","u_cnt","u_sum_qty","u_sum_spent","u_n_unique_items",
              "i_cnt","i_sum_qty","i_sum_spent","i_n_unique_users","uc_cnt","uc_sum_spent"]:
        if c in fl.columns: fl = fl.with_columns(pl.col(c).fill_null(0))
    for c in ["ui_recency_days","u_recency_days","i_recency_days","uc_recency_days"]:
        if c in fl.columns: fl = fl.with_columns(pl.col(c).fill_null(9999))

    recent_pairs = df_tx.filter(
        pl.col("created_date").is_between(pl.lit(recent_begin.date(), pl.Date), pl.lit(recent_end.date(), pl.Date), closed="both")
    ).select(["customer_id","item_id"]).unique().with_columns(pl.lit(1, pl.Int8).alias("Y"))
    fl = fl.join(recent_pairs, on=["customer_id","item_id"], how="left").with_columns(pl.col("Y").fill_null(0).cast(pl.Int8))

    # negative sampling per user
    rng = np.random.default_rng(int(cfg.random_state))
    fl_min = fl.select(["customer_id","item_id","Y"])
    users = fl_min["customer_id"].unique().to_list()
    if cfg.max_users_train and cfg.max_users_train > 0:
        users = users[:int(cfg.max_users_train)]

    keep = set()
    for u in tqdm(users, desc="NegSampling"):
        dfu = fl_min.filter(pl.col("customer_id")==u)
        pos = dfu.filter(pl.col("Y")==1)["item_id"].to_list()
        neg = dfu.filter(pl.col("Y")==0)["item_id"].to_list()
        for it in pos:
            keep.add((str(u), str(it)))
        if cfg.N_neg > 0 and len(neg) > 0:
            k = min(int(cfg.N_neg), len(neg))
            samp = rng.choice(np.array(neg, dtype=object), size=k, replace=False).tolist()
            for it in samp:
                keep.add((str(u), str(it)))

    keep_df = pl.DataFrame(list(keep), schema=[("customer_id", pl.Utf8), ("item_id", pl.Utf8)])
    fl_s = fl.join(keep_df, on=["customer_id","item_id"], how="inner")
    fl_s.write_parquet(sampled_path)
    log(f"[SAVE] Feature-label sampled -> {sampled_path}")
    return fl_s

def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

def train_model(cfg: Stage2Config, fl: pl.DataFrame) -> Any:
    log("[8/9] Train ranking model (L1 selection + Linear SVM) ...")
    key_cols = {"customer_id","item_id","Y","install_datetime","trend_month"}
    feat_cols = [c for c in fl.columns if c not in key_cols]

    numeric, categorical = [], []
    for c in feat_cols:
        dt = fl.schema[c]
        if dt in (pl.Int8,pl.Int16,pl.Int32,pl.Int64,pl.UInt32,pl.UInt64,pl.Float32,pl.Float64):
            numeric.append(c)
        else:
            categorical.append(c)

    dfp = fl.select(["Y"] + feat_cols).to_pandas()
    y = dfp["Y"].astype(int).values
    X = dfp.drop(columns=["Y"])

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler(with_mean=False))])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("onehot", make_onehot_encoder())])

    pre = ColumnTransformer([("num", num_pipe, numeric),
                             ("cat", cat_pipe, categorical)],
                            remainder="drop", sparse_threshold=0.3)

    l1 = LogisticRegression(penalty="l1", C=float(cfg.l1_C), solver="saga",
                            max_iter=int(cfg.max_iter_l1), n_jobs=-1,
                            class_weight=("balanced" if cfg.class_weight=="balanced" else None))
    selector = SelectFromModel(l1, prefit=False)
    svm = LinearSVC(C=float(cfg.svm_C), class_weight=("balanced" if cfg.class_weight=="balanced" else None))

    model = Pipeline([("pre", pre), ("select", selector), ("svm", svm)])
    model.fit(X, y)
    return model


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



def precision_at_k_userwise(pred, gt, hist, filter_bought_items=True, K=10):
    precisions = []
    ncold_start = 0
    cold_start_users = []

    for user in gt.keys():
        if (user not in hist) or (user not in pred):
            ncold_start += 1
            cold_start_users.append(user)
            continue

        relevant_items = set(gt[user])
        if filter_bought_items:
            relevant_items -= set(hist[user])

        hits = len(set(pred[user][:K]) & relevant_items)
        precisions.append(hits / K)

    mean_prec = float(np.mean(precisions)) if len(precisions) > 0 else 0.0
    return mean_prec, ncold_start, cold_start_users


def recall_at_k(pred: List[str], gt: set, k: int) -> float:
    return len(set(pred[:k]) & gt) / float(len(gt)) if gt else 0.0


def hit_at_k(pred: List[str], gt: set, k: int) -> float:
    return 1.0 if (set(pred[:k]) & gt) else 0.0


def score_topk(cfg: Stage2Config, model: Any, feature_table: pl.DataFrame, users: List[str]) -> Dict[str, List[str]]:
    topk = int(cfg.topk)
    key_cols = ["customer_id","item_id"]
    drop = {"Y","install_datetime","trend_month"}
    feat_cols = [c for c in feature_table.columns if c not in set(key_cols) and c not in drop]

    dfp = feature_table.select(key_cols + feat_cols).to_pandas()
    X = dfp[feat_cols]
    scores = model.decision_function(X)

    cust = dfp["customer_id"].astype(str).values
    item = dfp["item_id"].astype(str).values
    order = np.lexsort((-scores, cust))

    out: Dict[str, List[str]] = {}
    cur = None
    buf = []
    for u, it in zip(cust[order], item[order]):
        if cur is None:
            cur = u
        if u != cur:
            out[cur] = buf[:topk]
            cur = u
            buf = []
        if len(buf) < topk:
            buf.append(it)
    if cur is not None:
        out[cur] = buf[:topk]
    for u in users:
        out.setdefault(str(u), [])
    return out


def eval_metrics(cfg: Stage2Config, pred: Dict[str, List[str]], gt_recent: Dict[str, set],
                 hist: Dict[str, set], mode: str) -> dict:
    k = int(cfg.topk)
    precs, recs, hits, ndcgs = [], [], [], []
    n = 0
    for u, gt in gt_recent.items():
        gt_set = set(gt)
        p = pred.get(u, [])
        if mode == "filtered":
            h = hist.get(u, set())
            gt_set = gt_set - h
            p = [it for it in p if it not in h]
        if not gt_set:
            continue
        n += 1
        precs.append(precision_at_k(p, gt_set, k))
        recs.append(recall_at_k(p, gt_set, k))
        hits.append(hit_at_k(p, gt_set, k))
        ndcgs.append(ndcg_at_k(p, gt_set, k))
    if n == 0:
        return {"n_users":0,"precision":0.0,"recall":0.0,"hit":0.0,"ndcg":0.0,"k":k}
    return {"n_users":n,"precision":float(np.mean(precs)),"recall":float(np.mean(recs)),
            "hit":float(np.mean(hits)),"ndcg":float(np.mean(ndcgs)),"k":k}


def eval_private(cfg: Stage2Config, model: Any, feature_table: pl.DataFrame) -> Optional[dict]:
    if not cfg.groundtruth_pkl or not cfg.evaluate_private_test:
        return None
    if not os.path.exists(cfg.groundtruth_pkl):
        log(f"[WARN] groundtruth_pkl not found: {cfg.groundtruth_pkl}")
        return None

    gt_raw = pickle.load(open(cfg.groundtruth_pkl, "rb"))
    gt: Dict[str, set] = {}
    if isinstance(gt_raw, dict):
        for u, items in gt_raw.items():
            if isinstance(items, (list,set,tuple)):
                gt[str(u)] = set(map(str, items))
    elif isinstance(gt_raw, list):
        for row in gt_raw:
            u = str(row.get("customer_id"))
            items = row.get("item_id")
            if isinstance(items, (list,set,tuple)):
                gt[u] = set(map(str, items))
    else:
        log("[WARN] Unknown groundtruth.pkl format; skip.")
        return None

    users = list(gt.keys())
    if cfg.max_users_eval and cfg.max_users_eval > 0:
        users = users[:int(cfg.max_users_eval)]
        gt = {u: gt[u] for u in users}

    ft = feature_table.filter(pl.col("customer_id").is_in(users))
    pred = score_topk(cfg, model, ft, users)

    k = int(cfg.topk)
    precs, ndcgs = [], []
    n = 0
    for u, gt_set in gt.items():
        n += 1
        p = pred.get(u, [])
        precs.append(precision_at_k(p, gt_set, k))
        ndcgs.append(ndcg_at_k(p, gt_set, k))
    if n == 0:
        return {"n_users":0,"precision":0.0,"ndcg":0.0,"k":k}
    return {"n_users":n,"precision":float(np.mean(precs)),"ndcg":float(np.mean(ndcgs)),"k":k}


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    _ensure_dir(cfg.out_dir); _ensure_dir(cfg.cache_dir)
    run_dir = os.path.join(cfg.out_dir, cfg.run_name); _ensure_dir(run_dir)

    log("==== Stage2 Ranking (SVM + L1 selection) ====")
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
    ui_agg, u_agg, i_agg, uc_agg = build_hist_aggs(cfg, df_tx, items_df, train_begin, train_end)

    fl = build_feature_label(cfg, candidates, df_tx, items_df, users_df, ui_agg, u_agg, i_agg, uc_agg,
                             feat_tables, train_begin, train_end, recent_begin, recent_end)

    model = train_model(cfg, fl)
    model_path = os.path.join(run_dir, "stage2_model.joblib")
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
        ft = candidates.join(items_df, on="item_id", how="left").join(users_df, on="customer_id", how="left")
        ft = ft.with_columns(((pl.lit(train_end.date()) - pl.col("install_datetime")).dt.total_days().cast(pl.Int32)).alias("tenure_days")).with_columns(pl.col("tenure_days").fill_null(-1))
        if "price_segment" in feat_tables: ft = ft.join(feat_tables["price_segment"], on="item_id", how="left")
        if "buy_segment" in feat_tables: ft = ft.join(feat_tables["buy_segment"], on="customer_id", how="left")
        if "luxury_level" in feat_tables: ft = ft.join(feat_tables["luxury_level"], on="customer_id", how="left")
        if "age_final" in feat_tables: ft = ft.join(feat_tables["age_final"], on="customer_id", how="left")
        if "brand_segment" in feat_tables: ft = ft.join(feat_tables["brand_segment"], on="category_l1", how="left")
        if "top10_by_cat_month" in feat_tables:
            mdt = _shift_months(train_end, -int(cfg.top10_month_lag))
            mkey = _month_key(mdt)
            topm = feat_tables["top10_by_cat_month"].with_columns(pl.col("month").alias("month_key"))
            ft = ft.with_columns(pl.lit(mkey).alias("trend_month")).join(
                topm.select(["month_key","category_l1","item_id","rank"]),
                left_on=["trend_month","category_l1","item_id"],
                right_on=["month_key","category_l1","item_id"],
                how="left"
            ).rename({"rank":"rank_top10_by_cat_month"}).with_columns(pl.col("rank_top10_by_cat_month").fill_null(9999).cast(pl.Int32))
        else:
            ft = ft.with_columns(pl.lit(9999).alias("rank_top10_by_cat_month"))

        ft = ft.join(ui_agg, on=["customer_id","item_id"], how="left").join(u_agg, on="customer_id", how="left") \
               .join(i_agg, on="item_id", how="left").join(uc_agg, on=["customer_id","category_l1"], how="left")

        for c in ["ui_cnt","ui_sum_qty","ui_sum_spent","u_cnt","u_sum_qty","u_sum_spent","u_n_unique_items",
                  "i_cnt","i_sum_qty","i_sum_spent","i_n_unique_users","uc_cnt","uc_sum_spent"]:
            if c in ft.columns: ft = ft.with_columns(pl.col(c).fill_null(0))
        for c in ["ui_recency_days","u_recency_days","i_recency_days","uc_recency_days"]:
            if c in ft.columns: ft = ft.with_columns(pl.col(c).fill_null(9999))

        ft.write_parquet(full_path)
        log(f"[SAVE] Full feature table -> {full_path}")

    pred = score_topk(cfg, model, ft, users)
    hist = build_user_item_sets(df_tx, train_begin, train_end)

    m_f = eval_metrics(cfg, pred, gt_recent, hist, "filtered")
    m_u = eval_metrics(cfg, pred, gt_recent, hist, "unfiltered")

    metrics = {"offline_filtered": m_f, "offline_unfiltered": m_u}
    log("===== OFFLINE METRICS (RECENT) =====")
    log(f"FILTERED   P@{cfg.topk}={m_f['precision']:.6f}  R@{cfg.topk}={m_f['recall']:.6f}  Hit@{cfg.topk}={m_f['hit']:.6f}  NDCG@{cfg.topk}={m_f['ndcg']:.6f}  n_users={m_f['n_users']:,}")
    log(f"UNFILTERED P@{cfg.topk}={m_u['precision']:.6f}  R@{cfg.topk}={m_u['recall']:.6f}  Hit@{cfg.topk}={m_u['hit']:.6f}  NDCG@{cfg.topk}={m_u['ndcg']:.6f}  n_users={m_u['n_users']:,}")

    priv = eval_private(cfg, model, ft)
    if priv is not None:
        metrics["private_test"] = priv
        log("===== PRIVATE TEST (groundtruth.pkl) =====")
        log(f"P@{cfg.topk}={priv['precision']:.6f}  NDCG@{cfg.topk}={priv['ndcg']:.6f}  n_users={priv['n_users']:,}")

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
