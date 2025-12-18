# -*- coding: utf-8 -*-
"""
stage1_retrieval_implicit_dual.py

Stage-1 Retrieval (Candidate Generation) using BOTH:
  - implicit.nearest_neighbours.TFIDFRecommender
  - implicit.nearest_neighbours.CosineRecommender

Key behaviors:
  - Train (fit) on history window ending at train_end (inclusive).
  - Evaluate Recall@K and Hit@K on recent window starting at recent_begin (inclusive).
  - Reports BOTH:
      * FILTERED (new-item): remove train-history items from GT and filter already-liked items at recommend-time.
      * UNFILTERED (overall): keep repeats in GT and DO NOT filter already-liked items at recommend-time.
  - Counts cold-start users in eval (users with relevant GT but absent from train user index).
  - Trending (popular) backfill appended to candidates.
  - Random search over: len_hist, len_recent, N_cand, N_trend (plus optional weight_type, K_model_mult).
  - Caching to accelerate repeated runs:
      * Minimal transactions parquet (typed, filtered)
      * CSR + mappings + trending indices
      * Fitted implicit models (TFIDF/Cosine) + meta

Output files (under artifacts_dir/runs/run_name):
  - random_search_results.csv
  - best_params.json
  - best_metrics.json
  - best_stage1_meta.npz
  - best_stage1_tfidf.npz
  - best_stage1_cosine.npz
  - best_stage1_metrics.json
  - cold_start_users_filtered.pkl
  - cold_start_users_unfiltered.pkl
"""

from __future__ import annotations

import os
import json
import time
import pickle
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix, save_npz, load_npz
from tqdm.auto import tqdm

from implicit.nearest_neighbours import TFIDFRecommender, CosineRecommender


# -------------------------
# Logging
# -------------------------
def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# -------------------------
# Config
# -------------------------
@dataclass
class Stage1Config:
    # paths
    transactions_path_glob: str
    items_path_glob: Optional[str] = None
    users_path_glob: Optional[str] = None
    artifacts_dir: str = "./artifacts_stage1"
    run_name: str = "run"

    # time anchors (ISO date)
    train_end: str = "2024-11-30"       # inclusive
    recent_begin: str = "2024-12-01"    # inclusive

    # defaults (used when do_random_search=False or as fallbacks)
    len_hist: int = 120
    len_recent: int = 28
    N_cand: int = 100
    N_trend: int = 100

    # modeling knobs
    weight_type: str = "log_count"      # binary|count|log_count|log_qty|log_spent
    K_model_mult: int = 3               # K_model = clip(20, 800, K_model_mult*N_cand)
    num_threads: int = 0                # 0 => implicit default; set >0 to cap threads

    # evaluation
    batch_users: int = 5000
    max_rows_train: int = 0             # 0 => no cap
    max_rows_eval: int = 0              # 0 => no cap
    filter_train_history_in_gt: bool = True  # kept for backward-compat; eval reports both modes anyway


    # random search
    do_random_search: bool = True
    n_trials: int = 90
    random_state: int = 42
    search_space: Optional[dict] = None  # if provided, overrides the space_* lists
    space_len_hist: Optional[List[int]] = None
    space_len_recent: Optional[List[int]] = None
    space_N_cand: Optional[List[int]] = None
    space_N_trend: Optional[List[int]] = None
    space_weight_type: Optional[List[str]] = None
    space_K_model_mult: Optional[List[int]] = None

    # quick mode
    quick: bool = False
    quick_n_trials: int = 5
    quick_max_rows_train: int = 300_000
    quick_max_rows_eval: int = 200_000
    quick_max_users_eval: int = 5000


def load_config(path: str) -> Stage1Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Backward/forward compatible: ignore unexpected keys
    allowed = set(Stage1Config.__dataclass_fields__.keys())
    raw = {k: v for k, v in raw.items() if k in allowed}
    return Stage1Config(**raw)


# -------------------------
# Helpers
# -------------------------
def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _hash_dict(d: dict) -> str:
    s = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]


def _glob_scan(glob_path: str) -> pl.LazyFrame:
    return pl.scan_parquet(glob_path)


def _clip(a: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, a))


# -------------------------
# Data prep: minimal typed transactions + optional item filter
# -------------------------
def build_min_transactions(
    cfg: Stage1Config,
    train_begin: datetime,
    eval_end: datetime,
    cache_dir: str,
) -> Tuple[str, pl.DataFrame]:
    """
    Minimal transactions: customer_id(Utf8), item_id(Utf8), created_date(Date), quantity(Float64), price(Float64)
    Filtered to [train_begin, eval_end] inclusive.
    Cached to speed up repeated runs.
    """
    key = _hash_dict({
        "transactions_path_glob": cfg.transactions_path_glob,
        "items_path_glob": cfg.items_path_glob,
        "train_begin": train_begin.strftime("%Y-%m-%d"),
        "eval_end": eval_end.strftime("%Y-%m-%d"),
    })
    cache_path = os.path.join(cache_dir, f"tx_min_{key}.parquet")
    if os.path.exists(cache_path):
        log(f"[CACHE] Load minimal transactions: {cache_path}")
        return cache_path, pl.read_parquet(cache_path)

    log("[1/6] Scan transactions (lazy) ...")
    tx = _glob_scan(cfg.transactions_path_glob)

    need = {"customer_id", "item_id", "created_date"}
    cols = set(tx.columns)
    missing = [c for c in need if c not in cols]
    if missing:
        raise ValueError(f"Transactions missing required columns: {missing}. Available: {tx.columns}")

    has_qty = "quantity" in cols
    has_price = "price" in cols

    tx = tx.with_columns([
        pl.col("customer_id").cast(pl.Utf8),
        pl.col("item_id").cast(pl.Utf8),
        pl.col("created_date").cast(pl.Date, strict=False),
        (pl.col("quantity").cast(pl.Float64, strict=False).fill_null(1.0).fill_nan(1.0)
         if has_qty else pl.lit(1.0).alias("quantity")),
        (pl.col("price").cast(pl.Float64, strict=False).fill_null(0.0).fill_nan(0.0)
         if has_price else pl.lit(0.0).alias("price")),
    ]).select(["customer_id", "item_id", "created_date", "quantity", "price"])

    tx = tx.filter(
        pl.col("created_date").is_between(
            pl.lit(train_begin.date(), dtype=pl.Date),
            pl.lit(eval_end.date(), dtype=pl.Date),
            closed="both",
        )
    )

    if cfg.items_path_glob:
        log("[1/6] Scan items for valid item_id filter ...")
        it = _glob_scan(cfg.items_path_glob)
        if "item_id" in it.columns:
            valid = it.select(pl.col("item_id").cast(pl.Utf8)).unique()
            tx = tx.join(valid, on="item_id", how="inner")
        else:
            log("[WARN] items table has no item_id; skip item filter.")

    log("[1/6] Materialize minimal transactions ...")
    df = tx.collect(streaming=True)

    _ensure_dir(cache_dir)
    log(f"[SAVE] Minimal transactions cache -> {cache_path}")
    df.write_parquet(cache_path)
    return cache_path, df


# -------------------------
# CSR building
# -------------------------
def build_user_items_csr(
    df: pl.DataFrame,
    begin: datetime,
    end: datetime,
    weight_type: str,
    cache_dir: str,
    cache_prefix: str,
    max_rows: int = 0,
) -> Tuple[csr_matrix, List[str], List[str], Dict[str, int], Dict[str, int], np.ndarray]:
    """
    Build CSR for window [begin, end] inclusive.
    Returns:
      mat, users, items, u2i, it2i, trending_idx (descending by popularity)
    """
    key = _hash_dict({
        "cache_prefix": cache_prefix,
        "begin": begin.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "weight_type": weight_type,
        "max_rows": int(max_rows),
    })
    csr_path = os.path.join(cache_dir, f"csr_{key}.npz")
    maps_path = os.path.join(cache_dir, f"maps_{key}.npz")
    trend_path = os.path.join(cache_dir, f"trending_{key}.npy")

    if os.path.exists(csr_path) and os.path.exists(maps_path) and os.path.exists(trend_path):
        log(f"[CACHE] Load CSR: {csr_path}")
        mat = load_npz(csr_path).tocsr()
        maps = np.load(maps_path, allow_pickle=True)
        users = maps["users"].tolist()
        items = maps["items"].tolist()
        trending_idx = np.load(trend_path)
        u2i = {u: i for i, u in enumerate(users)}
        it2i = {it: j for j, it in enumerate(items)}
        return mat, users, items, u2i, it2i, trending_idx

    log("[2/6] Build CSR from transactions window ...")
    d = df.filter(
        pl.col("created_date").is_between(
            pl.lit(begin.date(), dtype=pl.Date),
            pl.lit(end.date(), dtype=pl.Date),
            closed="both",
        )
    ).select(["customer_id", "item_id", "quantity", "price"])

    if max_rows and max_rows > 0:
        d = d.head(max_rows)

    d = d.with_columns((pl.col("quantity") * pl.col("price")).alias("spent_row"))
    ui = d.group_by(["customer_id", "item_id"]).agg([
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

    users = ui.select("customer_id").unique().sort("customer_id").to_series().to_list()
    items = ui.select("item_id").unique().sort("item_id").to_series().to_list()
    u2i = {u: i for i, u in enumerate(users)}
    it2i = {it: j for j, it in enumerate(items)}

    u_idx = ui.select("customer_id").to_series().to_list()
    it_idx = ui.select("item_id").to_series().to_list()
    data = ui.select("value").to_series().to_numpy().astype(np.float32)

    row = np.fromiter((u2i[u] for u in u_idx), dtype=np.int32, count=len(u_idx))
    col = np.fromiter((it2i[it] for it in it_idx), dtype=np.int32, count=len(it_idx))

    mat = csr_matrix((data, (row, col)), shape=(len(users), len(items)), dtype=np.float32)

    pop = np.asarray(mat.sum(axis=0)).ravel()
    trending_idx = np.argsort(-pop).astype(np.int32)

    _ensure_dir(cache_dir)
    log(f"[SAVE] CSR -> {csr_path}")
    save_npz(csr_path, mat)
    log(f"[SAVE] Mappings -> {maps_path}")
    np.savez_compressed(maps_path, users=np.array(users, dtype=object), items=np.array(items, dtype=object))
    log(f"[SAVE] Trending idx -> {trend_path}")
    np.save(trend_path, trending_idx)

    return mat, users, items, u2i, it2i, trending_idx


# -------------------------
# Fit models + cache
# -------------------------
@dataclass
class FittedStage1:
    tfidf: TFIDFRecommender
    cosine: CosineRecommender
    user_items: csr_matrix
    users: List[str]
    items: List[str]
    u2i: Dict[str, int]
    it2i: Dict[str, int]
    trending_idx: np.ndarray


def fit_or_load_models(
    cfg: Stage1Config,
    mat: csr_matrix,
    users: List[str],
    items: List[str],
    trending_idx: np.ndarray,
    cache_dir: str,
    cache_prefix: str,
) -> FittedStage1:
    K_model = _clip(int(cfg.K_model_mult) * int(cfg.N_cand), 20, 800)

    key = _hash_dict({
        "cache_prefix": cache_prefix,
        "K_model": K_model,
        "N_cand": int(cfg.N_cand),
        "N_trend": int(cfg.N_trend),
        "weight_type": cfg.weight_type,
        "num_threads": int(cfg.num_threads),
        "mat_shape": [int(mat.shape[0]), int(mat.shape[1]), int(mat.nnz)],
    })
    tfidf_path = os.path.join(cache_dir, f"model_tfidf_{key}.npz")
    cosine_path = os.path.join(cache_dir, f"model_cosine_{key}.npz")
    meta_path = os.path.join(cache_dir, f"meta_{key}.npz")

    if os.path.exists(tfidf_path) and os.path.exists(cosine_path) and os.path.exists(meta_path):
        log(f"[CACHE] Load implicit models: {tfidf_path}, {cosine_path}")
        tfidf = TFIDFRecommender.load(tfidf_path)
        cosine = CosineRecommender.load(cosine_path)
        return FittedStage1(
            tfidf=tfidf,
            cosine=cosine,
            user_items=mat,
            users=users,
            items=items,
            u2i={u: i for i, u in enumerate(users)},
            it2i={it: j for j, it in enumerate(items)},
            trending_idx=trending_idx,
        )

    log("[3/6] Fit implicit models (TFIDF + Cosine) ...")
    tfidf = TFIDFRecommender(K=K_model, num_threads=int(cfg.num_threads))
    cosine = CosineRecommender(K=K_model, num_threads=int(cfg.num_threads))

    t0 = time.time()
    tfidf.fit(mat, show_progress=True)
    log(f"[3/6] TFIDF fit done in {time.time()-t0:.1f}s")

    t0 = time.time()
    cosine.fit(mat, show_progress=True)
    log(f"[3/6] Cosine fit done in {time.time()-t0:.1f}s")

    _ensure_dir(cache_dir)
    log(f"[SAVE] Save TFIDF -> {tfidf_path}")
    tfidf.save(tfidf_path)
    log(f"[SAVE] Save Cosine -> {cosine_path}")
    cosine.save(cosine_path)

    meta = {
        "K_model": K_model,
        "N_cand": int(cfg.N_cand),
        "N_trend": int(cfg.N_trend),
        "weight_type": cfg.weight_type,
        "num_threads": int(cfg.num_threads),
        "users_n": len(users),
        "items_n": len(items),
    }
    log(f"[SAVE] Save meta -> {meta_path}")
    np.savez_compressed(meta_path, config_json=json.dumps(meta, ensure_ascii=False))

    return FittedStage1(
        tfidf=tfidf,
        cosine=cosine,
        user_items=mat,
        users=users,
        items=items,
        u2i={u: i for i, u in enumerate(users)},
        it2i={it: j for j, it in enumerate(items)},
        trending_idx=trending_idx,
    )


# -------------------------
# Retrieval (user context) + merge
# -------------------------
def recommend_for_user(
    fitted: FittedStage1,
    user_id: str,
    N_cand: int,
    N_trend: int,
    filter_already_liked_items: bool,
) -> List[str]:
    """
    Return candidates length up to K=2*N_cand+N_trend:
      TFIDF top-N_cand + Cosine top-N_cand + trending top-N_trend, dedup in order.
    """
    K = 2 * int(N_cand) + int(N_trend)

    if user_id not in fitted.u2i:
        idxs = fitted.trending_idx[:K].tolist()
        return [fitted.items[i] for i in idxs]

    uidx = fitted.u2i[user_id]
    row = fitted.user_items[uidx]
    if row.nnz == 0:
        idxs = fitted.trending_idx[:K].tolist()
        return [fitted.items[i] for i in idxs]

    ids_t, _ = fitted.tfidf.recommend(uidx, row, N=int(N_cand), filter_already_liked_items=filter_already_liked_items)
    ids_c, _ = fitted.cosine.recommend(uidx, row, N=int(N_cand), filter_already_liked_items=filter_already_liked_items)

    cand_idx = []
    cand_idx.extend(ids_t.tolist() if hasattr(ids_t, "tolist") else list(ids_t))
    cand_idx.extend(ids_c.tolist() if hasattr(ids_c, "tolist") else list(ids_c))
    if int(N_trend) > 0:
        cand_idx.extend(fitted.trending_idx[:int(N_trend)].tolist())

    seen = set()
    out = []
    for ix in cand_idx:
        ix = int(ix)
        if ix not in seen:
            seen.add(ix)
            out.append(fitted.items[ix])
        if len(out) >= K:
            break
    return out


# -------------------------
# Evaluation: Recall@K, Hit@K, Cold-start count
# -------------------------
@dataclass
class EvalReport:
    recall: float
    hit: float
    n_users_eval: int
    n_cold_start: int
    K: int


def build_groundtruth(
    df: pl.DataFrame,
    eval_begin: datetime,
    eval_end: datetime,
    max_rows: int = 0,
) -> Dict[str, set]:
    d = df.filter(
        pl.col("created_date").is_between(
            pl.lit(eval_begin.date(), dtype=pl.Date),
            pl.lit(eval_end.date(), dtype=pl.Date),
            closed="both",
        )
    ).select(["customer_id", "item_id"])

    if max_rows and max_rows > 0:
        d = d.head(max_rows)

    gt: Dict[str, set] = {}
    for u, it in zip(d["customer_id"].to_list(), d["item_id"].to_list()):
        u = str(u)
        gt.setdefault(u, set()).add(str(it))
    return gt


def build_train_hist_sets(
    df: pl.DataFrame,
    train_begin: datetime,
    train_end: datetime,
) -> Dict[str, set]:
    d = df.filter(
        pl.col("created_date").is_between(
            pl.lit(train_begin.date(), dtype=pl.Date),
            pl.lit(train_end.date(), dtype=pl.Date),
            closed="both",
        )
    ).select(["customer_id", "item_id"]).unique()

    hist: Dict[str, set] = {}
    for u, it in zip(d["customer_id"].to_list(), d["item_id"].to_list()):
        u = str(u)
        hist.setdefault(u, set()).add(str(it))
    return hist


def eval_recall_hit(
    fitted: FittedStage1,
    gt: Dict[str, set],
    train_hist: Optional[Dict[str, set]],
    N_cand: int,
    N_trend: int,
    filter_train_history_in_gt: bool,
    batch_users: int,
) -> EvalReport:
    """
    FILTERED mode:
      - remove train_hist items from GT relevant set
      - recommend(filter_already_liked_items=True)
    UNFILTERED mode:
      - keep GT as-is
      - recommend(filter_already_liked_items=False)
    """
    K = 2 * int(N_cand) + int(N_trend)
    users = list(gt.keys())

    recalls = []
    hits = []
    n_cold = 0
    n_eval = 0

    for i in tqdm(range(0, len(users), int(batch_users)), desc=f"Eval @K={K} (batch_users={batch_users})"):
        batch = users[i:i + int(batch_users)]
        for u in batch:
            relevant = set(gt[u])
            if filter_train_history_in_gt and train_hist is not None and u in train_hist:
                relevant = relevant - train_hist[u]
            if not relevant:
                continue

            n_eval += 1
            if u not in fitted.u2i:
                n_cold += 1

            rec = recommend_for_user(
                fitted=fitted,
                user_id=u,
                N_cand=N_cand,
                N_trend=N_trend,
                filter_already_liked_items=filter_train_history_in_gt,
            )
            inter = set(rec[:K]) & relevant
            recalls.append(len(inter) / len(relevant))
            hits.append(1.0 if inter else 0.0)

    if n_eval == 0:
        return EvalReport(recall=0.0, hit=0.0, n_users_eval=0, n_cold_start=n_cold, K=K)

    return EvalReport(
        recall=float(np.mean(recalls)) if recalls else 0.0,
        hit=float(np.mean(hits)) if hits else 0.0,
        n_users_eval=int(n_eval),
        n_cold_start=int(n_cold),
        K=K,
    )


def compute_cold_users(
    fitted: FittedStage1,
    gt: Dict[str, set],
    train_hist: Optional[Dict[str, set]],
    filter_train_history_in_gt: bool,
) -> List[str]:
    cold = []
    for u, items in gt.items():
        relevant = set(items)
        if filter_train_history_in_gt and train_hist is not None and u in train_hist:
            relevant = relevant - train_hist[u]
        if not relevant:
            continue
        if u not in fitted.u2i:
            cold.append(u)
    return cold


# -------------------------
# Random search
# -------------------------
def _default_space(cfg: Stage1Config) -> dict:
    if cfg.search_space:
        return cfg.search_space

    def _or(v, default):
        return v if (v is not None and len(v) > 0) else default

    return {
        "len_hist": _or(cfg.space_len_hist, [cfg.len_hist]),
        "len_recent": _or(cfg.space_len_recent, [cfg.len_recent]),
        "N_cand": _or(cfg.space_N_cand, [cfg.N_cand]),
        "N_trend": _or(cfg.space_N_trend, [cfg.N_trend]),
        "weight_type": _or(cfg.space_weight_type, [cfg.weight_type]),
        "K_model_mult": _or(cfg.space_K_model_mult, [cfg.K_model_mult]),
    }


def _sample(space: dict, rng: np.random.Generator) -> dict:
    out = {}
    for k, vals in space.items():
        if not isinstance(vals, list) or len(vals) == 0:
            raise ValueError(f"search_space[{k}] must be non-empty list")
        out[k] = vals[int(rng.integers(0, len(vals)))]
    return out


def random_search_stage1(
    cfg: Stage1Config,
    df_min: pl.DataFrame,
    train_end: datetime,
    recent_begin: datetime,
    out_run_dir: str,
    cache_dir: str,
) -> Tuple[dict, EvalReport, EvalReport]:
    """
    Returns:
      best_params, best_report_filtered, best_report_unfiltered
    """
    _ensure_dir(out_run_dir)
    results_csv = os.path.join(out_run_dir, "random_search_results.csv")
    best_params_path = os.path.join(out_run_dir, "best_params.json")
    best_metrics_path = os.path.join(out_run_dir, "best_metrics.json")

    space = _default_space(cfg)
    n_trials = int(cfg.n_trials)
    if cfg.quick:
        n_trials = min(n_trials, int(cfg.quick_n_trials))

    if not os.path.exists(results_csv):
        with open(results_csv, "w", encoding="utf-8") as f:
            f.write("trial,recall_f,hit_f,recall_u,hit_u,n_users_eval_f,n_users_eval_u,n_cold_f,n_cold_u,params_json\n")

    rng = np.random.default_rng(int(cfg.random_state))

    best_recall_f = -1.0
    best_hit_f = -1.0
    best_params: Optional[dict] = None
    best_f: Optional[EvalReport] = None
    best_u: Optional[EvalReport] = None

    log(f"[RS] Random search n_trials={n_trials} (quick={cfg.quick})")
    for trial in tqdm(range(1, n_trials + 1), desc="Stage1 random search"):
        p = _sample(space, rng)

        # windows
        len_hist = int(p["len_hist"])
        len_recent = int(p["len_recent"])
        train_begin = train_end - timedelta(days=len_hist - 1)
        eval_begin = recent_begin
        eval_end = recent_begin + timedelta(days=len_recent - 1)

        # trial cfg (only fields that matter)
        trial_cfg = Stage1Config(**{**cfg.__dict__})
        trial_cfg.len_hist = len_hist
        trial_cfg.len_recent = len_recent
        trial_cfg.N_cand = int(p["N_cand"])
        trial_cfg.N_trend = int(p["N_trend"])
        trial_cfg.weight_type = str(p["weight_type"])
        trial_cfg.K_model_mult = int(p["K_model_mult"])

        max_rows_train = cfg.quick_max_rows_train if cfg.quick else cfg.max_rows_train
        max_rows_eval = cfg.quick_max_rows_eval if cfg.quick else cfg.max_rows_eval

        # CSR + models
        mat, users, items, u2i, it2i, trending_idx = build_user_items_csr(
            df=df_min,
            begin=train_begin,
            end=train_end,
            weight_type=trial_cfg.weight_type,
            cache_dir=cache_dir,
            cache_prefix=f"train_{train_begin:%Y%m%d}_{train_end:%Y%m%d}",
            max_rows=max_rows_train,
        )
        fitted = fit_or_load_models(
            cfg=trial_cfg,
            mat=mat,
            users=users,
            items=items,
            trending_idx=trending_idx,
            cache_dir=cache_dir,
            cache_prefix=f"models_{train_begin:%Y%m%d}_{train_end:%Y%m%d}",
        )

        gt = build_groundtruth(df_min, eval_begin, eval_end, max_rows=max_rows_eval)
        train_hist = build_train_hist_sets(df_min, train_begin, train_end)

        if cfg.quick and cfg.quick_max_users_eval > 0:
            # deterministic cap for smoke test
            users_eval = list(gt.keys())[:int(cfg.quick_max_users_eval)]
            gt = {u: gt[u] for u in users_eval}

        rep_f = eval_recall_hit(
            fitted=fitted, gt=gt, train_hist=train_hist,
            N_cand=trial_cfg.N_cand, N_trend=trial_cfg.N_trend,
            filter_train_history_in_gt=True,
            batch_users=trial_cfg.batch_users,
        )
        rep_u = eval_recall_hit(
            fitted=fitted, gt=gt, train_hist=train_hist,
            N_cand=trial_cfg.N_cand, N_trend=trial_cfg.N_trend,
            filter_train_history_in_gt=False,
            batch_users=trial_cfg.batch_users,
        )

        params_json = json.dumps({
            "len_hist": trial_cfg.len_hist,
            "len_recent": trial_cfg.len_recent,
            "N_cand": trial_cfg.N_cand,
            "N_trend": trial_cfg.N_trend,
            "weight_type": trial_cfg.weight_type,
            "K_model_mult": trial_cfg.K_model_mult,
        }, ensure_ascii=False)

        with open(results_csv, "a", encoding="utf-8") as f:
            f.write(
                f"{trial},{rep_f.recall:.10f},{rep_f.hit:.10f},{rep_u.recall:.10f},{rep_u.hit:.10f},"
                f"{rep_f.n_users_eval},{rep_u.n_users_eval},{rep_f.n_cold_start},{rep_u.n_cold_start},"
                f"{params_json}\n"
            )

        if (rep_f.recall > best_recall_f) or (np.isclose(rep_f.recall, best_recall_f) and rep_f.hit > best_hit_f):
            best_recall_f = rep_f.recall
            best_hit_f = rep_f.hit
            best_params = json.loads(params_json)
            best_f = rep_f
            best_u = rep_u
            with open(best_params_path, "w", encoding="utf-8") as fp:
                json.dump(best_params, fp, ensure_ascii=False, indent=2)
            with open(best_metrics_path, "w", encoding="utf-8") as fp:
                json.dump({"filtered": best_f.__dict__, "unfiltered": best_u.__dict__}, fp, ensure_ascii=False, indent=2)
            log(f"[RS] New best trial={trial}: RecallF@{rep_f.K}={rep_f.recall:.6f}, HitF={rep_f.hit:.6f} | params={best_params}")

    if best_params is None or best_f is None or best_u is None:
        raise RuntimeError("Random search did not produce a valid best result.")

    return best_params, best_f, best_u


# -------------------------
# Train best + save artifacts for serving
# -------------------------
def train_best_and_save(
    cfg: Stage1Config,
    df_min: pl.DataFrame,
    best_params: dict,
    train_end: datetime,
    recent_begin: datetime,
    out_run_dir: str,
    cache_dir: str,
) -> None:
    """
    Retrain using best params (caches are reused) and write run artifacts + metrics + cold-start lists.
    """
    log("[6/6] Train BEST model and save run artifacts ...")
    _ensure_dir(out_run_dir)

    cfg2 = Stage1Config(**{**cfg.__dict__})
    cfg2.len_hist = int(best_params["len_hist"])
    cfg2.len_recent = int(best_params["len_recent"])
    cfg2.N_cand = int(best_params["N_cand"])
    cfg2.N_trend = int(best_params["N_trend"])
    cfg2.weight_type = str(best_params.get("weight_type", cfg.weight_type))
    cfg2.K_model_mult = int(best_params.get("K_model_mult", cfg.K_model_mult))

    train_begin = train_end - timedelta(days=cfg2.len_hist - 1)
    eval_begin = recent_begin
    eval_end = recent_begin + timedelta(days=cfg2.len_recent - 1)

    max_rows_train = cfg.quick_max_rows_train if cfg.quick else cfg.max_rows_train
    max_rows_eval = cfg.quick_max_rows_eval if cfg.quick else cfg.max_rows_eval

    mat, users, items, u2i, it2i, trending_idx = build_user_items_csr(
        df=df_min,
        begin=train_begin,
        end=train_end,
        weight_type=cfg2.weight_type,
        cache_dir=cache_dir,
        cache_prefix=f"train_{train_begin:%Y%m%d}_{train_end:%Y%m%d}",
        max_rows=max_rows_train,
    )
    fitted = fit_or_load_models(
        cfg=cfg2,
        mat=mat,
        users=users,
        items=items,
        trending_idx=trending_idx,
        cache_dir=cache_dir,
        cache_prefix=f"models_{train_begin:%Y%m%d}_{train_end:%Y%m%d}",
    )

    gt = build_groundtruth(df_min, eval_begin, eval_end, max_rows=max_rows_eval)
    train_hist = build_train_hist_sets(df_min, train_begin, train_end)

    if cfg.quick and cfg.quick_max_users_eval > 0:
        users_eval = list(gt.keys())[:int(cfg.quick_max_users_eval)]
        gt = {u: gt[u] for u in users_eval}

    rep_f = eval_recall_hit(
        fitted=fitted, gt=gt, train_hist=train_hist,
        N_cand=cfg2.N_cand, N_trend=cfg2.N_trend,
        filter_train_history_in_gt=True,
        batch_users=cfg2.batch_users,
    )
    rep_u = eval_recall_hit(
        fitted=fitted, gt=gt, train_hist=train_hist,
        N_cand=cfg2.N_cand, N_trend=cfg2.N_trend,
        filter_train_history_in_gt=False,
        batch_users=cfg2.batch_users,
    )

    cold_filtered = compute_cold_users(fitted=fitted, gt=gt, train_hist=train_hist, filter_train_history_in_gt=True)
    cold_unfiltered = compute_cold_users(fitted=fitted, gt=gt, train_hist=train_hist, filter_train_history_in_gt=False)

    # save cold-start lists
    cold_f_path = os.path.join(out_run_dir, "cold_start_users_filtered.pkl")
    cold_u_path = os.path.join(out_run_dir, "cold_start_users_unfiltered.pkl")
    with open(cold_f_path, "wb") as f:
        pickle.dump(cold_filtered, f)
    with open(cold_u_path, "wb") as f:
        pickle.dump(cold_unfiltered, f)
    log(f"[SAVE] Cold-start users (filtered)  -> {cold_f_path}  (n={len(cold_filtered):,})")
    log(f"[SAVE] Cold-start users (unfiltered)-> {cold_u_path}  (n={len(cold_unfiltered):,})")

    # save run artifacts
    meta_path = os.path.join(out_run_dir, "best_stage1_meta.npz")
    tfidf_path = os.path.join(out_run_dir, "best_stage1_tfidf.npz")
    cosine_path = os.path.join(out_run_dir, "best_stage1_cosine.npz")
    metrics_path = os.path.join(out_run_dir, "best_stage1_metrics.json")

    np.savez_compressed(
        meta_path,
        users=np.array(users, dtype=object),
        items=np.array(items, dtype=object),
        trending_idx=trending_idx.astype(np.int32),
        config_json=json.dumps({
            "train_end": cfg.train_end,
            "recent_begin": cfg.recent_begin,
            "len_hist": cfg2.len_hist,
            "len_recent": cfg2.len_recent,
            "N_cand": cfg2.N_cand,
            "N_trend": cfg2.N_trend,
            "weight_type": cfg2.weight_type,
            "K_model_mult": cfg2.K_model_mult,
            "num_threads": int(cfg2.num_threads),
        }, ensure_ascii=False),
    )
    fitted.tfidf.save(tfidf_path)
    fitted.cosine.save(cosine_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"filtered": rep_f.__dict__, "unfiltered": rep_u.__dict__}, f, ensure_ascii=False, indent=2)

    log(f"[SAVE] Run meta    : {meta_path}")
    log(f"[SAVE] TFIDF model : {tfidf_path}")
    log(f"[SAVE] Cosine model: {cosine_path}")
    log(f"[SAVE] Metrics     : {metrics_path}")

    log("===== BEST EVAL REPORT =====")
    log("[FILTERED] (new-item) remove train-history items from GT; recommend(filter_already_liked_items=True)")
    log(f"          Recall@{rep_f.K}={rep_f.recall:.6f}  Hit@{rep_f.K}={rep_f.hit:.6f}  "
        f"n_users_eval={rep_f.n_users_eval:,}  cold_start_users={rep_f.n_cold_start:,}")
    log("[UNFILTERED] (overall) keep repeats in GT; recommend(filter_already_liked_items=False)")
    log(f"           Recall@{rep_u.K}={rep_u.recall:.6f}  Hit@{rep_u.K}={rep_u.hit:.6f}  "
        f"n_users_eval={rep_u.n_users_eval:,}  cold_start_users={rep_u.n_cold_start:,}")


# -------------------------
# Main
# -------------------------
def main(config_path: str) -> None:
    cfg = load_config(config_path)

    artifacts_dir = cfg.artifacts_dir
    run_dir = os.path.join(artifacts_dir, "runs", cfg.run_name)
    cache_dir = os.path.join(artifacts_dir, "cache")
    _ensure_dir(run_dir)
    _ensure_dir(cache_dir)

    if cfg.quick:
        log("[MODE] QUICK = True (smoke test)")

    train_end = _parse_date(cfg.train_end)
    recent_begin = _parse_date(cfg.recent_begin)

    # Determine maximal coverage window needed (for tx_min cache)
    space = _default_space(cfg) if cfg.do_random_search else {
        "len_hist": [cfg.len_hist],
        "len_recent": [cfg.len_recent],
        "N_cand": [cfg.N_cand],
        "N_trend": [cfg.N_trend],
        "weight_type": [cfg.weight_type],
        "K_model_mult": [cfg.K_model_mult],
    }
    max_len_hist = int(max(space["len_hist"]))
    max_len_recent = int(max(space["len_recent"]))

    earliest_train_begin = train_end - timedelta(days=max_len_hist - 1)
    latest_eval_end = recent_begin + timedelta(days=max_len_recent - 1)

    log("==== Stage1 Retrieval (implicit TFIDF + Cosine) ====")
    log(f"Config path   : {config_path}")
    log(f"Artifacts dir : {artifacts_dir}")
    log(f"Run dir       : {run_dir}")
    log(f"Cache dir     : {cache_dir}")
    log(f"Anchors       : train_end={cfg.train_end} | recent_begin={cfg.recent_begin}")
    log(f"Max windows   : train_begin={earliest_train_begin.date()} -> train_end={train_end.date()} | "
        f"eval_begin={recent_begin.date()} -> eval_end={latest_eval_end.date()}")

    # Step 1: minimal tx cache
    _, df_min = build_min_transactions(
        cfg=cfg,
        train_begin=earliest_train_begin,
        eval_end=latest_eval_end,
        cache_dir=cache_dir,
    )
    log(f"[INFO] Minimal transactions rows: {df_min.height:,}")

    # Step 2: random search and train best (or single run)
    if cfg.do_random_search:
        best_params, _, _ = random_search_stage1(
            cfg=cfg,
            df_min=df_min,
            train_end=train_end,
            recent_begin=recent_begin,
            out_run_dir=run_dir,
            cache_dir=cache_dir,
        )
        log(f"[RS] Best params: {best_params}")
        train_best_and_save(
            cfg=cfg,
            df_min=df_min,
            best_params=best_params,
            train_end=train_end,
            recent_begin=recent_begin,
            out_run_dir=run_dir,
            cache_dir=cache_dir,
        )
    else:
        best_params = {
            "len_hist": cfg.len_hist,
            "len_recent": cfg.len_recent,
            "N_cand": cfg.N_cand,
            "N_trend": cfg.N_trend,
            "weight_type": cfg.weight_type,
            "K_model_mult": cfg.K_model_mult,
        }
        with open(os.path.join(run_dir, "best_params.json"), "w", encoding="utf-8") as f:
            json.dump(best_params, f, ensure_ascii=False, indent=2)

        train_best_and_save(
            cfg=cfg,
            df_min=df_min,
            best_params=best_params,
            train_end=train_end,
            recent_begin=recent_begin,
            out_run_dir=run_dir,
            cache_dir=cache_dir,
        )

    log("DONE: Stage1 retrieval pipeline complete.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    main(args.config)
