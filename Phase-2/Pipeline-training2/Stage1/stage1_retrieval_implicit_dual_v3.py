# -*- coding: utf-8 -*-
"""
stage1_retrieval_implicit_dual_v3.py

Stage-1 Retrieval (Candidate Generation) using implicit:
  - TFIDFRecommender
  - CosineRecommender

Added (per user request):
  1) Recency weighting for weight(u,i) using transaction.created_date (train window).
  2) Discount-aware weighting using transaction.discount_rate (if present).
  3) Additional retrieval channel: user–category_l2 (anti-sparsity) + mapping back to items.
  4) Optional cohort post-filter using user profile + item attributes (kept conservative).

Also includes:
  - Random search across len_hist, len_recent, N_cand, N_trend, weight_type, recency, discount, and category channel toggles.
  - Standard metrics: Recall@K / Hit@K (filtered per-user new-item, and unfiltered overall)
  - NEW-ITEMS metrics: items never seen in global history [seen_begin, seen_end]
  - Caching for speed: minimal transactions, seen_items, CSR matrices, fitted models, category mappings.

Run:
  python stage1_retrieval_implicit_dual_v3.py --config stage1_quick_config_v3.json
  python stage1_retrieval_implicit_dual_v3.py --config stage1_full_config_v3.json
"""

from __future__ import annotations

import os
import json
import time
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set

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

    # global history for NEW-ITEMS evaluation (independent of len_hist)
    seen_begin: str = "2024-01-01"      # inclusive
    seen_end: Optional[str] = None      # if None, defaults to train_end

    # base params
    len_hist: int = 120
    len_recent: int = 28
    N_cand: int = 100
    N_trend: int = 100

    # modeling knobs
    weight_type: str = "log_count"      # binary|count|log_count|log_qty|log_spent
    K_model_mult: int = 3               # K_model = clip(20, 800, K_model_mult*N_cand)
    num_threads: int = 0                # 0 => implicit default

    # recency weighting
    use_recency: bool = True
    recency_halflife_days: float = 30.0  # weight *= 0.5 ** (age_days / halflife)

    # discount weighting
    use_discount: bool = True
    discount_beta: float = 0.7          # weight *= clip(1 - beta*discount_rate, min_discount_factor, 1)
    min_discount_factor: float = 0.2

    # category channel
    use_category_channel: bool = True
    N_cat: int = 50                      # how many categories to retrieve per user
    cat_items_per_cat: int = 30          # items to expand per category (before dedup)

    # cohort post-filter (conservative; optional)
    cohort_mode: str = "none"            # none|gender_target

    # evaluation
    batch_users: int = 5000
    max_rows_train: int = 0
    max_rows_eval: int = 0
    filter_train_history_in_gt: bool = True
    # after-eval refit & export (optional)
    refit_end_at_eval_end: bool = True
    export_all_users_out: Optional[str] = None  # e.g. "./artifacts_stage1/candidates_all_users_refit_202412.parquet"
    # groundtruth & offline metrics/candidates (optional)
    groundtruth_pkl_path: Optional[str] = None  # path to PRIVATE test gt JSON (customer_id, item_id) for Jan/2025
    export_gt_users_out: Optional[str] = None  # export candidates only for users in groundtruth
    compute_recall_from_groundtruth: bool = True
    recall_k: int = 0  # 0 => use K = 2*N_cand + N_trend

    # random search
    do_random_search: bool = True
    n_trials: int = 90
    random_state: int = 42
    search_space: Optional[dict] = None

    # quick
    quick: bool = False
    quick_n_trials: int = 5
    quick_max_rows_train: int = 300_000
    quick_max_rows_eval: int = 200_000
    quick_max_users_eval: int = 5000


def load_config(path: str) -> Stage1Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return Stage1Config(**raw)


# -------------------------
# Helpers
# -------------------------
def load_private_gt_json_2cols(path: str) -> Dict[str, Set[str]]:
    """
    Load private test groundtruth (Jan/2025) from a JSON file with 2 columns:
      - customer_id
      - item_id

    Supported JSON formats:
      A) list of objects: [{"customer_id": "...", "item_id": "..."}, ...]
      B) dict of lists: {"customer_id":[...], "item_id":[...]}

    Returns:
      gt: Dict[user_id -> Set[item_id]]
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    gt: Dict[str, Set[str]] = {}

    if isinstance(raw, list):
        # list of rows
        for row in raw:
            if row is None:
                continue
            u = row.get("customer_id", None)
            it = row.get("item_id", None)
            if u is None or it is None:
                continue
            gt.setdefault(str(u), set()).add(str(it))
        return gt

    if isinstance(raw, dict):
        # columnar dict
        if ("customer_id" not in raw) or ("item_id" not in raw):
            raise ValueError(f"JSON dict must contain keys customer_id and item_id. Found keys={list(raw.keys())}")
        cu = raw["customer_id"]
        ci = raw["item_id"]
        if len(cu) != len(ci):
            raise ValueError(f"Length mismatch: customer_id={len(cu)} vs item_id={len(ci)}")
        for u, it in zip(cu, ci):
            if u is None or it is None:
                continue
            gt.setdefault(str(u), set()).add(str(it))
        return gt

    raise ValueError(f"Unsupported JSON structure in {path}: {type(raw)}")

def _save_pickle(obj, path: str) -> None:
    import pickle
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def recall_at_k(
    pred: Dict[str, List[str]],
    gt: Dict[str, Set[str]],
    hist: Dict[str, Set[str]],
    K: int,
    filter_bought_items: bool = True,
    show_progress: bool = True,
) -> Tuple[float, List[str]]:
    """
    Recall@K:
      - relevant = gt[u] (unfiltered) OR gt[u] \ hist[u] (filtered)
      - recall_u = |pred[:K] ∩ relevant| / |relevant|
    Cold-start definition (matches your Precision@K style):
      - user missing in hist OR missing in pred -> counted as cold-start and skipped from mean.
    """
    recalls: List[float] = []
    cold_start_users: List[str] = []

    users_iter = list(gt.keys())
    if show_progress:
        users_iter = tqdm(users_iter, desc=f"Recall@{K} (filtered={filter_bought_items})")

    for u in users_iter:
        if (u not in hist) or (u not in pred):
            cold_start_users.append(u)
            continue

        relevant = set(gt[u])
        if filter_bought_items:
            relevant -= set(hist[u])

        # avoid division by zero; skip user if no relevant items
        if not relevant:
            continue

        hits = len(set(pred[u][:K]) & relevant)
        recalls.append(hits / len(relevant))

    mean_recall = float(np.mean(recalls)) if recalls else 0.0
    return mean_recall, cold_start_users


def recall_at_k_both(
    pred: Dict[str, List[str]],
    gt: Dict[str, Set[str]],
    hist: Dict[str, Set[str]],
    K: int,
    show_progress: bool = True,
) -> Dict[str, object]:
    r_u, cold_u = recall_at_k(pred, gt, hist, K=K, filter_bought_items=False, show_progress=show_progress)
    r_f, cold_f = recall_at_k(pred, gt, hist, K=K, filter_bought_items=True, show_progress=show_progress)
    cold = sorted(set(cold_u) | set(cold_f))
    return {
        "recall_unfiltered": r_u,
        "recall_filtered": r_f,
        "n_cold_start": len(cold),
        "cold_start_users": cold,
    }


def build_pred_dict_for_users(
    cfg: Stage1Config,
    fitted: "FittedStage1",
    users: List[str],
    N_cand: int,
    N_trend: int,
    K: int,
    show_progress: bool = True,
) -> Dict[str, List[str]]:
    """
    Build prediction dict: pred[u] = ranked list of item_ids (length up to K).
    """
    pred: Dict[str, List[str]] = {}
    it = users
    if show_progress:
        it = tqdm(users, desc=f"Build pred_dict @K={K}")
    for u in it:
        rec = recommend_for_user(
            cfg=cfg,
            fitted=fitted,
            user_id=str(u),
            N_cand=int(N_cand),
            N_trend=int(N_trend),
            filter_already_liked_items=False,
        )
        pred[str(u)] = rec[:K]
    return pred


def export_candidates_for_users_parquet(
    cfg: Stage1Config,
    fitted: "FittedStage1",
    users: List[str],
    out_path: str,
    N_cand: int,
    N_trend: int,
) -> None:
    """
    Export minimal candidates (customer_id, item_id, rank) for a given user list.
    Intended usage: users = list(gt.keys()) (i.e., only users that exist in groundtruth).
    """
    t0 = time.time()
    K = 2 * int(N_cand) + int(N_trend)

    rows: List[Tuple[str, str, int]] = []
    for u in tqdm(users, desc=f"Export candidates (gt users) @K={K}"):
        rec = recommend_for_user(cfg, fitted, str(u), N_cand=int(N_cand), N_trend=int(N_trend), filter_already_liked_items=False)
        for r, it in enumerate(rec[:K], start=1):
            rows.append((str(u), str(it), int(r)))

    df = pl.DataFrame(rows, schema=[("customer_id", pl.Utf8), ("item_id", pl.Utf8), ("rank", pl.Int32)])
    _ensure_dir(os.path.dirname(out_path) or ".")
    df.write_parquet(out_path)
    log(f"[SAVE] GT-users candidates -> {out_path} (users={len(users):,}, rows={df.height:,}, elapsed={time.time()-t0:.1f}s)")

def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _hash_dict(d: dict) -> str:
    s = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]

def _glob_scan(glob_path: str) -> pl.LazyFrame:
    return pl.scan_parquet(glob_path)

def _clip_int(a: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, a))


# -------------------------
# Load minimal transactions (typed) with optional discount_rate
# -------------------------
def build_min_transactions(cfg: Stage1Config, begin_all: datetime, end_all: datetime, cache_dir: str) -> pl.DataFrame:
    key = _hash_dict({
        "transactions_path_glob": cfg.transactions_path_glob,
        "items_path_glob": cfg.items_path_glob,
        "begin_all": begin_all.strftime("%Y-%m-%d"),
        "end_all": end_all.strftime("%Y-%m-%d"),
        "need_discount": True,  # keep stable, even if absent in source
    })
    cache_path = os.path.join(cache_dir, f"tx_min_{key}.parquet")
    if os.path.exists(cache_path):
        log(f"[CACHE] Load minimal transactions: {cache_path}")
        return pl.read_parquet(cache_path)

    log("[1/8] Scan transactions (lazy) ...")
    tx = _glob_scan(cfg.transactions_path_glob)
    cols = set(tx.columns)

    need = {"customer_id", "item_id", "created_date"}
    missing = [c for c in need if c not in cols]
    if missing:
        raise ValueError(f"Transactions missing required columns: {missing}. Available: {tx.columns}")

    has_qty = "quantity" in cols
    has_price = "price" in cols
    has_disc_rate = "discount_rate" in cols

    tx = tx.with_columns([
        pl.col("customer_id").cast(pl.Utf8),
        pl.col("item_id").cast(pl.Utf8),
        pl.col("created_date").cast(pl.Date, strict=False),
        (pl.col("quantity").cast(pl.Float64, strict=False).fill_null(1.0).fill_nan(1.0) if has_qty else pl.lit(1.0).alias("quantity")),
        (pl.col("price").cast(pl.Float64, strict=False).fill_null(0.0).fill_nan(0.0) if has_price else pl.lit(0.0).alias("price")),
        (pl.col("discount_rate").cast(pl.Float64, strict=False).fill_null(0.0).fill_nan(0.0) if has_disc_rate else pl.lit(0.0).alias("discount_rate")),
    ]).select(["customer_id", "item_id", "created_date", "quantity", "price", "discount_rate"])

    tx = tx.filter(
        pl.col("created_date").is_between(pl.lit(begin_all.date(), dtype=pl.Date), pl.lit(end_all.date(), dtype=pl.Date), closed="both")
    )

    if cfg.items_path_glob:
        log("[1/8] Scan items (lazy) for valid item_id filter ...")
        it = _glob_scan(cfg.items_path_glob)
        if "item_id" in it.columns:
            valid = it.select(pl.col("item_id").cast(pl.Utf8)).unique()
            tx = tx.join(valid, on="item_id", how="inner")
        else:
            log("[WARN] items table has no item_id; skip item filter.")

    log("[1/8] Materialize minimal transactions ...")
    df = tx.collect(streaming=True)

    _ensure_dir(cache_dir)
    log(f"[SAVE] Minimal transactions -> {cache_path}")
    df.write_parquet(cache_path)
    return df


# -------------------------
# Seen items cache
# -------------------------
def compute_seen_items(df_min: pl.DataFrame, seen_begin: datetime, seen_end: datetime, cache_dir: str) -> Set[str]:
    key = _hash_dict({
        "seen_begin": seen_begin.strftime("%Y-%m-%d"),
        "seen_end": seen_end.strftime("%Y-%m-%d"),
        "rows": int(df_min.height),
    })
    cache_path = os.path.join(cache_dir, f"seen_items_{key}.pkl")
    if os.path.exists(cache_path):
        log(f"[CACHE] Load seen_items: {cache_path}")
        import pickle
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    log("[2/8] Compute seen_items ...")
    d = df_min.filter(
        pl.col("created_date").is_between(pl.lit(seen_begin.date(), dtype=pl.Date), pl.lit(seen_end.date(), dtype=pl.Date), closed="both")
    ).select("item_id").unique()

    seen = set(d["item_id"].to_list())
    _ensure_dir(cache_dir)
    import pickle
    with open(cache_path, "wb") as f:
        pickle.dump(seen, f, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"[SAVE] seen_items -> {cache_path} (n={len(seen):,})")
    return seen


# -------------------------
# Item + user tables (for category channel and cohort)
# -------------------------
def load_item_table(items_path_glob: Optional[str], cache_dir: str) -> Optional[pl.DataFrame]:
    if not items_path_glob:
        return None
    key = _hash_dict({"items_path_glob": items_path_glob})
    cache_path = os.path.join(cache_dir, f"items_min_{key}.parquet")
    if os.path.exists(cache_path):
        log(f"[CACHE] Load items_min: {cache_path}")
        return pl.read_parquet(cache_path)

    log("[3/8] Scan items (lazy) ...")
    it = _glob_scan(items_path_glob)
    cols = set(it.columns)
    # minimal columns we need: item_id, category_l2, gender_target_final (for cohort)
    use_cols = []
    if "item_id" in cols: use_cols.append(pl.col("item_id").cast(pl.Utf8))
    else: 
        log("[WARN] items table missing item_id; category channel & cohort disabled.")
        return None
    if "category_l2" in cols: use_cols.append(pl.col("category_l2").cast(pl.Utf8))
    else: use_cols.append(pl.lit(None, dtype=pl.Utf8).alias("category_l2"))
    if "gender_target_final" in cols: use_cols.append(pl.col("gender_target_final").cast(pl.Utf8))
    else: use_cols.append(pl.lit(None, dtype=pl.Utf8).alias("gender_target_final"))

    it2 = it.select(use_cols).unique()
    df = it2.collect(streaming=True)
    _ensure_dir(cache_dir)
    log(f"[SAVE] items_min -> {cache_path}")
    df.write_parquet(cache_path)
    return df


def load_user_table(users_path_glob: Optional[str], cache_dir: str) -> Optional[pl.DataFrame]:
    if not users_path_glob:
        return None
    key = _hash_dict({"users_path_glob": users_path_glob})
    cache_path = os.path.join(cache_dir, f"users_min_{key}.parquet")
    if os.path.exists(cache_path):
        log(f"[CACHE] Load users_min: {cache_path}")
        return pl.read_parquet(cache_path)

    log("[3/8] Scan users (lazy) ...")
    us = _glob_scan(users_path_glob)
    cols = set(us.columns)
    if "customer_id" not in cols:
        log("[WARN] users table missing customer_id; cohort disabled.")
        return None
    # minimal: customer_id, gender
    if "gender" in cols:
        us2 = us.select([pl.col("customer_id").cast(pl.Utf8), pl.col("gender").cast(pl.Utf8)])
    else:
        us2 = us.select([pl.col("customer_id").cast(pl.Utf8), pl.lit(None, dtype=pl.Utf8).alias("gender")])
    df = us2.unique().collect(streaming=True)
    _ensure_dir(cache_dir)
    log(f"[SAVE] users_min -> {cache_path}")
    df.write_parquet(cache_path)
    return df


# -------------------------
# CSR for items with recency + discount
# -------------------------
def build_user_items_csr(
    df_min: pl.DataFrame,
    train_begin: datetime,
    train_end: datetime,
    cfg: Stage1Config,
    cache_dir: str,
    cache_prefix: str,
    max_rows: int = 0,
) -> Tuple[csr_matrix, List[str], List[str], np.ndarray]:
    """
    Build user-item CSR for [train_begin, train_end] inclusive with:
      base_event * recency_factor * discount_factor summed per (u,i),
      then optional log1p depending on weight_type.
    """
    key = _hash_dict({
        "cache_prefix": cache_prefix,
        "train_begin": train_begin.strftime("%Y-%m-%d"),
        "train_end": train_end.strftime("%Y-%m-%d"),
        "weight_type": cfg.weight_type,
        "use_recency": cfg.use_recency,
        "recency_halflife_days": float(cfg.recency_halflife_days),
        "use_discount": cfg.use_discount,
        "discount_beta": float(cfg.discount_beta),
        "min_discount_factor": float(cfg.min_discount_factor),
        "max_rows": int(max_rows),
    })
    csr_path = os.path.join(cache_dir, f"csr_item_{key}.npz")
    maps_path = os.path.join(cache_dir, f"maps_item_{key}.npz")
    trend_path = os.path.join(cache_dir, f"trending_item_{key}.npy")

    if os.path.exists(csr_path) and os.path.exists(maps_path) and os.path.exists(trend_path):
        log(f"[CACHE] Load item-CSR: {csr_path}")
        mat = load_npz(csr_path).tocsr()
        maps = np.load(maps_path, allow_pickle=True)
        users = maps["users"].tolist()
        items = maps["items"].tolist()
        trending_idx = np.load(trend_path)
        return mat, users, items, trending_idx

    log("[4/8] Build user-item CSR (recency/discount aware) ...")
    d = df_min.filter(
        pl.col("created_date").is_between(pl.lit(train_begin.date(), dtype=pl.Date), pl.lit(train_end.date(), dtype=pl.Date), closed="both")
    )
    if max_rows and max_rows > 0:
        d = d.head(max_rows)

    # base event value
    d = d.with_columns([
        (pl.col("quantity") * pl.col("price")).alias("spent_row"),
    ])

    # age_days = (train_end - created_date)
    d = d.with_columns([
        (pl.lit(train_end.date(), dtype=pl.Date) - pl.col("created_date")).dt.total_days().cast(pl.Float64).alias("age_days")
    ])

    # recency factor
    if cfg.use_recency:
        # factor = 0.5 ** (age / halflife) = exp(ln(0.5)*age/halflife)
        ln_half = float(np.log(0.5))
        hl = float(cfg.recency_halflife_days) if cfg.recency_halflife_days > 0 else 1.0
        d = d.with_columns((pl.col("age_days") * (ln_half / hl)).exp().alias("recency_factor"))
    else:
        d = d.with_columns(pl.lit(1.0).alias("recency_factor"))

    # discount factor
    if cfg.use_discount:
        beta = float(cfg.discount_beta)
        minf = float(cfg.min_discount_factor)
        d = d.with_columns([
            (1.0 - beta * pl.col("discount_rate").fill_null(0.0)).clip(minf, 1.0).alias("discount_factor")
        ])
    else:
        d = d.with_columns(pl.lit(1.0).alias("discount_factor"))

    # base per event according to weight_type (before log)
    wt = cfg.weight_type
    if wt in ("binary", "count", "log_count"):
        d = d.with_columns(pl.lit(1.0).alias("base_event"))
        use_log = (wt == "log_count")
    elif wt in ("log_qty",):
        d = d.with_columns(pl.col("quantity").alias("base_event"))
        use_log = True
    elif wt in ("log_spent",):
        d = d.with_columns(pl.col("spent_row").alias("base_event"))
        use_log = True
    elif wt in ("log_spent", "log_qty"):
        use_log = True
    elif wt == "count":
        d = d.with_columns(pl.lit(1.0).alias("base_event"))
        use_log = False
    else:
        raise ValueError(f"Unknown weight_type: {wt}")

    d = d.with_columns((pl.col("base_event") * pl.col("recency_factor") * pl.col("discount_factor")).alias("event_weight"))

    ui = d.group_by(["customer_id", "item_id"]).agg([
        pl.col("event_weight").sum().alias("value_raw"),
    ])

    if wt == "binary":
        ui = ui.with_columns(pl.lit(1.0).alias("value"))
    elif wt == "count":
        # sum of event_weight is already discounted/recency; keep raw
        ui = ui.with_columns(pl.col("value_raw").cast(pl.Float64).alias("value"))
    else:
        # log variants
        ui = ui.with_columns(pl.col("value_raw").cast(pl.Float64).log1p().alias("value"))

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
    save_npz(csr_path, mat)
    np.savez_compressed(maps_path, users=np.array(users, dtype=object), items=np.array(items, dtype=object))
    np.save(trend_path, trending_idx)
    log(f"[SAVE] item-CSR + maps + trending cached (key={key})")

    return mat, users, items, trending_idx


# -------------------------
# Category channel: user-category_l2 CSR + category->items mapping
# -------------------------
def build_user_category_csr_and_mapping(
    df_min: pl.DataFrame,
    items_df: Optional[pl.DataFrame],
    train_begin: datetime,
    train_end: datetime,
    cfg: Stage1Config,
    cache_dir: str,
    cache_prefix: str,
    max_rows: int = 0,
) -> Tuple[Optional[csr_matrix], Optional[List[str]], Optional[List[str]], Optional[dict]]:
    """
    Returns:
      user_cat_csr, users, categories, cat_to_items(list of item_ids sorted by pop within cat)
    If items_df is None or category_l2 is missing, returns all None.
    """
    if (not cfg.use_category_channel) or (items_df is None) or ("category_l2" not in items_df.columns):
        return None, None, None, None

    key = _hash_dict({
        "cache_prefix": cache_prefix,
        "train_begin": train_begin.strftime("%Y-%m-%d"),
        "train_end": train_end.strftime("%Y-%m-%d"),
        "weight_type": cfg.weight_type,
        "use_recency": cfg.use_recency,
        "recency_halflife_days": float(cfg.recency_halflife_days),
        "use_discount": cfg.use_discount,
        "discount_beta": float(cfg.discount_beta),
        "min_discount_factor": float(cfg.min_discount_factor),
        "max_rows": int(max_rows),
    })
    csr_path = os.path.join(cache_dir, f"csr_cat_{key}.npz")
    maps_path = os.path.join(cache_dir, f"maps_cat_{key}.npz")
    map_path = os.path.join(cache_dir, f"cat_to_items_{key}.json")

    if os.path.exists(csr_path) and os.path.exists(maps_path) and os.path.exists(map_path):
        log(f"[CACHE] Load category channel caches (key={key})")
        mat = load_npz(csr_path).tocsr()
        maps = np.load(maps_path, allow_pickle=True)
        users = maps["users"].tolist()
        cats = maps["cats"].tolist()
        with open(map_path, "r", encoding="utf-8") as f:
            cat_to_items = json.load(f)
        return mat, users, cats, cat_to_items

    log("[5/8] Build user-category CSR + category->items mapping ...")

    # join transactions with items to get category_l2
    it_map = items_df.select(["item_id", "category_l2"]).unique()
    d = df_min.filter(
        pl.col("created_date").is_between(pl.lit(train_begin.date(), dtype=pl.Date), pl.lit(train_end.date(), dtype=pl.Date), closed="both")
    ).join(it_map, on="item_id", how="left")

    d = d.filter(pl.col("category_l2").is_not_null())
    if max_rows and max_rows > 0:
        d = d.head(max_rows)

    # reuse event_weight logic (same as item CSR)
    d = d.with_columns([(pl.col("quantity") * pl.col("price")).alias("spent_row")])
    d = d.with_columns([(pl.lit(train_end.date(), dtype=pl.Date) - pl.col("created_date")).dt.total_days().cast(pl.Float64).alias("age_days")])

    if cfg.use_recency:
        ln_half = float(np.log(0.5))
        hl = float(cfg.recency_halflife_days) if cfg.recency_halflife_days > 0 else 1.0
        d = d.with_columns((pl.col("age_days") * (ln_half / hl)).exp().alias("recency_factor"))
    else:
        d = d.with_columns(pl.lit(1.0).alias("recency_factor"))

    if cfg.use_discount:
        beta = float(cfg.discount_beta)
        minf = float(cfg.min_discount_factor)
        d = d.with_columns((1.0 - beta * pl.col("discount_rate").fill_null(0.0)).clip(minf, 1.0).alias("discount_factor"))
    else:
        d = d.with_columns(pl.lit(1.0).alias("discount_factor"))

    wt = cfg.weight_type
    if wt in ("binary", "count", "log_count"):
        d = d.with_columns(pl.lit(1.0).alias("base_event"))
    elif wt == "log_qty":
        d = d.with_columns(pl.col("quantity").alias("base_event"))
    elif wt == "log_spent":
        d = d.with_columns(pl.col("spent_row").alias("base_event"))
    else:
        raise ValueError(f"Unknown weight_type: {wt}")

    d = d.with_columns((pl.col("base_event") * pl.col("recency_factor") * pl.col("discount_factor")).alias("event_weight"))

    uc = d.group_by(["customer_id", "category_l2"]).agg(pl.col("event_weight").sum().alias("value_raw"))
    if wt == "binary":
        uc = uc.with_columns(pl.lit(1.0).alias("value"))
    elif wt == "count":
        uc = uc.with_columns(pl.col("value_raw").cast(pl.Float64).alias("value"))
    else:
        uc = uc.with_columns(pl.col("value_raw").cast(pl.Float64).log1p().alias("value"))

    users = uc.select("customer_id").unique().sort("customer_id").to_series().to_list()
    cats = uc.select("category_l2").unique().sort("category_l2").to_series().to_list()
    u2i = {u: i for i, u in enumerate(users)}
    c2i = {c: j for j, c in enumerate(cats)}

    u_idx = uc.select("customer_id").to_series().to_list()
    c_idx = uc.select("category_l2").to_series().to_list()
    data = uc.select("value").to_series().to_numpy().astype(np.float32)
    row = np.fromiter((u2i[u] for u in u_idx), dtype=np.int32, count=len(u_idx))
    col = np.fromiter((c2i[c] for c in c_idx), dtype=np.int32, count=len(c_idx))

    mat = csr_matrix((data, (row, col)), shape=(len(users), len(cats)), dtype=np.float32)

    # category -> items list sorted by item popularity within category (train window)
    # popularity measured by sum(event_weight) per (category,item)
    ci = d.group_by(["category_l2", "item_id"]).agg(pl.col("event_weight").sum().alias("w")).sort(["category_l2", "w"], descending=[False, True])
    cat_to_items: Dict[str, List[str]] = {}
    for c, it in zip(ci["category_l2"].to_list(), ci["item_id"].to_list()):
        c = str(c); it = str(it)
        cat_to_items.setdefault(c, []).append(it)

    _ensure_dir(cache_dir)
    save_npz(csr_path, mat)
    np.savez_compressed(maps_path, users=np.array(users, dtype=object), cats=np.array(cats, dtype=object))
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(cat_to_items, f, ensure_ascii=False)
    log(f"[SAVE] category channel cached (key={key})")

    return mat, users, cats, cat_to_items


# -------------------------
# Fit models + cache helper
# -------------------------
@dataclass
class ImplicitPair:
    tfidf: TFIDFRecommender
    cosine: CosineRecommender

def fit_or_load_pair(
    kind: str,
    mat: csr_matrix,
    cfg: Stage1Config,
    cache_dir: str,
    cache_key_dict: dict,
) -> ImplicitPair:
    K_model = _clip_int(int(cfg.K_model_mult) * int(cfg.N_cand), 20, 800)
    key = _hash_dict({**cache_key_dict, "K_model": K_model, "num_threads": int(cfg.num_threads), "shape": [int(mat.shape[0]), int(mat.shape[1]), int(mat.nnz)]})
    tfidf_path = os.path.join(cache_dir, f"{kind}_tfidf_{key}.npz")
    cosine_path = os.path.join(cache_dir, f"{kind}_cosine_{key}.npz")

    if os.path.exists(tfidf_path) and os.path.exists(cosine_path):
        tfidf = TFIDFRecommender.load(tfidf_path)
        cosine = CosineRecommender.load(cosine_path)
        return ImplicitPair(tfidf=tfidf, cosine=cosine)

    log(f"[6/8] Fit implicit pair for {kind} ...")
    tfidf = TFIDFRecommender(K=K_model, num_threads=int(cfg.num_threads))
    cosine = CosineRecommender(K=K_model, num_threads=int(cfg.num_threads))

    t0 = time.time()
    tfidf.fit(mat, show_progress=True)
    log(f"[6/8] {kind} TFIDF fit done in {time.time()-t0:.1f}s")
    t0 = time.time()
    cosine.fit(mat, show_progress=True)
    log(f"[6/8] {kind} Cosine fit done in {time.time()-t0:.1f}s")

    _ensure_dir(cache_dir)
    tfidf.save(tfidf_path)
    cosine.save(cosine_path)
    log(f"[SAVE] Saved {kind} models -> {tfidf_path} / {cosine_path}")
    return ImplicitPair(tfidf=tfidf, cosine=cosine)


# -------------------------
# Fitted stage1 bundle
# -------------------------
@dataclass
class FittedStage1:
    # item channel
    item_pair: ImplicitPair
    user_items: csr_matrix
    users_item: List[str]
    items: List[str]
    u2i_item: Dict[str, int]
    trending_idx: np.ndarray

    # category channel (optional)
    cat_pair: Optional[ImplicitPair]
    user_cats: Optional[csr_matrix]
    users_cat: Optional[List[str]]
    cats: Optional[List[str]]
    u2i_cat: Optional[Dict[str, int]]
    cat_to_items: Optional[Dict[str, List[str]]]

    # cohort tables (optional)
    users_df: Optional[pl.DataFrame]
    items_df: Optional[pl.DataFrame]


def fit_stage1(
    cfg: Stage1Config,
    df_min: pl.DataFrame,
    items_df: Optional[pl.DataFrame],
    users_df: Optional[pl.DataFrame],
    train_begin: datetime,
    train_end: datetime,
    cache_dir: str,
    max_rows_train: int,
) -> FittedStage1:
    # item CSR
    mat_item, users_item, items, trending_idx = build_user_items_csr(
        df_min=df_min,
        train_begin=train_begin,
        train_end=train_end,
        cfg=cfg,
        cache_dir=cache_dir,
        cache_prefix=f"train_{train_begin:%Y%m%d}_{train_end:%Y%m%d}",
        max_rows=max_rows_train,
    )
    item_pair = fit_or_load_pair(
        kind="item",
        mat=mat_item,
        cfg=cfg,
        cache_dir=cache_dir,
        cache_key_dict={
            "train_begin": train_begin.strftime("%Y-%m-%d"),
            "train_end": train_end.strftime("%Y-%m-%d"),
            "weight_type": cfg.weight_type,
            "use_recency": cfg.use_recency,
            "hl": float(cfg.recency_halflife_days),
            "use_discount": cfg.use_discount,
            "beta": float(cfg.discount_beta),
            "minf": float(cfg.min_discount_factor),
        },
    )

    # category channel CSR + models
    user_cats, users_cat, cats, cat_to_items = build_user_category_csr_and_mapping(
        df_min=df_min,
        items_df=items_df,
        train_begin=train_begin,
        train_end=train_end,
        cfg=cfg,
        cache_dir=cache_dir,
        cache_prefix=f"train_{train_begin:%Y%m%d}_{train_end:%Y%m%d}",
        max_rows=max_rows_train,
    )

    cat_pair = None
    if user_cats is not None:
        cat_pair = fit_or_load_pair(
            kind="cat",
            mat=user_cats,
            cfg=cfg,
            cache_dir=cache_dir,
            cache_key_dict={
                "train_begin": train_begin.strftime("%Y-%m-%d"),
                "train_end": train_end.strftime("%Y-%m-%d"),
                "weight_type": cfg.weight_type,
                "use_recency": cfg.use_recency,
                "hl": float(cfg.recency_halflife_days),
                "use_discount": cfg.use_discount,
                "beta": float(cfg.discount_beta),
                "minf": float(cfg.min_discount_factor),
            },
        )

    # index maps
    u2i_item = {u: i for i, u in enumerate(users_item)}
    u2i_cat = {u: i for i, u in enumerate(users_cat)} if users_cat is not None else None

    return FittedStage1(
        item_pair=item_pair,
        user_items=mat_item,
        users_item=users_item,
        items=items,
        u2i_item=u2i_item,
        trending_idx=trending_idx,
        cat_pair=cat_pair,
        user_cats=user_cats,
        users_cat=users_cat,
        cats=cats,
        u2i_cat=u2i_cat,
        cat_to_items=cat_to_items,
        users_df=users_df,
        items_df=items_df,
    )


# -------------------------
# Cohort filtering (conservative)
# -------------------------
def _normalize_gender(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in ("m", "male", "nam"):
        return "male"
    if s in ("f", "female", "nữ", "nu", "woman", "women"):
        return "female"
    return s

def cohort_filter_items(
    cfg: Stage1Config,
    fitted: FittedStage1,
    user_id: str,
    item_ids: List[str],
) -> List[str]:
    if cfg.cohort_mode == "none":
        return item_ids
    if cfg.cohort_mode != "gender_target":
        return item_ids  # unknown mode -> no filter

    if fitted.users_df is None or fitted.items_df is None:
        return item_ids

    # build maps (cached in memory by lazy creation)
    if not hasattr(cohort_filter_items, "_u_gender"):
        ug = dict(zip(fitted.users_df["customer_id"].to_list(), fitted.users_df["gender"].to_list()))
        setattr(cohort_filter_items, "_u_gender", ug)
    if not hasattr(cohort_filter_items, "_it_gender_target"):
        ig = dict(zip(fitted.items_df["item_id"].to_list(), fitted.items_df["gender_target_final"].to_list()))
        setattr(cohort_filter_items, "_it_gender_target", ig)

    ug = getattr(cohort_filter_items, "_u_gender")
    ig = getattr(cohort_filter_items, "_it_gender_target")

    g_user = _normalize_gender(ug.get(user_id))
    if not g_user:
        return item_ids

    out = []
    for it in item_ids:
        gt = _normalize_gender(ig.get(it))
        if gt in (None, "", "unisex", "all", "both"):
            out.append(it); continue
        # if gt mentions female/male and matches user -> keep
        if (g_user == "male" and "male" in gt and "female" not in gt) or (g_user == "female" and "female" in gt and "male" not in gt):
            out.append(it); continue
        # otherwise drop
    return out


# -------------------------
# Recommendation for a user (merge channels)
# -------------------------
def recommend_for_user(
    cfg: Stage1Config,
    fitted: FittedStage1,
    user_id: str,
    N_cand: int,
    N_trend: int,
    filter_already_liked_items: bool = True,
) -> List[str]:
    top_k = 2 * int(N_cand) + int(N_trend)

    # cold-start -> trending only
    if user_id not in fitted.u2i_item:
        idxs = fitted.trending_idx[:top_k].tolist()
        out = [fitted.items[i] for i in idxs]
        return cohort_filter_items(cfg, fitted, user_id, out) if cfg.cohort_mode != "none" else out

    uidx = fitted.u2i_item[user_id]
    row = fitted.user_items[uidx]
    if row.nnz == 0:
        idxs = fitted.trending_idx[:top_k].tolist()
        out = [fitted.items[i] for i in idxs]
        return cohort_filter_items(cfg, fitted, user_id, out) if cfg.cohort_mode != "none" else out

    ids_t, _ = fitted.item_pair.tfidf.recommend(uidx, row, N=int(N_cand), filter_already_liked_items=filter_already_liked_items)
    ids_c, _ = fitted.item_pair.cosine.recommend(uidx, row, N=int(N_cand), filter_already_liked_items=filter_already_liked_items)

    cand: List[str] = []
    cand.extend([fitted.items[int(i)] for i in (ids_t.tolist() if hasattr(ids_t, "tolist") else list(ids_t))])
    cand.extend([fitted.items[int(i)] for i in (ids_c.tolist() if hasattr(ids_c, "tolist") else list(ids_c))])

    # category channel expansion
    if cfg.use_category_channel and fitted.cat_pair is not None and fitted.user_cats is not None and fitted.u2i_cat is not None:
        if user_id in fitted.u2i_cat:
            uc_idx = fitted.u2i_cat[user_id]
            uc_row = fitted.user_cats[uc_idx]
            if uc_row.nnz > 0 and fitted.cats is not None and fitted.cat_to_items is not None:
                # retrieve categories
                cat_ids_t, _ = fitted.cat_pair.tfidf.recommend(uc_idx, uc_row, N=int(cfg.N_cat), filter_already_liked_items=False)
                cat_ids_c, _ = fitted.cat_pair.cosine.recommend(uc_idx, uc_row, N=int(cfg.N_cat), filter_already_liked_items=False)
                cat_list = []
                cat_list.extend([fitted.cats[int(i)] for i in (cat_ids_t.tolist() if hasattr(cat_ids_t, "tolist") else list(cat_ids_t))])
                cat_list.extend([fitted.cats[int(i)] for i in (cat_ids_c.tolist() if hasattr(cat_ids_c, "tolist") else list(cat_ids_c))])
                # expand categories to items
                per_cat = int(cfg.cat_items_per_cat)
                for c in cat_list:
                    items_in_c = fitted.cat_to_items.get(str(c), [])
                    cand.extend(items_in_c[:per_cat])

    # trending backfill
    if int(N_trend) > 0:
        cand.extend([fitted.items[int(i)] for i in fitted.trending_idx[:int(N_trend)].tolist()])

    # dedup, truncate
    seen = set()
    out = []
    for it in cand:
        if it not in seen:
            seen.add(it)
            out.append(it)
        if len(out) >= top_k:
            break

    # cohort post-filter (optional)
    if cfg.cohort_mode != "none":
        out = cohort_filter_items(cfg, fitted, user_id, out)
        # refill with trending if filtered too much
        if len(out) < top_k:
            for ix in fitted.trending_idx.tolist():
                it = fitted.items[int(ix)]
                if it not in out:
                    out.append(it)
                if len(out) >= top_k:
                    break

    return out


# -------------------------
# Evaluation
# -------------------------
@dataclass
class EvalReport:
    # UNFILTERED overall
    recall: float
    hit: float
    n_users_eval: int
    n_cold_start: int
    K: int
    # NEW-ITEMS (global) — treated as FILTERED per user definition
    recall_new_items: float
    hit_new_items: float
    n_users_eval_new_items: int
    n_new_items_in_eval: int


def build_groundtruth(df_min: pl.DataFrame, eval_begin: datetime, eval_end: datetime, max_rows: int = 0) -> Dict[str, Set[str]]:
    d = df_min.filter(
        pl.col("created_date").is_between(pl.lit(eval_begin.date(), dtype=pl.Date), pl.lit(eval_end.date(), dtype=pl.Date), closed="both")
    ).select(["customer_id", "item_id"])
    if max_rows and max_rows > 0:
        d = d.head(max_rows)
    gt: Dict[str, Set[str]] = {}
    for u, it in zip(d["customer_id"].to_list(), d["item_id"].to_list()):
        gt.setdefault(str(u), set()).add(str(it))
    return gt


def build_train_hist_sets(df_min: pl.DataFrame, train_begin: datetime, train_end: datetime) -> Dict[str, Set[str]]:
    d = df_min.filter(
        pl.col("created_date").is_between(pl.lit(train_begin.date(), dtype=pl.Date), pl.lit(train_end.date(), dtype=pl.Date), closed="both")
    ).select(["customer_id", "item_id"]).unique()
    hist: Dict[str, Set[str]] = {}
    for u, it in zip(d["customer_id"].to_list(), d["item_id"].to_list()):
        hist.setdefault(str(u), set()).add(str(it))
    return hist


def eval_recall_hit(
    cfg: Stage1Config,
    fitted: FittedStage1,
    gt: Dict[str, Set[str]],
    seen_items: Set[str],
    N_cand: int,
    N_trend: int,
    batch_users: int,
) -> EvalReport:
    """
    Metrics:
      1) UNFILTERED overall: relevant = gt[u]
      2) NEW-ITEMS (global) — user's "FILTERED": relevant_new = {it in gt[u] | it not in seen_items}
         where seen_items are items that appeared in global history [seen_begin, seen_end] (e.g., Jan–Nov 2024).

    Cold-start user: user not present in training user->index map.
    """
    K = 2 * int(N_cand) + int(N_trend)
    users = list(gt.keys())

    # global count new items in eval
    all_eval_items: Set[str] = set()
    for s in gt.values():
        all_eval_items |= set(s)
    n_new_items_in_eval = sum(1 for it in all_eval_items if it not in seen_items)

    recalls, hits = [], []
    recalls_new, hits_new = [], []
    n_eval = 0
    n_eval_new = 0
    n_cold = 0

    for i in tqdm(range(0, len(users), int(batch_users)), desc=f"Eval @K={K} (batch_users={batch_users})"):
        batch = users[i:i+int(batch_users)]
        for u in batch:
            # UNFILTERED overall
            relevant = set(gt[u])
            if relevant:
                n_eval += 1
                if u not in fitted.u2i_item:
                    n_cold += 1
                rec = recommend_for_user(cfg, fitted, u, N_cand, N_trend, filter_already_liked_items=False)
                inter = set(rec[:K]) & relevant
                recalls.append(len(inter) / len(relevant))
                hits.append(1.0 if inter else 0.0)

            # NEW-ITEMS (global) — treated as FILTERED
            relevant_new = set(it for it in gt[u] if it not in seen_items)
            if relevant_new:
                n_eval_new += 1
                rec = recommend_for_user(cfg, fitted, u, N_cand, N_trend, filter_already_liked_items=False)
                inter_new = set(rec[:K]) & relevant_new
                recalls_new.append(len(inter_new) / len(relevant_new))
                hits_new.append(1.0 if inter_new else 0.0)

    recall = float(np.mean(recalls)) if n_eval and recalls else 0.0
    hit = float(np.mean(hits)) if n_eval and hits else 0.0
    recall_new = float(np.mean(recalls_new)) if n_eval_new and recalls_new else 0.0
    hit_new = float(np.mean(hits_new)) if n_eval_new and hits_new else 0.0

    return EvalReport(
        recall=recall,
        hit=hit,
        n_users_eval=int(n_eval),
        n_cold_start=int(n_cold),
        K=int(K),
        recall_new_items=recall_new,
        hit_new_items=hit_new,
        n_users_eval_new_items=int(n_eval_new),
        n_new_items_in_eval=int(n_new_items_in_eval),
    )



# -------------------------
# Random search
# -------------------------
def _default_space(cfg: Stage1Config) -> dict:
    if cfg.search_space:
        return cfg.search_space
    return {
        "len_hist": [cfg.len_hist],
        "len_recent": [cfg.len_recent],
        "N_cand": [cfg.N_cand],
        "N_trend": [cfg.N_trend],
        "weight_type": [cfg.weight_type],
        "use_recency": [cfg.use_recency],
        "recency_halflife_days": [cfg.recency_halflife_days],
        "use_discount": [cfg.use_discount],
        "discount_beta": [cfg.discount_beta],
        "use_category_channel": [cfg.use_category_channel],
        "K_model_mult": [cfg.K_model_mult],
    }


def _sample(space: dict, rng: np.random.Generator) -> dict:
    return {k: vals[int(rng.integers(0, len(vals)))] for k, vals in space.items()}


def random_search(
    cfg: Stage1Config,
    df_min: pl.DataFrame,
    items_df: Optional[pl.DataFrame],
    users_df: Optional[pl.DataFrame],
    seen_items: Set[str],
    train_end: datetime,
    recent_begin: datetime,
    out_run_dir: str,
    cache_dir: str,
) -> dict:
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
            f.write(
                "trial,recall_u,hit_u,recall_new,hit_new,"
                "n_users_eval_u,n_users_eval_new,n_cold_u,K,params_json\n"
            )

    rng = np.random.default_rng(int(cfg.random_state))
    best = {"recall_new": -1.0, "hit_new": -1.0, "recall_u": -1.0, "params": None, "rep": None}

    log(f"[RS] Random search n_trials={n_trials} (quick={cfg.quick})")
    for t in tqdm(range(1, n_trials + 1), desc="Stage1 random search"):
        p = _sample(space, rng)

        # trial cfg clone
        trial = Stage1Config(**{**cfg.__dict__})
        for k, v in p.items():
            setattr(trial, k, v)

        len_hist = int(trial.len_hist)
        len_recent = int(trial.len_recent)
        train_begin = train_end - timedelta(days=len_hist - 1)
        eval_begin = recent_begin
        eval_end = recent_begin + timedelta(days=len_recent - 1)

        max_rows_train = cfg.quick_max_rows_train if cfg.quick else cfg.max_rows_train
        max_rows_eval = cfg.quick_max_rows_eval if cfg.quick else cfg.max_rows_eval

        fitted = fit_stage1(
            cfg=trial, df_min=df_min, items_df=items_df, users_df=users_df,
            train_begin=train_begin, train_end=train_end, cache_dir=cache_dir, max_rows_train=max_rows_train
        )

        gt = build_groundtruth(df_min, eval_begin, eval_end, max_rows=max_rows_eval)
        if cfg.quick and cfg.quick_max_users_eval > 0:
            keys = list(gt.keys())[:int(cfg.quick_max_users_eval)]
            gt = {u: gt[u] for u in keys}

        train_hist = build_train_hist_sets(df_min, train_begin, train_end)

        rep = eval_recall_hit(cfg=trial, fitted=fitted, gt=gt, seen_items=seen_items,
                      N_cand=trial.N_cand, N_trend=trial.N_trend, batch_users=trial.batch_users)

        params_json = json.dumps({
            "len_hist": trial.len_hist,
            "len_recent": trial.len_recent,
            "N_cand": trial.N_cand,
            "N_trend": trial.N_trend,
            "weight_type": trial.weight_type,
            "use_recency": trial.use_recency,
            "recency_halflife_days": trial.recency_halflife_days,
            "use_discount": trial.use_discount,
            "discount_beta": trial.discount_beta,
            "use_category_channel": trial.use_category_channel,
            "K_model_mult": trial.K_model_mult,
        }, ensure_ascii=False)

        with open(results_csv, "a", encoding="utf-8") as f:
            f.write(
                f"{t},{rep.recall:.10f},{rep.hit:.10f},{rep.recall_new_items:.10f},{rep.hit_new_items:.10f},"
                f"{rep.n_users_eval},{rep.n_users_eval_new_items},{rep.n_cold_start},{rep.K},{params_json}"
            )

        if (
            rep.recall_new_items > best["recall_new"]
            or (
                np.isclose(rep.recall_new_items, best["recall_new"])
                and rep.hit_new_items > best["hit_new"]
            )
            or (
                np.isclose(rep.recall_new_items, best["recall_new"])
                and np.isclose(rep.hit_new_items, best["hit_new"])
                and rep.recall > best["recall_u"]
            )
        ):
            best["recall_new"] = rep.recall_new_items
            best["hit_new"] = rep.hit_new_items
            best["recall_u"] = rep.recall
            best["params"] = json.loads(params_json)
            best["rep"] = rep

            with open(best_params_path, "w", encoding="utf-8") as fp:
                json.dump(best["params"], fp, ensure_ascii=False, indent=2)

            with open(best_metrics_path, "w", encoding="utf-8") as fp:
                json.dump({"best": rep.__dict__}, fp, ensure_ascii=False, indent=2)

            log(
                f"[RS] New best trial={t}: "
                f"NEW-ITEMS Recall@{rep.K}={rep.recall_new_items:.6f}, "
                f"Hit={rep.hit_new_items:.6f} | "
                f"UNFILTERED Recall={rep.recall:.6f}"
            )


    if best["params"] is None:
        raise RuntimeError("Random search produced no best params.")
    return best["params"]


# -------------------------
# Train best + save artifacts
# -------------------------
def train_best_and_save(
    cfg: Stage1Config,
    df_min: pl.DataFrame,
    items_df: Optional[pl.DataFrame],
    users_df: Optional[pl.DataFrame],
    seen_items: Set[str],
    best_params: dict,
    train_end: datetime,
    recent_begin: datetime,
    out_run_dir: str,
    cache_dir: str,
) -> None:
    log("[8/8] Train BEST model and save artifacts ...")

    best = Stage1Config(**{**cfg.__dict__})
    for k, v in best_params.items():
        setattr(best, k, v)

    train_begin = train_end - timedelta(days=int(best.len_hist) - 1)
    eval_begin = recent_begin
    eval_end = recent_begin + timedelta(days=int(best.len_recent) - 1)

    max_rows_train = cfg.quick_max_rows_train if cfg.quick else cfg.max_rows_train
    max_rows_eval = cfg.quick_max_rows_eval if cfg.quick else cfg.max_rows_eval

    fitted = fit_stage1(best, df_min, items_df, users_df, train_begin, train_end, cache_dir, max_rows_train)

    gt = build_groundtruth(df_min, eval_begin, eval_end, max_rows=max_rows_eval)
    if cfg.quick and cfg.quick_max_users_eval > 0:
        keys = list(gt.keys())[:int(cfg.quick_max_users_eval)]
        gt = {u: gt[u] for u in keys}

    train_hist = build_train_hist_sets(df_min, train_begin, train_end)

    
    # Save groundtruth for downstream usage (and for your request to compute Recall@K from groundtruth.pkl)
    log(f"[INFO] Dec/2024 groundtruth (eval) users={len(gt):,} (NOT saved to groundtruth_pkl_path)")

    # Optional: compute Recall@K on this groundtruth using pred/hist dicts (both unfiltered & filtered)
    if best.compute_recall_from_groundtruth:
        K_recall = int(best.recall_k) if int(best.recall_k) > 0 else (2 * int(best.N_cand) + int(best.N_trend))
        gt_users = list(gt.keys())
        pred_dict = build_pred_dict_for_users(
            cfg=best,
            fitted=fitted,
            users=gt_users,
            N_cand=best.N_cand,
            N_trend=best.N_trend,
            K=K_recall,
            show_progress=True,
        )
        recall_res = recall_at_k_both(pred=pred_dict, gt=gt, hist=train_hist, K=K_recall, show_progress=True)

        recall_path = os.path.join(out_run_dir, "groundtruth_recall_at_k.json")
        with open(recall_path, "w", encoding="utf-8") as f:
            json.dump({"K": K_recall, **recall_res}, f, ensure_ascii=False, indent=2)
        log(f"[SAVE] recall@k (gt) -> {recall_path} | unfiltered={recall_res['recall_unfiltered']:.6f} filtered={recall_res['recall_filtered']:.6f} cold={recall_res['n_cold_start']:,}")

        rep = eval_recall_hit(
            cfg=best,
            fitted=fitted,
            gt=gt,
            seen_items=seen_items,
            N_cand=best.N_cand,
            N_trend=best.N_trend,
            batch_users=best.batch_users,
        )

    # -------------------------
    # AFTER-EVAL: refit retrieval to include RECENT (train_end := eval_end)
    # -------------------------
    if getattr(cfg, "refit_end_at_eval_end", False):
        train_end_2 = eval_end
        train_begin_2 = train_end_2 - timedelta(days=int(best.len_hist) - 1)
        log("==== AFTER-EVAL REFIT (include RECENT) ====")
        log(f"Refit window    : {train_begin_2.date()} -> {train_end_2.date()}")
        fitted_refit = fit_stage1(
            best, df_min, items_df, users_df,
            train_begin_2, train_end_2,
            cache_dir, max_rows_train
        )

        
    # Optional export candidates ONLY for users in groundtruth (your request)
    out_gt = getattr(best, "export_gt_users_out", None) or getattr(best, "export_all_users_out", None)
    if out_gt:
        gt_users = list(gt.keys())
        export_candidates_for_users_parquet(
            cfg=best,
            fitted=fitted_refit,
            users=gt_users,
            out_path=out_gt,
            N_cand=best.N_cand,
            N_trend=best.N_trend,
        )
    else:
        log("[INFO] export_gt_users_out is None -> skip exporting GT-users candidates.")


        _ensure_dir(out_run_dir)
        meta_path = os.path.join(out_run_dir, "best_stage1_meta.json")
        metrics_path = os.path.join(out_run_dir, "best_stage1_metrics.json")

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(best_params, f, ensure_ascii=False, indent=2)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"unfiltered": rep.__dict__}, f, ensure_ascii=False, indent=2)

    log("===== BEST EVAL REPORT =====")
    log(
        f"UNFILTERED (overall)   Recall@{rep.K}={rep.recall:.6f}  "
        f"Hit@{rep.K}={rep.hit:.6f}  "
        f"n_users_eval={rep.n_users_eval:,}  "
        f"cold_start_users={rep.n_cold_start:,}"
    )
    log(
        f"NEW-ITEMS (FILTERED)   Recall@{rep.K}={rep.recall_new_items:.6f}  "
        f"Hit@{rep.K}={rep.hit_new_items:.6f}  "
        f"n_users_eval={rep.n_users_eval_new_items:,}  "
        f"n_new_items_in_eval={rep.n_new_items_in_eval:,}"
    )


# -------------------------
# Main
# -------------------------
def main(config_path: str):
    cfg = load_config(config_path)

    artifacts_dir = cfg.artifacts_dir
    run_dir = os.path.join(artifacts_dir, "runs", cfg.run_name)
    cache_dir = os.path.join(artifacts_dir, "cache")
    _ensure_dir(run_dir)
    _ensure_dir(cache_dir)

    if cfg.quick:
        log("[MODE] QUICK=True")

    train_end = _parse_date(cfg.train_end)
    recent_begin = _parse_date(cfg.recent_begin)
    seen_begin = _parse_date(cfg.seen_begin)
    seen_end = _parse_date(cfg.seen_end) if cfg.seen_end else train_end

    # Build one minimal tx cache spanning all needed dates
    space = cfg.search_space if (cfg.do_random_search and cfg.search_space) else {"len_hist":[cfg.len_hist], "len_recent":[cfg.len_recent]}
    max_len_hist = int(max(space.get("len_hist", [cfg.len_hist])))
    max_len_recent = int(max(space.get("len_recent", [cfg.len_recent])))

    earliest_train_begin = train_end - timedelta(days=max_len_hist - 1)
    latest_eval_end = recent_begin + timedelta(days=max_len_recent - 1)

    begin_all = min(seen_begin, earliest_train_begin)
    end_all = max(seen_end, latest_eval_end)

    log("==== Stage1 Retrieval v3 (implicit TFIDF + Cosine) ====")
    log(f"Config path     : {config_path}")
    log(f"Run dir         : {run_dir}")
    log(f"Cache dir       : {cache_dir}")
    log(f"Anchors         : train_end={cfg.train_end} | recent_begin={cfg.recent_begin}")
    log(f"Global seen win : {seen_begin.date()} -> {seen_end.date()} (NEW-ITEMS)")
    log(f"Min tx cache    : {begin_all.date()} -> {end_all.date()}")

    df_min = build_min_transactions(cfg, begin_all, end_all, cache_dir)
    log(f"[INFO] tx_min rows: {df_min.height:,}")

    seen_items = compute_seen_items(df_min, seen_begin, seen_end, cache_dir)

    items_df = load_item_table(cfg.items_path_glob, cache_dir)
    users_df = load_user_table(cfg.users_path_glob, cache_dir)

    if cfg.do_random_search:
        best_params = random_search(
            cfg, df_min, items_df, users_df, seen_items,
            train_end, recent_begin, run_dir, cache_dir
        )
    else:
        best_params = {
            "len_hist": cfg.len_hist, "len_recent": cfg.len_recent,
            "N_cand": cfg.N_cand, "N_trend": cfg.N_trend,
            "weight_type": cfg.weight_type, "K_model_mult": cfg.K_model_mult,
            "use_recency": cfg.use_recency, "recency_halflife_days": cfg.recency_halflife_days,
            "use_discount": cfg.use_discount, "discount_beta": cfg.discount_beta,
            "use_category_channel": cfg.use_category_channel,
        }
        with open(os.path.join(run_dir, "best_params.json"), "w", encoding="utf-8") as f:
            json.dump(best_params, f, ensure_ascii=False, indent=2)

    train_best_and_save(cfg, df_min, items_df, users_df, seen_items, best_params, train_end, recent_begin, run_dir, cache_dir)
    log("DONE.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    main(args.config)
