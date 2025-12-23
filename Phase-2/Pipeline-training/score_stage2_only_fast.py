# -*- coding: utf-8 -*-
"""
Score Stage 2 (fast) from already-trained model.

Key optimizations vs train_stage2_lgbm.py:
  1) Precompute HIST aggregates ONCE (brand_counts, age_counts, category_counts, target_user_group_counts, last_cat_purchase)
  2) Load extra feature tables ONCE (price_segment, buy_segment, luxury_level, age_final, brand_segment, top10_by_cat_month prefiltered)
  3) In scoring loop, only:
        - build (customer_id, item_id) pairs
        - join cached tables
        - predict scores
        - rank per user

Outputs (saved progressively):
  - pred_stage2.pkl
  - cold_start_users.pkl
  - stage2_precision_report.json
  - pred_stage2_partial.pkl (periodic checkpoint)

Run:
  python score_stage2_only_fast.py --config stage2_train_config.json

Optional overrides:
  --batch_users 2000
  --save_every 10
  --max_users 0          (0 = use all)
  --num_threads 1        (force LightGBM predict threads; also sets common env vars)
"""

from __future__ import annotations
import os
import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Set

import numpy as np
import pandas as pd
import polars as pl
from tqdm.auto import tqdm
import joblib

from stage1_implicit_itemitem import load_stage1_from_artifacts


# ------------------------------
# Utils
# ------------------------------
def _parse_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_json(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _load_groundtruth(path: str) -> Dict[Any, List[Any]]:
    try:
        with open(path, "rb") as f:
            gt = pickle.load(f)
    except Exception:
        with open(path, "r", encoding="utf-8") as f:
            gt = json.load(f)
    if isinstance(gt, dict):
        return gt
    raise ValueError(f"Unsupported groundtruth format: {type(gt)}")

def _as_str_id(x) -> str:
    return str(x)

def precision_at_k(pred, gt, hist, filter_bought_items=True, K=10):
    precisions = []
    cold_start_users = []
    for user in gt.keys():
        if (user not in hist) or (user not in pred):
            cold_start_users.append(user)
            continue
        relevant_items = set(gt[user])
        if filter_bought_items:
            relevant_items -= set(hist[user])
        hits = len(set(pred[user][:K]) & relevant_items)
        precisions.append(hits / K)
    return float(np.mean(precisions)) if precisions else 0.0, cold_start_users

def _scan_parquet_glob(glob_path: str) -> pl.LazyFrame:
    return pl.scan_parquet(glob_path)

def _colset(lf: pl.LazyFrame) -> Set[str]:
    # avoids PerformanceWarning and is faster than lf.columns
    return set(lf.collect_schema().names())

def _canonical_item_attrs(items_lf: pl.LazyFrame) -> pl.LazyFrame:
    cols = _colset(items_lf)

    def col_or_lit(name, alt=None):
        if name in cols:
            return pl.col(name)
        if alt and alt in cols:
            return pl.col(alt)
        return pl.lit("__UNK__")

    return items_lf.select([
        pl.col("item_id").cast(pl.Utf8).alias("item_id"),
        col_or_lit("brand_final", "brand").cast(pl.Utf8).alias("brand_final"),
        col_or_lit("age_bucket_final", "age_group_final").cast(pl.Utf8).alias("age_bucket_final"),
        col_or_lit("category", "category_l2").cast(pl.Utf8).alias("category"),
        (pl.col("category_l1") if "category_l1" in cols else pl.lit("__UNK__")).cast(pl.Utf8).alias("category_l1"),
        col_or_lit("target_user_group_final", "gender_target_final").cast(pl.Utf8).alias("target_user_group_final"),
    ])

def _shift_month_yyyy_mm(dt: datetime, lag: int) -> str:
    y, m = dt.year, dt.month
    m2 = m - lag
    while m2 <= 0:
        y -= 1
        m2 += 12
    return f"{y:04d}-{m2:02d}"


# ------------------------------
# Cached context for scoring
# ------------------------------
@dataclass
class FeatureContext:
    item_attrs: pl.LazyFrame

    # hist aggregates (LazyFrames; already independent of pairs)
    brand_counts: pl.LazyFrame
    age_counts: pl.LazyFrame
    category_counts: pl.LazyFrame
    target_user_group_counts: pl.LazyFrame
    last_cat_purchase: pl.LazyFrame

    # extra feature tables (LazyFrames)
    price_segment: Optional[pl.LazyFrame]
    buy_segment: Optional[pl.LazyFrame]
    luxury_level: Optional[pl.LazyFrame]
    age_final: Optional[pl.LazyFrame]
    brand_segment: Optional[pl.LazyFrame]
    top10_by_cat_month: Optional[pl.LazyFrame]

    # precomputed join keys
    trend_month_key: str
    top10m_has_cat_l1: bool


def prepare_feature_context(
    transactions_glob: str,
    items_glob: str,
    begin_hist: datetime,
    end_hist: datetime,
    extra_feature_dir: str,
    trend_month_lag: int,
) -> FeatureContext:
    print("[ScoreStage2] Preparing cached feature context (HIST aggregates + extra tables) ...")

    # 1) Base scan
    tx = _scan_parquet_glob(transactions_glob).with_columns([
        pl.col("created_date").cast(pl.Datetime, strict=False).alias("created_date"),
        pl.col("customer_id").cast(pl.Utf8).alias("customer_id"),
        pl.col("item_id").cast(pl.Utf8).alias("item_id"),
    ])
    items_lf = _scan_parquet_glob(items_glob)
    item_attrs = _canonical_item_attrs(items_lf).cache()

    # 2) HIST slice once
    hist = tx.filter(
        pl.col("created_date").is_between(
            pl.lit(begin_hist, dtype=pl.Datetime),
            pl.lit(end_hist, dtype=pl.Datetime),
            closed="both",
        )
    )

    # 3) HIST enriched once
    hist_enriched = hist.join(item_attrs, on="item_id", how="left").cache()

    # 4) Aggregates
    brand_counts = hist_enriched.group_by(["customer_id", "brand_final"]).agg(pl.len().alias("brand_counts")).cache()
    age_counts = hist_enriched.group_by(["customer_id", "age_bucket_final"]).agg(pl.len().alias("age_counts")).cache()
    category_counts = hist_enriched.group_by(["customer_id", "category"]).agg(pl.len().alias("category_counts")).cache()
    target_user_group_counts = hist_enriched.group_by(["customer_id", "target_user_group_final"]).agg(pl.len().alias("target_user_group_counts")).cache()
    last_cat_purchase = hist_enriched.group_by(["customer_id", "category"]).agg(pl.col("created_date").max().alias("last_purchase_date")).cache()

    # 5) Extra feature tables (scan once)
    def _try_scan(path: str) -> Optional[pl.LazyFrame]:
        try:
            return pl.scan_parquet(path)
        except Exception:
            return None

    ps = _try_scan(os.path.join(extra_feature_dir, "price_segment.parquet"))
    if ps is not None:
        ps_cols = _colset(ps)
        if ("item_id" in ps_cols) and ("price_segment" in ps_cols):
            ps = ps.select([pl.col("item_id").cast(pl.Utf8), pl.col("price_segment").cast(pl.Utf8)]).cache()
        else:
            ps = None

    cb = _try_scan(os.path.join(extra_feature_dir, "customer_behavior.parquet"))
    if cb is not None:
        cb_cols = _colset(cb)
        if ("customer_id" in cb_cols) and ("buy_segment" in cb_cols):
            cb = cb.select([pl.col("customer_id").cast(pl.Utf8), pl.col("buy_segment").cast(pl.Utf8)]).cache()
        else:
            cb = None

    lux = _try_scan(os.path.join(extra_feature_dir, "customer_luxury.parquet"))
    if lux is not None:
        lux_cols = _colset(lux)
        lux_col = "luxury_level" if "luxury_level" in lux_cols else ("customer_luxury" if "customer_luxury" in lux_cols else None)
        if ("customer_id" in lux_cols) and lux_col:
            lux = lux.select([pl.col("customer_id").cast(pl.Utf8), pl.col(lux_col).cast(pl.Utf8).alias("luxury_level")]).cache()
        else:
            lux = None

    agef = _try_scan(os.path.join(extra_feature_dir, "customer_age_features.parquet"))
    if agef is not None:
        agef_cols = _colset(agef)
        if ("customer_id" in agef_cols) and ("age_final" in agef_cols):
            agef = agef.select([pl.col("customer_id").cast(pl.Utf8), pl.col("age_final")]).cache()
        else:
            agef = None

    bs = _try_scan(os.path.join(extra_feature_dir, "brand_segment.parquet"))
    if bs is not None:
        bs_cols = _colset(bs)
        if ("customer_id" in bs_cols) and ("brand_segment" in bs_cols):
            bs = bs.select([pl.col("customer_id").cast(pl.Utf8), pl.col("brand_segment").cast(pl.Utf8)]).cache()
        else:
            bs = None

    # by-month top10 file
    top10m = _try_scan(os.path.join(extra_feature_dir, "top10_by_cat_month.parquet"))
    trend_month_key = _shift_month_yyyy_mm(end_hist, trend_month_lag)
    top10m_has_cat_l1 = False

    if top10m is not None:
        top10m_cols = _colset(top10m)
        if ("item_id" in top10m_cols) and ("month" in top10m_cols):
            top10m_has_cat_l1 = ("category_l1" in top10m_cols)

            schema = top10m.collect_schema()
            mtype = schema.get("month")
            t = top10m
            if mtype in (pl.Date, pl.Datetime):
                t = t.with_columns(pl.col("month").dt.strftime("%Y-%m").alias("month_key"))
            else:
                t = t.with_columns(pl.col("month").cast(pl.Utf8).str.slice(0, 7).alias("month_key"))

            # Prefilter to needed month only
            t = t.filter(pl.col("month_key") == pl.lit(trend_month_key))

            sel = [pl.col("item_id").cast(pl.Utf8), pl.col("month_key").cast(pl.Utf8)]
            if top10m_has_cat_l1:
                sel.append(pl.col("category_l1").cast(pl.Utf8))
            if "rank" in top10m_cols:
                sel.append(pl.col("rank").cast(pl.Int32).alias("rank_top10_by_cat_month"))
            else:
                sel.append(pl.lit(None).cast(pl.Int32).alias("rank_top10_by_cat_month"))
            if "total_sold" in top10m_cols:
                sel.append(pl.col("total_sold").cast(pl.Float32).alias("total_sold_top10_by_cat_month"))
            else:
                sel.append(pl.lit(None).cast(pl.Float32).alias("total_sold_top10_by_cat_month"))

            top10m = t.select(sel).cache()
        else:
            top10m = None

    print("[ScoreStage2] Context prepared.")
    print(f"[ScoreStage2] trend_month_key for top10_by_cat_month: {trend_month_key}")

    return FeatureContext(
        item_attrs=item_attrs,
        brand_counts=brand_counts,
        age_counts=age_counts,
        category_counts=category_counts,
        target_user_group_counts=target_user_group_counts,
        last_cat_purchase=last_cat_purchase,
        price_segment=ps,
        buy_segment=cb,
        luxury_level=lux,
        age_final=agef,
        brand_segment=bs,
        top10_by_cat_month=top10m,
        trend_month_key=trend_month_key,
        top10m_has_cat_l1=top10m_has_cat_l1,
    )


def build_features_for_pairs_cached(
    pairs_df: pl.DataFrame,
    ctx: FeatureContext,
    end_hist: datetime,
) -> pl.DataFrame:
    cand = pairs_df.lazy().with_columns([
        pl.col("customer_id").cast(pl.Utf8),
        pl.col("item_id").cast(pl.Utf8),
    ])

    features = (
        cand
        .join(ctx.item_attrs, on="item_id", how="left")
        .join(ctx.brand_counts, on=["customer_id", "brand_final"], how="left")
        .join(ctx.age_counts, on=["customer_id", "age_bucket_final"], how="left")
        .join(ctx.category_counts, on=["customer_id", "category"], how="left")
        .join(ctx.target_user_group_counts, on=["customer_id", "target_user_group_final"], how="left")
        .join(ctx.last_cat_purchase, on=["customer_id", "category"], how="left")
        .with_columns([
            pl.col("brand_counts").fill_null(0).cast(pl.Int32),
            pl.col("age_counts").fill_null(0).cast(pl.Int32),
            pl.col("category_counts").fill_null(0).cast(pl.Int32),
            pl.col("target_user_group_counts").fill_null(0).cast(pl.Int32),
        ])
        .with_columns(
            (
                (pl.lit(end_hist, dtype=pl.Datetime) - pl.col("last_purchase_date").cast(pl.Datetime, strict=False))
                .dt.total_days()
                .cast(pl.Int32)
            ).alias("time_since_last_purchase_in_B_category")
        )
        .with_columns(pl.col("time_since_last_purchase_in_B_category").fill_null(9999))
    )

    if ctx.price_segment is not None:
        features = features.join(ctx.price_segment, on="item_id", how="left")
    if ctx.buy_segment is not None:
        features = features.join(ctx.buy_segment, on="customer_id", how="left")
    if ctx.luxury_level is not None:
        features = features.join(ctx.luxury_level, on="customer_id", how="left")
    if ctx.age_final is not None:
        features = features.join(ctx.age_final, on="customer_id", how="left")
    if ctx.brand_segment is not None:
        features = features.join(ctx.brand_segment, on="customer_id", how="left")

    if ctx.top10_by_cat_month is not None:
        if ctx.top10m_has_cat_l1:
            features = features.join(ctx.top10_by_cat_month, on=["item_id", "category_l1"], how="left")
        else:
            features = features.join(ctx.top10_by_cat_month, on=["item_id"], how="left")
        features = features.with_columns([
            pl.col("rank_top10_by_cat_month").fill_null(9999),
            pl.col("total_sold_top10_by_cat_month").fill_null(0.0),
        ])

    fcols = set(features.collect_schema().names())
    features = features.with_columns([
        (pl.col("price_segment").fill_null("__MISSING__") if "price_segment" in fcols else pl.lit("__MISSING__").alias("price_segment")),
        (pl.col("buy_segment").fill_null("__MISSING__") if "buy_segment" in fcols else pl.lit("__MISSING__").alias("buy_segment")),
        (pl.col("luxury_level").fill_null("__MISSING__") if "luxury_level" in fcols else pl.lit("__MISSING__").alias("luxury_level")),
        (pl.col("brand_segment").fill_null("__MISSING__") if "brand_segment" in fcols else pl.lit("__MISSING__").alias("brand_segment")),
    ])
    if "age_final" in fcols:
        features = features.with_columns(pl.col("age_final").fill_null(-1).cast(pl.Int32))

    try:
        return features.collect(streaming=True)
    except Exception:
        return features.collect()


def build_hist_from_transactions(transactions_glob: str, end_dt: datetime) -> Dict[str, List[str]]:
    lf = _scan_parquet_glob(transactions_glob).with_columns([
        pl.col("created_date").cast(pl.Datetime, strict=False).alias("created_date"),
        pl.col("customer_id").cast(pl.Utf8).alias("customer_id"),
        pl.col("item_id").cast(pl.Utf8).alias("item_id"),
    ]).filter(pl.col("created_date") < pl.lit(end_dt, dtype=pl.Datetime))

    df = lf.select(["customer_id", "item_id"]).unique().collect(streaming=True)
    grouped = df.group_by("customer_id").agg(pl.col("item_id").alias("items")).to_dict(as_series=False)
    out = {}
    for u, items in zip(grouped["customer_id"], grouped["items"]):
        out[u] = items
    return out


def score_candidates_to_pred_fast(
    stage1_model,
    stage2_model,
    feature_cols: List[str],
    cat_cols: List[str],
    gt_users: List[str],
    transactions_glob: str,
    items_glob: str,
    extra_feature_dir: str,
    begin_hist: datetime,
    end_hist: datetime,
    N_total_cand: int,
    batch_users: int,
    trend_month_lag: int,
    filter_bought_items_for_candidates: bool,
    out_dir: str,
    save_every: int,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    print("[ScoreStage2] Building hist dict for evaluation ...")
    hist_dict = build_hist_from_transactions(transactions_glob, end_hist + timedelta(days=1))
    print(f"[ScoreStage2] Hist users: {len(hist_dict):,}")

    ctx = prepare_feature_context(
        transactions_glob=transactions_glob,
        items_glob=items_glob,
        begin_hist=begin_hist,
        end_hist=end_hist,
        extra_feature_dir=extra_feature_dir,
        trend_month_lag=trend_month_lag,
    )

    pred: Dict[str, List[str]] = {}
    partial_path = os.path.join(out_dir, "pred_stage2_partial.pkl")

    nbatches = (len(gt_users) + batch_users - 1) // batch_users
    for b, start in enumerate(tqdm(range(0, len(gt_users), batch_users), total=nbatches, desc="Scoring users (batch)")):
        users_batch = gt_users[start:start + batch_users]

        # candidates
        user_col: List[str] = []
        item_col: List[str] = []
        for u in users_batch:
            cand = stage1_model.recommend_for_user_id(u, top_k=N_total_cand)
            if filter_bought_items_for_candidates and (u in hist_dict):
                hist_set = set(hist_dict[u])
                cand = [it for it in cand if it not in hist_set]
            if not cand:
                continue
            cand = cand[:N_total_cand]
            user_col.extend([u] * len(cand))
            item_col.extend(cand)

        if not user_col:
            continue

        pairs_df = pl.DataFrame({"customer_id": user_col, "item_id": item_col})

        # features (cached)
        feat_df = build_features_for_pairs_cached(pairs_df=pairs_df, ctx=ctx, end_hist=end_hist)

        # ensure feature cols exist
        present = set(feat_df.columns)
        missing = [c for c in feature_cols if c not in present]
        if missing:
            feat_df = feat_df.with_columns([pl.lit(0).alias(c) for c in missing])

        # predict
        X_pd = feat_df.select(feature_cols).to_pandas()
        for c in cat_cols:
            if c in X_pd.columns:
                X_pd[c] = X_pd[c].astype("category")
            else:
                X_pd[c] = "__MISSING__"
                X_pd[c] = X_pd[c].astype("category")

        scores = stage2_model.predict_proba(X_pd)[:, 1]

        # rank in polars
        scored = feat_df.select(["customer_id", "item_id"]).with_columns(
            pl.Series("score", scores.astype(np.float32))
        )
        ranked = (
            scored
            .sort(by=["customer_id", "score"], descending=[False, True])
            .group_by("customer_id")
            .agg(pl.col("item_id").alias("items"))
        )

        d = ranked.to_dict(as_series=False)
        for u, items in zip(d["customer_id"], d["items"]):
            pred[str(u)] = [str(x) for x in items]

        if save_every and ((b + 1) % save_every == 0):
            with open(partial_path, "wb") as f:
                pickle.dump(pred, f)
            print(f"[ScoreStage2] Checkpoint saved: {partial_path} (batches={b+1}/{nbatches})")

    with open(partial_path, "wb") as f:
        pickle.dump(pred, f)
    print(f"[ScoreStage2] Final checkpoint saved: {partial_path}")

    return pred, hist_dict


def _set_safe_threads(num_threads: int):
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(num_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(num_threads))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(num_threads))
    os.environ.setdefault("POLARS_MAX_THREADS", str(num_threads))


def main(config_path: str, batch_users: int, save_every: int, max_users: int, num_threads: int):
    import warnings
    warnings.filterwarnings("ignore", category=pl.exceptions.PerformanceWarning)

    if num_threads and num_threads > 0:
        _set_safe_threads(num_threads)

    cfg = _load_json(config_path)
    paths = cfg["paths"]
    params = cfg["params"]
    time_cfg = cfg.get("time", {})

    out_dir = paths["out_dir"]
    _ensure_dir(out_dir)
    _save_json(
        {"config": cfg, "batch_users": batch_users, "save_every": save_every, "max_users": max_users, "num_threads": num_threads},
        os.path.join(out_dir, "stage2_scoring_config_used.json")
    )

    print(f"[ScoreStage2] Using config: {config_path}")
    print(f"[ScoreStage2] Outputs to: {out_dir}")
    print(f"[ScoreStage2] batch_users={batch_users} save_every={save_every} max_users={max_users} num_threads={num_threads}")

    # Stage1
    print("[ScoreStage2] Loading Stage1 candidate model ...")
    prefix = paths["stage1_prefix"]
    art_dir = paths["stage1_artifacts_dir"]
    stage1 = load_stage1_from_artifacts(
        meta_npz_path=os.path.join(art_dir, f"{prefix}_meta.npz"),
        user_items_npz_path=os.path.join(art_dir, f"{prefix}_user_items.npz"),
        tfidf_npz_path=os.path.join(art_dir, f"{prefix}_tfidf.npz"),
        cosine_npz_path=os.path.join(art_dir, f"{prefix}_cosine.npz"),
    )
    print("[ScoreStage2] Stage1 loaded OK.")

    # Stage2 model + feature cols
    model_path = os.path.join(out_dir, "stage2_model.pkl")
    feat_path = os.path.join(out_dir, "stage2_feature_cols.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing trained model: {model_path}")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Missing feature cols: {feat_path}")

    stage2_model = joblib.load(model_path)
    meta = _load_json(feat_path)
    feature_cols = meta["feature_cols"]
    cat_cols = meta.get("cat_cols", [])
    print(f"[ScoreStage2] Loaded Stage2 model: {model_path}")
    print(f"[ScoreStage2] #features={len(feature_cols)} | #cat_cols={len(cat_cols)}")

    # groundtruth
    print("[ScoreStage2] Loading groundtruth ...")
    gt = _load_groundtruth(paths["groundtruth_path"])
    gt = {_as_str_id(k): [str(x) for x in v] for k, v in gt.items()}
    gt_users = list(gt.keys())
    if max_users and max_users > 0:
        gt_users = gt_users[:max_users]
        gt = {u: gt[u] for u in gt_users}
    print(f"[ScoreStage2] Groundtruth users used: {len(gt_users):,}")

    # windows
    test_begin = _parse_dt(time_cfg.get("test_begin", "2025-01-01"))
    len_hist = int(params.get("len_hist", 120))
    begin_hist = test_begin - timedelta(days=len_hist)
    end_hist = test_begin - timedelta(days=1)
    trend_month_lag = int(time_cfg.get("trend_month_lag", 2))
    N_total_cand = int(params.get("N_trend", 0)) + 2 * int(params.get("N_cand", 300))
    print(f"[ScoreStage2] Eval window: begin_hist={begin_hist.date()} end_hist={end_hist.date()} | N_total_cand={N_total_cand}")

    pred, hist_dict = score_candidates_to_pred_fast(
        stage1_model=stage1,
        stage2_model=stage2_model,
        feature_cols=feature_cols,
        cat_cols=cat_cols,
        gt_users=gt_users,
        transactions_glob=paths["transactions_path_glob"],
        items_glob=paths["items_path_glob"],
        extra_feature_dir=paths["extra_feature_dir"],
        begin_hist=begin_hist,
        end_hist=end_hist,
        N_total_cand=N_total_cand,
        batch_users=batch_users,
        trend_month_lag=trend_month_lag,
        filter_bought_items_for_candidates=bool(params.get("filter_bought_items", False)),
        out_dir=out_dir,
        save_every=save_every,
    )

    # save final pred
    pred_path = os.path.join(out_dir, "pred_stage2.pkl")
    with open(pred_path, "wb") as f:
        pickle.dump(pred, f)
    print(f"[ScoreStage2] Saved predictions: {pred_path}")

    # metrics
    K = int(params.get("Topk", 10))
    p_unf, cold_unf = precision_at_k(pred, gt, hist_dict, filter_bought_items=False, K=K)
    p_flt, cold_flt = precision_at_k(pred, gt, hist_dict, filter_bought_items=True,  K=K)
    cold_users = sorted(set(cold_unf) | set(cold_flt))

    cold_path = os.path.join(out_dir, "cold_start_users.pkl")
    with open(cold_path, "wb") as f:
        pickle.dump(cold_users, f)
    print(f"[ScoreStage2] Saved cold-start users: {cold_path}")

    report = {
        "precision_at_k_unfiltered": p_unf,
        "precision_at_k_filtered": p_flt,
        "K": K,
        "n_users_gt": len(gt_users),
        "n_users_pred": len(pred),
        "n_cold_start": len(cold_users),
        "filter_bought_items_for_candidates": bool(params.get("filter_bought_items", False)),
        "begin_hist": begin_hist.strftime("%Y-%m-%d"),
        "end_hist": end_hist.strftime("%Y-%m-%d"),
        "N_total_cand": N_total_cand,
        "batch_users": batch_users,
    }
    report_path = os.path.join(out_dir, "stage2_precision_report.json")
    _save_json(report, report_path)

    print("[ScoreStage2] ===== Precision@K Report =====")
    print(f"Precision@{K} (unfiltered): {p_unf:.6f}")
    print(f"Precision@{K} (filtered):   {p_flt:.6f}")
    print(f"Cold-start users: {len(cold_users):,} / {len(gt_users):,}")
    print(f"[ScoreStage2] Saved report: {report_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--batch_users", type=int, default=4000)
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--max_users", type=int, default=0)
    ap.add_argument("--num_threads", type=int, default=64)
    args = ap.parse_args()
    main(args.config, batch_users=args.batch_users, save_every=args.save_every, max_users=args.max_users, num_threads=args.num_threads)
