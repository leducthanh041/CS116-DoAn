
# -*- coding: utf-8 -*-
"""
Train Stage 2 (reranker) using:
  - Stage1 candidate model (implicit TFIDF + Cosine ensemble) for inference candidates
  - Feature-label parquet created previously for training

Outputs (saved progressively):
  - stage2_model.pkl
  - stage2_feature_cols.json
  - stage2_metrics.json
  - pred_stage2.pkl (dict user -> ranked item_ids)
  - cold_start_users.pkl

Notes:
  - min_trans_items: chưa có spec rõ ràng => KHÔNG dùng trong script (không biết dùng đúng nghĩa ở pipeline hiện tại).
  - filter_fashion: chưa có rule category nào là fashion => KHÔNG dùng (không biết rule).
  - min_coo/session_window: tạo feature co-occurrence đúng nghĩa cần định nghĩa rõ feature per (user,item) => KHÔNG dùng (không biết spec).
"""

from __future__ import annotations
import os
import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import polars as pl
from tqdm.auto import tqdm
import joblib

# Stage1 loader (expects your Stage1 module file exists in project)
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
    # groundtruth.pkl is assumed to be pickle. If fails, try json.
    try:
        with open(path, "rb") as f:
            gt = pickle.load(f)
    except Exception:
        with open(path, "r", encoding="utf-8") as f:
            gt = json.load(f)
    if isinstance(gt, dict):
        return gt
    raise ValueError(f"Unsupported groundtruth format: {type(gt)}")

def precision_at_k(pred, gt, hist, filter_bought_items=True, K=10):
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
    return float(np.mean(precisions)) if len(precisions) else 0.0, cold_start_users

def _as_str_id(x):
    # unify keys to string for robust joining between parquet/json/pickle
    return str(x)

# ------------------------------
# Feature building for inference
# (re-implements the "no segment_name" feature set)
# ------------------------------
def _scan_parquet_glob(glob_path: str) -> pl.LazyFrame:
    return pl.scan_parquet(glob_path)

def _canonical_item_attrs(items_lf: pl.LazyFrame) -> pl.LazyFrame:
    cols = items_lf.columns
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

def build_features_for_pairs(
    transactions_lf: pl.LazyFrame,
    items_lf: pl.LazyFrame,
    pairs_df: pl.DataFrame,  # columns: customer_id, item_id (both string)
    begin_hist: datetime,
    end_hist: datetime,
    extra_feature_dir: str,
    trend_month_lag: int,
) -> pl.DataFrame:
    """
    Build the same feature set used in feature-label, for arbitrary (customer_id, item_id) pairs.
    """
    # ensure datetime
    tx = transactions_lf.with_columns([
        pl.col("created_date").cast(pl.Datetime, strict=False).alias("created_date"),
        pl.col("customer_id").cast(pl.Utf8).alias("customer_id"),
        pl.col("item_id").cast(pl.Utf8).alias("item_id"),
    ])

    hist = tx.filter(
        pl.col("created_date").is_between(
            pl.lit(begin_hist, dtype=pl.Datetime),
            pl.lit(end_hist, dtype=pl.Datetime),
            closed="both"
        )
    )

    item_attrs = _canonical_item_attrs(items_lf)

    hist_enriched = hist.join(item_attrs, on="item_id", how="left")

    # Aggregate features
    brand_counts = hist_enriched.group_by(["customer_id", "brand_final"]).agg(pl.len().alias("brand_counts"))
    age_counts = hist_enriched.group_by(["customer_id", "age_bucket_final"]).agg(pl.len().alias("age_counts"))
    category_counts = hist_enriched.group_by(["customer_id", "category"]).agg(pl.len().alias("category_counts"))
    target_user_group_counts = hist_enriched.group_by(["customer_id", "target_user_group_final"]).agg(pl.len().alias("target_user_group_counts"))
    last_cat_purchase = hist_enriched.group_by(["customer_id", "category"]).agg(pl.col("created_date").max().alias("last_purchase_date"))

    cand = pairs_df.lazy().with_columns([
        pl.col("customer_id").cast(pl.Utf8),
        pl.col("item_id").cast(pl.Utf8),
    ])

    features = (
        cand
        .join(item_attrs, on="item_id", how="left")
        .join(brand_counts, on=["customer_id", "brand_final"], how="left")
        .join(age_counts, on=["customer_id", "age_bucket_final"], how="left")
        .join(category_counts, on=["customer_id", "category"], how="left")
        .join(target_user_group_counts, on=["customer_id", "target_user_group_final"], how="left")
        .join(last_cat_purchase, on=["customer_id", "category"], how="left")
        .with_columns([
            pl.col("brand_counts").fill_null(0).cast(pl.Int32),
            pl.col("age_counts").fill_null(0).cast(pl.Int32),
            pl.col("category_counts").fill_null(0).cast(pl.Int32),
            pl.col("target_user_group_counts").fill_null(0).cast(pl.Int32),
        ])
        .with_columns(
            (
                (
                    pl.lit(end_hist, dtype=pl.Datetime)
                    - pl.col("last_purchase_date").cast(pl.Datetime, strict=False)
                )
                .dt.total_days()
                .cast(pl.Int32)
            ).alias("time_since_last_purchase_in_B_category")
        )
        .with_columns(
            pl.col("time_since_last_purchase_in_B_category").fill_null(9999)
        )
    )

    # ---- Extra features joins (all optional; if file missing => skip)
    def _try_scan(path):
        try:
            return pl.scan_parquet(path)
        except Exception:
            return None

    # item-level: price_segment
    ps = _try_scan(os.path.join(extra_feature_dir, "price_segment.parquet"))
    if ps is not None and ("item_id" in ps.columns) and ("price_segment" in ps.columns):
        features = features.join(
            ps.select([pl.col("item_id").cast(pl.Utf8), pl.col("price_segment").cast(pl.Utf8)]),
            on="item_id", how="left"
        )
    # user-level: buy_segment
    cb = _try_scan(os.path.join(extra_feature_dir, "customer_behavior.parquet"))
    if cb is not None and ("customer_id" in cb.columns) and ("buy_segment" in cb.columns):
        features = features.join(
            cb.select([pl.col("customer_id").cast(pl.Utf8), pl.col("buy_segment").cast(pl.Utf8)]),
            on="customer_id", how="left"
        )
    # user-level: luxury_level
    lux = _try_scan(os.path.join(extra_feature_dir, "customer_luxury.parquet"))
    if lux is not None:
        # accept both luxury_level or customer_luxury naming if exists
        lux_col = "luxury_level" if "luxury_level" in lux.columns else ("customer_luxury" if "customer_luxury" in lux.columns else None)
        if ("customer_id" in lux.columns) and lux_col:
            features = features.join(
                lux.select([pl.col("customer_id").cast(pl.Utf8), pl.col(lux_col).cast(pl.Utf8).alias("luxury_level")]),
                on="customer_id", how="left"
            )
    # user-level: age_final
    agef = _try_scan(os.path.join(extra_feature_dir, "customer_age_features.parquet"))
    if agef is not None and ("customer_id" in agef.columns) and ("age_final" in agef.columns):
        features = features.join(
            agef.select([pl.col("customer_id").cast(pl.Utf8), pl.col("age_final")]),
            on="customer_id", how="left"
        )
    # user-level: brand_segment
    bs = _try_scan(os.path.join(extra_feature_dir, "brand_segment.parquet"))
    if bs is not None and ("customer_id" in bs.columns) and ("brand_segment" in bs.columns):
        features = features.join(
            bs.select([pl.col("customer_id").cast(pl.Utf8), pl.col("brand_segment").cast(pl.Utf8)]),
            on="customer_id", how="left"
        )
    # item-level: top10_by_cat
    # top10 = _try_scan(os.path.join(extra_feature_dir, "top10_by_cat_month.parquet"))
    # if top10 is not None and ("item_id" in top10.columns):
    #     keep_cols = [c for c in ["item_id", "top10_by_cat", "rank", "total_sold"] if c in top10.columns]
    #     if len(keep_cols) > 1:
    #         sel = [pl.col("item_id").cast(pl.Utf8)] + [pl.col(c) for c in keep_cols if c != "item_id"]
    #         features = features.join(top10.select(sel), on="item_id", how="left")

    # item-level by month: top10_by_cat_month (single file with month, category_l1, item_id, total_sold, rank)
    top10m = _try_scan(os.path.join(extra_feature_dir, "top10_by_cat_month.parquet"))
    if top10m is not None and ("item_id" in top10m.columns) and ("month" in top10m.columns):
        # build trend_month_key = YYYY-MM from end_hist shifted by lag months
        def shift_month(dt: datetime, lag: int) -> str:
            y, m = dt.year, dt.month
            m2 = m - lag
            while m2 <= 0:
                y -= 1
                m2 += 12
            return f"{y:04d}-{m2:02d}"

        trend_month_key = shift_month(end_hist, trend_month_lag)

        # normalize month column to YYYY-MM string
        mcol = top10m.select([pl.col("month")]).collect_schema().get("month")
        t = top10m
        if mcol == pl.Date:
            t = t.with_columns(pl.col("month").dt.strftime("%Y-%m").alias("month_key"))
        elif mcol == pl.Datetime:
            t = t.with_columns(pl.col("month").dt.strftime("%Y-%m").alias("month_key"))
        else:
            # string/int: take first 7 chars if looks like YYYY-MM-...
            t = t.with_columns(
                pl.col("month").cast(pl.Utf8).str.slice(0, 7).alias("month_key")
            )

        join_keys_right = ["item_id", "month_key"]
        join_keys_left = ["item_id", "trend_month_key"]
        if "category_l1" in top10m.columns:
            join_keys_right = ["item_id", "category_l1", "month_key"]
            join_keys_left = ["item_id", "category_l1", "trend_month_key"]

        features = (
            features
            .with_columns(pl.lit(trend_month_key).cast(pl.Utf8).alias("trend_month_key"))
            .join(
                t.select([
                    pl.col("item_id").cast(pl.Utf8),
                    *( [pl.col("category_l1").cast(pl.Utf8)] if "category_l1" in top10m.columns else [] ),
                    pl.col("month_key").cast(pl.Utf8),
                    (pl.col("rank").cast(pl.Int32).alias("rank_top10_by_cat_month") if "rank" in top10m.columns else pl.lit(None).cast(pl.Int32).alias("rank_top10_by_cat_month")),
                    (pl.col("total_sold").cast(pl.Float32).alias("total_sold_top10_by_cat_month") if "total_sold" in top10m.columns else pl.lit(None).cast(pl.Float32).alias("total_sold_top10_by_cat_month")),
                ]),
                left_on=join_keys_left,
                right_on=join_keys_right,
                how="left"
            )
            .with_columns([
                pl.col("rank_top10_by_cat_month").fill_null(9999),
                pl.col("total_sold_top10_by_cat_month").fill_null(0.0),
            ])
            .drop(["trend_month_key"])
        )

    # fill missing categoricals
    features = features.with_columns([
        pl.col("price_segment").fill_null("__MISSING__") if "price_segment" in features.columns else pl.lit("__MISSING__").alias("price_segment"),
        pl.col("buy_segment").fill_null("__MISSING__") if "buy_segment" in features.columns else pl.lit("__MISSING__").alias("buy_segment"),
        pl.col("luxury_level").fill_null("__MISSING__") if "luxury_level" in features.columns else pl.lit("__MISSING__").alias("luxury_level"),
        pl.col("brand_segment").fill_null("__MISSING__") if "brand_segment" in features.columns else pl.lit("__MISSING__").alias("brand_segment"),
    ])

    if "age_final" in features.columns:
        features = features.with_columns(pl.col("age_final").fill_null(-1).cast(pl.Int32))

    return features.collect()


# ------------------------------
# Training
# ------------------------------
def build_train_table(feature_label_path: str, N_neg: int) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Load feature_label_final.parquet and perform per-user negative sampling.
    Returns: (train_df, feature_cols, cat_cols)
    """
    print(f"[Stage2] Loading feature-label: {feature_label_path}")
    fl = pl.scan_parquet(feature_label_path).collect()

    # normalize id types to string for stable joins with groundtruth
    fl = fl.with_columns([
        pl.col("customer_id").cast(pl.Utf8).alias("customer_id"),
        pl.col("item_id").cast(pl.Utf8).alias("item_id"),
    ])

    # identify label column
    if "Y" not in fl.columns:
        raise ValueError("feature_label must contain column `Y`.")

    # Fill defaults for numerics if any null
    numeric_fill0 = [c for c, dt in zip(fl.columns, fl.dtypes) if dt in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64) and c not in ("Y",)]
    if numeric_fill0:
        fl = fl.with_columns([pl.col(c).fill_null(0) for c in numeric_fill0])

    # Categorical columns: object-like
    cat_cols = [c for c, dt in zip(fl.columns, fl.dtypes) if dt == pl.Utf8 and c not in ("customer_id", "item_id")]
    if cat_cols:
        fl = fl.with_columns([pl.col(c).fill_null("__MISSING__") for c in cat_cols])

    # Pandas for sampling & LightGBM sklearn API
    df = fl.to_pandas()
    df["Y"] = df["Y"].astype(int)

    # negative sampling per user
    if N_neg is not None and N_neg > 0:
        parts = []
        print(f"[Stage2] Negative sampling: N_neg={N_neg} (per user, keep all positives)")
        for u, g in tqdm(df.groupby("customer_id"), desc="Neg-sampling by user"):
            pos = g[g["Y"] == 1]
            neg = g[g["Y"] == 0]
            if len(neg) > N_neg:
                neg = neg.sample(n=N_neg, random_state=42)
            parts.append(pd.concat([pos, neg], axis=0, ignore_index=True))
        df = pd.concat(parts, axis=0, ignore_index=True)

    # Feature columns (exclude ids and label)
    feature_cols = [c for c in df.columns if c not in ("customer_id", "item_id", "Y")]

    # Cast categoricals to pandas 'category' for LightGBM
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")

    return df, feature_cols, cat_cols


def train_lgbm(df: pd.DataFrame, feature_cols: List[str], cat_cols: List[str], cfg: dict, out_dir: str):
    from lightgbm import LGBMClassifier

    X = df[feature_cols]
    y = df["Y"].astype(int).values

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    spw = (n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0

    params = dict(cfg)
    # enforce a few safety defaults
    params.setdefault("objective", "binary")
    params.setdefault("random_state", 42)
    params.setdefault("n_estimators", 800)
    params.setdefault("learning_rate", 0.05)
    params.setdefault("num_threads", -1)

    model = LGBMClassifier(**params, scale_pos_weight=spw)

    print(f"[Stage2] Training LightGBM. n_pos={n_pos}, n_neg={n_neg}, scale_pos_weight={spw:.3f}")
    model.fit(X, y, categorical_feature=cat_cols if cat_cols else "auto")

    # save
    _ensure_dir(out_dir)
    model_path = os.path.join(out_dir, "stage2_model.pkl")
    joblib.dump(model, model_path)
    print(f"[Stage2] Saved model: {model_path}")

    feat_path = os.path.join(out_dir, "stage2_feature_cols.json")
    _save_json({"feature_cols": feature_cols, "cat_cols": cat_cols}, feat_path)
    print(f"[Stage2] Saved feature cols: {feat_path}")

    return model, spw, {"n_pos": n_pos, "n_neg": n_neg, "scale_pos_weight": spw}


# ------------------------------
# Inference: Stage1 candidates + Stage2 scoring
# ------------------------------
def build_hist_from_transactions(transactions_glob: str, end_dt: datetime) -> Dict[str, List[str]]:
    """
    Build hist dict: user -> list of item_ids with created_date < end_dt (exclusive)
    """
    lf = _scan_parquet_glob(transactions_glob).with_columns([
        pl.col("created_date").cast(pl.Datetime, strict=False).alias("created_date"),
        pl.col("customer_id").cast(pl.Utf8).alias("customer_id"),
        pl.col("item_id").cast(pl.Utf8).alias("item_id"),
    ]).filter(pl.col("created_date") < pl.lit(end_dt, dtype=pl.Datetime))

    df = lf.select(["customer_id", "item_id"]).unique().collect()
    # group
    hist = (
        df.group_by("customer_id")
          .agg(pl.col("item_id").alias("items"))
          .to_dict(as_series=False)
    )
    # to python dict
    out = {}
    for u, items in zip(hist["customer_id"], hist["items"]):
        out[u] = items
    return out

def score_candidates_to_pred(
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
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Return:
      pred: user -> ranked item list (length N_total_cand)
      hist_dict: user -> history item list (for precision_at_k)
    """
    print("[Stage2] Building hist dict for evaluation ...")
    hist_dict = build_hist_from_transactions(transactions_glob, end_hist + timedelta(days=1))
    print(f"[Stage2] Hist users: {len(hist_dict):,}")

    print("[Stage2] Loading transactions/items lazyframes for feature building ...")
    tx_lf = _scan_parquet_glob(transactions_glob)
    items_lf = _scan_parquet_glob(items_glob)

    pred = {}

    # batching users to keep memory stable
    for i in tqdm(range(0, len(gt_users), batch_users), desc="Scoring users (batch)"):
        users_batch = gt_users[i:i+batch_users]

        # 1) generate candidates per user
        batch_pairs = []
        for u in users_batch:
            # stage1 cold-start fallback is handled inside stage1 model
            cand = stage1_model.recommend_for_user_id(u, top_k=N_total_cand)
            if filter_bought_items_for_candidates and (u in hist_dict):
                hist_set = set(hist_dict[u])
                cand = [it for it in cand if it not in hist_set]
                if len(cand) < N_total_cand:
                    # keep as-is; stage1 already had trending/pop fallback, but we filtered after
                    pass
            for it in cand[:N_total_cand]:
                batch_pairs.append((u, it))

        if not batch_pairs:
            continue

        pairs_df = pl.DataFrame(batch_pairs, schema=["customer_id", "item_id"])

        # 2) build features for these pairs
        feat_df = build_features_for_pairs(
            transactions_lf=tx_lf,
            items_lf=items_lf,
            pairs_df=pairs_df,
            begin_hist=begin_hist,
            end_hist=end_hist,
            extra_feature_dir=extra_feature_dir,
            trend_month_lag=trend_month_lag,
        )

        # 3) to pandas for model predict
        pdf = feat_df.to_pandas()

        # ensure all expected columns exist
        for c in feature_cols:
            if c not in pdf.columns:
                # default fill
                pdf[c] = 0
        # fill categorical missing
        for c in cat_cols:
            if c in pdf.columns:
                pdf[c] = pdf[c].astype("category")
            else:
                pdf[c] = "__MISSING__"
                pdf[c] = pdf[c].astype("category")

        X = pdf[feature_cols]
        scores = stage2_model.predict_proba(X)[:, 1]

        pdf["score"] = scores

        # 4) build per-user ranked list
        for u, g in pdf.groupby("customer_id"):
            g2 = g.sort_values("score", ascending=False)
            pred[u] = g2["item_id"].astype(str).tolist()

    return pred, hist_dict


# ------------------------------
# Main
# ------------------------------
def main(config_path: str):
    cfg = _load_json(config_path)
    paths = cfg["paths"]
    params = cfg["params"]
    model_cfg = cfg.get("stage2_model", {})
    time_cfg = cfg.get("time", {})

    out_dir = paths["out_dir"]
    _ensure_dir(out_dir)
    _save_json(cfg, os.path.join(out_dir, "stage2_config_used.json"))
    print(f"[Stage2] Using config: {config_path}")
    print(f"[Stage2] Outputs to: {out_dir}")

    # 1) Load Stage1 candidate model
    print("[Stage2] Loading Stage1 candidate model ...")
    prefix = paths["stage1_prefix"]
    art_dir = paths["stage1_artifacts_dir"]

    stage1 = load_stage1_from_artifacts(
        meta_npz_path=os.path.join(art_dir, f"{prefix}_meta.npz"),
        user_items_npz_path=os.path.join(art_dir, f"{prefix}_user_items.npz"),
        tfidf_npz_path=os.path.join(art_dir, f"{prefix}_tfidf.npz"),
        cosine_npz_path=os.path.join(art_dir, f"{prefix}_cosine.npz"),
    )
    print("[Stage2] Stage1 loaded OK.")

    # 2) Load feature label and build train table
    df_train, feature_cols, cat_cols = build_train_table(
        feature_label_path=paths["feature_label_path"],
        N_neg=int(params.get("N_neg", 10)),
    )

    # 3) Train Stage2 model
    print("[Stage2] Training Stage2 model ...")
    stage2_model, spw, train_stats = train_lgbm(
        df=df_train,
        feature_cols=feature_cols,
        cat_cols=cat_cols,
        cfg=model_cfg,
        out_dir=out_dir
    )

    # save training stats
    metrics_path = os.path.join(out_dir, "stage2_train_stats.json")
    _save_json(train_stats, metrics_path)
    print(f"[Stage2] Saved train stats: {metrics_path}")

    # 4) Evaluate precision@k on groundtruth
    print("[Stage2] Loading groundtruth ...")
    gt = _load_groundtruth(paths["groundtruth_path"])
    # unify keys to string
    gt = {_as_str_id(k): [str(x) for x in v] for k, v in gt.items()}
    gt_users = list(gt.keys())
    print(f"[Stage2] Groundtruth users: {len(gt_users):,}")

    # define test time windows
    test_begin = _parse_dt(time_cfg.get("test_begin", "2025-01-01"))
    # hist for features: last len_hist days before test_begin
    len_hist = int(params.get("len_hist", 120))
    begin_hist = test_begin - timedelta(days=len_hist)
    end_hist = test_begin - timedelta(days=1)
    trend_month_lag = int(time_cfg.get("trend_month_lag", 2))

    N_total_cand = int(params.get("N_trend", 0)) + 2 * int(params.get("N_cand", 300))
    print(f"[Stage2] Eval window: begin_hist={begin_hist.date()} end_hist={end_hist.date()} | N_total_cand={N_total_cand}")

    pred, hist_dict = score_candidates_to_pred(
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
        batch_users=2000,  # adjust if memory tight
        trend_month_lag=trend_month_lag,
        filter_bought_items_for_candidates=bool(params.get("filter_bought_items", False)),
    )

    # save predictions
    pred_path = os.path.join(out_dir, "pred_stage2.pkl")
    with open(pred_path, "wb") as f:
        pickle.dump(pred, f)
    print(f"[Stage2] Saved predictions: {pred_path}")

    # 5) Precision@K: both filtered/unfiltered + cold-start count
    K = int(params.get("Topk", 10))
    p_unf, cold_unf = precision_at_k(pred, gt, hist_dict, filter_bought_items=False, K=K)
    p_flt, cold_flt = precision_at_k(pred, gt, hist_dict, filter_bought_items=True,  K=K)

    # cold-start users: union (they should be identical lists in most cases)
    cold_users = sorted(set(cold_unf) | set(cold_flt))
    cold_path = os.path.join(out_dir, "cold_start_users.pkl")
    with open(cold_path, "wb") as f:
        pickle.dump(cold_users, f)
    print(f"[Stage2] Saved cold-start users: {cold_path}")

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
    }
    report_path = os.path.join(out_dir, "stage2_precision_report.json")
    _save_json(report, report_path)

    print("[Stage2] ===== Precision@K Report =====")
    print(f"Precision@{K} (unfiltered): {p_unf:.6f}")
    print(f"Precision@{K} (filtered):   {p_flt:.6f}")
    print(f"Cold-start users: {len(cold_users):,} / {len(gt_users):,}")
    print(f"[Stage2] Saved report: {report_path}")


if __name__ == "__main__":
    import argparse
    import warnings

    warnings.filterwarnings("ignore", category=pl.exceptions.PerformanceWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    main(args.config)
