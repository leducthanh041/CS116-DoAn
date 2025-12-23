
# -*- coding: utf-8 -*-
"""
Quick test (smoke test) for Stage 2 pipeline BEFORE full training.

What this script does (fast):
1) Load config JSON (same schema as train_stage2_lgbm.py)
2) Load Stage1 candidate model artifacts
3) Load feature_label_final.parquet and sample a small subset of rows for training
4) Negative sampling per user (small N_neg)
5) Train a small model quickly:
   - Prefer LightGBM if available
   - Fallback to sklearn LogisticRegression if LightGBM is not installed
6) Evaluate precision@K on a small subset of groundtruth users:
   - Print Precision@K (unfiltered) and Precision@K (filtered)
   - Print number of cold-start users
7) Save quick-test outputs to a dedicated folder.

Run:
  python test_stage2_quick.py --config stage2_train_config.json

Optional overrides:
  --max_train_rows 200000
  --max_users_gt 200
  --n_estimators 80
  --N_neg 5
  --batch_users 50
"""

from __future__ import annotations
import os
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import polars as pl
from tqdm.auto import tqdm
import joblib

# Stage1 loader (must exist in your project)
from stage1_implicit_itemitem import load_stage1_from_artifacts


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

def _parse_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")

def _as_str(x) -> str:
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

def build_hist_from_transactions(transactions_glob: str, end_dt: datetime) -> Dict[str, List[str]]:
    lf = _scan_parquet_glob(transactions_glob).with_columns([
        pl.col("created_date").cast(pl.Datetime, strict=False).alias("created_date"),
        pl.col("customer_id").cast(pl.Utf8).alias("customer_id"),
        pl.col("item_id").cast(pl.Utf8).alias("item_id"),
    ]).filter(pl.col("created_date") < pl.lit(end_dt, dtype=pl.Datetime))

    df = lf.select(["customer_id", "item_id"]).unique().collect()
    grouped = df.group_by("customer_id").agg(pl.col("item_id").alias("items"))
    out = {}
    d = grouped.to_dict(as_series=False)
    for u, items in zip(d["customer_id"], d["items"]):
        out[u] = items
    return out

def build_features_for_pairs(
    transactions_lf: pl.LazyFrame,
    items_lf: pl.LazyFrame,
    pairs_df: pl.DataFrame,
    begin_hist: datetime,
    end_hist: datetime,
    extra_feature_dir: str,
    trend_month_lag: int,
) -> pl.DataFrame:
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
        .with_columns(pl.col("time_since_last_purchase_in_B_category").fill_null(9999))
    )

    # Optional extra features
    def _try_scan(path):
        try:
            return pl.scan_parquet(path)
        except Exception:
            return None

    ps = _try_scan(os.path.join(extra_feature_dir, "price_segment.parquet"))
    if ps is not None and ("item_id" in ps.columns) and ("price_segment" in ps.columns):
        features = features.join(ps.select([pl.col("item_id").cast(pl.Utf8), pl.col("price_segment").cast(pl.Utf8)]),
                                on="item_id", how="left")

    cb = _try_scan(os.path.join(extra_feature_dir, "customer_behavior.parquet"))
    if cb is not None and ("customer_id" in cb.columns) and ("buy_segment" in cb.columns):
        features = features.join(cb.select([pl.col("customer_id").cast(pl.Utf8), pl.col("buy_segment").cast(pl.Utf8)]),
                                on="customer_id", how="left")

    lux = _try_scan(os.path.join(extra_feature_dir, "customer_luxury.parquet"))
    if lux is not None:
        lux_col = "luxury_level" if "luxury_level" in lux.columns else ("customer_luxury" if "customer_luxury" in lux.columns else None)
        if ("customer_id" in lux.columns) and lux_col:
            features = features.join(lux.select([pl.col("customer_id").cast(pl.Utf8), pl.col(lux_col).cast(pl.Utf8).alias("luxury_level")]),
                                    on="customer_id", how="left")

    agef = _try_scan(os.path.join(extra_feature_dir, "customer_age_features.parquet"))
    if agef is not None and ("customer_id" in agef.columns) and ("age_final" in agef.columns):
        features = features.join(agef.select([pl.col("customer_id").cast(pl.Utf8), pl.col("age_final")]),
                                on="customer_id", how="left")

    bs = _try_scan(os.path.join(extra_feature_dir, "brand_segment.parquet"))
    if bs is not None and ("customer_id" in bs.columns) and ("brand_segment" in bs.columns):
        features = features.join(bs.select([pl.col("customer_id").cast(pl.Utf8), pl.col("brand_segment").cast(pl.Utf8)]),
                                on="customer_id", how="left")

    # top10 = _try_scan(os.path.join(extra_feature_dir, "top10_by_cat_month.parquet"))
    # if top10 is not None and ("item_id" in top10.columns):
    #     keep = [c for c in ["top10_by_cat", "rank", "total_sold"] if c in top10.columns]
    #     if keep:
    #         features = features.join(top10.select([pl.col("item_id").cast(pl.Utf8)] + [pl.col(c) for c in keep]),
    #                                 on="item_id", how="left")

    # by-month top10 file (month, category_l1, item_id, total_sold, rank)
    top10m = _try_scan(os.path.join(extra_feature_dir, "top10_by_cat_month.parquet"))
    if top10m is not None and ("item_id" in top10m.columns) and ("month" in top10m.columns):
        def shift_month(dt: datetime, lag: int) -> str:
            y, m = dt.year, dt.month
            m2 = m - lag
            while m2 <= 0:
                y -= 1
                m2 += 12
            return f"{y:04d}-{m2:02d}"

        trend_month_key = shift_month(end_hist, trend_month_lag)
        schema = top10m.collect_schema()
        mtype = schema.get("month")
        t = top10m
        if mtype in (pl.Date, pl.Datetime):
            t = t.with_columns(pl.col("month").dt.strftime("%Y-%m").alias("month_key"))
        else:
            t = t.with_columns(pl.col("month").cast(pl.Utf8).str.slice(0, 7).alias("month_key"))

        if "category_l1" in top10m.columns:
            features = (
                features
                .with_columns(pl.lit(trend_month_key).cast(pl.Utf8).alias("trend_month_key"))
                .join(
                    t.select([
                        pl.col("item_id").cast(pl.Utf8),
                        pl.col("category_l1").cast(pl.Utf8),
                        pl.col("month_key").cast(pl.Utf8),
                        (pl.col("rank").cast(pl.Int32).alias("rank_top10_by_cat_month") if "rank" in top10m.columns else pl.lit(None).cast(pl.Int32).alias("rank_top10_by_cat_month")),
                        (pl.col("total_sold").cast(pl.Float32).alias("total_sold_top10_by_cat_month") if "total_sold" in top10m.columns else pl.lit(None).cast(pl.Float32).alias("total_sold_top10_by_cat_month")),
                    ]),
                    left_on=["item_id", "category_l1", "trend_month_key"],
                    right_on=["item_id", "category_l1", "month_key"],
                    how="left",
                )
                .with_columns([
                    pl.col("rank_top10_by_cat_month").fill_null(9999),
                    pl.col("total_sold_top10_by_cat_month").fill_null(0.0),
                ])
                .drop(["trend_month_key"])
            )
        else:
            features = (
                features
                .with_columns(pl.lit(trend_month_key).cast(pl.Utf8).alias("trend_month_key"))
                .join(
                    t.select([
                        pl.col("item_id").cast(pl.Utf8),
                        pl.col("month_key").cast(pl.Utf8),
                        (pl.col("rank").cast(pl.Int32).alias("rank_top10_by_cat_month") if "rank" in top10m.columns else pl.lit(None).cast(pl.Int32).alias("rank_top10_by_cat_month")),
                        (pl.col("total_sold").cast(pl.Float32).alias("total_sold_top10_by_cat_month") if "total_sold" in top10m.columns else pl.lit(None).cast(pl.Float32).alias("total_sold_top10_by_cat_month")),
                    ]),
                    left_on=["item_id", "trend_month_key"],
                    right_on=["item_id", "month_key"],
                    how="left",
                )
                .with_columns([
                    pl.col("rank_top10_by_cat_month").fill_null(9999),
                    pl.col("total_sold_top10_by_cat_month").fill_null(0.0),
                ])
                .drop(["trend_month_key"])
            )

    for cname in ["price_segment", "buy_segment", "luxury_level", "brand_segment"]:
        if cname in features.columns:
            features = features.with_columns(pl.col(cname).fill_null("__MISSING__"))
        else:
            features = features.with_columns(pl.lit("__MISSING__").alias(cname))

    if "age_final" in features.columns:
        features = features.with_columns(pl.col("age_final").fill_null(-1).cast(pl.Int32))

    return features.collect()

def build_quick_train_table(feature_label_path: str, max_rows: int, N_neg: int) -> Tuple[pd.DataFrame, List[str], List[str]]:
    print(f"[QuickTest] Loading feature-label: {feature_label_path}")
    fl = pl.scan_parquet(feature_label_path).collect()
    fl = fl.with_columns([pl.col("customer_id").cast(pl.Utf8), pl.col("item_id").cast(pl.Utf8)])

    if "Y" not in fl.columns:
        raise ValueError("feature_label must contain column `Y`")

    df = fl.to_pandas()
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
    df["Y"] = df["Y"].astype(int)

    cat_cols = [c for c in df.columns if c not in ("customer_id", "item_id", "Y") and df[c].dtype == object]
    for c in cat_cols:
        df[c] = df[c].fillna("__MISSING__").astype("category")
    for c in df.columns:
        if c in ("customer_id", "item_id", "Y"): 
            continue
        if c not in cat_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    if N_neg is not None and N_neg > 0:
        parts = []
        print(f"[QuickTest] Negative sampling per user: N_neg={N_neg}")
        for u, g in tqdm(df.groupby("customer_id"), desc="Neg-sampling"):
            pos = g[g["Y"] == 1]
            neg = g[g["Y"] == 0]
            if len(neg) > N_neg:
                neg = neg.sample(n=N_neg, random_state=42)
            parts.append(pd.concat([pos, neg], axis=0, ignore_index=True))
        df = pd.concat(parts, axis=0, ignore_index=True)

    feature_cols = [c for c in df.columns if c not in ("customer_id", "item_id", "Y")]
    return df, feature_cols, cat_cols

def train_quick_model(df: pd.DataFrame, feature_cols: List[str], cat_cols: List[str], n_estimators: int, out_dir: str):
    X = df[feature_cols]
    y = df["Y"].astype(int).values
    n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum())
    spw = (n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0
    print(f"[QuickTest] Train rows={len(df):,} | n_pos={n_pos:,} n_neg={n_neg:,} scale_pos_weight={spw:.3f}")

    try:
        from lightgbm import LGBMClassifier
        params = dict(
            n_estimators=n_estimators,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            objective="binary",
        )
        model = LGBMClassifier(**params, scale_pos_weight=spw)
        model.fit(X, y, categorical_feature=cat_cols if cat_cols else "auto")
        algo = "lightgbm"
    except Exception as e:
        print(f"[QuickTest][WARN] LightGBM unavailable or failed ({e}). Falling back to LogisticRegression.")
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression

        cat_features = [c for c in cat_cols if c in feature_cols]
        num_features = [c for c in feature_cols if c not in cat_features]
        pre = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
                ("num", "passthrough", num_features),
            ]
        )
        model = Pipeline(steps=[
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=200, n_jobs=-1, class_weight="balanced", solver="saga"))
        ])
        model.fit(X, y)
        algo = "logreg"

    _ensure_dir(out_dir)
    joblib.dump(model, os.path.join(out_dir, "stage2_quick_model.pkl"))
    _save_json({"algo": algo, "feature_cols": feature_cols, "cat_cols": cat_cols}, os.path.join(out_dir, "stage2_quick_meta.json"))
    return model, algo

def score_and_eval_quick(
    stage1,
    stage2_model,
    algo: str,
    feature_cols: List[str],
    cat_cols: List[str],
    gt: Dict[str, List[str]],
    transactions_glob: str,
    items_glob: str,
    extra_feature_dir: str,
    begin_hist: datetime,
    end_hist: datetime,
    N_total_cand: int,
    batch_users: int,
    trend_month_lag: int,
    filter_bought_items_for_candidates: bool,
    Topk: int,
):
    gt_users = list(gt.keys())

    print("[QuickTest] Building hist dict ...")
    hist = build_hist_from_transactions(transactions_glob, end_hist + timedelta(days=1))
    print(f"[QuickTest] Hist users: {len(hist):,}")

    tx_lf = _scan_parquet_glob(transactions_glob)
    items_lf = _scan_parquet_glob(items_glob)

    pred: Dict[str, List[str]] = {}

    for i in tqdm(range(0, len(gt_users), batch_users), desc="Scoring users"):
        users_batch = gt_users[i:i+batch_users]
        pairs = []
        for u in users_batch:
            cand = stage1.recommend_for_user_id(u, top_k=N_total_cand)
            if filter_bought_items_for_candidates and (u in hist):
                hs = set(hist[u])
                cand = [it for it in cand if it not in hs]
            for it in cand[:N_total_cand]:
                pairs.append((u, it))
        if not pairs:
            continue

        pairs_df = pl.DataFrame(pairs, schema=["customer_id", "item_id"])
        feat = build_features_for_pairs(
            transactions_lf=tx_lf,
            items_lf=items_lf,
            pairs_df=pairs_df,
            begin_hist=begin_hist,
            end_hist=end_hist,
            extra_feature_dir=extra_feature_dir,
            trend_month_lag=trend_month_lag,
        )
        pdf = feat.to_pandas()

        for c in feature_cols:
            if c not in pdf.columns:
                pdf[c] = 0
        for c in cat_cols:
            if c in pdf.columns:
                pdf[c] = pdf[c].fillna("__MISSING__").astype("category")
            else:
                pdf[c] = "__MISSING__"
                pdf[c] = pdf[c].astype("category")

        X = pdf[feature_cols]
        scores = stage2_model.predict_proba(X)[:, 1]
        pdf["score"] = scores

        for u, g in pdf.groupby("customer_id"):
            g2 = g.sort_values("score", ascending=False)
            pred[u] = g2["item_id"].astype(str).tolist()

    p_unf, cold_unf = precision_at_k(pred, gt, hist, filter_bought_items=False, K=Topk)
    p_flt, cold_flt = precision_at_k(pred, gt, hist, filter_bought_items=True,  K=Topk)
    cold_users = sorted(set(cold_unf) | set(cold_flt))

    report = {
        "precision_at_k_unfiltered": p_unf,
        "precision_at_k_filtered": p_flt,
        "K": Topk,
        "n_users_gt": len(gt_users),
        "n_users_pred": len(pred),
        "n_cold_start": len(cold_users),
        "filter_bought_items_for_candidates": filter_bought_items_for_candidates,
    }
    return report, pred, cold_users

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--max_train_rows", type=int, default=200_000)
    ap.add_argument("--max_users_gt", type=int, default=200)
    ap.add_argument("--n_estimators", type=int, default=80)
    ap.add_argument("--N_neg", type=int, default=5)
    ap.add_argument("--batch_users", type=int, default=50)
    args = ap.parse_args()

    cfg = _load_json(args.config)
    paths = cfg["paths"]
    params = cfg.get("params", {})
    time_cfg = cfg.get("time", {})

    base_out = paths.get("out_dir", "./artifacts/stage2_model")
    quick_out = os.path.join(base_out, "_quicktest")
    _ensure_dir(quick_out)
    _save_json({"config": cfg, "overrides": vars(args)}, os.path.join(quick_out, "quicktest_config_used.json"))
    print(f"[QuickTest] Output dir: {quick_out}")

    print("[QuickTest] Loading Stage1 model ...")
    prefix = paths["stage1_prefix"]
    art_dir = paths["stage1_artifacts_dir"]
    stage1 = load_stage1_from_artifacts(
        meta_npz_path=os.path.join(art_dir, f"{prefix}_meta.npz"),
        user_items_npz_path=os.path.join(art_dir, f"{prefix}_user_items.npz"),
        tfidf_npz_path=os.path.join(art_dir, f"{prefix}_tfidf.npz"),
        cosine_npz_path=os.path.join(art_dir, f"{prefix}_cosine.npz"),
    )
    print("[QuickTest] Stage1 loaded OK.")

    df_train, feature_cols, cat_cols = build_quick_train_table(
        feature_label_path=paths["feature_label_path"],
        max_rows=args.max_train_rows,
        N_neg=args.N_neg,
    )

    model, algo = train_quick_model(df_train, feature_cols, cat_cols, n_estimators=args.n_estimators, out_dir=quick_out)
    print(f"[QuickTest] Stage2 quick model trained: {algo}")

    gt = _load_groundtruth(paths["groundtruth_path"])
    gt = {_as_str(k): [str(x) for x in v] for k, v in gt.items()}
    gt_users = list(gt.keys())
    if args.max_users_gt and len(gt_users) > args.max_users_gt:
        gt_users = gt_users[:args.max_users_gt]
        gt = {u: gt[u] for u in gt_users}
    print(f"[QuickTest] Groundtruth users used: {len(gt_users):,}")

    test_begin = _parse_dt(time_cfg.get("test_begin", "2025-01-01"))
    len_hist = int(params.get("len_hist", 120))
    begin_hist = test_begin - timedelta(days=len_hist)
    end_hist = test_begin - timedelta(days=1)

    N_total_cand = int(params.get("N_trend", 0)) + 2 * int(params.get("N_cand", 300))
    trend_month_lag = int(time_cfg.get("trend_month_lag", 2))
    Topk = int(params.get("Topk", 10))
    filter_bought_items_for_candidates = bool(params.get("filter_bought_items", False))

    print(f"[QuickTest] begin_hist={begin_hist.date()} end_hist={end_hist.date()} | N_total_cand={N_total_cand} | Topk={Topk}")

    report, pred, cold_users = score_and_eval_quick(
        stage1=stage1,
        stage2_model=model,
        algo=algo,
        feature_cols=feature_cols,
        cat_cols=cat_cols,
        gt=gt,
        transactions_glob=paths["transactions_path_glob"],
        items_glob=paths["items_path_glob"],
        extra_feature_dir=paths["extra_feature_dir"],
        begin_hist=begin_hist,
        end_hist=end_hist,
        N_total_cand=N_total_cand,
        batch_users=args.batch_users,
        trend_month_lag=trend_month_lag,
        filter_bought_items_for_candidates=filter_bought_items_for_candidates,
        Topk=Topk,
    )

    with open(os.path.join(quick_out, "pred_quick.pkl"), "wb") as f:
        pickle.dump(pred, f)
    with open(os.path.join(quick_out, "cold_start_users_quick.pkl"), "wb") as f:
        pickle.dump(cold_users, f)
    _save_json(report, os.path.join(quick_out, "quicktest_precision_report.json"))

    print("[QuickTest] ===== QUICKTEST REPORT =====")
    print(f"Precision@{report['K']} (unfiltered): {report['precision_at_k_unfiltered']:.6f}")
    print(f"Precision@{report['K']} (filtered):   {report['precision_at_k_filtered']:.6f}")
    print(f"Cold-start users: {report['n_cold_start']:,} / {report['n_users_gt']:,}")
    print(f"[QuickTest] Saved report to: {os.path.join(quick_out, 'quicktest_precision_report.json')}")

if __name__ == "__main__":
    main()
