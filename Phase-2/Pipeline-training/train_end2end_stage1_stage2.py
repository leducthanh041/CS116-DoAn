# -*- coding: utf-8 -*-
"""
train_end2end_stage1_stage2.py

End-to-end pipeline:
  (1) Stage 1: Random Search (default 90 trials) over key params for implicit TFIDF + Cosine item-item candidate model
      - Select best by Recall@K (then Hit@K)
      - Save best Stage1 artifacts to {stage1_artifacts_dir}/{stage1_prefix}_*.{npz,json,csv}

  (2) Stage 2: Build feature-label (HIST=120d, RECENT=Dec/2024 via ref_datetime=2024-12-31)
      - Uses stage2_feature_label.py (already stable in your pipeline)

  (3) Stage 2: Train LightGBM as a RANKER (LambdaRank) to optimize NDCG@K
      - Uses feature_label_final.parquet
      - Supports quick smoke test via --quick

  (4) Evaluate on groundtruth.pkl (Jan/2025):
      - Precision@K (filter/unfilter)
      - NDCG@K (filter/unfilter)
      - Cold-start user list

Notes on params with unclear spec:
  - min_trans_items, session_window, min_coo, filter_fashion are NOT applied here (không biết spec chính xác).
    We keep them in config for future extension and print an explicit notice.

Requires local modules:
  - stage1_implicit_itemitem.py
  - stage2_feature_label.py
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import polars as pl
from tqdm.auto import tqdm

# local modules (must be in same working directory or PYTHONPATH)
from stage1_implicit_itemitem import (
    split_train_valid_by_days,
    build_user_items_csr,
    Stage1ImplicitItemItem,
)
import stage2_feature_label as s2fl


# -------------------------
# Utils
# -------------------------

def log(msg: str) -> None:
    print(msg, flush=True)

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def parse_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")

def to_iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")

def scan_parquet_glob(glob_path: str) -> pl.LazyFrame:
    return pl.scan_parquet(glob_path)

def month_key(dt: datetime) -> str:
    return f"{dt.year:04d}-{dt.month:02d}"

def shift_month_key(dt: datetime, lag_months: int) -> str:
    y, m = dt.year, dt.month
    m2 = m - lag_months
    while m2 <= 0:
        y -= 1
        m2 += 12
    return f"{y:04d}-{m2:02d}"

def load_groundtruth(path: str) -> Dict[str, List[str]]:
    with open(path, "rb") as f:
        gt = pickle.load(f)
    if not isinstance(gt, dict):
        raise ValueError(f"Unsupported groundtruth type: {type(gt)}")
    out = {}
    for k, v in gt.items():
        out[str(k)] = [str(x) for x in v]
    return out

def precision_at_k(pred: Dict[str, List[str]],
                   gt: Dict[str, List[str]],
                   hist: Dict[str, List[str]],
                   filter_bought_items: bool,
                   K: int) -> Tuple[float, List[str]]:
    precisions = []
    cold = []
    for u in gt.keys():
        if (u not in pred) or (u not in hist):
            cold.append(u)
            continue
        relevant = set(gt[u])
        if filter_bought_items:
            relevant -= set(hist[u])
        hits = len(set(pred[u][:K]) & relevant)
        precisions.append(hits / K)
    return (float(np.mean(precisions)) if precisions else 0.0), cold

def ndcg_at_k(pred: Dict[str, List[str]],
              gt: Dict[str, List[str]],
              hist: Dict[str, List[str]],
              filter_bought_items: bool,
              K: int) -> Tuple[float, List[str]]:
    ndcgs = []
    cold = []
    denom = np.log2(np.arange(2, K + 2))  # positions 1..K => log2(i+1)
    for u in gt.keys():
        if (u not in pred) or (u not in hist):
            cold.append(u)
            continue
        relevant = set(gt[u])
        if filter_bought_items:
            relevant -= set(hist[u])
        if not relevant:
            # if no relevant after filtering, skip from metric (consistent with earlier recall behavior)
            continue
        rec = pred[u][:K]
        rel_vec = np.array([1.0 if it in relevant else 0.0 for it in rec], dtype=np.float32)
        dcg = float((rel_vec / denom).sum())
        ideal_len = min(len(relevant), K)
        idcg = float((np.ones(ideal_len, dtype=np.float32) / denom[:ideal_len]).sum())
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return (float(np.mean(ndcgs)) if ndcgs else 0.0), cold


# -------------------------
# Stage 1: Random Search
# -------------------------

def sample_from_space(space: dict, rng: random.Random) -> dict:
    out = {}
    for k, v in space.items():
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError(f"search_space[{k}] must be non-empty list")
        out[k] = rng.choice(v)
    return out

def stage1_random_search(
    lf_all: pl.LazyFrame,
    test_begin: datetime,
    artifacts_dir: str,
    prefix: str,
    base_user_col: str,
    base_item_col: str,
    created_col: str,
    qty_col: str,
    price_col: str,
    num_threads: int,
    n_trials: int,
    random_state: int,
    space: dict,
    quick: bool,
    quick_max_rows: int,
) -> dict:
    """
    Returns best_params dict.
    Saves results CSV and best params JSON progressively.
    """
    ensure_dir(artifacts_dir)
    results_csv = os.path.join(artifacts_dir, f"{prefix}_random_search_results.csv")
    best_json = os.path.join(artifacts_dir, f"{prefix}_best_params.json")
    metrics_json = os.path.join(artifacts_dir, f"{prefix}_best_metrics.json")

    # Filter out any interactions after 2024-12-31 to avoid leakage into test month
    ref_dt = test_begin - timedelta(days=1)
    lf = lf_all.with_columns(pl.col(created_col).cast(pl.Datetime, strict=False).alias(created_col))
    lf = lf.filter(pl.col(created_col) <= pl.lit(ref_dt, dtype=pl.Datetime))

    if quick:
        log(f"[Stage1][QUICK] Limiting transactions to first {quick_max_rows:,} rows for smoke test.")
        lf = lf.limit(quick_max_rows)

    # Prepare CSV header
    if not os.path.exists(results_csv):
        with open(results_csv, "w", encoding="utf-8") as f:
            f.write("trial,recall,hit,n_users_eval,params_json\n")

    rng = random.Random(int(random_state))

    best = {"recall": -1.0, "hit": -1.0, "params": None}

    log(f"[Stage1] Random search: n_trials={n_trials} (quick={quick})")
    for t in tqdm(range(1, n_trials + 1), desc="Stage1 random search"):
        p = sample_from_space(space, rng)

        len_hist = int(p["len_hist"])
        len_val = int(p["len_val"])
        weight_type = str(p["weight_type"])
        N_cand = int(p["N_cand"])
        N_trend = int(p["N_trend"])
        allow_repeat = bool(p.get("allow_repeat", False))
        K_mult = int(p["K_mult"])
        K_model = max(20, min(800, K_mult * N_cand))

        # split by windows relative to ref_dt
        # We cannot change split_train_valid_by_days ref_dt directly, so we filter by <= ref_dt above.
        lf_train, lf_valid, train_start, valid_start, ref_dt_pol = split_train_valid_by_days(
            lf, created_col=created_col, len_hist=len_hist, len_val=len_val
        )

        # build CSR
        user_items, users, items, _, _ = build_user_items_csr(
            lf_train,
            user_col=base_user_col,
            item_col=base_item_col,
            qty_col=qty_col,
            price_col=price_col,
            weight_type=weight_type,
        )

        model = Stage1ImplicitItemItem(
            N_cand=N_cand,
            N_trend=N_trend,
            num_threads=num_threads,
            K_model=K_model,
            n_iter=1,
            filter_already_liked_items=(not allow_repeat),
        )
        model.fit_from_csr(user_items, users, items)

        K_eval = 2 * N_cand + N_trend
        m = model.eval_recall_hit(
            lf_valid,
            user_col=base_user_col,
            item_col=base_item_col,
            filter_train_history=True,
            k=K_eval,
        )

        # write trial result
        row = {
            "trial": t,
            "recall": m.recall,
            "hit": m.hit,
            "n_users_eval": m.n_users_eval,
            "params_json": json.dumps(
                {
                    "len_hist": len_hist,
                    "len_val": len_val,
                    "weight_type": weight_type,
                    "N_cand": N_cand,
                    "N_trend": N_trend,
                    "allow_repeat": allow_repeat,
                    "K_model": K_model,
                    "num_threads": num_threads,
                    "K_eval": K_eval,
                },
                ensure_ascii=False
            ),
        }
        with open(results_csv, "a", encoding="utf-8") as f:
            f.write(f"{row['trial']},{row['recall']:.10f},{row['hit']:.10f},{row['n_users_eval']},{row['params_json']}\n")

        # update best
        if (m.recall > best["recall"]) or (np.isclose(m.recall, best["recall"]) and m.hit > best["hit"]):
            best["recall"] = m.recall
            best["hit"] = m.hit
            best["params"] = json.loads(row["params_json"])
            save_json(best["params"], best_json)
            save_json({"recall": best["recall"], "hit": best["hit"], "n_users_eval": m.n_users_eval}, metrics_json)
            log(f"[Stage1] New best @trial={t}: recall={m.recall:.6f} hit={m.hit:.6f} params={best['params']}")

    if best["params"] is None:
        raise RuntimeError("[Stage1] Random search produced no valid trial result.")

    log(f"[Stage1] Best params saved: {best_json}")
    log(f"[Stage1] Search results saved: {results_csv}")
    return best["params"]

def stage1_train_best(
    lf_all: pl.LazyFrame,
    test_begin: datetime,
    artifacts_dir: str,
    prefix: str,
    best_params: dict,
    base_user_col: str,
    base_item_col: str,
    created_col: str,
    qty_col: str,
    price_col: str,
    num_threads: int,
    quick: bool,
    quick_max_rows: int,
) -> None:
    """
    Retrain Stage1 with best params and save artifacts in the same format as train_stage1_implicit_itemitem.py.
    """
    from scipy.sparse import save_npz

    ensure_dir(artifacts_dir)

    ref_dt = test_begin - timedelta(days=1)
    lf = lf_all.with_columns(pl.col(created_col).cast(pl.Datetime, strict=False).alias(created_col))
    lf = lf.filter(pl.col(created_col) <= pl.lit(ref_dt, dtype=pl.Datetime))

    if quick:
        lf = lf.limit(quick_max_rows)

    len_hist = int(best_params["len_hist"])
    len_val = int(best_params["len_val"])

    lf_train, lf_valid, train_start, valid_start, ref_dt_pol = split_train_valid_by_days(
        lf, created_col=created_col, len_hist=len_hist, len_val=len_val
    )

    user_items, users, items, _, _ = build_user_items_csr(
        lf_train,
        user_col=base_user_col,
        item_col=base_item_col,
        qty_col=qty_col,
        price_col=price_col,
        weight_type=str(best_params["weight_type"]),
    )

    model = Stage1ImplicitItemItem(
        N_cand=int(best_params["N_cand"]),
        N_trend=int(best_params["N_trend"]),
        num_threads=num_threads,
        K_model=int(best_params["K_model"]),
        n_iter=1,
        filter_already_liked_items=(not bool(best_params.get("allow_repeat", False))),
    )
    model.fit_from_csr(user_items, users, items)

    K_eval = int(best_params.get("K_eval", 2 * int(best_params["N_cand"]) + int(best_params["N_trend"])))
    m = model.eval_recall_hit(
        lf_valid,
        user_col=base_user_col,
        item_col=base_item_col,
        filter_train_history=True,
        k=K_eval,
    )

    meta_path = os.path.join(artifacts_dir, f"{prefix}_meta.npz")
    user_items_path = os.path.join(artifacts_dir, f"{prefix}_user_items.npz")
    tfidf_path = os.path.join(artifacts_dir, f"{prefix}_tfidf.npz")
    cosine_path = os.path.join(artifacts_dir, f"{prefix}_cosine.npz")
    metrics_path = os.path.join(artifacts_dir, f"{prefix}_metrics.json")

    save_npz(user_items_path, user_items)

    # save meta (compatible with load_stage1_from_artifacts)
    np.savez_compressed(
        meta_path,
        user_ids=np.array(users, dtype=object),
        item_ids=np.array(items, dtype=object),
        trending_item_idx=model.trending_item_idx_.astype(np.int32),
        config=json.dumps(
            {
                "len_hist": len_hist,
                "len_val": len_val,
                "N_trend": int(best_params["N_trend"]),
                "N_cand": int(best_params["N_cand"]),
                "n_iter": 1,
                "weight_type": str(best_params["weight_type"]),
                "num_threads": num_threads,
                "K_model": int(best_params["K_model"]),
                "allow_repeat": bool(best_params.get("allow_repeat", False)),
                "user_col": base_user_col,
                "item_col": base_item_col,
                "created_col": created_col,
                "qty_col": qty_col,
                "price_col": price_col,
            },
            ensure_ascii=False
        ),
        train_start=str(train_start),
        valid_start=str(valid_start),
        ref_dt=str(ref_dt_pol),
    )

    model.model_tfidf_.save(tfidf_path)
    model.model_cosine_.save(cosine_path)

    save_json(
        {
            "recall": m.recall,
            "hit": m.hit,
            "n_users_eval": m.n_users_eval,
            "train_start": str(train_start),
            "valid_start": str(valid_start),
            "ref_dt": str(ref_dt_pol),
        },
        metrics_path
    )

    log("[Stage1] === Saved best Stage1 artifacts ===")
    log(f"  meta:       {meta_path}")
    log(f"  user_items:  {user_items_path}")
    log(f"  tfidf:       {tfidf_path}")
    log(f"  cosine:      {cosine_path}")
    log(f"  metrics:     {metrics_path}")
    log(f"  Recall@K={K_eval}: {m.recall:.6f} | Hit@K={K_eval}: {m.hit:.6f}")


# -------------------------
# Stage 2: Build feature label
# -------------------------

def stage2_build_feature_label(cfg: dict, rebuild: bool) -> str:
    paths = cfg["paths"]
    params = cfg["params"]
    time_cfg = cfg["time"]

    feature_label_path = paths["feature_label_path"]
    out_dir = os.path.dirname(feature_label_path) if feature_label_path.endswith(".parquet") else feature_label_path
    ensure_dir(out_dir)

    if (not rebuild) and os.path.exists(feature_label_path):
        log(f"[Stage2-FL] Feature-label exists, skip build: {feature_label_path}")
        return feature_label_path

    test_begin = parse_dt(time_cfg["test_begin"])
    ref_dt = test_begin - timedelta(days=1)  # 2024-12-31 for Jan/2025 test
    ref_iso = to_iso(ref_dt)

    # Derive data_dir from transaction glob: "./preprocessed-dataset/...."
    trx_glob = paths["transactions_path_glob"]
    data_dir = os.path.dirname(trx_glob) if os.path.dirname(trx_glob) else "./preprocessed-dataset"

    # Build Stage2Params for stage2_feature_label.py
    p = s2fl.Stage2Params(
        data_dir=data_dir,
        new_feature_dir=paths["extra_feature_dir"],
        len_hist=int(params["len_hist"]),
        len_recent=int(params["len_recent"]),
        ref_datetime=ref_iso,
        out_dir=out_dir,
        save_intermediate=True,
        compute_stats=False,

        price_segment_path=os.path.join(paths["extra_feature_dir"], "price_segment.parquet"),
        customer_behavior_path=os.path.join(paths["extra_feature_dir"], "customer_behavior.parquet"),
        customer_luxury_path=os.path.join(paths["extra_feature_dir"], "customer_luxury.parquet"),
        customer_age_features_path=os.path.join(paths["extra_feature_dir"], "customer_age_features.parquet"),
        brand_segment_path=os.path.join(paths["extra_feature_dir"], "brand_segment.parquet"),
        # top10_by_cat_path=os.path.join(paths["extra_feature_dir"], "top10_by_cat.parquet"),
        top10_by_cat_month_path=os.path.join(paths["extra_feature_dir"], "top10_by_cat_month.parquet"),

        top10_month_lag=int(time_cfg.get("trend_month_lag", 2)),
    )

    # Print computed windows (for transparency)
    end_recent = parse_dt(ref_iso)
    begin_recent = end_recent - timedelta(days=int(params["len_recent"]) - 1)
    end_hist = begin_recent - timedelta(days=1)
    begin_hist = end_hist - timedelta(days=int(params["len_hist"]) - 1)
    log(f"[Stage2-FL] ref_datetime={ref_iso}")
    log(f"[Stage2-FL] HIST  : {begin_hist.date()} -> {end_hist.date()}  (len_hist={params['len_hist']}d)")
    log(f"[Stage2-FL] RECENT: {begin_recent.date()} -> {end_recent.date()} (len_recent={params['len_recent']}d)")

    params_used_path = os.path.join(out_dir, "stage2_feature_label_params_used.json")
    save_json(asdict(p), params_used_path)
    log(f"[Stage2-FL] Saved params used: {params_used_path}")

    final_path = s2fl.build_feature_label_pipeline(p)
    if final_path != feature_label_path:
        # keep config path consistent by copying/renaming
        try:
            os.replace(final_path, feature_label_path)
            log(f"[Stage2-FL] Renamed {final_path} -> {feature_label_path}")
            final_path = feature_label_path
        except Exception:
            log(f"[Stage2-FL] WARN: cannot rename {final_path} to {feature_label_path}, keep original.")
    return final_path


# -------------------------
# Stage 2: Train ranker
# -------------------------

def stage2_train_ranker(cfg: dict, feature_label_path: str, quick: bool, quick_max_rows: int) -> Tuple[Any, List[str], List[str]]:
    paths = cfg["paths"]
    params = cfg["params"]
    s2 = cfg.get("stage2_model", {})
    s2train = cfg.get("stage2_training", {})

    out_dir = paths["out_dir"]
    ensure_dir(out_dir)

    # load feature label
    log(f"[Stage2] Loading feature-label: {feature_label_path}")
    lf = pl.scan_parquet(feature_label_path)

    if quick:
        log(f"[Stage2][QUICK] Limiting feature-label rows to {quick_max_rows:,}")
        lf = lf.limit(quick_max_rows)

    df = lf.collect(streaming=True)

    # normalize id types
    df = df.with_columns([
        pl.col("customer_id").cast(pl.Utf8).alias("customer_id"),
        pl.col("item_id").cast(pl.Utf8).alias("item_id"),
        pl.col("Y").cast(pl.Int8).alias("Y"),
    ])

    # Determine categorical columns (Utf8) excluding ids
    cat_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt == pl.Utf8 and c not in ("customer_id", "item_id")]
    num_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt != pl.Utf8 and c not in ("Y",)]

    # Fill missing
    if cat_cols:
        df = df.with_columns([pl.col(c).fill_null("__MISSING__") for c in cat_cols])
    if num_cols:
        df = df.with_columns([pl.col(c).fill_null(0) for c in num_cols])

    # Convert to pandas
    pdf = df.to_pandas()
    pdf["Y"] = pdf["Y"].astype(int)
    for c in cat_cols:
        pdf[c] = pdf[c].astype("category")

    N_neg = int(params.get("N_neg", 10))
    if N_neg > 0:
        log(f"[Stage2] Negative sampling per user: N_neg={N_neg} (keep all positives)")
        parts = []
        for u, g in tqdm(pdf.groupby("customer_id"), desc="Neg-sampling by user"):
            pos = g[g["Y"] == 1]
            neg = g[g["Y"] == 0]
            if len(neg) > N_neg:
                neg = neg.sample(n=N_neg, random_state=int(s2train.get("random_state", 42)))
            parts.append(pd.concat([pos, neg], axis=0, ignore_index=True))
        pdf = pd.concat(parts, axis=0, ignore_index=True)

    # Filter groups with at least 2 items and at least 1 positive (ranker stability)
    grp = pdf.groupby("customer_id")["Y"].agg(["size", "sum"]).reset_index()
    keep_users = set(grp[(grp["size"] >= 2) & (grp["sum"] >= 1)]["customer_id"].tolist())
    before = len(pdf)
    pdf = pdf[pdf["customer_id"].isin(keep_users)].reset_index(drop=True)
    log(f"[Stage2] Keep users for ranking: {len(keep_users):,} | rows: {before:,} -> {len(pdf):,}")

    feature_cols = [c for c in pdf.columns if c not in ("customer_id", "item_id", "Y")]

    task = str(cfg.get("stage2_training", {}).get("task", "rank")).lower()
    if task not in ("rank", "binary"):
        task = "rank"

    # Train/valid split by user for early stopping (optional)
    valid_frac = float(s2train.get("valid_user_frac", 0.0))
    rstate = int(s2train.get("random_state", 42))

    users = pdf["customer_id"].unique().tolist()
    rng = np.random.default_rng(rstate)
    rng.shuffle(users)
    n_valid = int(len(users) * valid_frac)
    valid_users = set(users[:n_valid])
    train_mask = ~pdf["customer_id"].isin(valid_users)
    valid_mask = ~train_mask

    pdf_train = pdf[train_mask].reset_index(drop=True)
    pdf_valid = pdf[valid_mask].reset_index(drop=True) if n_valid > 0 else None

    log(f"[Stage2] Train users: {pdf_train['customer_id'].nunique():,} | Valid users: {0 if pdf_valid is None else pdf_valid['customer_id'].nunique():,}")

    # Build group arrays
    def group_sizes(df_: pd.DataFrame) -> np.ndarray:
        return df_.groupby("customer_id").size().to_numpy(dtype=np.int32)

    group_train = group_sizes(pdf_train)
    group_valid = group_sizes(pdf_valid) if pdf_valid is not None and len(pdf_valid) else None

    import lightgbm as lgb

    if task == "rank":
        log("[Stage2] Training LightGBM RANKER (objective=lambdarank)")
        model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            eval_at=[int(cfg["params"].get("Topk", 10))],
            random_state=int(s2.get("random_state", 42)),
            num_threads=int(s2.get("num_threads", -1)),
            n_estimators=int(s2.get("n_estimators", 300)),
            learning_rate=float(s2.get("learning_rate", 0.05)),
            num_leaves=int(s2.get("num_leaves", 63)),
            min_data_in_leaf=int(s2.get("min_data_in_leaf", 50)),
            subsample=float(s2.get("subsample", 0.8)),
            colsample_bytree=float(s2.get("colsample_bytree", 0.8)),
        )

        X_tr = pdf_train[feature_cols]
        y_tr = pdf_train["Y"].values
        fit_kwargs = dict(
            group=group_train,
            categorical_feature=cat_cols if cat_cols else "auto",
        )
        if pdf_valid is not None and len(pdf_valid):
            X_va = pdf_valid[feature_cols]
            y_va = pdf_valid["Y"].values
            fit_kwargs.update(dict(
                eval_set=[(X_va, y_va)],
                eval_group=[group_valid],
            ))
        model.fit(X_tr, y_tr, **fit_kwargs)

        # save model
        model_path = os.path.join(out_dir, "lgb_stage2_ranking.txt")
        model.booster_.save_model(model_path)
        log(f"[Stage2] Saved ranker model: {model_path}")

    else:
        log("[Stage2] Training LightGBM CLASSIFIER (objective=binary) [ranking disabled]")
        model = lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            random_state=int(s2.get("random_state", 42)),
            num_threads=int(s2.get("num_threads", -1)),
            n_estimators=int(s2.get("n_estimators", 300)),
            learning_rate=float(s2.get("learning_rate", 0.05)),
            num_leaves=int(s2.get("num_leaves", 63)),
            min_data_in_leaf=int(s2.get("min_data_in_leaf", 50)),
            subsample=float(s2.get("subsample", 0.8)),
            colsample_bytree=float(s2.get("colsample_bytree", 0.8)),
        )
        X_tr = pdf_train[feature_cols]
        y_tr = pdf_train["Y"].values
        model.fit(X_tr, y_tr, categorical_feature=cat_cols if cat_cols else "auto")
        model_path = os.path.join(out_dir, "lgb_stage2_binary.txt")
        model.booster_.save_model(model_path)
        log(f"[Stage2] Saved classifier model: {model_path}")

    save_json({"feature_cols": feature_cols, "cat_cols": cat_cols, "task": task}, os.path.join(out_dir, "stage2_feature_cols.json"))
    return model, feature_cols, cat_cols


# -------------------------
# Stage 2: Score & Evaluate on GT
# -------------------------

def build_hist_dict(transactions_glob: str, cutoff_dt: datetime) -> Dict[str, List[str]]:
    lf = scan_parquet_glob(transactions_glob).with_columns([
        pl.col("created_date").cast(pl.Datetime, strict=False).alias("created_date"),
        pl.col("customer_id").cast(pl.Utf8).alias("customer_id"),
        pl.col("item_id").cast(pl.Utf8).alias("item_id"),
    ]).filter(pl.col("created_date") < pl.lit(cutoff_dt, dtype=pl.Datetime))

    df = lf.select(["customer_id", "item_id"]).unique().collect(streaming=True)
    d = df.group_by("customer_id").agg(pl.col("item_id").alias("items")).to_dict(as_series=False)
    out = {}
    for u, items in zip(d["customer_id"], d["items"]):
        out[u] = [str(x) for x in items]
    return out

def build_features_for_pairs(
    transactions_lf: pl.LazyFrame,
    items_lf: pl.LazyFrame,
    pairs_df: pl.DataFrame,
    begin_hist: datetime,
    end_hist: datetime,
    extra_feature_dir: str,
    trend_month_key: str,
) -> pl.DataFrame:
    """
    Build feature set for arbitrary (customer_id,item_id) pairs for inference.
    This follows the same logic as stage2_feature_label.py (no segment_name).
    """
    # canonical item attrs (align to stage2_feature_label)
    cols = items_lf.columns
    def col_or(name: str, alt: str, default="__UNK__"):
        if name in cols:
            return pl.col(name)
        if alt in cols:
            return pl.col(alt)
        return pl.lit(default)

    item_attrs = items_lf.select([
        pl.col("item_id").cast(pl.Utf8).alias("item_id"),
        col_or("brand_final", "brand").cast(pl.Utf8).alias("brand_final"),
        col_or("age_bucket_final", "age_group_final").cast(pl.Utf8).alias("age_bucket_final"),
        col_or("category", "category_l2").cast(pl.Utf8).alias("category"),
        (pl.col("category_l1") if "category_l1" in cols else pl.lit("__UNK__")).cast(pl.Utf8).alias("category_l1"),
        col_or("target_user_group_final", "gender_target_final").cast(pl.Utf8).alias("target_user_group_final"),
    ])

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

    hist_enriched = hist.join(item_attrs, on="item_id", how="left")

    # aggregates
    brand_counts = hist_enriched.group_by(["customer_id", "brand_final"]).agg(pl.len().alias("brand_counts"))
    age_counts = hist_enriched.group_by(["customer_id", "age_bucket_final"]).agg(pl.len().alias("age_counts"))
    category_counts = hist_enriched.group_by(["customer_id", "category"]).agg(pl.len().alias("category_counts"))
    tug_counts = hist_enriched.group_by(["customer_id", "target_user_group_final"]).agg(pl.len().alias("target_user_group_counts"))
    last_cat = hist_enriched.group_by(["customer_id", "category"]).agg(pl.col("created_date").max().alias("last_purchase_date"))

    features = (
        pairs_df.lazy()
        .with_columns([pl.col("customer_id").cast(pl.Utf8), pl.col("item_id").cast(pl.Utf8)])
        .join(item_attrs, on="item_id", how="left")
        .join(brand_counts, on=["customer_id", "brand_final"], how="left")
        .join(age_counts, on=["customer_id", "age_bucket_final"], how="left")
        .join(category_counts, on=["customer_id", "category"], how="left")
        .join(tug_counts, on=["customer_id", "target_user_group_final"], how="left")
        .join(last_cat, on=["customer_id", "category"], how="left")
        .with_columns([
            pl.col("brand_counts").fill_null(0).cast(pl.Int32),
            pl.col("age_counts").fill_null(0).cast(pl.Int32),
            pl.col("category_counts").fill_null(0).cast(pl.Int32),
            pl.col("target_user_group_counts").fill_null(0).cast(pl.Int32),
            (
                (
                    pl.lit(end_hist, dtype=pl.Datetime)
                    - pl.col("last_purchase_date").cast(pl.Datetime, strict=False)
                )
                .dt.total_days()
                .cast(pl.Int32)
            ).alias("time_since_last_purchase_in_B_category"),
        ])
        .with_columns(pl.col("time_since_last_purchase_in_B_category").fill_null(9999))
    )

    # Extra features (optional joins)
    def try_scan(p):
        try:
            return pl.scan_parquet(p)
        except Exception:
            return None

    ps = try_scan(os.path.join(extra_feature_dir, "price_segment.parquet"))
    if ps is not None and ("item_id" in ps.columns) and ("price_segment" in ps.columns):
        features = features.join(
            ps.select([pl.col("item_id").cast(pl.Utf8), pl.col("price_segment").cast(pl.Utf8)]),
            on="item_id", how="left"
        )

    cb = try_scan(os.path.join(extra_feature_dir, "customer_behavior.parquet"))
    if cb is not None and ("customer_id" in cb.columns) and ("buy_segment" in cb.columns):
        features = features.join(
            cb.select([pl.col("customer_id").cast(pl.Utf8), pl.col("buy_segment").cast(pl.Utf8)]),
            on="customer_id", how="left"
        )

    lux = try_scan(os.path.join(extra_feature_dir, "customer_luxury.parquet"))
    if lux is not None and ("customer_id" in lux.columns):
        lux_col = "luxury_level" if "luxury_level" in lux.columns else ("customer_luxury" if "customer_luxury" in lux.columns else None)
        if lux_col is not None:
            features = features.join(
                lux.select([pl.col("customer_id").cast(pl.Utf8), pl.col(lux_col).cast(pl.Utf8).alias("luxury_level")]),
                on="customer_id", how="left"
            )

    agef = try_scan(os.path.join(extra_feature_dir, "customer_age_features.parquet"))
    if agef is not None and ("customer_id" in agef.columns) and ("age_final" in agef.columns):
        features = features.join(
            agef.select([pl.col("customer_id").cast(pl.Utf8), pl.col("age_final").cast(pl.Int32, strict=False)]),
            on="customer_id", how="left"
        )

    bs = try_scan(os.path.join(extra_feature_dir, "brand_segment.parquet"))
    if bs is not None and ("customer_id" in bs.columns) and ("brand_segment" in bs.columns):
        features = features.join(
            bs.select([pl.col("customer_id").cast(pl.Utf8), pl.col("brand_segment").cast(pl.Utf8)]),
            on="customer_id", how="left"
        )

    # top10 = try_scan(os.path.join(extra_feature_dir, "top10_by_cat.parquet"))
    # if top10 is not None and ("item_id" in top10.columns):
    #     if "top10_by_cat" in top10.columns:
    #         features = features.join(
    #             top10.select([pl.col("item_id").cast(pl.Utf8), pl.col("top10_by_cat").cast(pl.Float32, strict=False)]).unique(),
    #             on="item_id", how="left"
    #         ).with_columns(pl.col("top10_by_cat").fill_null(0.0))
    #     elif "rank" in top10.columns:
    #         features = features.join(
    #             top10.select([pl.col("item_id").cast(pl.Utf8), pl.col("rank").cast(pl.Int32, strict=False)]).unique(),
    #             on="item_id", how="left"
    #         ).rename({"rank": "rank_top10_by_cat"}).with_columns(pl.col("rank_top10_by_cat").fill_null(9999))

    top10m = try_scan(os.path.join(extra_feature_dir, "top10_by_cat_month.parquet"))
    if top10m is not None and ("item_id" in top10m.columns) and ("month" in top10m.columns):
        # Normalize month to key string
        m_schema = top10m.collect_schema()
        mdt = m_schema.get("month")
        t = top10m
        if mdt == pl.Date:
            t = t.with_columns(pl.col("month").dt.strftime("%Y-%m").alias("month_key"))
        elif mdt == pl.Datetime:
            t = t.with_columns(pl.col("month").dt.strftime("%Y-%m").alias("month_key"))
        else:
            t = t.with_columns(pl.col("month").cast(pl.Utf8).str.slice(0, 7).alias("month_key"))

        # join by (item_id, month_key) and optionally category_l1
        if "category_l1" in t.columns:
            features = (
                features
                .with_columns(pl.lit(trend_month_key).cast(pl.Utf8).alias("trend_month_key"))
                .join(
                    t.select([
                        pl.col("item_id").cast(pl.Utf8),
                        pl.col("category_l1").cast(pl.Utf8),
                        pl.col("month_key").cast(pl.Utf8),
                        (pl.col("rank").cast(pl.Int32).alias("rank_top10_by_cat_month") if "rank" in t.columns else pl.lit(None).cast(pl.Int32).alias("rank_top10_by_cat_month")),
                        (pl.col("total_sold").cast(pl.Float32).alias("total_sold_top10_by_cat_month") if "total_sold" in t.columns else pl.lit(None).cast(pl.Float32).alias("total_sold_top10_by_cat_month")),
                    ]),
                    left_on=["item_id", "category_l1", "trend_month_key"],
                    right_on=["item_id", "category_l1", "month_key"],
                    how="left"
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
                        (pl.col("rank").cast(pl.Int32).alias("rank_top10_by_cat_month") if "rank" in t.columns else pl.lit(None).cast(pl.Int32).alias("rank_top10_by_cat_month")),
                        (pl.col("total_sold").cast(pl.Float32).alias("total_sold_top10_by_cat_month") if "total_sold" in t.columns else pl.lit(None).cast(pl.Float32).alias("total_sold_top10_by_cat_month")),
                    ]),
                    left_on=["item_id", "trend_month_key"],
                    right_on=["item_id", "month_key"],
                    how="left"
                )
                .with_columns([
                    pl.col("rank_top10_by_cat_month").fill_null(9999),
                    pl.col("total_sold_top10_by_cat_month").fill_null(0.0),
                ])
                .drop(["trend_month_key"])
            )

    # fill categorical missing
    features = features.with_columns([
        (pl.col("price_segment").fill_null("__MISSING__") if "price_segment" in features.columns else pl.lit("__MISSING__").alias("price_segment")),
        (pl.col("buy_segment").fill_null("__MISSING__") if "buy_segment" in features.columns else pl.lit("__MISSING__").alias("buy_segment")),
        (pl.col("luxury_level").fill_null("__MISSING__") if "luxury_level" in features.columns else pl.lit("__MISSING__").alias("luxury_level")),
        (pl.col("brand_segment").fill_null("__MISSING__") if "brand_segment" in features.columns else pl.lit("__MISSING__").alias("brand_segment")),
    ])
    if "age_final" in features.columns:
        features = features.with_columns(pl.col("age_final").fill_null(-1).cast(pl.Int32))

    return features.collect(streaming=True)

def stage2_score_and_eval(
    cfg: dict,
    stage1_model: Any,
    stage2_model: Any,
    feature_cols: List[str],
    cat_cols: List[str],
    quick: bool,
    max_users_eval: int,
    batch_users: int,
) -> dict:
    paths = cfg["paths"]
    params = cfg["params"]
    time_cfg = cfg["time"]

    gt = load_groundtruth(paths["groundtruth_path"])
    users = list(gt.keys())
    if quick and max_users_eval > 0:
        users = users[:max_users_eval]
        gt = {u: gt[u] for u in users}
        log(f"[Eval][QUICK] Evaluate only first {len(users):,} users from groundtruth.")

    test_begin = parse_dt(time_cfg["test_begin"])
    # For inference on Jan/2025: hist window is the 120 days immediately before test_begin
    begin_hist = test_begin - timedelta(days=int(params["len_hist"]))
    end_hist = test_begin - timedelta(days=1)

    # trend month key based on test month (NOT end_hist month)
    lag = int(time_cfg.get("trend_month_lag", 2))
    trend_key = shift_month_key(test_begin, lag)

    log(f"[Eval] Feature window HIST: {begin_hist.date()} -> {end_hist.date()}")
    log(f"[Eval] trend_month_key (test_month - lag): {trend_key} (lag={lag})")

    # hist dict for filtering
    hist = build_hist_dict(paths["transactions_path_glob"], cutoff_dt=test_begin)

    # lazyframes for feature building
    tx_lf = scan_parquet_glob(paths["transactions_path_glob"])
    items_lf = scan_parquet_glob(paths["items_path_glob"])

    K = int(params.get("Topk", 10))
    N_total_cand = int(params.get("N_trend", 0)) + 2 * int(params.get("N_cand", 100))

    pred: Dict[str, List[str]] = {}

    for i in tqdm(range(0, len(users), batch_users), desc="Scoring users (batch)"):
        batch = users[i:i + batch_users]
        pairs = []
        for u in batch:
            cand = stage1_model.recommend_for_user_id(u, top_k=N_total_cand)
            # stage1 already filters repeats depending on config; no extra filter here
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
            extra_feature_dir=paths["extra_feature_dir"],
            trend_month_key=trend_key,
        )
        pdf = feat.to_pandas()

        # ensure columns
        for c in feature_cols:
            if c not in pdf.columns:
                pdf[c] = 0
        for c in cat_cols:
            if c not in pdf.columns:
                pdf[c] = "__MISSING__"
            pdf[c] = pdf[c].astype("category")

        X = pdf[feature_cols]

        # ranker/classifier compatibility
        if hasattr(stage2_model, "predict_proba"):
            scores = stage2_model.predict_proba(X)[:, 1]
        else:
            scores = stage2_model.predict(X)

        pdf["score"] = scores

        # rank per user
        for u, g in pdf.groupby("customer_id"):
            g2 = g.sort_values("score", ascending=False)
            pred[u] = g2["item_id"].astype(str).tolist()

    # metrics
    p_unf, cold_unf = precision_at_k(pred, gt, hist, filter_bought_items=False, K=K)
    p_flt, cold_flt = precision_at_k(pred, gt, hist, filter_bought_items=True, K=K)
    n_unf, cold2_unf = ndcg_at_k(pred, gt, hist, filter_bought_items=False, K=K)
    n_flt, cold2_flt = ndcg_at_k(pred, gt, hist, filter_bought_items=True, K=K)

    cold_users = sorted(set(cold_unf) | set(cold_flt) | set(cold2_unf) | set(cold2_flt))

    report = {
        "K": K,
        "precision_unfiltered": p_unf,
        "precision_filtered": p_flt,
        "ndcg_unfiltered": n_unf,
        "ndcg_filtered": n_flt,
        "n_users_eval": len(users),
        "n_users_pred": len(pred),
        "n_cold_start": len(cold_users),
        "trend_month_key": trend_key,
        "begin_hist": to_iso(begin_hist),
        "end_hist": to_iso(end_hist),
        "N_total_cand": N_total_cand,
    }

    # save artifacts
    out_dir = paths["out_dir"]
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "pred_stage2.pkl"), "wb") as f:
        pickle.dump(pred, f)
    with open(os.path.join(out_dir, "cold_start_users.pkl"), "wb") as f:
        pickle.dump(cold_users, f)
    save_json(report, os.path.join(out_dir, "stage2_eval_report.json"))

    log("[Eval] ===== Report =====")
    log(f"Precision@{K} (unfiltered): {p_unf:.6f}")
    log(f"Precision@{K} (filtered)  : {p_flt:.6f}")
    log(f"NDCG@{K}      (unfiltered): {n_unf:.6f}")
    log(f"NDCG@{K}      (filtered)  : {n_flt:.6f}")
    log(f"Cold-start users: {len(cold_users):,} / {len(users):,}")
    return report


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--quick", action="store_true", help="Smoke test mode: fewer trials, smaller samples.")
    ap.add_argument("--rebuild_feature_label", action="store_true")
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--stage1_trials", type=int, default=None, help="Override stage1_search.n_trials")
    ap.add_argument("--quick_max_rows", type=int, default=300_000, help="Row cap used in quick mode for Stage1/Stage2")
    ap.add_argument("--max_users_eval", type=int, default=2000, help="User cap used in quick mode for evaluation")
    ap.add_argument("--batch_users", type=int, default=10000, help="Batch size for scoring users")
    ap.add_argument("--num_threads", type=int, default=None, help="Override implicit/lightgbm threads")
    args = ap.parse_args()

    cfg = load_json(args.config)

    # Print explicit notice for unclear-spec params
    unclear = ["min_trans_items", "session_window", "min_coo", "filter_fashion"]
    for k in unclear:
        if k in cfg.get("params", {}):
            log(f"[NOTICE] Param `{k}` is present but not applied in this script (không biết spec chính xác).")

    # Stage1
    paths = cfg["paths"]
    ensure_dir(paths["stage1_artifacts_dir"])
    ensure_dir(paths["out_dir"])

    test_begin = parse_dt(cfg["time"]["test_begin"])
    num_threads = int(cfg["stage2_model"].get("num_threads", -1))
    if args.num_threads is not None:
        num_threads = int(args.num_threads)

    # scan transactions (lazy)
    lf_all = scan_parquet_glob(paths["transactions_path_glob"])

    # ensure created_col for stage1
    created_col = "created_datetime"
    schema = lf_all.collect_schema()
    if "created_datetime" in schema:
        created_col = "created_datetime"
    elif "created_date" in schema:
        created_col = "created_date"
    else:
        raise ValueError(f"Transactions parquet missing `created_datetime` and `created_date`. Columns: {list(schema.keys())}")

    # Stage1 search
    s1s = cfg.get("stage1_search", {})
    space = s1s.get("search_space", {})
    n_trials = int(s1s.get("n_trials", 90))
    if args.stage1_trials is not None:
        n_trials = int(args.stage1_trials)
    if args.quick:
        n_trials = min(n_trials, 5)

    best_params = stage1_random_search(
        lf_all=lf_all,
        test_begin=test_begin,
        artifacts_dir=paths["stage1_artifacts_dir"],
        prefix=paths["stage1_prefix"],
        base_user_col="customer_id",
        base_item_col="item_id",
        created_col=created_col,
        qty_col="quantity",
        price_col="price",
        num_threads=max(0, num_threads),
        n_trials=n_trials,
        random_state=int(s1s.get("random_state", 42)),
        space=space,
        quick=args.quick,
        quick_max_rows=args.quick_max_rows,
    )

    stage1_train_best(
        lf_all=lf_all,
        test_begin=test_begin,
        artifacts_dir=paths["stage1_artifacts_dir"],
        prefix=paths["stage1_prefix"],
        best_params=best_params,
        base_user_col="customer_id",
        base_item_col="item_id",
        created_col=created_col,
        qty_col="quantity",
        price_col="price",
        num_threads=max(0, num_threads),
        quick=args.quick,
        quick_max_rows=args.quick_max_rows,
    )

    # Load stage1 model back for stage2 scoring
    from stage1_implicit_itemitem import load_stage1_from_artifacts

    prefix = paths["stage1_prefix"]
    art_dir = paths["stage1_artifacts_dir"]
    stage1 = load_stage1_from_artifacts(
        meta_npz_path=os.path.join(art_dir, f"{prefix}_meta.npz"),
        user_items_npz_path=os.path.join(art_dir, f"{prefix}_user_items.npz"),
        tfidf_npz_path=os.path.join(art_dir, f"{prefix}_tfidf.npz"),
        cosine_npz_path=os.path.join(art_dir, f"{prefix}_cosine.npz"),
    )
    log("[Stage1] Loaded best Stage1 artifacts OK.")

    # Stage2 feature label
    fl_path = stage2_build_feature_label(cfg, rebuild=args.rebuild_feature_label)

    # Stage2 training
    stage2_model, feature_cols, cat_cols = stage2_train_ranker(
        cfg=cfg,
        feature_label_path=fl_path,
        quick=args.quick,
        quick_max_rows=args.quick_max_rows,
    )
    log("[Stage2] Training done.")

    # Stage2 evaluation
    if not args.skip_eval:
        stage2_score_and_eval(
            cfg=cfg,
            stage1_model=stage1,
            stage2_model=stage2_model,
            feature_cols=feature_cols,
            cat_cols=cat_cols,
            quick=args.quick,
            max_users_eval=args.max_users_eval,
            batch_users=args.batch_users,
        )
    else:
        log("[Eval] Skipped by --skip_eval.")

    log("DONE: end-to-end Stage1 -> Stage2 pipeline complete.")


if __name__ == "__main__":
    # suppress polars PerformanceWarning noise
    import warnings
    try:
        warnings.filterwarnings("ignore", category=pl.exceptions.PerformanceWarning)
    except Exception:
        pass
    main()
