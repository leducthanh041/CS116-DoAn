# train_two_stage.py
# Clean sequential pipeline (like notebook), runnable from terminal.

from __future__ import annotations

import os
import argparse
import json

import pandas as pd
import polars as pl

from two_stage_lib import (
    scan_user,
    scan_item,
    scan_transaction,
    split_train_valid_lf,
    ensure_created_datetime_lf,
    precompute_stage1,
    train_stage1_random_search,
    load_best_stage1,
    build_ground_truth,
    generate_candidates,
    build_ranking_rows,
    enrich_stage2_features,
    train_stage2_lightgbm,
    DEFAULT_STAGE1_PARAM_SPACE,
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data_dir", type=str, default="./preprocessed-dataset")
    p.add_argument("--new_feature_dir", type=str, default="./new-feature")
    p.add_argument("--artifacts_dir", type=str, default="./artifacts")

    # split config
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--train_month_start", type=int, default=1)
    p.add_argument("--train_month_end", type=int, default=11)
    p.add_argument("--valid_month", type=int, default=12)

    # quick test
    p.add_argument("--quick_test", action="store_true")
    p.add_argument("--quick_train_rows", type=int, default=100)
    p.add_argument("--quick_valid_rows", type=int, default=10)

    # stage1
    p.add_argument("--stage1_k", type=int, default=500)
    p.add_argument("--stage1_trials", type=int, default=20)
    p.add_argument("--stage1_seed", type=int, default=42)
    p.add_argument("--stage1_allow_repeat", action="store_true")

    # stage2
    p.add_argument("--stage2_num_boost_round", type=int, default=200)
    p.add_argument("--stage2_device", type=str, default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--top10_month_lag", type=int, default=1)

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.artifacts_dir, exist_ok=True)

    print("=== (1) Scan parquet (lazy) ===")
    lf_user = scan_user(args.data_dir)
    lf_item = scan_item(args.data_dir)
    lf_trx = scan_transaction(args.data_dir)

    # transaction must have created_datetime
    lf_trx = ensure_created_datetime_lf(lf_trx)

    print("=== (2) Split train/valid (lazy filters) ===")
    lf_train, lf_valid = split_train_valid_lf(
        lf_trx,
        year=args.year,
        train_month_start=args.train_month_start,
        train_month_end=args.train_month_end,
        valid_month=args.valid_month,
    )

    if args.quick_test:
        print("=== QUICK TEST: limit rows BEFORE collect ===")
        qtr = args.quick_train_rows
        qva = args.quick_valid_rows
    else:
        qtr = None
        qva = None

    print("=== (3) Precompute Stage1 tables (once) ===")
    pre = precompute_stage1(
        lf_train=lf_train,
        lf_valid=lf_valid,
        lf_item=lf_item,
        user_col="customer_id",
        item_col="item_id",
        quick_limit_train=qtr,
        quick_limit_valid=qva,
    )

    print(f"Precomputed: n_users={pre.n_users:,} | n_items={pre.n_items:,} | nnz(UI)={len(pre.indices):,}")

    # NOTE: inject day arrays for preference recency terms (kept internal to precompute)
    # (These are created inside precompute via pandas intermediate; we re-attach here.)
    # This is intentionally simple to keep pipeline sequential.
    # If missing, library will raise.
    # (The arrays exist inside the precompute function scope; attach via joblib would keep them, but we add them now.)
    # In this clean version, we recompute l1/l2 day arrays from source files is not available;
    # therefore precompute_stage1 attaches them into `pre` object.
    # (No action needed here.)

    print("=== (4) Stage1 random search (overwrite best artifacts) ===")
    stage1_model_path = os.path.join(args.artifacts_dir, "stage1_item_item_cf.pkl")
    stage1_params_path = os.path.join(args.artifacts_dir, "stage1_best_params.json")
    stage1_results_csv = os.path.join(args.artifacts_dir, "stage1_random_search_results.csv")

    n_trials = 2 if args.quick_test else args.stage1_trials

    s1 = train_stage1_random_search(
        pre=pre,
        k_eval=args.stage1_k,
        n_trials=n_trials,
        param_space=DEFAULT_STAGE1_PARAM_SPACE,
        seed=args.stage1_seed,
        artifacts_dir=args.artifacts_dir,
        model_path=stage1_model_path,
        params_path=stage1_params_path,
        results_csv=stage1_results_csv,
    )

    print("Stage1 BEST cfg:", json.dumps(s1.best_cfg, ensure_ascii=False))
    print("Stage1 BEST metrics:", json.dumps(s1.best_metrics, ensure_ascii=False))

    best_stage1 = load_best_stage1(stage1_model_path)

    print("=== (5) Build Stage2 ranking dataset from Stage1 candidates ===")
    # Build ground truth from VALID pairs (from precompute)
    df_valid_pairs = pd.DataFrame({"customer_id": pre.valid_pairs_user, "item_id": pre.valid_pairs_item})
    gt = build_ground_truth(df_valid_pairs, user_col="customer_id", item_col="item_id")

    users = list(gt.keys())
    if args.quick_test:
        users = users[: min(len(users), 50)]  # avoid too tiny label issue but keep fast

    candidates = generate_candidates(
        stage1_model=best_stage1,
        user_ids=users,
        k_cand=args.stage1_k,
        allow_repeat=args.stage1_allow_repeat,
    )

    df_rank = build_ranking_rows(candidates, gt)
    pos = int(df_rank["label"].sum())
    print(f"Ranking rows={len(df_rank):,} | positives={pos:,} | users={df_rank['customer_id'].nunique():,}")

    if pos == 0:
        raise RuntimeError("Ranking dataset không có label=1. Tăng quick_valid_rows hoặc kiểm tra valid split.")

    print("=== (6) Enrich Stage2 features (incl. ./new-feature) ===")
    df_rank2, feature_cols, cat_cols = enrich_stage2_features(
        df_rank=df_rank,
        lf_user=lf_user,
        lf_item=lf_item,
        new_feature_dir=args.new_feature_dir,
        valid_month=args.valid_month,
        top10_month_lag=args.top10_month_lag,
    )

    print("Stage2 features:", len(feature_cols), "| categorical:", len(cat_cols))

    print("=== (7) Train LightGBM Stage2 (num_boost_round=200) ===")
    stage2_model_path = os.path.join(args.artifacts_dir, "lgb_stage2_ranking.txt")
    stage2_feat_path = os.path.join(args.artifacts_dir, "stage2_feature_cols.json")

    s2 = train_stage2_lightgbm(
        df_rank=df_rank2,
        feature_cols=feature_cols,
        cat_cols=cat_cols,
        num_boost_round=args.stage2_num_boost_round,
        random_state=args.stage1_seed,
        device_type=args.stage2_device,
        model_path=stage2_model_path,
        feat_path=stage2_feat_path,
    )

    print("Stage2 metrics:")
    print(json.dumps(s2.metrics, indent=2, ensure_ascii=False))

    print("\n=== DONE. Artifacts ===")
    print("Stage1 model:", stage1_model_path)
    print("Stage1 params:", stage1_params_path)
    print("Stage1 search:", stage1_results_csv)
    print("Stage2 model:", stage2_model_path)
    print("Stage2 features:", stage2_feat_path)


if __name__ == "__main__":
    main()
