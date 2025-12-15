
"""
train_stage1_implicit_itemitem.py

Train Stage-1 using implicit TFIDFRecommender + CosineRecommender (item-item).
No sklearn NearestNeighbors is used.

Expected data layout (same as your existing pipeline):
  data_dir contains parquet(s) matching: "purchase_history_daily_chunk*.parquet"
  and has at least: customer_id, item_id, created_datetime (or a castable datetime col).

Outputs (artifacts_dir):
  - {prefix}_meta.npz         (np.savez with mappings + trending + config + time window)
  - {prefix}_user_items.npz   (scipy.sparse CSR matrix)
  - {prefix}_tfidf.npz        (implicit model)
  - {prefix}_cosine.npz       (implicit model)
  - {prefix}_metrics.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import polars as pl
from scipy.sparse import save_npz
from tqdm.auto import tqdm

from stage1_implicit_itemitem import (
    split_train_valid_by_days,
    build_user_items_csr,
    Stage1ImplicitItemItem,
)


def _glob_parquets(data_dir: str, prefix: str) -> list[str]:
    import glob
    pats = [
        os.path.join(data_dir, f"{prefix}*.parquet"),
        os.path.join(data_dir, f"**/{prefix}*.parquet"),
    ]
    files: list[str] = []
    for p in pats:
        files.extend(glob.glob(p, recursive=True))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No parquet found for prefix={prefix} under {data_dir}")
    return files


def scan_transaction(data_dir: str) -> pl.LazyFrame:
    files = _glob_parquets(data_dir, "sale_pers.purchase_history_daily_chunk")
    return pl.scan_parquet(files)


def save_meta_npz(
    path: str,
    user_ids: list[str],
    item_ids: list[str],
    trending_item_idx: np.ndarray,
    config: dict,
    train_start: str,
    valid_start: str,
    ref_dt: str,
):
    np.savez_compressed(
        path,
        user_ids=np.array(user_ids, dtype=object),
        item_ids=np.array(item_ids, dtype=object),
        trending_item_idx=trending_item_idx.astype(np.int32),
        config=json.dumps(config, ensure_ascii=False),
        train_start=train_start,
        valid_start=valid_start,
        ref_dt=ref_dt,
    )


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data_dir", type=str, default="./preprocessed-dataset")
    p.add_argument("--artifacts_dir", type=str, default="./artifacts")
    p.add_argument("--prefix", type=str, default="stage1_implicit_itemitem")

    # key configs requested
    p.add_argument("--len_hist", type=int, default=180)
    p.add_argument("--len_val", type=int, default=30)
    p.add_argument("--N_trend", type=int, default=100)
    p.add_argument("--N_cand", type=int, default=300)
    p.add_argument("--n_iter", type=int, default=1)  # ignored by TFIDF/Cosine models

    # weight
    p.add_argument(
        "--weight_type",
        type=str,
        default="log_count",
        choices=["binary", "count", "log_count", "log_qty", "log_spent"],
    )

    # implicit
    p.add_argument("--num_threads", type=int, default=0)
    p.add_argument(
        "--K_model",
        type=int,
        default=0,
        help="implicit K (neighbors in item-item similarity). 0 => auto from N_cand",
    )
    p.add_argument(
        "--allow_repeat",
        action="store_true",
        help="If set, do NOT filter already-liked items in recommend()",
    )

    # columns
    p.add_argument("--user_col", type=str, default="customer_id")
    p.add_argument("--item_col", type=str, default="item_id")
    # Your raw tables typically use created_date; override if needed.
    p.add_argument("--created_col", type=str, default="created_date")
    p.add_argument("--qty_col", type=str, default="quantity")
    p.add_argument("--price_col", type=str, default="price")

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.artifacts_dir, exist_ok=True)

    # ---------------------------
    # Print config for traceability
    # ---------------------------
    cfg = {
        "data_dir": args.data_dir,
        "artifacts_dir": args.artifacts_dir,
        "prefix": args.prefix,
        "len_hist": args.len_hist,
        "len_val": args.len_val,
        "N_trend": args.N_trend,
        "N_cand": args.N_cand,
        "n_iter": args.n_iter,  # ignored by TFIDF/Cosine
        "weight_type": args.weight_type,
        "num_threads": args.num_threads,
        "K_model": (None if args.K_model == 0 else args.K_model),
        "allow_repeat": args.allow_repeat,
        "user_col": args.user_col,
        "item_col": args.item_col,
        "created_col": args.created_col,
        "qty_col": args.qty_col,
        "price_col": args.price_col,
    }
    print("=== Stage1 (implicit TFIDF + Cosine) - CONFIG ===")
    print(json.dumps(cfg, ensure_ascii=False, indent=2))

    # ---------------------------
    # Pipeline with visible progress
    # ---------------------------
    pipeline = tqdm(total=6, desc="Stage1 pipeline", leave=True)

    # (1) scan parquet(s)
    lf_all = scan_transaction(args.data_dir)
    # show schema early
    print("\n[1/6] Loaded LazyFrame")
    print("Columns:", lf_all.columns)
    pipeline.update(1)

    # (2) split train/valid
    print("\n[2/6] Split train/valid by days")
    lf_train, lf_valid, train_start, valid_start, ref_dt = split_train_valid_by_days(
        lf_all,
        created_col=args.created_col,
        len_hist=args.len_hist,
        len_val=args.len_val,
    )
    # materialize small counts for sanity
    n_train = lf_train.select(pl.len()).collect()[0, 0]
    n_valid = lf_valid.select(pl.len()).collect()[0, 0]
    print(f"Time windows: train=[{train_start}, {valid_start})  valid=[{valid_start}, {ref_dt}]")
    print(f"Rows: train={n_train:,}  valid={n_valid:,}")
    pipeline.update(1)

    # (3) build user-item matrix (CSR)
    print("\n[3/6] Build user-item CSR")
    user_items, users, items, _, _ = build_user_items_csr(
        lf_train,
        user_col=args.user_col,
        item_col=args.item_col,
        qty_col=args.qty_col,
        price_col=args.price_col,
        weight_type=args.weight_type,
    )
    print(f"CSR shape: users={user_items.shape[0]:,}  items={user_items.shape[1]:,}  nnz={user_items.nnz:,}")
    pipeline.update(1)

    model = Stage1ImplicitItemItem(
        N_cand=args.N_cand,
        N_trend=args.N_trend,
        num_threads=args.num_threads,
        K_model=(None if args.K_model == 0 else args.K_model),
        n_iter=args.n_iter,
        filter_already_liked_items=(not args.allow_repeat),
    )
    
    # (4) fit implicit models
    print("\n[4/6] Fit implicit models")
    print("- TFIDFRecommender.fit(show_progress=True)")
    print("- CosineRecommender.fit(show_progress=True)")
    model.fit_from_csr(user_items, users, items)
    print(f"Trending items appended: N_trend={args.N_trend}")
    pipeline.update(1)

    # (5) evaluate on valid
    print("\n[5/6] Evaluate Stage1 (Recall/Hit)")
    K_eval = 2 * args.N_cand + args.N_trend
    print(f"Eval K = {K_eval} | allow_repeat (serving) = {args.allow_repeat}")

    # FILTERED: new-item recall (exclude train history from GT AND filter already-liked in recommend)
    metrics_f = model.eval_recall_hit(
        lf_valid,
        user_col=args.user_col,
        item_col=args.item_col,
        filter_train_history=True,
        k=K_eval,
    )
    print(
        f"FILTERED (new-item)  Recall@K={metrics_f.recall:.6f}  Hit@K={metrics_f.hit:.6f}  "
        f"n_users_eval={metrics_f.n_users_eval:,}"
    )

    # UNFILTERED: overall recall (include repeats; do NOT filter already-liked in recommend)
    metrics_u = model.eval_recall_hit(
        lf_valid,
        user_col=args.user_col,
        item_col=args.item_col,
        filter_train_history=False,
        k=K_eval,
    )
    print(
        f"UNFILTERED (overall) Recall@K={metrics_u.recall:.6f}  Hit@K={metrics_u.hit:.6f}  "
        f"n_users_eval={metrics_u.n_users_eval:,}"
    )

    # Primary metric to save: keep FILTERED by default (Stage-1 candidate quality for novel items)
    metrics = metrics_f
    pipeline.update(1)

    prefix = args.prefix
    meta_path = os.path.join(args.artifacts_dir, f"{prefix}_meta.npz")
    user_items_path = os.path.join(args.artifacts_dir, f"{prefix}_user_items.npz")
    tfidf_path = os.path.join(args.artifacts_dir, f"{prefix}_tfidf.npz")
    cosine_path = os.path.join(args.artifacts_dir, f"{prefix}_cosine.npz")
    metrics_path = os.path.join(args.artifacts_dir, f"{prefix}_metrics.json")

    # (6) save artifacts
    print("\n[6/6] Save artifacts")
    save_npz(user_items_path, user_items)

    save_meta_npz(
        meta_path,
        user_ids=users,
        item_ids=items,
        trending_item_idx=model.trending_item_idx_,
        config={
            "len_hist": args.len_hist,
            "len_val": args.len_val,
            "N_trend": args.N_trend,
            "N_cand": args.N_cand,
            "n_iter": args.n_iter,
            "weight_type": args.weight_type,
            "num_threads": args.num_threads,
            "K_model": (None if args.K_model == 0 else args.K_model),
            "allow_repeat": args.allow_repeat,
            "user_col": args.user_col,
            "item_col": args.item_col,
            "created_col": args.created_col,
            "qty_col": args.qty_col,
            "price_col": args.price_col,
        },
        train_start=str(train_start),
        valid_start=str(valid_start),
        ref_dt=str(ref_dt),
    )

    model.model_tfidf_.save(tfidf_path)
    model.model_cosine_.save(cosine_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "recall": metrics.recall,
                "hit": metrics.hit,
                "n_users_eval": metrics.n_users_eval,
                "train_start": str(train_start),
                "valid_start": str(valid_start),
                "ref_dt": str(ref_dt),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=== Stage1 trained & saved ===")
    print("meta:", meta_path)
    print("user_items:", user_items_path)
    print("tfidf:", tfidf_path)
    print("cosine:", cosine_path)
    print("metrics:", metrics_path)
    print("Recall@K:", metrics.recall, "Hit@K:", metrics.hit, "n_users_eval:", metrics.n_users_eval)

    pipeline.update(1)
    pipeline.close()


if __name__ == "__main__":
    main()
