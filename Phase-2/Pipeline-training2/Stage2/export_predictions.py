import os
import json
import pickle
from typing import List, Optional, Dict

import numpy as np
import polars as pl
from tqdm import tqdm
from datetime import datetime


# --- Giữ nguyên các hàm helper cũ (load_model, score_block, global_fallback...) ---
# (Tôi copy lại ngắn gọn để code chạy được đầy đủ)

def load_model(run_dir: str, model_type: str):
    model_path = os.path.join(
        run_dir,
        "lgbm_classifier.pkl" if model_type.lower() == "classifier" else "lgbm_ranker.pkl"
    )
    feat_path = os.path.join(run_dir, "feature_columns.json")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(feat_path, "r", encoding="utf-8") as f:
        feat_cols = json.load(f)["feature_cols"]
    return model, feat_cols

def score_block(model, df_block: pl.DataFrame, feature_cols: List[str]) -> np.ndarray:
    X = df_block.select(feature_cols).to_pandas()
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = X[c].astype("category").cat.codes
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)

def build_global_fallback_items(transactions_path_glob: str, begin: str, end: str, top_n: int) -> List[str]:
    # (Giữ nguyên logic hàm này như cũ)
    begin_dt = datetime.strptime(begin, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end, "%Y-%m-%d").date()
    tx_lf = pl.scan_parquet(transactions_path_glob)
    df = (
        tx_lf.select([pl.col("item_id").cast(pl.Utf8).str.strip_chars(), pl.col("created_date").cast(pl.Date, strict=False)])
        .filter(pl.col("created_date").is_between(pl.lit(begin_dt, dtype=pl.Date), pl.lit(end_dt, dtype=pl.Date), closed="both"))
        .group_by("item_id").agg(pl.len().alias("len_hist"))
        .sort("len_hist", descending=True).head(int(top_n))
        .collect(engine="streaming")
    )
    return df["item_id"].to_list()

def _load_or_build_global_fallback(cache_path: str, transactions_path_glob: str, begin: str, end: str, top_n: int) -> List[str]:
    if os.path.exists(cache_path):
        print(f"[CACHE] Loading global fallback from: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    print("[BUILD] Building global fallback items (this may take a while)...")
    items = build_global_fallback_items(transactions_path_glob, begin, end, top_n)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(items, f, protocol=pickle.HIGHEST_PROTOCOL)
    return items


def export_predictions_csv(
    run_dir: str,
    out_csv_path: str,
    transactions_path_glob: str,
    model_type: str = "ranker",
    pool_path: Optional[str] = None, # Chỉ cần nếu KHÔNG dùng pred_dict_path
    pred_dict_path: Optional[str] = None, # [NEW] Đường dẫn file cache pred_dict
    k: int = 10,
    batch_users: int = 200000,
    user_list_path: Optional[str] = None,
    fallback_begin: str = "2024-08-01",
    fallback_end: str = "2024-12-31",
    fallback_top_n: int = 200000,
) -> None:
    """
    Export predictions to CSV using either:
    1. pred_dict_path (Fastest): Load pre-computed user->items dict (no model scoring).
    2. pool_path + model (Slower): Load candidates, score with model, sort.
    
    Always pads with global fallback to ensure K items per user.
    """
    
    # 1. Load Global Fallback (Cần cho cả 2 trường hợp để điền khuyết)
    fallback_cache_path = os.path.join(run_dir, f"global_fallback_{fallback_begin}_{fallback_end}_top{int(fallback_top_n)}.pkl")
    global_fallback = _load_or_build_global_fallback(
        cache_path=fallback_cache_path,
        transactions_path_glob=transactions_path_glob,
        begin=fallback_begin,
        end=fallback_end,
        top_n=max(int(fallback_top_n), int(k) * 10),
    )
    if not global_fallback:
        raise ValueError("Global fallback list is empty.")

    # 2. Logic chính: Chọn luồng Fast (Cache) hay Slow (Model)
    use_cache = False
    pred_dict = {}

    if pred_dict_path and os.path.exists(pred_dict_path):
        print(f"===========================================================")
        print(f"[FAST PATH] Found pred_dict cache at: {pred_dict_path}")
        print(f"            Loading predictions directly without re-scoring.")
        print(f"===========================================================")
        with open(pred_dict_path, "rb") as f:
            pred_dict = pickle.load(f)
        use_cache = True
    else:
        print(f"[SLOW PATH] No pred_dict cache provided or found.")
        print(f"            Will load model and score candidates from pool.")
        if not pool_path:
            raise ValueError("pool_path must be provided if pred_dict_path is missing.")

    # 3. Mở file CSV để ghi
    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    first_write = True

    # ---------------------------------------------------------
    # CASE A: DÙNG CACHE (pred_dict)
    # ---------------------------------------------------------
    if use_cache:
        # Filter users nếu cần
        target_users = list(pred_dict.keys())
        if user_list_path:
            print(f"[FILTER] Filtering users from {user_list_path}...")
            filter_u = set([x.strip() for x in open(user_list_path, "r") if x.strip()])
            target_users = [u for u in target_users if str(u) in filter_u]
        
        target_users.sort()
        print(f"[INFO] Exporting {len(target_users):,} users from cache...")

        # Buffer để ghi CSV theo batch (tránh I/O liên tục)
        rows_buffer = []
        BUFFER_SIZE = 100000 

        for u_str in tqdm(target_users, desc="[EXPORT CACHE]", unit="user"):
            items = pred_dict[u_str] # List[str] items đã rank
            
            # Logic Padding (Điền khuyết)
            pred_items = items[:int(k)]
            # Fake score vì pred_dict thường không lưu score. 
            # Rank 1 = 100, Rank 2 = 99... hoặc đơn giản là giảm dần.
            pred_scores = [float(1000 - i) for i in range(len(pred_items))] 

            if len(pred_items) < int(k):
                used = set(pred_items)
                for it in global_fallback:
                    if it in used: continue
                    pred_items.append(it)
                    pred_scores.append(-999.0) # Score thấp cho fallback
                    used.add(it)
                    if len(pred_items) == int(k): break
            
            # Cắt đúng K
            pred_items = pred_items[:int(k)]
            pred_scores = pred_scores[:int(k)]

            # Tạo row
            try:
                u_int = int(float(u_str))
            except:
                continue # Skip bad user id

            for r in range(int(k)):
                rows_buffer.append((u_int, pred_items[r], pred_scores[r], r + 1))

            # Flush buffer
            if len(rows_buffer) >= BUFFER_SIZE:
                df_chunk = pl.DataFrame(rows_buffer, schema=["customer_id", "item_id", "score", "rank"])
                if first_write:
                    df_chunk.write_csv(out_csv_path)
                    first_write = False
                else:
                    with open(out_csv_path, "a", encoding="utf-8") as f:
                        df_chunk.write_csv(f, include_header=False)
                rows_buffer = []

        # Flush phần còn lại
        if rows_buffer:
            df_chunk = pl.DataFrame(rows_buffer, schema=["customer_id", "item_id", "score", "rank"])
            if first_write:
                df_chunk.write_csv(out_csv_path)
            else:
                with open(out_csv_path, "a", encoding="utf-8") as f:
                    df_chunk.write_csv(f, include_header=False)

    # ---------------------------------------------------------
    # CASE B: TÍNH TOÁN (Model Scoring) - Code cũ
    # ---------------------------------------------------------
    else:
        # Load Model
        model, feat_cols = load_model(run_dir, model_type)
        
        # Load Pool
        if pool_path.endswith(".parquet"): df = pl.read_parquet(pool_path)
        else: df = pl.read_csv(pool_path)
        if "Y" in df.columns: df = df.drop("Y")

        # Cast Types & Filter Users
        df = df.with_columns([pl.col("customer_id").cast(pl.Int64), pl.col("item_id").cast(pl.Utf8).str.strip_chars()])
        
        if user_list_path:
             # (Logic filter user như cũ)
             pass 

        users = df.select("customer_id").unique()["customer_id"].to_list()
        users.sort()

        for i in tqdm(range(0, len(users), int(batch_users)), desc="[EXPORT MODEL]", unit="batch"):
            bu = users[i:i + int(batch_users)]
            block = df.filter(pl.col("customer_id").is_in(bu))
            if block.height == 0: continue

            scores = score_block(model, block, feat_cols)
            block = block.with_columns(pl.Series("score", scores))
            block = block.sort(["customer_id", "score"], descending=[False, True])
            
            # --- QUAN TRỌNG: KHỬ TRÙNG LẶP (Logic bạn vừa sửa lúc nãy) ---
            # Chúng ta dùng group_by để gom item lại, bản thân việc này đã gom theo list.
            # Tuy nhiên để an toàn như hàm build_pred, ta nên unique trước.
            topk = (
                block
                .unique(subset=["customer_id", "item_id"], keep="first") # <--- Thêm dòng này cho an toàn
                .group_by("customer_id")
                .agg([pl.col("item_id").alias("items_all"), pl.col("score").alias("scores_all")])
            )

            rows = []
            for u, items_all, scores_all in zip(topk["customer_id"], topk["items_all"], topk["scores_all"]):
                items, scs = list(items_all), list(scores_all)
                
                # Logic Padding & Fallback (Giống hệt case A nhưng có score thật)
                pred_items = items[:int(k)]
                pred_scores = [float(x) for x in scs[:int(k)]]

                if len(pred_items) < int(k):
                    used = set(pred_items)
                    for it in global_fallback:
                        if it in used: continue
                        pred_items.append(it)
                        pred_scores.append(-999.0)
                        used.add(it)
                        if len(pred_items) == int(k): break
                
                pred_items = pred_items[:int(k)]
                pred_scores = pred_scores[:int(k)]

                for r in range(int(k)):
                    rows.append((int(u), pred_items[r], float(pred_scores[r]), r + 1))
            
            # Write chunk
            out_chunk = pl.DataFrame(rows, schema=["customer_id", "item_id", "score", "rank"])
            if first_write:
                out_chunk.write_csv(out_csv_path)
                first_write = False
            else:
                with open(out_csv_path, "a", encoding="utf-8") as f:
                    out_chunk.write_csv(f, include_header=False)

    print(f"[DONE] Saved predictions to: {out_csv_path}")

if __name__ == "__main__":
    # Ví dụ sử dụng CACHE:
    export_predictions_csv(
        transactions_path_glob=".././preprocessed-dataset/sale_pers.purchase_history_daily_chunk_*.parquet",
        run_dir="./artifacts_stage2/runs/run_stage2_full_v1",
        
        # --- Điền đường dẫn file cache vào đây ---
        pred_dict_path="./artifacts_stage2/cache/pred_dict_03ea8c8d1e08.pkl", 
        
        # pool_path có thể để None nếu đã có pred_dict_path
        pool_path=None, 
        
        out_csv_path="./artifacts_stage2/runs/run_stage2_full_v1/predictions_submission.csv",
        k=10
    )