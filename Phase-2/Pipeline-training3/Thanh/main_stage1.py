import numpy as np
import pandas as pd
from pathlib import Path
import joblib 
import json
import pickle
from tqdm.auto import tqdm

# Load config từ settings.py
from config.settings import CFG, get_path

# Import các module từ src
from src import (
    load_and_split_data, 
    ItemItemCFStage1, 
    load_parquet_cache,
    save_parquet_cache
)

# =========================================================
# 1. HÀM LOAD GROUND TRUTH (TỪ FILE PKL THẬT)
# =========================================================
def load_groundtruth_pkl(path):
    print(f">>> Loading Ground Truth from: {path}")
    if not Path(path).exists():
        print(f"❌ File not found: {path}")
        return {}
        
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    gt_clean = {}
    if isinstance(data, pd.DataFrame):
        u_col, i_col = 'customer_id', 'item_id'
        if not data.empty:
            if isinstance(data[i_col].iloc[0], (list, np.ndarray, set)):
                data = data.explode(i_col)
        data = data.dropna(subset=[u_col, i_col])
        data[u_col] = data[u_col].astype(str)
        data[i_col] = data[i_col].astype(str)
        gt_clean = data.groupby(u_col)[i_col].apply(set).to_dict()
        
    elif isinstance(data, dict):
        for k in ['gt_test', 'test', 'groundtruth']:
            if k in data: data = data[k]; break
        for u, v in data.items():
            items = set(str(x) for x in v) if isinstance(v, (list, tuple, set)) else {str(v)}
            gt_clean[str(u)] = items
            
    print(f"   [SUCCESS] Loaded {len(gt_clean)} users in GT.")
    return gt_clean

# =========================================================
# 2. HÀM TÍNH METRICS (Recall & Hit Rate) - MỚI
# =========================================================
def calculate_stage1_metrics(candidates_df, gt_dict, stage1_model):
    """
    Tính Metrics Stage 1 (Retrieval Performance) theo logic Warm/Cold chuẩn.
    """
    print(f"\n[{stage1_model.__class__.__name__}] Calculating Stage 1 Metrics...")
    
    # Group candidates theo user: {user_id: {item1, item2, ...}}
    pred_dict = candidates_df.groupby("customer_id")["item_id"].apply(set).to_dict()
    
    stats = {
        "all":  {"recall": [], "hit": [], "count": 0},
        "warm": {"recall": [], "hit": [], "count": 0},
        "cold": {"recall": [], "hit": [], "count": 0}
    }
    
    users_no_pred = 0
    
    # Duyệt qua từng User trong Ground Truth
    for user_id, true_items in gt_dict.items():
        true_items_set = set(true_items)
        if not true_items_set: continue 
        
        user_str = str(user_id)
        
        # [QUAN TRỌNG] Xác định Warm/Cold dựa vào việc User có trong Stage 1 Model (Tập Train) hay không
        if user_str in stage1_model.user_id_to_index_:
            u_type = "warm"
        else:
            u_type = "cold"
        
        # Lấy tập item dự đoán
        pred_items_set = pred_dict.get(user_str, set())
        
        if not pred_items_set:
            users_no_pred += 1
            recall = 0.0
            hit = 0
        else:
            # Tính số lượng item trùng khớp
            match_cnt = len(true_items_set.intersection(pred_items_set))
            recall = match_cnt / len(true_items_set)
            hit = 1 if match_cnt > 0 else 0
        
        # Lưu chỉ số All
        stats["all"]["recall"].append(recall)
        stats["all"]["hit"].append(hit)
        stats["all"]["count"] += 1
        
        # Lưu chỉ số Warm/Cold
        stats[u_type]["recall"].append(recall)
        stats[u_type]["hit"].append(hit)
        stats[u_type]["count"] += 1

    # In Báo cáo dạng bảng
    print(f"\n{'='*55}")
    print(f"📊 STAGE 1 PERFORMANCE REPORT (Candidates Set)")
    print(f"{'='*55}")
    print(f"Total Test Users: {stats['all']['count']} (No Candidates: {users_no_pred})")
    print("-" * 55)
    print(f"{'Type':<10} | {'Count':<8} | {'Recall (Avg)':<12} | {'Hit Rate':<10}")
    print("-" * 55)
    
    for key in ["all", "warm", "cold"]:
        count = stats[key]["count"]
        if count > 0:
            avg_rec = sum(stats[key]["recall"]) / count
            avg_hit = sum(stats[key]["hit"]) / count
            print(f"{key.upper():<10} | {count:<8} | {avg_rec:.4f}       | {avg_hit:.4f}")
        else:
            print(f"{key.upper():<10} | 0        | N/A          | N/A")
    print("-" * 55)
    
    return stats

# =========================================================
# 3. MAIN FLOW
# =========================================================
def main():
    print(f"Project Root: {get_path('')}")
    GT_PATH = "./data/final_groundtruth.pkl" # Đường dẫn file test thật
    
    # ----------------------------------------------------
    # 1. LOAD DATA & TRAIN STAGE 1
    # ----------------------------------------------------
    print(">>> 1. Loading Data & Model...")
    df_train, df_valid, df_item, df_user = load_and_split_data(
        CFG.paths.raw_data_path,
        CFG.data_split.hist_end_date,
        CFG.data_split.hist_days,
        CFG.data_split.recent_days
    )
    
    # Load GT thật
    gt_test = load_groundtruth_pkl(GT_PATH)
    test_users = list(gt_test.keys())
    
    # Train/Load Stage 1
    stage1_path = get_path("artifacts/stage1_model_base.pkl") 
    
    if Path(stage1_path).exists():
        print("   -> Loading Stage 1 from cache...")
        stage1 = joblib.load(stage1_path)
    else:
        print("   -> Training Stage 1 (Full Data)...")
        df_full = pd.concat([df_train, df_valid], ignore_index=True)
        stage1 = ItemItemCFStage1(CFG.stage1, df_item, df_user)
        stage1.fit(df_full)
        joblib.dump(stage1, stage1_path)

    # ----------------------------------------------------
    # 2. EVALUATE STAGE 1 (NEW ITEMS - Filtered)
    # ----------------------------------------------------
    # print("\n>>> 2. Evaluating Recall for NEW ITEM REC (Filter Bought)...")
    
    # cand_path_new = get_path("artifacts/stage1_eval_candidates_new.parquet")
    # df_cand_new = load_parquet_cache(cand_path_new)
    
    # if df_cand_new is None:
    #     print("   -> Generating candidates (New Item Mode)...")
    #     df_cand_new = stage1.recommend_candidates(test_users, allow_repeat=False)
    #     save_parquet_cache(df_cand_new, cand_path_new)
    
    # # [CALL] Hàm tính metrics mới
    # calculate_stage1_metrics(df_cand_new, gt_test, stage1)

    # ----------------------------------------------------
    # 3. EVALUATE STAGE 1 (ALL ITEMS - Unfiltered)
    # ----------------------------------------------------
    print("\n>>> 3. Evaluating Recall for ALL ITEM REC (Repurchase Allowed)...")
    
    cand_path_all = get_path("artifacts/stage1_eval_candidates_all.parquet")
    df_cand_all = load_parquet_cache(cand_path_all)
    
    if df_cand_all is None:
        print("   -> Generating candidates (All Item Mode)...")
        df_cand_all = stage1.recommend_candidates(test_users, allow_repeat=True)
        save_parquet_cache(df_cand_all, cand_path_all)
    
    # [CALL] Hàm tính metrics mới
    calculate_stage1_metrics(df_cand_all, gt_test, stage1)

    print("\n>>> EVALUATION COMPLETED!")

if __name__ == "__main__":
    main()