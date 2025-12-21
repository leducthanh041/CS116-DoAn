import pandas as pd
import numpy as np
import joblib
import pickle
import lightgbm as lgb
import json
import os
import glob
from pathlib import Path
from tqdm.auto import tqdm

# Load config từ settings.py
from config.settings import CFG, get_path
from src import (
    ItemItemCFStage1, 
    predict_stage2, 
    load_parquet_cache, 
    save_parquet_cache,
    calculate_metrics_at_k
)

# =========================================================
# 1. HÀM LOAD GROUND TRUTH
# =========================================================
def load_groundtruth_pkl(path):
    print(f">>> Loading Ground Truth from: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    gt_clean = {}

    if isinstance(data, pd.DataFrame):
        u_col, i_col = 'customer_id', 'item_id'
        if u_col not in data.columns or i_col not in data.columns: return {}

        if not data.empty:
            first_val = data[i_col].iloc[0]
            if isinstance(first_val, (list, np.ndarray, set)):
                print("   [INFO] Exploding aggregated items in GT...")
                data = data.explode(i_col)
        
        data = data.dropna(subset=[u_col, i_col])
        data[u_col] = data[u_col].astype(str)
        try:
            data[i_col] = data[i_col].astype(float).astype(np.int64).astype(str)
        except:
            data[i_col] = data[i_col].astype(str)
            
        gt_clean = data.groupby(u_col)[i_col].apply(set).to_dict()
        
    elif isinstance(data, dict):
        for k in ['gt_test', 'test', 'groundtruth']:
            if k in data: data = data[k]; break
        for u, v in data.items():
            user_str = str(u)
            if isinstance(v, (list, tuple, set, np.ndarray)):
                items_str = set(str(x) for x in v)
            else:
                items_str = {str(v)}
            gt_clean[user_str] = items_str

    print(f"   [SUCCESS] Loaded {len(gt_clean)} users in GT.")
    return gt_clean

# =========================================================
# 2. HÀM LOAD FULL LỊCH SỬ GIAO DỊCH (TỪ RAW DATA)
# =========================================================
def load_full_history_from_raw(raw_path):
    print(f">>> Loading FULL Transaction History from: {raw_path}")
    pattern = os.path.join(raw_path, "*purchase_history_daily_chunk*.parquet")
    files = glob.glob(pattern)
    
    if not files:
        print("   ⚠️ WARNING: Không tìm thấy file transaction nào! History sẽ rỗng.")
        return {}
    
    dfs = []
    cols = ['customer_id', 'item_id']
    
    for f in tqdm(files, desc="Reading History Files"):
        try:
            df = pd.read_parquet(f, columns=cols)
            df['customer_id'] = df['customer_id'].astype(str)
            df['item_id'] = df['item_id'].astype(str)
            dfs.append(df)
        except Exception as e:
            print(f"   Skipping {f}: {e}")
            
    if not dfs: return {}
    
    df_all = pd.concat(dfs, ignore_index=True)
    full_hist = df_all.groupby('customer_id')['item_id'].apply(set).to_dict()
    print(f"   [SUCCESS] Indexed history for {len(full_hist)} users.")
    return full_hist

# =========================================================
# 3. HÀM TÍNH PRECISION (PHÂN LOẠI WARM/COLD)
# =========================================================
def precision_at_k_custom(pred, gt, hist, filter_bought_items=True, K=10):
    """
    Tính Precision@K cho: All Users, Warm Users, Cold Users.
    Trả về dictionary chứa các metrics.
    """
    precisions = []
    warm_precisions = []
    cold_precisions = []
    
    cold_users_list = []
    
    for user in gt.keys():
        user = str(user)
        
        # 1. Xác định Warm/Cold dựa trên lịch sử
        # Nếu user không có trong hist -> Cold
        user_hist = hist.get(user, set())
        is_cold = len(user_hist) == 0
        
        # 2. Xác định tập items mục tiêu (Relevant Items)
        gt_items = set(gt[user])
        relevant_items = gt_items.copy()
        
        # 3. Logic Lọc hàng đã mua
        if filter_bought_items:
            relevant_items -= set(user_hist)
            # Nếu sau khi lọc mà không còn item nào (user chỉ mua lại đồ cũ)
            # -> Bỏ qua user này khỏi mẫu đánh giá (theo đúng logic New Item Rec)
            if not relevant_items:
                continue 
        
        # 4. Tính Score
        if user in pred:
            pred_items = pred[user][:K]
            hits = len(set(pred_items) & relevant_items)
            val = hits / K
        else:
            # User có trong GT nhưng Model không đưa ra dự đoán nào -> P = 0
            val = 0.0
            
        # 5. Phân nhóm kết quả
        precisions.append(val)
        
        if is_cold:
            cold_precisions.append(val)
            cold_users_list.append(user)
        else:
            warm_precisions.append(val)
            
    # Tính trung bình
    res = {
        "all": np.mean(precisions) if precisions else 0.0,
        "warm": np.mean(warm_precisions) if warm_precisions else 0.0,
        "cold": np.mean(cold_precisions) if cold_precisions else 0.0,
        "cnt_all": len(precisions),
        "cnt_warm": len(warm_precisions),
        "cnt_cold": len(cold_precisions),
        "cold_users": cold_users_list
    }
    return res

# =========================================================
# 4. LOAD FEATURES
# =========================================================
def get_stage2_features():
    config_path = get_path("artifacts/feature_config.json")
    if Path(config_path).exists():
        with open(config_path, "r") as f: return json.load(f)
    return [
        "stage1_score", "stage1_rank", "sim_max", "sim_avg", "support_cnt",
        "item_hist_cnt", "brand_match_cnt", "cat2_match_cnt", 
        "cat_hist_cnt", "age_group_hist_cnt",
        "feat_days_since_item", "feat_days_since_cat",
        "feat_price_ratio", "feat_log_price", "feat_pop_30d"
    ]

# =========================================================
# 5. MAIN FLOW
# =========================================================
def run_test(flow_name="new_item_rec"):
    print(f"\n{'='*50}")
    print(f">>> STARTING TEST PIPELINE: {flow_name.upper()}")
    print(f"{'='*50}")
    
    GT_PATH = "./data/final_groundtruth.pkl" 
    METRIC_K = 10
    
    # -----------------------------------------------------
    # A. LOAD RESOURCES & FULL HISTORY
    # -----------------------------------------------------
    gt_test = load_groundtruth_pkl(GT_PATH)
    test_users = list(gt_test.keys())
    
    # [NEW] Load Full History từ Raw Data
    full_history = load_full_history_from_raw(CFG.paths.raw_data_path)
    
    try:
        # Stage 1 (chỉ để sinh candidates)
        s1_path = get_path("artifacts/stage1_model_base.pkl")
        stage1_model = joblib.load(s1_path)
        
        # Stage 2
        lgbm_path = get_path(f"artifacts/lgbm_model_{flow_name}.txt")
        lgbm_model = lgb.Booster(model_file=lgbm_path)
        
        feature_cols = get_stage2_features()
        print(f"   -> Loaded Models & {len(feature_cols)} Features.")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    # -----------------------------------------------------
    # B. PREDICT (CANDIDATES -> RANKING)
    # -----------------------------------------------------
    cache_cand_path = get_path(f"artifacts/test_candidates_{flow_name}.parquet")
    candidates = load_parquet_cache(cache_cand_path)
    
    if candidates is None:
        print(f"   [Cache Miss] Generating candidates...")
        allow_repeat = True if flow_name == "all_item_rec" else False
        candidates = stage1_model.recommend_candidates(test_users, allow_repeat=allow_repeat)
        save_parquet_cache(candidates, cache_cand_path)
    else:
        print(f"   [Cache Hit] Loaded candidates.")

    # Predict Stage 2
    for col in feature_cols:
        if col not in candidates.columns: candidates[col] = 0
            
    df_pred = predict_stage2(lgbm_model, candidates, feature_cols=feature_cols, top_k=50)
    
    # Convert to Dict
    pred_dict = df_pred.groupby("customer_id")["item_id"].apply(list).to_dict()
    
    # Save Prediction
    pkl_out_path = get_path(f"artifacts/predictions_{flow_name}.pkl")
    with open(pkl_out_path, "wb") as f:
        pickle.dump(pred_dict, f)
    print(f"✅ Saved predictions to: {pkl_out_path}")
    
    # -----------------------------------------------------
    # C. CALCULATE METRICS
    # -----------------------------------------------------
    print(f"\n>>> 4. Calculating Metrics (Precision@{METRIC_K})...")
    
    is_filter_bought = (flow_name == "new_item_rec")
    mode_label = "Filtered/New" if is_filter_bought else "Unfiltered/All"
    
    # 1. Tính Precision (All/Warm/Cold)
    p_stats = precision_at_k_custom(
        pred_dict, gt_test, full_history, 
        filter_bought_items=is_filter_bought, K=METRIC_K
    )
    
    # 2. Tính Recall & NDCG (Metric Function)
    other_metrics = calculate_metrics_at_k(
        df_pred, gt_test, full_history, 
        k=METRIC_K, filter_bought_items=is_filter_bought
    )
    
    # --- REPORT ---
    print("-" * 50)
    print(f"📊 RESULT REPORT: {flow_name} ({mode_label})")
    print("-" * 50)
    
    # Prefix key cho metrics Recall/NDCG
    m_prefix = f"all_{'NEW' if is_filter_bought else 'ALL'}"
    w_prefix = f"warm_{'NEW' if is_filter_bought else 'ALL'}"
    c_prefix = f"cold_{'NEW' if is_filter_bought else 'ALL'}"
    
    # Helper format
    def fmt(val): return f"{val:.4f}"

    print(f"{'METRIC':<15} | {'ALL USERS':<10} | {'WARM USERS':<10} | {'COLD USERS':<10}")
    print("-" * 55)
    
    # Precision
    print(f"{'Precision@'+str(METRIC_K):<15} | {fmt(p_stats['all']):<10} | {fmt(p_stats['warm']):<10} | {fmt(p_stats['cold']):<10}")
    
    # Recall
    r_all = other_metrics.get(f"{m_prefix}_R@{METRIC_K}", 0)
    r_warm = other_metrics.get(f"{w_prefix}_R@{METRIC_K}", 0)
    r_cold = other_metrics.get(f"{c_prefix}_R@{METRIC_K}", 0)
    print(f"{'Recall@'+str(METRIC_K):<15} | {fmt(r_all):<10} | {fmt(r_warm):<10} | {fmt(r_cold):<10}")
    
    # NDCG
    n_all = other_metrics.get(f"{m_prefix}_NDCG@{METRIC_K}", 0)
    n_warm = other_metrics.get(f"{w_prefix}_NDCG@{METRIC_K}", 0)
    n_cold = other_metrics.get(f"{c_prefix}_NDCG@{METRIC_K}", 0)
    print(f"{'NDCG@'+str(METRIC_K):<15} | {fmt(n_all):<10} | {fmt(n_warm):<10} | {fmt(n_cold):<10}")
    
    print("-" * 55)
    print(f"ℹ️  Sample Sizes -> All: {p_stats['cnt_all']} | Warm: {p_stats['cnt_warm']} | Cold: {p_stats['cnt_cold']}")
    print("-" * 55)

if __name__ == "__main__":
    run_test(flow_name="new_item_rec")
    run_test(flow_name="all_item_rec")