import pandas as pd
import numpy as np
import joblib
import pickle
import lightgbm as lgb
from pathlib import Path
from config.settings import CFG, get_path
from src import (
    load_and_split_data, 
    predict_stage2,
    calculate_metrics_at_k,
    ItemItemCFStage1
)

def load_groundtruth_pkl(path):
    """Hàm load groundtruth từ file .pkl (xử lý ID về string)"""
    print(f"Loading Ground Truth from: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    # Xử lý format nếu file pkl bọc trong dict con
    if isinstance(data, dict):
        for key in ['gt_test', 'test', 'groundtruth']:
            if key in data:
                data = data[key]
                break
                
    # Chuẩn hóa: User ID và Item ID phải là String
    gt_clean = {}
    for user, items in data.items():
        user_str = str(user)
        if isinstance(items, (list, tuple, set, np.ndarray)):
            items_str = set(str(x) for x in items)
        else:
            items_str = {str(items)}
        gt_clean[user_str] = items_str
        
    return gt_clean

def run_test(flow_name="new_item_rec"):
    """
    Chạy test cho một flow cụ thể.
    flow_name: 'new_item_rec' hoặc 'all_item_rec'
    """
    # 1. Cấu hình
    GT_PATH = "/datastore/uittogether2/LuuTru/tienptx/zDoAn/CS116_DoAnMonHoc/groundtruth.pkl"
    TOP_K = 200     # Số lượng candidate Stage 1
    METRIC_K = 10   # K để tính Precision/Recall
    
    # Xác định config dựa trên flow name
    if flow_name == "new_item_rec":
        ALLOW_REPEAT = False
        FILTER_BOUGHT_EVAL = True
    else: # all_item_rec
        ALLOW_REPEAT = True
        FILTER_BOUGHT_EVAL = False

    print(f"\n{'='*40}")
    print(f"Project Root: {get_path('')}")
    print(f">>> STARTING INFERENCE: {flow_name.upper()} ON JAN 2025 DATA")
    print(f">>> Config: ALLOW_REPEAT={ALLOW_REPEAT} | FILTER_EVAL={FILTER_BOUGHT_EVAL}")
    print(f"{'='*40}")

    # 2. Load Models
    print(">>> Loading Trained Models...")
    try:
        # Load Stage 1 Base Model (Dùng chung)
        model_path = get_path("artifacts/stage1_model_base.pkl")
        if not Path(model_path).exists():
             model_path = get_path("artifacts/stage1_model.pkl")
             
        stage1 = joblib.load(model_path)
        print(f"   -> Loaded Stage 1 from {model_path}")

        # Load Stage 2 Model tương ứng với Flow
        lgbm_path = get_path(f"artifacts/lgbm_model_{flow_name}.txt")
        if not Path(lgbm_path).exists():
             print(f"❌ Warning: Model for {flow_name} not found at {lgbm_path}. Trying default...")
             lgbm_path = get_path("artifacts/lgbm_model.txt")
             
        lgbm_model = lgb.Booster(model_file=lgbm_path)
        print(f"   -> Loaded Stage 2 from {lgbm_path}")
        
        # Load Feature Columns tương ứng với Flow
        # [FIX]: Load từ file .pkl chứ không phải .parquet
        feature_path = get_path(f"artifacts/stage2_features_{flow_name}.pkl")
        if not Path(feature_path).exists():
             # Fallback nếu dùng chung file features
             feature_path = get_path("artifacts/stage2_features.pkl")
             
        feature_cols = joblib.load(feature_path)
        print(f"   -> Loaded Features list from {feature_path}")
        
    except FileNotFoundError as e:
        print(f"❌ Lỗi: Không tìm thấy file artifact. Chi tiết: {e}")
        return

    # 3. Load Test Data (Ground Truth)
    gt_test = load_groundtruth_pkl(GT_PATH)
    test_users = list(gt_test.keys())
    print(f">>> Found {len(test_users)} users in Test Set.")

    # 4. Stage 1: Candidate Generation
    print(">>> 1. Generating Candidates (Stage 1)...")
    
    # Setup Top K
    stage1.top_k = TOP_K
    
    # Quan trọng: Set allow_repeat đúng theo flow config
    candidates = stage1.recommend_candidates(test_users, allow_repeat=ALLOW_REPEAT)
    
    print(f"   -> Candidates shape: {candidates.shape}")

    # 5. Stage 2: Ranking
    print(">>> 2. Ranking (Stage 2)...")
    
    # Nếu thiếu cột nào trong feature_cols thì thêm vào (fill 0)
    for col in feature_cols:
        if col not in candidates.columns:
            candidates[col] = 0
            
    df_pred = predict_stage2(
        lgbm_model, 
        candidates, 
        feature_cols=feature_cols, 
        top_k=METRIC_K
    )
    
    # 6. Evaluation
    print(">>> 3. Calculating Metrics...")
    
    # Dùng hàm lấy lịch sử
    train_history = stage1.get_user_history_dict()
    
    metrics = calculate_metrics_at_k(
        df_pred, 
        gt_test, 
        train_history, # Truyền dict lịch sử vào
        k=METRIC_K,
        filter_bought_items=FILTER_BOUGHT_EVAL 
    )
    
    print(f"\n--- TEST RESULTS: {flow_name.upper()} ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Save submission
    out_path = get_path(f"artifacts/submission_jan2025_{flow_name.replace('_item_rec', '')}.csv") # new.csv or all.csv
    df_pred[['customer_id', 'item_id', 'pred_score']].to_csv(out_path, index=False)
    print(f"\n>>> Predictions saved to: {out_path}")

if __name__ == "__main__":
    # Chạy lần lượt cả 2 flow để test
    run_test(flow_name="new_item_rec")
    run_test(flow_name="all_item_rec")