import numpy as np
import pandas as pd
from pathlib import Path
import joblib 
import lightgbm as lgb

# Load config từ settings.py
from config.settings import CFG, get_path

# Import các module từ src
from src import (
    load_and_split_data, 
    ItemItemCFStage1, 
    train_lgbm_ranker, 
    predict_stage2,
    calculate_metrics_at_k,
    load_parquet_cache,
    save_parquet_cache
)

def run_pipeline_flow(
    flow_name: str,
    stage1_model,
    df_valid: pd.DataFrame,
    df_train_user_set: set,
    train_history: dict,
    allow_repeat_stage1: bool,
    filter_bought_eval: bool
):
    """
    Hàm thực thi một Flow (Pipeline) hoàn chỉnh.
    """
    print(f"\n{'='*40}")
    print(f">>> RUNNING FLOW: {flow_name.upper()}")
    print(f">>> Config: allow_repeat={allow_repeat_stage1} | filter_eval={filter_bought_eval}")
    print(f"{'='*40}")

    # ----------------------------------------
    # 1. Generate Candidates (Candidate Gen)
    # ----------------------------------------
    print(f"[{flow_name}] 1. Generating Candidates...")
    
    stage2_train_file = get_path(f"artifacts/stage2_train_{flow_name}.parquet")
    df_stage2_train = load_parquet_cache(stage2_train_file)
    
    if df_stage2_train is None:
        print(f"   -> Cache not found. Generating candidates (allow_repeat={allow_repeat_stage1})...")
        valid_users = df_valid['customer_id'].unique()
        
        candidates = stage1_model.recommend_candidates(valid_users, allow_repeat=allow_repeat_stage1)
        
        print("   -> Labeling candidates...")
        gt_valid = df_valid.groupby("customer_id")["item_id"].apply(set).to_dict()
        
        candidates['label'] = [
            1 if item in gt_valid.get(user, set()) else 0
            for user, item in zip(candidates['customer_id'], candidates['item_id'])
        ]
        
        print(f"   -> Label stats: {candidates['label'].value_counts().to_dict()}")
        df_stage2_train = candidates
        save_parquet_cache(df_stage2_train, stage2_train_file)
    else:
        print("   -> Loaded candidates from cache.")

    # ----------------------------------------
    # 2. Train Stage 2 (Ranking)
    # ----------------------------------------
    print(f"[{flow_name}] 2. Training Stage 2 (LightGBM)...")
    
    unique_users = df_stage2_train['customer_id'].unique()
    np.random.seed(42)
    valid_users_lgbm = np.random.choice(unique_users, size=int(len(unique_users) * 0.1), replace=False)
    
    train_lgbm_df = df_stage2_train[~df_stage2_train['customer_id'].isin(valid_users_lgbm)]
    valid_lgbm_df = df_stage2_train[df_stage2_train['customer_id'].isin(valid_users_lgbm)]
    
    lgbm_model, feature_cols = train_lgbm_ranker(
        train_lgbm_df, 
        CFG.stage2, 
        valid_df=valid_lgbm_df
    )
    
    # ----------------------------------------
    # 3. Evaluation
    # ----------------------------------------
    print(f"[{flow_name}] 3. Evaluating...")
    
    df_pred = predict_stage2(lgbm_model, df_stage2_train, feature_cols=feature_cols, top_k=50)
    
    gt_valid = df_valid.groupby("customer_id")["item_id"].apply(set).to_dict()
    K_METRIC = CFG.evaluation.k_metric
    
    metrics = calculate_metrics_at_k(
        df_pred, 
        gt_valid, 
        train_history, 
        k=K_METRIC, 
        filter_bought_items=filter_bought_eval
    )
    
    print(f"\n--- RESULTS for {flow_name.upper()} (K={K_METRIC}) ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
        
    # ----------------------------------------
    # 4. Save Artifacts for this Flow
    # ----------------------------------------
    print(f"[{flow_name}] 4. Saving Models...")
    # Lưu Model Stage 2 (LightGBM học được ranking pattern)
    lgbm_model.save_model(get_path(f"artifacts/lgbm_model_{flow_name}.txt"))
    
    # Lưu danh sách features riêng cho flow này
    joblib.dump(feature_cols, get_path(f"artifacts/stage2_features_{flow_name}.pkl"))
    
    return metrics


def main():
    print(f"Project Root: {get_path('')}")

    # ==========================================
    # 1. LOAD DATA (Chung cho cả 2 flow)
    # ==========================================
    print(">>> 1. Loading Data...")
    # Load 1/8 -> 30/11 (Train) và 1/12 -> 31/12 (Valid)
    df_train, df_valid, df_item, df_user = load_and_split_data(
        CFG.paths.raw_data_path,
        CFG.data_split.hist_end_date,
        CFG.data_split.hist_days,
        CFG.data_split.recent_days
    )
    
    # ==========================================
    # 2. TRAIN STAGE 1 (Cho mục đích Validate)
    # ==========================================
    print(">>> 2. Training Stage 1 (For Validation)...")
    # Model này chỉ dùng để sinh candidate cho tập Valid (Tháng 12)
    stage1_val = ItemItemCFStage1(CFG.stage1, df_item, df_user)
    stage1_val.fit(df_train)
    
    # Chuẩn bị history cho việc đánh giá
    train_history = stage1_val.get_user_history_dict()
    train_users_set = set(df_train['customer_id'].astype(str).unique())

    # =================================================================
    # RUN FLOWS (Validate & Train Stage 2)
    # =================================================================
    # Flow 1: New Item
    run_pipeline_flow(
        flow_name="new_item_rec",
        stage1_model=stage1_val,
        df_valid=df_valid,
        df_train_user_set=train_users_set,
        train_history=train_history,
        allow_repeat_stage1=False,
        filter_bought_eval=True
    )
    
    # Flow 2: All Item
    run_pipeline_flow(
        flow_name="all_item_rec",
        stage1_model=stage1_val,
        df_valid=df_valid,
        df_train_user_set=train_users_set,
        train_history=train_history,
        allow_repeat_stage1=True,
        filter_bought_eval=False
    )
    
    # ==========================================
    # 3. [QUAN TRỌNG] RETRAIN STAGE 1 FULL DATA
    # ==========================================
    print("\n" + "="*50)
    print(">>> FINAL STEP: RETRAINING STAGE 1 ON FULL DATA (Aug-Dec)")
    print("="*50)
    
    # Gộp Train (Aug-Nov) + Valid (Dec) thành Full History
    df_full_train = pd.concat([df_train, df_valid], ignore_index=True)
    
    print(f"   -> Full Data Size: {len(df_full_train)} rows")
    print(f"   -> Date Range: {df_full_train['created_date'].min()} to {df_full_train['created_date'].max()}")
    
    # Khởi tạo model mới
    stage1_production = ItemItemCFStage1(CFG.stage1, df_item, df_user)
    
    # Train lại trên toàn bộ dữ liệu
    stage1_production.fit(df_full_train)
    
    # Lưu đè model base cũ bằng model mới xịn hơn
    # test.py sẽ load model này để có thông tin mới nhất đến hết tháng 12
    out_path = get_path("artifacts/stage1_model_base.pkl")
    joblib.dump(stage1_production, out_path)
    print(f"✅ Production Stage 1 Model saved to: {out_path}")
    
    print("\n>>> ALL PIPELINES COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()