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
    train_history: dict,
    allow_repeat_stage1: bool,
    filter_bought_eval: bool
):
    print(f"\n{'='*40}")
    print(f">>> RUNNING FLOW: {flow_name.upper()}")
    print(f">>> allow_repeat={allow_repeat_stage1} | filter_eval={filter_bought_eval}")
    print(f"{'='*40}")

    # ====================================================
    # 1. LOAD / BUILD STAGE 1 CANDIDATES
    # ====================================================
    cand_path = get_path(f"artifacts/candidates_{flow_name}.parquet")
    df_candidates = load_parquet_cache(cand_path)

    if df_candidates is None:
        print(f"[{flow_name}] Generating candidates...")
        valid_users = df_valid["customer_id"].unique()
        df_candidates = stage1_model.recommend_candidates(
            valid_users, allow_repeat=allow_repeat_stage1
        )
        save_parquet_cache(df_candidates, cand_path)
    else:
        print(f"[{flow_name}] Loaded candidates from cache.")

    # ====================================================
    # 2. LOAD / BUILD STAGE 2 TRAIN DATA (LABEL = REAL)
    # ====================================================
    stage2_train_path = get_path(f"artifacts/stage2_train_{flow_name}.parquet")
    df_stage2_train = load_parquet_cache(stage2_train_path)

    if df_stage2_train is None:
        print(f"[{flow_name}] Labeling candidates (NO FAKE LABEL)...")
        gt_valid = df_valid.groupby("customer_id")["item_id"].apply(set).to_dict()

        df_candidates = df_candidates.copy()
        df_candidates["label"] = [
            1 if item in gt_valid.get(user, set()) else 0
            for user, item in zip(df_candidates["customer_id"], df_candidates["item_id"])
        ]

        print(f"   Label stats: {df_candidates['label'].value_counts().to_dict()}")
        save_parquet_cache(df_candidates, stage2_train_path)
        df_stage2_train = df_candidates
    else:
        print(f"[{flow_name}] Loaded Stage2 train data from cache.")

    # ====================================================
    # 3. LOAD / TRAIN STAGE 2 MODEL
    # ====================================================
    model_path = get_path(f"artifacts/lgbm_model_{flow_name}.txt")
    feat_path = get_path(f"artifacts/stage2_features_{flow_name}.pkl")

    if Path(model_path).exists() and Path(feat_path).exists():
        print(f"[{flow_name}] Loading Stage 2 model from cache...")
        lgbm_model = lgb.Booster(model_file=model_path)
        feature_cols = joblib.load(feat_path)
    else:
        print(f"[{flow_name}] Training Stage 2 model...")

        users = df_stage2_train["customer_id"].unique()
        np.random.seed(42)
        valid_users_lgbm = np.random.choice(users, size=int(len(users) * 0.1), replace=False)

        train_df = df_stage2_train[~df_stage2_train["customer_id"].isin(valid_users_lgbm)]
        valid_df = df_stage2_train[df_stage2_train["customer_id"].isin(valid_users_lgbm)]

        lgbm_model, feature_cols = train_lgbm_ranker(
            train_df,
            CFG.stage2,
            valid_df=valid_df
        )

        lgbm_model.save_model(model_path)
        joblib.dump(feature_cols, feat_path)

    # ====================================================
    # 4. EVALUATION (KHÔNG CACHE)
    # ====================================================
    print(f"[{flow_name}] Evaluating...")
    df_pred = predict_stage2(
        lgbm_model,
        df_stage2_train,
        feature_cols=feature_cols,
        top_k=50
    )

    gt_valid = df_valid.groupby("customer_id")["item_id"].apply(set).to_dict()

    metrics = calculate_metrics_at_k(
        df_pred,
        gt_valid,
        train_history,
        k=CFG.evaluation.k_metric,
        filter_bought_items=filter_bought_eval
    )

    print(f"\n--- RESULTS {flow_name.upper()} ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    return metrics


def main():
    print(f"Project Root: {get_path('')}")

    # ==========================================
    # 1. LOAD DATA (Chung cho cả 2 flow)
    # ==========================================
    print(">>> 1. Loading Data...")
    # Load 1/9 -> 31/12 (Train) và 01/01 -> 31/01 (Valid)
    df_train, df_valid, df_item, df_user = load_and_split_data(
        CFG.paths.raw_data_path,
        CFG.data_split.hist_end_date,
        CFG.data_split.hist_days,
        CFG.data_split.recent_days
    )
    
    # ==========================================
    # 2. TRAIN STAGE 1 (Cho mục đích Validate)
    # ==========================================
    print(">>> 2. Training / Loading Stage 1 (Validation)...")
    stage1_val_path = get_path("artifacts/stage1_model_val.pkl")

    if Path(stage1_val_path).exists():
        print("   -> Loading Stage 1 (validation) from cache...")
        stage1_val = joblib.load(stage1_val_path)
    else:
        stage1_val = ItemItemCFStage1(CFG.stage1, df_item, df_user)
        stage1_val.fit(df_train)
        joblib.dump(stage1_val, stage1_val_path)
        print("   -> Saved Stage 1 (validation).")

    train_history = stage1_val.get_user_history_dict()

    # =================================================================
    # RUN FLOWS (Validate & Train Stage 2)
    # =================================================================
    # Flow 1: New Item
    run_pipeline_flow(
        flow_name="new_item_rec",
        stage1_model=stage1_val,
        df_valid=df_valid,
        train_history=train_history,
        allow_repeat_stage1=False,
        filter_bought_eval=True
    )

    run_pipeline_flow(
        flow_name="all_item_rec",
        stage1_model=stage1_val,
        df_valid=df_valid,
        train_history=train_history,
        allow_repeat_stage1=True,
        filter_bought_eval=False
    )

    
    # ==========================================
    # 3. [QUAN TRỌNG] RETRAIN STAGE 1 FULL DATA
    # ==========================================
    print("\n>>> RETRAIN / LOAD STAGE 1 PRODUCTION")
    prod_path = get_path("artifacts/stage1_model_base.pkl")

    if Path(prod_path).exists():
        print("   -> Production Stage 1 already exists. Skip retrain.")
    else:
        df_full_train = pd.concat([df_train, df_valid], ignore_index=True)
        stage1_prod = ItemItemCFStage1(CFG.stage1, df_item, df_user)
        stage1_prod.fit(df_full_train)
        joblib.dump(stage1_prod, prod_path)
        print("   -> Saved Production Stage 1 model.")

    
    print("\n>>> ALL PIPELINES COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()