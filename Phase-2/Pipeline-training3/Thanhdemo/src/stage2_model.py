import lightgbm as lgb
import pandas as pd
import numpy as np

def train_lgbm_classifier(train_df, config, valid_df=None, feature_cols=None):
    """
    Train Binary Classification Model (Pointwise) cho RecSys.
    Mục tiêu: Dự đoán xác suất user mua item (0 hoặc 1).
    
    Args:
        train_df: DataFrame train
        config: Config object (CFG.stage2)
        valid_df: DataFrame valid
        feature_cols: (Optional) Danh sách feature cụ thể cần dùng
    """
    
    # 1. Xác định Features
    if feature_cols is None:
        ignore_cols = {'customer_id', 'item_id', 'label', 'created_date', 'created_datetime', 'pred_score'}
        feature_cols = [c for c in train_df.columns if c not in ignore_cols]
    
    print(f"[Stage2] Training Classifier with {len(feature_cols)} features.")

    # 2. Chuẩn bị dữ liệu (KHÔNG CẦN GROUP cho Binary)
    X_train = train_df[feature_cols]
    y_train = train_df["label"]
    
    # Tạo Dataset cho LightGBM
    lgb_train = lgb.Dataset(X_train, y_train)
    
    # Valid set setup
    valid_sets = [lgb_train]
    valid_names = ['train']
    
    if valid_df is not None:
        X_valid = valid_df[feature_cols]
        y_valid = valid_df["label"]
        lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
        valid_sets.append(lgb_eval)
        valid_names.append('valid')

    # 3. Setup Parameters (Binary Classification)
    params = {
        "objective": "binary",       # <--- THAY ĐỔI QUAN TRỌNG
        "metric": "auc",             # Dùng AUC để tối ưu khả năng ranking (AUC cao -> Precision cao)
        "boosting_type": "gbdt",
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "min_data_in_leaf": config.min_data_in_leaf,
        
        # Xử lý mất cân bằng mẫu (Quan trọng cho RecSys vì số 0 >> số 1)
        "is_unbalance": True,        
        
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "random_state": 42
    }

    # 4. Train
    callbacks = [
        lgb.log_evaluation(period=50)
    ]
    
    if valid_df is not None:
        callbacks.append(lgb.early_stopping(stopping_rounds=50))

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=config.n_estimators,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks
    )
    
    return model, feature_cols

def train_lgbm_ranker(train_df, config, valid_df=None):
    """
    Train LambdaRank Model.
    
    Args:
        train_df: DataFrame train (đã có features + label)
        config: Object config (CFG.stage2)
        valid_df: DataFrame valid (optional, dùng để early stopping)
    """
    
    # 1. Xác định Features (Loại bỏ các cột định danh/label)
    ignore_cols = {'customer_id', 'item_id', 'label', 'created_date', 'created_datetime'}
    feature_cols = [c for c in train_df.columns if c not in ignore_cols]
    
    print(f"[Stage2] Training with {len(feature_cols)} features: {feature_cols}")

    # 2. Chuẩn bị dữ liệu cho LightGBM (Cần sort theo Group/Query)
    # Train set
    train_df = train_df.sort_values("customer_id")
    q_train = train_df.groupby("customer_id").size().values
    X_train = train_df[feature_cols]
    y_train = train_df["label"]
    
    lgb_train = lgb.Dataset(X_train, y_train, group=q_train)
    
    # Valid set setup
    valid_sets = [lgb_train]
    valid_names = ['train']
    
    if valid_df is not None:
        valid_df = valid_df.sort_values("customer_id")
        q_valid = valid_df.groupby("customer_id").size().values
        X_valid = valid_df[feature_cols]
        y_valid = valid_df["label"]
        
        lgb_eval = lgb.Dataset(X_valid, y_valid, group=q_valid, reference=lgb_train)
        valid_sets.append(lgb_eval)
        valid_names.append('valid')

    # 3. Setup Parameters (Map từ Config Object)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10, 20],  # Quan trọng: Đánh giá NDCG tại top 10, 20
        "boosting_type": "gbdt",
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "min_data_in_leaf": config.min_data_in_leaf,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "random_state": 42
    }

    # 4. Train với Callbacks
    callbacks = [
        lgb.log_evaluation(period=50)
    ]
    
    # Chỉ thêm early_stopping nếu có tập valid
    if valid_df is not None:
        callbacks.append(lgb.early_stopping(stopping_rounds=50))

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=config.n_estimators,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks
    )
    
    # Trả về cả list features để đảm bảo lúc predict dùng đúng thứ tự
    return model, feature_cols

def predict_stage2(model, test_df, feature_cols=None, top_k=None):
    """
    Predict và Rank lại items.
    
    Args:
        feature_cols: List tên cột features (bắt buộc phải khớp lúc train)
        top_k: Nếu set, chỉ trả về Top K items mỗi user để giảm nhẹ output
    """
    df = test_df.copy()
    
    # Nếu không truyền feature_cols, tự động lấy (rủi ro nếu thứ tự sai)
    if feature_cols is None:
        ignore_cols = {'customer_id', 'item_id', 'label', 'created_date', 'pred_score'}
        feature_cols = [c for c in df.columns if c not in ignore_cols]
    
    # Predict
    df['pred_score'] = model.predict(df[feature_cols])
    
    # Sort theo Score giảm dần
    df_sorted = df.sort_values(['customer_id', 'pred_score'], ascending=[True, False])
    
    # Filter Top K (nếu cần)
    if top_k:
        df_sorted = df_sorted.groupby('customer_id').head(top_k)
        
    return df_sorted