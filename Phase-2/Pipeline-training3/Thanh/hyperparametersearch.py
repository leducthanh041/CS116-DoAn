import random
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from tqdm.auto import tqdm

# Import từ project của bạn
from config.settings import CFG, get_path
from src import (
    load_and_split_data, 
    ItemItemCFStage1, 
    calculate_metrics_at_k
)

class OverrideConfig:
    """
    Helper class để giả lập object config của Stage 1.
    """
    def __init__(self, base_config_dict, overrides):
        self.__dict__.update(base_config_dict)
        self.__dict__.update(overrides)

    def __getattr__(self, item):
        return self.__dict__.get(item)

def run_random_search():
    # ==========================================
    # 1. CẤU HÌNH RANDOM SEARCH
    # ==========================================
    N_TRIALS = 30  # Số lần chạy thử nghiệm (Budget)
    
    # Định nghĩa không gian tìm kiếm (Search Space)
    # Bạn liệt kê các giá trị có thể chọn, thuật toán sẽ random trong list này
    param_space = {
        "n_neighbors": [50, 100, 150, 200, 300],
        "alpha_l1_rec": [0.01, 0.05, 0.1, 0.15, 0.2],
        "weight_type": ["log_count", "rel_freq"],
        "ui_recency_lambda": [0.005, 0.01, 0.02, 0.05],
        # "alpha_l1_cnt": [0.05, 0.1, 0.2], # Thêm nếu cần
    }

    print(f"Project Root: {get_path('')}")
    print(f"Mode: Random Search | Budget: {N_TRIALS} trials")

    # ==========================================
    # 2. LOAD DATA (Chỉ 1 lần)
    # ==========================================
    print(">>> Loading Data...")
    df_train, df_valid, df_item, df_user = load_and_split_data(
        CFG.paths.raw_data_path,
        CFG.data_split.hist_end_date,
        CFG.data_split.hist_days,
        CFG.data_split.recent_days
    )
    
    print(">>> Preparing Evaluation Data...")
    gt_valid = df_valid.groupby("customer_id")["item_id"].apply(set).to_dict()
    valid_users = list(gt_valid.keys())
    train_users_set = set(df_train['customer_id'].astype(str).unique())
    
    # Config gốc
    base_stage1_cfg = vars(CFG.stage1) if hasattr(CFG.stage1, '__dict__') else dict(CFG.stage1)
    
    results = []
    tried_configs = set() # Để tránh random trùng lặp
    
    # ==========================================
    # 3. LOOP RANDOM SEARCH
    # ==========================================
    print(f"\n>>> Starting Random Search on {len(valid_users)} validation users...")
    
    # Dùng tqdm để hiện thanh tiến trình
    pbar = tqdm(total=N_TRIALS, desc="Random Search")
    
    count = 0
    while count < N_TRIALS:
        # A. Sampling Parameters
        current_params = {}
        for k, v in param_space.items():
            current_params[k] = random.choice(v)
            
        # Kiểm tra trùng lặp (Hash dict thành string để lưu vào set)
        config_key = json.dumps(current_params, sort_keys=True)
        if config_key in tried_configs:
            continue # Nếu đã chạy rồi thì skip, random lại
        
        tried_configs.add(config_key)
        count += 1
        pbar.update(1)
        
        try:
            # B. Config Override & Train
            run_cfg = OverrideConfig(base_stage1_cfg, current_params)
            
            model = ItemItemCFStage1(run_cfg, df_item, df_user)
            model.fit(df_train)
            
            # C. Recommend (Top K nhỏ để test nhanh)
            eval_k = 100 
            model.top_k = eval_k 
            preds = model.recommend_candidates(valid_users)
            
            # D. Evaluate
            metrics = calculate_metrics_at_k(
                preds, gt_valid, train_users_set, k=eval_k
            )
            
            # E. Log Result
            record = current_params.copy()
            record.update({
                "recall_all": metrics.get(f"all_R@{eval_k}", 0),
                "ndcg_all": metrics.get(f"all_NDCG@{eval_k}", 0),
                "recall_cold": metrics.get(f"cold_R@{eval_k}", 0),
                "recall_warm": metrics.get(f"warm_R@{eval_k}", 0),
                # "trial_id": count
            })
            
            # In ngắn gọn kết quả
            tqdm.write(f"Trial {count}: {current_params} -> Result: Recall@{eval_k}={record['recall_all']:.4f} | Cold Recall={record['recall_cold']:.4f}")
            results.append(record)
            
        except Exception as e:
            tqdm.write(f"   [Error] Trial {count} failed: {e}")
            continue
            
    pbar.close()

    # ==========================================
    # 4. SAVE RESULTS
    # ==========================================
    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = get_path("logs")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        out_file = os.path.join(out_dir, f"random_search_stage1_{timestamp}.csv")
        
        df_res = pd.DataFrame(results)
        df_res = df_res.sort_values(by="recall_all", ascending=False)
        
        df_res.to_csv(out_file, index=False)
        print(f"\n>>> Search Complete! Results saved to: {out_file}")
        print("Top 3 Best Configs:")
        print(df_res.head(3))
    else:
        print("\n>>> No results collected.")

if __name__ == "__main__":
    run_random_search()