import numpy as np
import pandas as pd

def calculate_metrics_at_k(pred_df, gt_dict, train_history, k=10, filter_bought_items=True):
    """
    Tính Recall@K và NDCG@K (Bỏ qua Precision vì đã tính riêng).
    
    Args:
        pred_df: DataFrame [customer_id, item_id, pred_score]
        gt_dict: Dict {user_id: set(item_ids)} - Ground Truth
        train_history: Dict {user_id: set(item_ids)} - Lịch sử mua hàng (để lọc)
        k: Top K items
        filter_bought_items: True/False
    """
    
    # [QUAN TRỌNG] Ép kiểu ID về String để tránh lỗi lệch kiểu (Int vs Str)
    pred_df = pred_df.copy()
    pred_df["customer_id"] = pred_df["customer_id"].astype(str)
    pred_df["item_id"] = pred_df["item_id"].astype(str)

    # 1. Convert Prediction DataFrame to Dict (Sort by Score -> Top K)
    pred_map = (
        pred_df.sort_values(["customer_id", "pred_score"], ascending=[True, False])
        .groupby("customer_id")["item_id"]
        .apply(lambda x: list(x)[:k])
        .to_dict()
    )
    
    # Chỉ lưu trữ Recall và NDCG
    metrics = {
        "all": {"r": [], "ndcg": []},
        "warm": {"r": [], "ndcg": []},
        "cold": {"r": [], "ndcg": []}
    }
    
    # 2. Iterate qua Ground Truth
    for user, truth_items in gt_dict.items():
        user = str(user)
        truth_items = set(str(x) for x in truth_items)
        
        # Lấy lịch sử mua hàng (để xác định Cold/Warm và lọc)
        hist_items = set(str(x) for x in train_history.get(user, []))
        is_cold = len(hist_items) == 0
        
        # --- Logic Filter Bought Items ---
        relevant_items = truth_items.copy()
        if filter_bought_items:
            relevant_items = relevant_items - hist_items
            # Nếu sau khi lọc mà không còn item nào (User chỉ mua lại đồ cũ) -> Skip
            if len(relevant_items) == 0:
                continue
        # ---------------------------------

        # Lấy danh sách dự đoán
        recs = pred_map.get(user, []) # List đã được cắt Top K ở bước 1
        
        # --- Tính Recall ---
        hits = len(set(recs) & relevant_items)
        recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0.0
        
        # --- Tính NDCG ---
        dcg = 0.0
        for i, item in enumerate(recs):
            if item in relevant_items:
                dcg += 1.0 / np.log2(i + 2)
        
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_items), k)))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        # --- Aggregation ---
        group = "cold" if is_cold else "warm"
        
        # Add to All
        metrics["all"]["r"].append(recall)
        metrics["all"]["ndcg"].append(ndcg)
        
        # Add to Group (Warm/Cold)
        metrics[group]["r"].append(recall)
        metrics[group]["ndcg"].append(ndcg)

    # 3. Summarize Results
    final_res = {}
    mode_str = "NEW" if filter_bought_items else "ALL"
    
    for group in ["all", "warm", "cold"]:
        # Chỉ tính trung bình nếu list không rỗng
        if metrics[group]["r"]:
            final_res[f"{group}_{mode_str}_R@{k}"] = np.mean(metrics[group]["r"])
            final_res[f"{group}_{mode_str}_NDCG@{k}"] = np.mean(metrics[group]["ndcg"])
        else:
            final_res[f"{group}_{mode_str}_R@{k}"] = 0.0
            final_res[f"{group}_{mode_str}_NDCG@{k}"] = 0.0
            
    return final_res

def precision_at_k_custom(pred_dict, gt_dict, hist_dict, filter_bought_items=True, K=10):
    """
    Tính Precision@K tùy chỉnh theo logic của bạn.
    Input đều là Dictionary: {user_id: [item_id_1, item_id_2, ...]}
    """
    precisions = []
    cold_start_users = []
    
    # Duyệt qua tất cả user có trong Ground Truth (Tập Test)
    for user in gt_dict.keys():
        user = str(user)
        
        # Lấy danh sách dự đoán (Pred)
        if user not in pred_dict:
            # Nếu user không được predict -> P@K = 0
            precisions.append(0.0)
            continue
            
        pred_items = pred_dict[user][:K]
        
        # Lấy danh sách lịch sử (Hist)
        # [MODIFIED] Dùng .get() để không skip User Cold Start (Hist rỗng)
        user_hist = hist_dict.get(user, set())
        
        # Ground Truth Items
        gt_items = set(gt_dict[user])
        
        # Logic lọc hàng đã mua
        relevant_items = gt_items.copy()
        if filter_bought_items:
            relevant_items = relevant_items - set(user_hist)
            # Nếu sau khi lọc mà rỗng (User chỉ mua lại đồ cũ) -> Skip hoặc tính là 0 tùy logic
            if not relevant_items:
                continue

        # Tính Hits
        hits = len(set(pred_items) & relevant_items)
        precisions.append(hits / K)
        
        # Note user cold start (không có lịch sử)
        if not user_hist:
            cold_start_users.append(user)
            
    avg_precision = np.mean(precisions) if precisions else 0.0
    return avg_precision, cold_start_users