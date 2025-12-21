import numpy as np
import pandas as pd

def calculate_metrics_at_k(pred_df, gt_dict, train_history, k=10, filter_bought_items=True):
    """
    Tính Precision/Recall/NDCG với tùy chọn lọc bỏ hàng đã mua.
    
    Args:
        pred_df: DataFrame [customer_id, item_id, pred_score]
        gt_dict: Dict {user_id: set(item_ids)} - Ground Truth (Valid/Test)
        train_history: Dict {user_id: set(item_ids)} - Lịch sử mua hàng trong tập Train
        k: Top K items
        filter_bought_items: 
            - True: Không tính items đã mua trong quá khứ (Recommend New Items)
            - False: Chấp nhận items đã mua (Recommend All/Re-purchase)
    """
    
    # 1. Convert Prediction DataFrame to Dict
    # Sort by score desc và lấy top K
    pred_map = (
        pred_df.sort_values(["customer_id", "pred_score"], ascending=[True, False])
        .groupby("customer_id")["item_id"]
        .apply(lambda x: list(x)[:k])
        .to_dict()
    )
    
    metrics = {
        "all": {"p": [], "r": [], "ndcg": []},
        "warm": {"p": [], "r": [], "ndcg": []},
        "cold": {"p": [], "r": [], "ndcg": []}
    }
    
    # 2. Iterate qua Ground Truth (User trong tập Valid/Test)
    for user, truth_items in gt_dict.items():
        user = str(user)
        truth_items = set(str(x) for x in truth_items)
        
        # Lấy lịch sử mua hàng của user (nếu có)
        hist_items = set(str(x) for x in train_history.get(user, []))
        is_cold = len(hist_items) == 0
        
        # --- LOGIC QUAN TRỌNG: Filter Bought Items ---
        relevant_items = truth_items.copy()
        if filter_bought_items:
            # Loại bỏ những item đã có trong lịch sử khỏi tập Ground Truth mong muốn
            # (Tức là: Nếu user mua lại món cũ, ta KHÔNG tính đó là thành công cho bài toán New Item Rec)
            relevant_items = relevant_items - hist_items
            
            # Nếu sau khi lọc mà không còn item nào để recommend (user chỉ toàn mua lại đồ cũ)
            # Thì skip user này (hoặc coi như mẫu số = 0 tùy định nghĩa business)
            if len(relevant_items) == 0:
                continue
        # ---------------------------------------------

        # Lấy danh sách dự đoán
        recs = [str(x) for x in pred_map.get(user, [])]
        
        # Cắt Top K (đã cắt ở trên rồi nhưng check lại cho chắc)
        recs = recs[:k]
        
        # --- Tính Metrics ---
        hits = len(set(recs) & relevant_items)
        
        precision = hits / k
        recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0
        
        # NDCG
        dcg = 0.0
        for i, item in enumerate(recs):
            if item in relevant_items:
                dcg += 1.0 / np.log2(i + 2)
        
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_items), k)))
        ndcg = dcg / idcg if idcg > 0 else 0
        
        # --- Aggregation ---
        group = "cold" if is_cold else "warm"
        
        # Add to All
        metrics["all"]["p"].append(precision)
        metrics["all"]["r"].append(recall)
        metrics["all"]["ndcg"].append(ndcg)
        
        # Add to Group
        metrics[group]["p"].append(precision)
        metrics[group]["r"].append(recall)
        metrics[group]["ndcg"].append(ndcg)

    # 3. Summarize Results
    final_res = {}
    mode_str = "NEW" if filter_bought_items else "ALL"
    
    for group in ["all", "warm", "cold"]:
        if metrics[group]["p"]:
            final_res[f"{group}_{mode_str}_P@{k}"] = np.mean(metrics[group]["p"])
            final_res[f"{group}_{mode_str}_R@{k}"] = np.mean(metrics[group]["r"])
            final_res[f"{group}_{mode_str}_NDCG@{k}"] = np.mean(metrics[group]["ndcg"])
        else:
            final_res[f"{group}_{mode_str}_P@{k}"] = 0.0
            
    final_res["n_eval_users"] = len(metrics["all"]["p"])
    
    return final_res