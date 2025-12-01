import os
import polars as pl
import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

# ===== PATH FILE =====
TRAIN_PATH = "./sale_pers.train_data_0.parquet"
EVAL_PATH  = "./sale_pers.eva_data_0.parquet"
K = 10


# ============================================================
# 1. LOAD DATA
# ============================================================
def load_train_dataset(path: str):
    """
    Đọc parquet train, tách X_train, y_train và trả thêm danh sách feature_cols.
    Bỏ 2 cột ID customer_id, item_id và cột Y khỏi X.
    """
    print(f"[LOAD TRAIN] {path}")
    df = pl.read_parquet(path)

    target_col = "Y"
    drop_cols = ["customer_id", "item_id", target_col]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    print("  #rows      :", df.height)
    print("  #features  :", len(feature_cols))
    print("  features   :", feature_cols)

    X_train = df.select(feature_cols).to_pandas()
    y_train = df[target_col].to_pandas()

    return X_train, y_train, feature_cols


def load_eval_dataset(path: str, feature_cols):
    """
    Đọc parquet eval, tách X_eval, y_eval, customer_id, item_id.
    Giả định eval có cùng schema feature như train (hoặc superset).
    """
    print(f"[LOAD EVAL] {path}")
    df = pl.read_parquet(path)

    target_col = "Y"

    # Đảm bảo chỉ lấy các cột feature có trong eval
    used_feature_cols = [c for c in feature_cols if c in df.columns]
    if len(used_feature_cols) != len(feature_cols):
        missing = set(feature_cols) - set(used_feature_cols)
        print("[CẢNH BÁO] Eval thiếu một số feature, thiếu:", missing)

    print("  #rows      :", df.height)
    print("  #features used:", len(used_feature_cols))

    X_eval = df.select(used_feature_cols).to_pandas()
    y_eval = df[target_col].to_pandas()
    customer_ids = df["customer_id"].to_pandas().values
    item_ids     = df["item_id"].to_pandas().values

    return X_eval, y_eval, customer_ids, item_ids


# ============================================================
# 2. MODEL
# ============================================================
def build_model(max_iter: int = 500) -> Pipeline:
    """
    Logistic Regression trong pipeline với StandardScaler.
    Dùng class_weight='balanced' vì label vẫn lệch.
    """
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=max_iter,
                    class_weight="balanced",
                    solver="lbfgs",
                    verbose=1,  # in tiến trình train theo iteration
                ),
            ),
        ]
    )
    return pipe


# ============================================================
# 3. METRICS: DCG, Precision@K, NDCG@K
# ============================================================
def dcg_at_k(relevances: np.ndarray) -> float:
    """
    Tính DCG cho 1 vector relevance (0/1) đã cắt tới top-k.
    """
    if len(relevances) == 0:
        return 0.0
    rel = np.asarray(relevances, dtype=float)
    discounts = np.log2(np.arange(2, rel.size + 2))
    gains = (2.0 ** rel - 1.0)
    return float(np.sum(gains / discounts))


def build_user_topk_table(
    customer_ids: np.ndarray,
    item_ids: np.ndarray,
    y_true: np.ndarray,
    scores: np.ndarray,
    k: int = 10,
) -> pd.DataFrame:
    """
    Tạo bảng với mỗi dòng là 1 user:

    Columns:
        - customer_id
        - top_k_items: dict {item_id: score}
        - precision_at_k
        - ndcg_at_k
    """
    # Gom vào DataFrame pandas để groupby dễ
    df_scores = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "item_id": item_ids,
            "y_true": y_true,
            "score": scores,
        }
    )

    rows = []
    precisions = []
    ndcgs = []

    # Group theo user
    for user_id, group in tqdm(
        df_scores.groupby("customer_id"),
        desc=f"Tính top-{k}, Precision@{k}, NDCG@{k}",
        ncols=100,
    ):
        # sort theo score giảm dần
        group_sorted = group.sort_values("score", ascending=False)

        if group_sorted.shape[0] == 0:
            continue

        # Lấy top-k
        topk = group_sorted.head(k)
        actual_k = topk.shape[0]
        rel = topk["y_true"].values  # 0/1

        # Precision@k
        prec = rel.sum() / float(actual_k)
        precisions.append(prec)

        # NDCG@k
        dcg = dcg_at_k(rel)
        # IDCG: ranking tốt nhất theo y_true
        ideal_rel = np.sort(group["y_true"].values)[::-1]
        ideal_rel_at_k = ideal_rel[:k]
        idcg = dcg_at_k(ideal_rel_at_k)
        if idcg == 0:
            ndcg = 0.0
        else:
            ndcg = dcg / idcg
        ndcgs.append(ndcg)

        # Top-k items dictionary: {item_id: score}
        topk_items_dict = {
            int(row.item_id): float(row.score)
            for _, row in topk.iterrows()
        }

        rows.append(
            {
                "customer_id": user_id,
                "top_k_items": topk_items_dict,
                "precision_at_k": prec,
                "ndcg_at_k": ndcg,
            }
        )

    user_topk_df = pd.DataFrame(rows)

    # In thêm global mean metric
    mean_prec = float(np.mean(precisions)) if precisions else 0.0
    mean_ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0

    print(f"\n[GLOBAL] Mean Precision@{k}: {mean_prec:.4f}")
    print(f"[GLOBAL] Mean NDCG@{k}     : {mean_ndcg:.4f}")

    return user_topk_df


# ============================================================
# 4. TRAIN & EVAL
# ============================================================
def train_and_eval(train_path: str, eval_path: str, k: int = 10):
    # ----- LOAD DATA -----
    X_train, y_train, feature_cols = load_train_dataset(train_path)
    X_eval, y_eval, cust_eval, item_eval = load_eval_dataset(eval_path, feature_cols)

    # ----- BUILD MODEL -----
    model = build_model(max_iter=500)

    # ----- TRAIN -----
    print("\n=== BẮT ĐẦU TRAIN LOGISTIC REGRESSION ===")
    model.fit(X_train, y_train)
    print("=== TRAIN XONG ===\n")

    # ----- EVAL CLASSIFICATION (THAM KHẢO) -----
    print("=== EVAL (Classification metrics trên EVA) ===")
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1]

    auc = roc_auc_score(y_eval, y_prob)
    print(f"ROC-AUC: {auc:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_eval, y_pred))

    print("\nClassification report:")
    print(classification_report(y_eval, y_pred))

    # ----- EVAL RANKING (TOP-K) -----
    print(f"\n=== EVAL RANKING (Precision@{k}, NDCG@{k}) ===")
    user_topk_df = build_user_topk_table(
        customer_ids=cust_eval,
        item_ids=item_eval,
        y_true=y_eval.values,
        scores=y_prob,
        k=k,
    )

    # In thử vài dòng đầu
    print("\n=== MỘT VÀI DÒNG ĐẦU CỦA BẢNG USER TOP-K ===")
    print(user_topk_df.head(10))

    # OPTIONAL: lưu ra file nếu bạn muốn
    # user_topk_df.to_parquet("./sale_pers.eval_user_topk.parquet", index=False)

    return model, user_topk_df


# ============================================================
# 5. ENTRYPOINT
# ============================================================
if __name__ == "__main__":
    model, user_topk_df = train_and_eval(TRAIN_PATH, EVAL_PATH, k=K)