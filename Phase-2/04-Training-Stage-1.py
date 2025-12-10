import os
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GridSearchCV

from tqdm.auto import tqdm  # <- thêm dòng này


def read_parquet_user(train_path: str):
    # Lấy tất cả các file parquet trong thư mục
    files = [os.path.join(train_path, f) for f in os.listdir(train_path) if f.endswith('.parquet')]
    
    # Phân loại các file theo loại tên
    user_chunk_files = [file for file in files if 'user_chunk' in file]
        
    # Đọc các file riêng biệt thành DataFrame
    user_chunk_df = pl.concat([pl.read_parquet(file) for file in user_chunk_files]) if user_chunk_files else None
        
    # Trả về một dictionary chứa các DataFrame
    return user_chunk_df

def read_parquet_item(train_path: str):
    # Lấy tất cả các file parquet trong thư mục
    files = [os.path.join(train_path, f) for f in os.listdir(train_path) if f.endswith('.parquet')]
    
    # Phân loại các file theo loại tên
    item_chunk_files = [file for file in files if 'item_chunk' in file]
        
    # Đọc các file riêng biệt thành DataFrame
    item_chunk_df = pl.concat([pl.read_parquet(file) for file in item_chunk_files]) if item_chunk_files else None
        
    # Trả về một dictionary chứa các DataFrame
    return item_chunk_df

def read_parquet_transaction(train_path: str):
    # Lấy tất cả các file parquet trong thư mục
    files = [os.path.join(train_path, f) for f in os.listdir(train_path) if f.endswith('.parquet')]
    
    # Phân loại các file theo loại tên
    purchase_chunk_files = [file for file in files if 'purchase_history_daily_chunk' in file]
        
    # Đọc các file riêng biệt thành DataFrame
    purchase_chunk_df = pl.concat([pl.read_parquet(file) for file in purchase_chunk_files]) if purchase_chunk_files else None
        
    # Trả về một dictionary chứa các DataFrame
    return purchase_chunk_df

def split_and_save_parquet(df, num_files, output_dir, type):
    """
    Tách DataFrame thành nhiều file Parquet và lưu vào thư mục đích.
    
    :param df: DataFrame cần tách
    :param num_files: Số lượng file Parquet muốn tách
    :param output_dir: Thư mục lưu các file Parquet
    """
    # Đảm bảo thư mục tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Tính số dòng mỗi file sẽ có
    num_rows = df.height
    rows_per_file = num_rows // num_files

    # Tách DataFrame thành các phần và lưu mỗi phần vào một file Parquet
    for i in range(num_files):
        start_row = i * rows_per_file
        # Đảm bảo phần cuối cùng sẽ chứa tất cả các dòng còn lại
        end_row = (i + 1) * rows_per_file if i < num_files - 1 else num_rows
        
        # Tách phần DataFrame
        split_df = df[start_row:end_row]
        
        # Lưu phần DataFrame vào file .parquet
        file_path = os.path.join(output_dir, f"sale_pers.{type}_chunk_{i}.parquet")
        split_df.write_parquet(file_path)
        print(f"Đã lưu file: {file_path}")

df_user = read_parquet_user("./preprocessed-dataset")
df_user.head()

df_transaction = read_parquet_transaction("./preprocessed-dataset")
df_transaction.head()

df_item = read_parquet_item("./preprocessed-dataset")
df_item.head()

# df_transaction: Polars DataFrame gốc

df_trx = (
    df_transaction
    .select(["customer_id", "item_id", "created_date"])
    .drop_nulls(["customer_id", "item_id", "created_date"])
    .with_columns(
        pl.col("created_date")
        .cast(pl.Datetime)              # chuyển Date/String → Datetime
        .alias("created_datetime")
    )
    .sort("created_datetime")          # sắp xếp thời gian tăng dần
)

print("Tổng số dòng transaction:", df_trx.height)
print(df_trx.head())
print(df_trx.dtypes)

# Chỉ lấy năm 2024
df_trx_2024 = df_trx.filter(
    pl.col("created_datetime").dt.year() == 2024
)

# Train = tháng 1 → 11/2024
df_train_pl = df_trx_2024.filter(
    pl.col("created_datetime").dt.month().is_between(1, 11, closed="both")
)

# Valid = tháng 12/2024
df_valid_pl = df_trx_2024.filter(
    pl.col("created_datetime").dt.month() == 12
)

print("Số dòng train (Polars):", df_train_pl.height)
print("Số dòng valid (Polars):", df_valid_pl.height)

df_train = df_train_pl.to_pandas()
df_valid = df_valid_pl.to_pandas()

print("Số dòng train (pandas):", len(df_train))
print("Số dòng valid (pandas):", len(df_valid))
print(df_train.head())
print(df_valid.head())


class ItemItemCFStage1(BaseEstimator):
    """
    Stage 1: Item-Item Collaborative Filtering với:
      - Ma trận user-item có trọng số (log_count + ưu tiên category/age_group)
      - TF-IDF weighting trên ma trận user-item
      - Cosine similarity trên vector item sau TF-IDF
      - Cold-start user -> top-k item phổ biến nhất

    df_train : pandas DataFrame, chứa transaction train
    df_valid : pandas DataFrame, chứa transaction valid
    df_item  : pandas DataFrame, chứa metadata item

    Cột bắt buộc:
      - df_train: [user_col, item_col], nên có "quantity" nếu muốn trọng số tốt hơn
      - df_item : [item_col, "category_l2", "age_group_final"]
    """

    def __init__(
        self,
        df_train,
        df_valid,
        df_item,
        user_col="customer_id",
        item_col="item_id",
        weight_type="log_count",      # "binary" | "count" | "log_count" | "rel_freq"
        alpha_cat=0.5,
        alpha_age=0.5,
        n_neighbors=50,
        k_eval=100,
        use_tqdm=True,
    ):
        # dữ liệu & cấu hình
        self.df_train = df_train
        self.df_valid = df_valid
        self.df_item = df_item

        self.user_col = user_col
        self.item_col = item_col

        # siêu tham số ma trận user-item
        self.weight_type = weight_type    # base: binary/count/log_count/rel_freq
        self.alpha_cat = alpha_cat        # độ mạnh ưu tiên category_l2
        self.alpha_age = alpha_age        # độ mạnh ưu tiên age_group_final

        # siêu tham số CF
        self.n_neighbors = n_neighbors
        self.k_eval = k_eval
        self.use_tqdm = use_tqdm

    # =========================
    # Helper: build ma trận user-item có trọng số + side info item
    # =========================
    def _build_user_item_matrix(self):
        """
        Tạo ma trận user-item (CSR) từ df_train với trọng số w_ui:

            base = log1p(sum_qty(u,i)) hoặc biến thể
            w_ui = base * (1 + alpha_cat * cat_pref + alpha_age * age_pref)

        Trong đó:
          - cat_pref = P(category_l2(i) | user u)
          - age_pref = P(age_group_final(i) | user u)
        """

        df = self.df_train.copy()

        # chỉ giữ cột cần thiết
        cols_needed = [self.user_col, self.item_col]
        if "quantity" in df.columns:
            cols_needed.append("quantity")
        df = df[cols_needed]

        # aggregate theo (user, item)
        if "quantity" in df.columns:
            agg = (
                df.groupby([self.user_col, self.item_col], as_index=False)
                .agg(
                    n_interactions=(self.item_col, "size"),
                    sum_qty=("quantity", "sum"),
                )
            )
        else:
            agg = (
                df.groupby([self.user_col, self.item_col], as_index=False)
                .agg(
                    n_interactions=(self.item_col, "size"),
                )
            )
            agg["sum_qty"] = agg["n_interactions"]

        # ===== join thêm thông tin item: category_l2, age_group_final =====
        item_cols = [self.item_col, "category_l2", "age_group_final"]
        df_item_small = self.df_item[item_cols].drop_duplicates(subset=[self.item_col])
        agg = agg.merge(df_item_small, on=self.item_col, how="left")

        # fill NA cho feature dùng trong groupby
        agg["category_l2"] = agg["category_l2"].fillna("__UNK_CAT2__")
        agg["age_group_final"] = agg["age_group_final"].fillna("__UNK_AGE__")

        # ===== tính base weight từ sum_qty =====
        base_raw = agg["sum_qty"].astype(float)

        if self.weight_type == "binary":
            base = np.ones_like(base_raw, dtype=np.float32)
        elif self.weight_type == "count":
            base = base_raw.values.astype(np.float32)
        elif self.weight_type == "log_count":
            base = np.log1p(base_raw.values).astype(np.float32)
        elif self.weight_type == "rel_freq":
            # freq tương đối trong lịch sử user: freq_ui / tổng freq user
            user_total = agg.groupby(self.user_col)["sum_qty"].transform("sum")
            base = (base_raw / user_total).values.astype(np.float32)
        else:
            raise ValueError(f"Unknown weight_type: {self.weight_type}")

        agg["base_weight"] = base

        # ===== tính cat_pref & age_pref trên sum_qty =====
        # tổng quantity theo user (mẫu số)
        agg["user_total_qty"] = agg.groupby(self.user_col)["sum_qty"].transform("sum")

        # tổng quantity theo (user, category_l2)
        agg["user_cat_l2_qty"] = agg.groupby(
            [self.user_col, "category_l2"]
        )["sum_qty"].transform("sum")

        # tổng quantity theo (user, age_group_final)
        agg["user_ageg_qty"] = agg.groupby(
            [self.user_col, "age_group_final"]
        )["sum_qty"].transform("sum")

        # preference = freq_ui_feature / tổng của user
        agg["cat_pref"] = agg["user_cat_l2_qty"] / agg["user_total_qty"]
        agg["age_pref"] = agg["user_ageg_qty"] / agg["user_total_qty"]

        # fill NA nếu có
        agg["cat_pref"] = agg["cat_pref"].fillna(0.0)
        agg["age_pref"] = agg["age_pref"].fillna(0.0)

        # ===== final weight w_ui =====
        factor = 1.0 + self.alpha_cat * agg["cat_pref"] + self.alpha_age * agg["age_pref"]
        w_ui = agg["base_weight"].values * factor.values.astype(np.float32)

        agg["value"] = w_ui.astype(np.float32)

        # ===== mã hóa user_id, item_id -> index =====
        user_cat = agg[self.user_col].astype("category")
        item_cat = agg[self.item_col].astype("category")

        self.user_index_to_id_ = list(user_cat.cat.categories)
        self.item_index_to_id_ = list(item_cat.cat.categories)

        self.user_id_to_index_ = {uid: idx for idx, uid in enumerate(self.user_index_to_id_)}
        self.item_id_to_index_ = {iid: idx for idx, iid in enumerate(self.item_index_to_id_)}

        user_codes = user_cat.cat.codes.to_numpy()
        item_codes = item_cat.cat.codes.to_numpy()
        data = agg["value"].to_numpy(dtype=np.float32)

        n_users = len(self.user_index_to_id_)
        n_items = len(self.item_index_to_id_)

        ui_matrix = csr_matrix(
            (data, (user_codes, item_codes)),
            shape=(n_users, n_items),
            dtype=np.float32
        )

        # lịch sử train cho từng user (set index item)
        self.user_history_ = {}
        for u_idx, i_idx in zip(user_codes, item_codes):
            self.user_history_.setdefault(u_idx, set()).add(i_idx)

        # độ phổ biến item (để fallback cold-start user)
        item_popularity = np.asarray(ui_matrix.sum(axis=0)).ravel()
        self.item_popularity_ = item_popularity
        self.popular_item_indices_ = np.argsort(-item_popularity)

        return ui_matrix

    # =========================
    # Helper: build hàng xóm item trên TF-IDF
    # =========================
    def _build_item_neighbors(self, ui_matrix):
        """
        Áp TF-IDF lên ma trận user-item rồi fit NearestNeighbors trên vector item.
        """

        self.tfidf_ = TfidfTransformer(
            norm="l2",
            use_idf=True,
            sublinear_tf=True,
        )
        ui_tfidf = self.tfidf_.fit_transform(ui_matrix)

        # Mỗi item = 1 vector = 1 dòng trong ma trận chuyển vị
        X_items = ui_tfidf.T

        self.nn_model_ = NearestNeighbors(
            n_neighbors=self.n_neighbors + 1,  # +1 để bao gồm chính nó
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )
        self.nn_model_.fit(X_items)

        distances, indices = self.nn_model_.kneighbors(X_items, return_distance=True)
        sims = 1.0 - distances  # cosine similarity

        # Bỏ neighbor thứ 0 (chính item)
        self.item_neighbors_ = indices[:, 1:]
        self.item_neighbor_sims_ = sims[:, 1:]

    # =========================
    # Helper: recommend cho 1 user-index (warm-start)
    # =========================
    def _recommend_for_user_index(self, user_idx, top_k=None):
        """
        Recommend cho 1 user warm-start (có history trong train).
        Trả về list index item nội bộ.
        """
        if top_k is None:
            top_k = self.k_eval

        history = self.user_history_.get(user_idx, None)
        if not history:
            return []

        candidate_scores = {}

        for item_i in history:
            neighbors = self.item_neighbors_[item_i]
            sims = self.item_neighbor_sims_[item_i]

            for nbr_idx, sim in zip(neighbors, sims):
                if nbr_idx in history:
                    continue
                candidate_scores[nbr_idx] = candidate_scores.get(nbr_idx, 0.0) + float(sim)

        if not candidate_scores:
            return []

        sorted_candidates = sorted(
            candidate_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        top_items = [idx for idx, _ in sorted_candidates[:top_k]]
        return top_items

    # =========================
    # API sklearn
    # =========================
    def fit(self, X=None, y=None):
        """
        Build model từ df_train:
        - Xây ma trận user-item có trọng số (dùng item metadata)
        - Áp TF-IDF
        - Fit item-item NearestNeighbors
        """
        ui_matrix = self._build_user_item_matrix()
        self._build_item_neighbors(ui_matrix)
        return self

    def score(self, X=None, y=None):
        """
        Tính mean Recall@k_eval trên df_valid.

        - Warm-start user (có lịch sử train)  -> dùng item-item CF
        - Cold-start user (chỉ có ở valid)   -> fallback top-k item phổ biến nhất
        """
        if not hasattr(self, "user_history_"):
            raise RuntimeError("Model chưa fit. Hãy gọi fit() trước score().")

        if self.df_valid.empty:
            return 0.0

        df_val = self.df_valid[[self.user_col, self.item_col]].drop_duplicates().copy()

        # map item_id valid sang index item train
        item_cat_val = pd.Categorical(df_val[self.item_col], categories=self.item_index_to_id_)
        df_val["item_idx"] = item_cat_val.codes

        # bỏ item cold-start (chưa có trong train)
        df_val = df_val[df_val["item_idx"] != -1]

        if df_val.empty:
            return 0.0

        # ground truth: user_id -> set(item_idx) trong valid
        user_to_true_items = (
            df_val.groupby(self.user_col)["item_idx"]
            .apply(lambda s: set(s.to_list()))
            .to_dict()
        )

        recalls = []
        n_eval_users = 0

        iterator = user_to_true_items.items()
        if self.use_tqdm:
            iterator = tqdm(
                iterator,
                total=len(user_to_true_items),
                desc=f"Evaluating users (k={self.k_eval}, w={self.weight_type}, nn={self.n_neighbors})",
                leave=False,
            )

        for user_id, true_items in iterator:
            # warm vs cold
            if user_id in self.user_id_to_index_:
                user_idx = self.user_id_to_index_[user_id]
                rec_items = self._recommend_for_user_index(user_idx, top_k=self.k_eval)
            else:
                # cold-start user -> top-k item phổ biến nhất
                rec_items = list(self.popular_item_indices_[: self.k_eval])

            if not rec_items:
                recalls.append(0.0)
                n_eval_users += 1
                continue

            rec_set = set(rec_items)
            inter = len(rec_set & true_items)
            recall_u = inter / len(true_items)
            recalls.append(recall_u)
            n_eval_users += 1

        if n_eval_users == 0:
            return 0.0

        mean_recall = float(np.mean(recalls))
        return mean_recall

    # =========================
    # Convenience: recommend theo user_id gốc (online)
    # =========================
    def recommend_for_user_id(self, user_id, top_k=None):
        """
        Recommend top_k item_id cho một user_id (id gốc).
        - Warm-start: dùng CF
        - Cold-start: top-k item phổ biến nhất
        """
        if top_k is None:
            top_k = self.k_eval

        if user_id in self.user_id_to_index_:
            user_idx = self.user_id_to_index_[user_id]
            item_idx_list = self._recommend_for_user_index(user_idx, top_k=top_k)
        else:
            item_idx_list = list(self.popular_item_indices_[: top_k])

        item_ids = [self.item_index_to_id_[idx] for idx in item_idx_list]
        return item_ids

df_item_pd = df_item.to_pandas()

X_dummy = np.zeros((1, 1))


base_estimator = ItemItemCFStage1(
    df_train=df_train,
    df_valid=df_valid,
    df_item=df_item_pd,
    user_col="customer_id",
    item_col="item_id",
    use_tqdm=True,
)

param_grid = {
    "weight_type": ["log_count", "rel_freq", "count", "binary"],   # có thể thử thêm "count", "binary" nếu muốn
    "n_neighbors": [20, 50, 100],
    "k_eval": [200, 500, 1000],
    # bạn cũng có thể thử tune alpha_cat, alpha_age nhưng nên để sau
}

# param_grid = {
#     "weight_type": ["rel_freq"],   # có thể thử thêm "count", "binary" nếu muốn
#     "n_neighbors": [100],
#     "k_eval": [1000],
#     # bạn cũng có thể thử tune alpha_cat, alpha_age nhưng nên để sau
# }
cv = [(np.arange(len(X_dummy)), np.arange(len(X_dummy)))]

grid = GridSearchCV(
    estimator=base_estimator,
    param_grid=param_grid,
    cv=cv,
    scoring=None,
    refit=True,
    verbose=2,
    n_jobs=1,
)

grid.fit(X_dummy)

print("Best params:", grid.best_params_)
print("Best mean Recall@K:", grid.best_score_)

best_model = grid.best_estimator_

def recommend_for_user(model: ItemItemCFStage1, user_id, top_k=100):
    """
    Gợi ý top_k item_id cho 1 user_id (id gốc).
    Sử dụng model đã được fit (best_model).
    """
    # Map user_id → user_idx (index nội bộ)
    user_idx = model.user_id_to_index_.get(user_id, None)
    if user_idx is None:
        # cold-start user: hiện Stage 1 này không xử lý
        # có thể return danh sách item phổ biến ở ngoài
        return []

    item_idx_list = model._recommend_for_user_index(user_idx, top_k=top_k)
    item_ids = [model.item_index_to_id_[idx] for idx in item_idx_list]
    return item_ids

# Ví dụ test nhanh:
if len(df_train) > 0:
    some_user = df_train["customer_id"].iloc[2]
    candidates = recommend_for_user(best_model, some_user, top_k=10)
    print("User:", some_user)
    print("Top-10 candidate:", candidates)
