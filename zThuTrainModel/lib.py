import os
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# region: Read Parquet Files
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

def read_parquet_purchase(train_path: str):
    # Lấy tất cả các file parquet trong thư mục
    files = [os.path.join(train_path, f) for f in os.listdir(train_path) if f.endswith('.parquet')]
    
    # Phân loại các file theo loại tên
    purchase_chunk_files = [file for file in files if 'purchase_history_daily_chunk' in file]
        
    # Đọc các file riêng biệt thành DataFrame
    purchase_chunk_df = pl.concat([pl.read_parquet(file) for file in purchase_chunk_files]) if purchase_chunk_files else None
        
    # Trả về một dictionary chứa các DataFrame
    return purchase_chunk_df
# endregion

# region: Save Parquet Files
def split_and_save_parquet(df, num_files, output_dir, type_file):
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
        file_path = os.path.join(output_dir, f"sale_pers.{type_file}_data_{i}.parquet")
        split_df.write_parquet(file_path)
        print(f"Đã lưu file: {file_path}")
# endregion

import polars as pl
from datetime import datetime

def build_feature_label(
    transactions_lf: pl.LazyFrame,  # transaction_df.lazy()
    items_lf: pl.LazyFrame,         # item_df.lazy()
    users_lf: pl.LazyFrame,         # user_df.lazy(), hiện không dùng
    begin: datetime,
    end: datetime,
    begin_recent: datetime,
    end_recent: datetime,
) -> pl.LazyFrame:
    """
    Tạo bảng Feature - Label rút gọn cho bài toán khuyến nghị top-k.

    Output:
        - customer_id
        - item_id
        - brand_count      (X_1)
        - age_group_count  (X_2)
        - category_count   (X_3)
        - Y (0/1)
    """

    # 1. Filter giai đoạn HIST và RECENT trên bảng giao dịch
    hist_lf = (
        transactions_lf
        .filter(
            pl.col("created_date").is_between(begin, end, closed="both")
        )
    )

    recent_lf = (
        transactions_lf
        .filter(
            pl.col("created_date").is_between(begin_recent, end_recent, closed="both")
        )
    )

    # 2. Thuộc tính item cần thiết từ item_df
    #    - dùng: brand, age_group_final, category
    item_attrs = (
        items_lf
        .select([
            "item_id",
            "brand",
            "age_group_final",
            "category",
        ])
    )

    # 3. Gắn thuộc tính item vào HIST
    hist_enriched = (
        hist_lf
        .join(item_attrs, on="item_id", how="left")
    )

    # 4. Tính 3 feature trong HIST

    # 4.1. X_1: số lần user mua brand của item hiện tại trong HIST
    brand_count = (
        hist_enriched
        .group_by(["customer_id", "brand"])
        .agg(pl.len().alias("brand_count"))
    )

    # 4.2. X_2: số lần user mua age_group_final của item hiện tại trong HIST
    age_group_count = (
        hist_enriched
        .group_by(["customer_id", "age_group_final"])
        .agg(pl.len().alias("age_group_count"))
    )

    # 4.3. X_3: số lần user mua category của item hiện tại trong HIST
    category_count = (
        hist_enriched
        .group_by(["customer_id", "category"])
        .agg(pl.len().alias("category_count"))
    )

    # 5. Xây tập candidate (customer_id, item_id) từ HIST ∪ RECENT
    hist_pairs = hist_lf.select(["customer_id", "item_id"]).unique()
    recent_pairs = recent_lf.select(["customer_id", "item_id"]).unique()

    candidate_pairs = pl.concat([hist_pairs, recent_pairs]).unique()

    # 6. Gắn thông tin item (brand, age_group_final, category) vào candidate
    candidate_enriched = (
        candidate_pairs
        .join(item_attrs, on="item_id", how="left")
    )

    # 7. Join 3 bảng đếm vào candidate
    features = (
        candidate_enriched
        # join brand_count
        .join(
            brand_count,
            on=["customer_id", "brand"],
            how="left",
        )
        # join age_group_count
        .join(
            age_group_count,
            on=["customer_id", "age_group_final"],
            how="left",
        )
        # join category_count
        .join(
            category_count,
            on=["customer_id", "category"],
            how="left",
        )
        # fill null = 0 cho các count (user chưa mua brand/age_group/category đó trong HIST)
        .with_columns([
            pl.col("brand_count").fill_null(0),
            pl.col("age_group_count").fill_null(0),
            pl.col("category_count").fill_null(0),
        ])
    )

    # 8. Tạo label Y từ RECENT: (customer_id, item_id) có giao dịch trong RECENT -> Y=1
    labels = recent_pairs.with_columns(
        pl.lit(1).alias("Y")
    )

    feature_label_lf = (
        features
        .join(labels, on=["customer_id", "item_id"], how="left")
        .with_columns(
            pl.col("Y").fill_null(0).cast(pl.Int8)
        )
        .select([
            "customer_id",
            "item_id",
            "brand_count",      # X_1
            "age_group_count",  # X_2
            "category_count",   # X_3
            "Y",
        ])
    )

    return feature_label_lf

