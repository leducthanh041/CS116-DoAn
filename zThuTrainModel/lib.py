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

def build_feature_label(
    transactions_lf: pl.LazyFrame,  # purchase_df.lazy()
    items_lf: pl.LazyFrame,         # item_df.lazy()
    users_lf: pl.LazyFrame,         # chưa dùng, để đúng spec
    begin_hist: datetime,
    end_hist: datetime,
    begin_recent: datetime,
    end_recent: datetime,
) -> pl.LazyFrame:
    """
    Tạo bảng Feature - Label cho bài toán khuyến nghị top-k.

    Output columns:
        - customer_id
        - item_id
        - brand_counts
        - age_counts
        - category_counts
        - segment_counts            (# lần mua cùng (category_l1, segment_name) với item hiện tại)
        - target_user_group_counts
        - time_since_last_purchase_in_B_category
        - Y
    """

    # 0. Làm sạch transactions_lf: bỏ cột segment_name_right nếu tồn tại (do join trước đó sinh ra)
    tx_cols = transactions_lf.columns
    if "segment_name_right" in tx_cols:
        base_tx = transactions_lf.drop("segment_name_right")
    else:
        base_tx = transactions_lf

    # 1. Filter giai đoạn HIST và RECENT trên bảng giao dịch
    hist_lf = (
        base_tx
        .filter(
            pl.col("created_date").is_between(begin_hist, end_hist, closed="both")
        )
    )

    recent_lf = (
        base_tx
        .filter(
            pl.col("created_date").is_between(begin_recent, end_recent, closed="both")
        )
    )

    # 2. Thuộc tính item cơ bản (từ item_df)
    item_attrs = items_lf.select([
        "item_id",
        "brand_final",
        "age_bucket_final",
        "category",
        "category_l1",
        "target_user_group_final",
    ])

    # 3. Mapping item_id -> segment_name đại diện (mode trong HIST)
    item_segment = (
        hist_lf
        .group_by("item_id")
        .agg(pl.col("segment_name").mode().alias("segment_name_list"))
        .with_columns(
            pl.col("segment_name_list").list.first().alias("segment_name")
        )
        .select(["item_id", "segment_name"])
    )

    # 4. HIST đã gắn đầy đủ: brand/age/category/category_l1/segment_name
    hist_enriched = (
        hist_lf
        .join(item_attrs,   on="item_id", how="left")
        .join(item_segment, on="item_id", how="left")
    )

    # 5. Feature 1: brand_counts
    brand_counts = (
        hist_enriched
        .group_by(["customer_id", "brand_final"])
        .agg(pl.len().alias("brand_counts"))
    )

    # 6. Feature 2: age_counts
    age_counts = (
        hist_enriched
        .group_by(["customer_id", "age_bucket_final"])
        .agg(pl.len().alias("age_counts"))
    )

    # 7. Feature 3: category_counts
    category_counts = (
        hist_enriched
        .group_by(["customer_id", "category"])
        .agg(pl.len().alias("category_counts"))
    )

    # 8. Feature: target_user_group_counts
    target_user_group_counts = (
        hist_enriched
        .group_by(["customer_id", "target_user_group_final"])
        .agg(pl.len().alias("target_user_group_counts"))
    )

    # 9. Feature 4: segment_counts theo (customer, category_l1, segment_name)
    segment_counts = (
        hist_enriched
        .group_by(["customer_id", "category_l1", "segment_name"])
        .agg(pl.len().alias("segment_counts"))
    )

    # 10. Feature: time_since_last_purchase_in_B_category
    #     - lấy lần mua gần nhất theo (customer_id, category) trong HIST
    last_cat_purchase = (
        hist_enriched
        .group_by(["customer_id", "category"])
        .agg(
            pl.col("created_date").max().alias("last_purchase_date")
        )
    )

    # 11. Xây tập candidate (customer_id, item_id) từ HIST ∪ RECENT
    hist_pairs = hist_lf.select(["customer_id", "item_id"]).unique()
    recent_pairs = recent_lf.select(["customer_id", "item_id"]).unique()

    candidate_pairs = pl.concat([hist_pairs, recent_pairs]).unique()

    # 12. Enrich candidate với thuộc tính item & segment_name đại diện
    candidate_enriched = (
        candidate_pairs
        .join(item_attrs,   on="item_id", how="left")
        .join(item_segment, on="item_id", how="left")
    )

    # 13. Join tất cả các bảng count + last_cat_purchase để tạo feature set
    features = (
        candidate_enriched
        # brand_counts
        .join(
            brand_counts,
            on=["customer_id", "brand_final"],
            how="left",
        )
        # age_counts
        .join(
            age_counts,
            on=["customer_id", "age_bucket_final"],
            how="left",
        )
        # category_counts
        .join(
            category_counts,
            on=["customer_id", "category"],
            how="left",
        )
        # segment_counts — join theo (customer_id, category_l1, segment_name)
        .join(
            segment_counts,
            on=["customer_id", "category_l1", "segment_name"],
            how="left",
        )
        # target_user_group_counts
        .join(
            target_user_group_counts,
            on=["customer_id", "target_user_group_final"],
            how="left",
        )
        # last_cat_purchase để tính time_since_last_purchase_in_B_category
        .join(
            last_cat_purchase,
            on=["customer_id", "category"],
            how="left",
        )
        # fill null = 0 cho các count
        .with_columns([
            pl.col("brand_counts").fill_null(0),
            pl.col("age_counts").fill_null(0),
            pl.col("category_counts").fill_null(0),
            pl.col("segment_counts").fill_null(0),
            pl.col("target_user_group_counts").fill_null(0),
        ])
        # tính số ngày từ end_hist đến lần mua gần nhất trong cùng category
        .with_columns(
            (
                (pl.lit(end_hist) - pl.col("last_purchase_date"))
                .dt.total_days()
            ).alias("time_since_last_purchase_in_B_category")
        )
        # khách chưa từng mua category đó → 9999
        .with_columns(
            pl.col("time_since_last_purchase_in_B_category").fill_null(9999)
        )
    )

    # 14. Tạo label Y từ RECENT: (customer_id, item_id) có giao dịch trong RECENT -> Y=1
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
            "brand_counts",
            "age_counts",
            "category_counts",
            "segment_counts",
            "target_user_group_counts",
            "time_since_last_purchase_in_B_category",
            "Y",
        ])
    )

    return feature_label_lf