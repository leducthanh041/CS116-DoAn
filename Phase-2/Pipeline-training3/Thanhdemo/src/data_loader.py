import os
import polars as pl
import pandas as pd
from datetime import datetime, timedelta
from src.utils import memory

from glob import glob

def read_parquet_folder(path, keyword): 
    """Hàm đọc file parquet generic""" 
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.parquet') and keyword in f] 
    if not files: return None 
    return pl.concat([pl.read_parquet(f) for f in files])


@memory.cache # Cache lại kết quả split để lần sau chạy nhanh hơn
def load_and_split_data(raw_data_path, hist_end_date, hist_days, recent_days):
    print("Loading data from parquet...")
    df_user = read_parquet_folder(raw_data_path, 'sale_pers.user_chunk_')
    df_item = read_parquet_folder(raw_data_path, 'sale_pers.item_chunk_')
    df_trx = read_parquet_folder(raw_data_path, 'sale_pers.purchase_history_daily_chunk_')

    # --- SỬA ĐOẠN NÀY: Ép kiểu customer_id và item_id về String (Utf8) ---
    df_trx = (
        df_trx
        .select(["customer_id", "item_id", "created_date"])
        .drop_nulls()
        .with_columns([
            pl.col("created_date").cast(pl.Datetime).alias("created_datetime"),
            pl.col("customer_id").cast(pl.Utf8), # Ép về String
            pl.col("item_id").cast(pl.Utf8)      # Ép về String
        ])
        .sort("created_datetime")
    )
    
    # Ép kiểu cho df_item và df_user luôn để khớp
    if df_item is not None:
        df_item = df_item.with_columns(pl.col("item_id").cast(pl.Utf8))
    if df_user is not None:
        df_user = df_user.with_columns(pl.col("customer_id").cast(pl.Utf8))

    # --- SPLIT LOGIC (Đồng bộ) ---
    hist_end = datetime.strptime(hist_end_date, "%Y-%m-%d")
    hist_start = hist_end - timedelta(days=hist_days)
    
    recent_start = hist_end + timedelta(days=1)
    recent_end = recent_start + timedelta(days=recent_days)

    print(f"Splitting Data: Train[{hist_start.date()} -> {hist_end.date()}] | Valid[{recent_start.date()} -> {recent_end.date()}]")

    df_train_pl = df_trx.filter(
        (pl.col("created_datetime") >= hist_start) & (pl.col("created_datetime") <= hist_end)
    )
    
    df_valid_pl = df_trx.filter(
        (pl.col("created_datetime") >= recent_start) & (pl.col("created_datetime") <= recent_end)
    )

    # Convert to Pandas for Model Processing
    return df_train_pl.to_pandas(), df_valid_pl.to_pandas(), df_item.to_pandas(), df_user.to_pandas()