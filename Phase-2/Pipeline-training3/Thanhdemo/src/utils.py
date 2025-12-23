import json
import os
import joblib
import pandas as pd
from pathlib import Path

# Setup caching
CACHE_DIR = "./artifacts/cache"
os.makedirs(CACHE_DIR, exist_ok=True)
memory = joblib.Memory(CACHE_DIR, verbose=0)

class Config:
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.cfg = json.load(f)

    def __getitem__(self, item):
        return self.cfg[item]

    @property
    def data_split(self):
        return self.cfg['data_split']
    
    @property
    def stage1_params(self):
        return self.cfg['stage1']

def save_parquet_cache(df: pd.DataFrame, path: str):
    """Lưu dataframe ra parquet để cache thủ công nếu cần"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def load_parquet_cache(path: str):
    if os.path.exists(path):
        print(f"Loading cached file: {path}")
        return pd.read_parquet(path)
    return None