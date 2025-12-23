import sys
import os
import glob  # <--- Cần thêm thư viện này để quét file chunk

# --- SETUP PATH ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# ------------------

import pandas as pd
import numpy as np
import re
import ast
import joblib
from tqdm import tqdm
from config.settings import get_path, CFG  # Import CFG để lấy đường dẫn data

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def read_parquet_pattern(base_path, keyword):
    """Đọc và gộp tất cả các file parquet chứa keyword"""
    search_pattern = os.path.join(base_path, f"*{keyword}*.parquet")
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"❌ Warning: No files found for keyword '{keyword}' in {base_path}")
        return pd.DataFrame()
        
    print(f"   -> Found {len(files)} files for '{keyword}'")
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)

def parse_age_group(age_str):
    """Parse chuỗi age_group thành (min, max)"""
    if pd.isna(age_str) or age_str == "":
        return (0, 999)
    try:
        if str(age_str).startswith("["):
            lst = ast.literal_eval(age_str)
            text = " ".join(lst)
        else:
            text = str(age_str)
        text = text.upper()
        
        # 0-12M
        range_match = re.search(r'(\d+)\s*-\s*(\d+)\s*M', text)
        if range_match:
            return (int(range_match.group(1)), int(range_match.group(2)))
        # Từ 12M
        from_match = re.search(r'TỪ\s*(\d+)\s*M', text)
        if from_match:
            return (int(from_match.group(1)), 999)
        # Dưới 12M
        under_match = re.search(r'DƯỚI\s*(\d+)\s*M', text)
        if under_match:
            return (0, int(under_match.group(1)))
            
        return (0, 999)
    except:
        return (0, 999)

def estimate_baby_dob(df_full_history, df_item):
    """Ước tính ngày sinh của bé từ lịch sử mua hàng"""
    print(">>> Estimating Baby DOB from full history...")
    
    # Chỉ lấy các cột cần thiết để merge cho nhẹ
    item_cols = ['item_id', 'category']
    if 'age_group_final' in df_item.columns:
        item_cols.append('age_group_final')
        
    df_merged = df_full_history.merge(df_item[item_cols].drop_duplicates('item_id'), on='item_id', how='inner')
    
    if 'created_date' in df_merged.columns:
        df_merged['created_datetime'] = pd.to_datetime(df_merged['created_date'])
    
    keywords = {
        'STEP 1': 1, 'SỐ 1': 1, '0-6M': 1, 'NEWBORN': 0, 'SƠ SINH': 0,
        'STEP 2': 7, 'SỐ 2': 7, '6-12M': 7,
        'STEP 3': 13, 'SỐ 3': 13, '1-3Y': 13,
        'BẦU': -9
    }
    
    user_dob = {}
    df_merged = df_merged.sort_values('created_datetime')
    
    # GroupBy User
    for uid, group in tqdm(df_merged.groupby('customer_id'), desc="Estimating DOB"):
        estimated_dob = None
        for _, row in group.iterrows():
            # Ghép text để search keyword
            full_text = str(row.get('category', '')).upper()
            if 'age_group_final' in row:
                full_text += " " + str(row['age_group_final']).upper()
            
            detected_months = None
            for kw, val in keywords.items():
                if kw in full_text:
                    detected_months = val
                    break
            
            if detected_months is not None:
                # DOB = Ngày mua - Số tháng tuổi
                estimated_dob = row['created_datetime'] - pd.DateOffset(months=detected_months)
                break 
        
        if estimated_dob:
            user_dob[str(uid)] = estimated_dob
            
    return user_dob

# ==========================================
# 2. MAIN LOGIC
# ==========================================
def build_features(df_item, df_history):
    """
    Hàm chính để xử lý và lưu artifact.
    Nhận vào DataFrame thay vì đường dẫn file để linh hoạt với chunk.
    """
    # 1. Item Target Age Parsing
    print(">>> Parsing Item Target Age...")
    item_age_dict = {}
    
    df_item['item_id'] = df_item['item_id'].astype(str)
    
    for _, row in tqdm(df_item.iterrows(), total=len(df_item), desc="Parsing Item Ages"):
        iid = str(row['item_id'])
        if 'age_group_final' in row:
            item_age_dict[iid] = parse_age_group(row['age_group_final'])
        else:
            item_age_dict[iid] = (0, 999)
            
    # 2. User Baby DOB Estimation
    user_dob_dict = estimate_baby_dob(df_history, df_item)
    
    # 3. Save Artifacts
    os.makedirs(get_path("artifacts"), exist_ok=True)
    
    path_age = get_path("artifacts/item_target_age.pkl")
    path_dob = get_path("artifacts/user_baby_dob.pkl")
    
    joblib.dump(item_age_dict, path_age)
    joblib.dump(user_dob_dict, path_dob)
    
    print(f"\n✅ SUCCESS! Artifacts saved:")
    print(f"   - {path_age} ({len(item_age_dict)} items)")
    print(f"   - {path_dob} ({len(user_dob_dict)} users estimated)")

if __name__ == "__main__":
    print(">>> LOADING DATA CHUNKS...")
    
    # 1. Load Item Data
    df_item = read_parquet_pattern(CFG.paths.raw_data_path, "item_chunk")
    
    # 2. Load History Data
    df_history = read_parquet_pattern(CFG.paths.raw_data_path, "purchase_history_daily_chunk")
    
    if df_item.empty or df_history.empty:
        print("❌ Error: Could not load data. Check paths in config.")
    else:
        # 3. Run Build
        build_features(df_item, df_history)