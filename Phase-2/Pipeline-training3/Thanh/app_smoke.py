import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import glob
import joblib
from config.settings import CFG, get_path

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
PATH_PRED_NEW = get_path("artifacts/predictions_new_item_rec.pkl")
PATH_PRED_ALL = get_path("artifacts/predictions_all_item_rec.pkl")
PATH_GT = "./data/final_groundtruth.pkl"
PATH_MODEL_S1 = get_path("artifacts/stage1_model_base.pkl") 
PATH_ITEM_DIR = CFG.paths.raw_data_path 
PATH_TRX_DIR = CFG.paths.raw_data_path

# ==========================================
# CẤU HÌNH CỘT HIỂN THỊ (ĐÃ BỎ ITEM NAME)
# ==========================================
FIELD_MAP = {
    # "Item Name": "item_name",  <-- ĐÃ BỎ
    "Category": "category",
    "L1": "category_l1",
    "L2": "category_l2",
    "Brand": "brand_final",
    "Age Group": "age_group_final",
    "Price": "price"
}

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def read_parquet_pattern(base_path, keyword, columns=None):
    search_pattern = os.path.join(base_path, f"*{keyword}*.parquet")
    files = glob.glob(search_pattern)
    if not files: return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            # Đọc parquet
            df = pd.read_parquet(f, columns=columns)
            # [FIX] Ép kiểu string ngay lập tức sau khi đọc để tránh mất số 0
            # Nếu cột item_id có trong df, ép về str
            if 'item_id' in df.columns:
                df['item_id'] = df['item_id'].astype(str)
            if 'customer_id' in df.columns:
                df['customer_id'] = df['customer_id'].astype(str)
            dfs.append(df)
        except Exception: continue
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def convert_dict_to_df(pkl_path):
    if not os.path.exists(pkl_path): return None
    with open(pkl_path, "rb") as f: pred_dict = pickle.load(f)
        
    rows = []
    for user, items in pred_dict.items():
        for rank, item in enumerate(items, start=1):
            rows.append({
                "customer_id": str(user),
                "item_id": str(item), # Giữ nguyên string từ pickle
                "rank": rank,
                "pred_score": 1.0 / rank 
            })
            
    if not rows: return pd.DataFrame(columns=["customer_id", "item_id", "rank", "pred_score"])
    return pd.DataFrame(rows)

# ==========================================
# LOAD DATA
# ==========================================
@st.cache_data
def load_data_pkl_v4(): # Đổi tên v4 để clear cache
    """Load Predictions & GT (Fix ID format)"""
    df_new = convert_dict_to_df(PATH_PRED_NEW)
    if df_new is None: st.error(f"❌ Thiếu file: {PATH_PRED_NEW}"); st.stop()
        
    df_all = convert_dict_to_df(PATH_PRED_ALL)
    if df_all is None: st.error(f"❌ Thiếu file: {PATH_PRED_ALL}"); st.stop()

    if not os.path.exists(PATH_GT): st.error(f"❌ Thiếu file GT: {PATH_GT}"); st.stop()
    with open(PATH_GT, "rb") as f: gt_data = pickle.load(f)
        
    gt_clean = {}
    
    # --- Xử lý Ground Truth ---
    if isinstance(gt_data, pd.DataFrame):
        u_col, i_col = 'customer_id', 'item_id'
        if not gt_data.empty:
            first = gt_data[i_col].iloc[0]
            if isinstance(first, (list, np.ndarray, set)):
                gt_data = gt_data.explode(i_col)
        
        gt_data = gt_data.dropna(subset=[u_col, i_col])
        
        # [FIX QUAN TRỌNG] Chỉ ép về str, KHÔNG ép về float/int trung gian
        # Điều này giữ nguyên '00123' thay vì biến thành '123'
        gt_data[u_col] = gt_data[u_col].astype(str)
        gt_data[i_col] = gt_data[i_col].astype(str)
        
        # Nếu data bị dính .0 (do file gốc lưu float), dùng regex để xóa
        # gt_data[i_col] = gt_data[i_col].str.replace(r'\.0$', '', regex=True)

        gt_clean = gt_data.groupby(u_col)[i_col].apply(set).to_dict()
        
    elif isinstance(gt_data, dict):
        for key in ['gt_test', 'test', 'groundtruth']:
            if key in gt_data: gt_data = gt_data[key]; break
        for u, v in gt_data.items():
            u_str = str(u)
            if isinstance(v, (list, tuple, set, np.ndarray)):
                # Đảm bảo item trong list cũng là string chuẩn
                gt_clean[u_str] = set(str(x) for x in v)
            else:
                gt_clean[u_str] = {str(v)}
                
    return df_new, df_all, gt_clean

@st.cache_resource
def load_train_user_set():
    try:
        if os.path.exists(PATH_MODEL_S1):
            stage1 = joblib.load(PATH_MODEL_S1)
            # Ép kiểu set keys về string để so sánh chuẩn
            return set(str(k) for k in stage1.user_id_to_index_.keys())
        return set()
    except: return set()

@st.cache_data
def load_metadata():
    # Chỉ load item_id và các cột trong FIELD_MAP
    cols_needed = ["item_id"] + list(FIELD_MAP.values())
    
    pattern = os.path.join(PATH_ITEM_DIR, "*item_chunk*.parquet")
    files = glob.glob(pattern)
    if not files: return {}
    
    try:
        sample_cols = pd.read_parquet(files[0]).columns.tolist()
        final_cols = [c for c in cols_needed if c in sample_cols]
        
        df_item = read_parquet_pattern(PATH_ITEM_DIR, "item_chunk", columns=final_cols)
        if df_item.empty: return {}
        
        # [FIX] Ép kiểu string ngay khi load metadata
        df_item['item_id'] = df_item['item_id'].astype(str)
        
        df_item = df_item.drop_duplicates("item_id")
        return df_item.set_index("item_id").to_dict(orient="index")
    except: return {}

def get_user_history_dynamic(user_id):
    search_pattern = os.path.join(PATH_TRX_DIR, "*purchase_history_daily_chunk*.parquet")
    files = glob.glob(search_pattern)
    user_history_dfs = []
    cols = ['customer_id', 'item_id', 'created_date']
    
    # user_id input đã là string
    target_uid = str(user_id)
    
    for f in files:
        try:
            df_chunk = pd.read_parquet(f, columns=cols)
            # [FIX] Ép kiểu trước khi so sánh
            df_chunk['customer_id'] = df_chunk['customer_id'].astype(str)
            df_chunk['item_id'] = df_chunk['item_id'].astype(str)
            
            df_user = df_chunk[df_chunk['customer_id'] == target_uid]
            if not df_user.empty: user_history_dfs.append(df_user)
        except: continue
        
    if not user_history_dfs: return pd.DataFrame()
    df_hist = pd.concat(user_history_dfs, ignore_index=True)
    if 'created_date' in df_hist.columns:
        df_hist = df_hist.sort_values('created_date', ascending=False)
    return df_hist

# ==========================================
# UI LOGIC
# ==========================================
def highlight_hit(row, gt_items):
    return ['background-color: #d4edda' if str(row['item_id']) in gt_items else '' for _ in row]

def display_recs(df, user_id, item_map, gt_items, title, top_k=10):
    st.subheader(title)
    user_recs = df[df['customer_id'] == user_id].copy()
    
    if user_recs.empty:
        st.info("No recommendations generated.")
        return

    if 'rank' in user_recs.columns:
        user_recs = user_recs.sort_values("rank", ascending=True).head(top_k)
    else:
        user_recs = user_recs.head(top_k)
    
    # Map Metadata
    for display_name, meta_key in FIELD_MAP.items():
        user_recs[display_name] = user_recs['item_id'].apply(
            lambda x: item_map.get(x, {}).get(meta_key, 'N/A')
        )
        
    user_recs['Is Hit 🎯'] = user_recs['item_id'].apply(lambda x: '✅' if x in gt_items else '')
    
    # Chọn cột hiển thị (bỏ Item Name)
    cols_show = ['Is Hit 🎯', 'rank', 'item_id'] + list(FIELD_MAP.keys())
    final_cols = [c for c in cols_show if c in user_recs.columns]
    
    st.dataframe(
        user_recs[final_cols].style.apply(lambda x: highlight_hit(x, gt_items), axis=1),
        hide_index=True,
        use_container_width=True
    )
    
    hits = user_recs[user_recs['Is Hit 🎯'] == '✅'].shape[0]
    st.metric(f"Precision @ {top_k}", f"{hits}/{top_k}", delta=f"{hits/top_k:.1%}")

# ==========================================
# MAIN
# ==========================================
def main():
    st.set_page_config(page_title="RecSys Auditor", layout="wide")
    st.title("❄️ RecSys Cold Start Auditor")
    
    if st.sidebar.button("🧹 Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    with st.spinner("Loading Data..."):
        df_new, df_all, gt_clean = load_data_pkl_v4()
        item_map = load_metadata()
        train_user_set = load_train_user_set()
        
    test_users = set(df_new['customer_id'].unique())
    cold_users = sorted(list(test_users - train_user_set))
    warm_users = sorted(list(test_users.intersection(train_user_set)))
    
    st.sidebar.header("Filter")
    filter_type = st.sidebar.radio("Show:", [f"Cold ({len(cold_users)})", f"Warm ({len(warm_users)})", "All"])
    
    if "Cold" in filter_type: pool = cold_users
    elif "Warm" in filter_type: pool = warm_users
    else: pool = sorted(list(test_users))
    
    user_mode = st.sidebar.radio("Mode:", ["Random", "Input ID"])
    selected_user = None
    
    if user_mode == "Random":
        if st.sidebar.button("🎲 Pick Random"):
            if pool: st.session_state['u'] = np.random.choice(pool)
        if 'u' in st.session_state and st.session_state['u'] in pool:
            selected_user = st.session_state['u']
    else:
        selected_user = st.sidebar.text_input("Customer ID:")

    if selected_user:
        selected_user = str(selected_user).strip()
        is_cold = selected_user not in train_user_set
        st.markdown(f"## 👤 `{selected_user}` ({'🥶 Cold' if is_cold else '🔥 Warm'})")
        
        # --- History Section ---
        if not is_cold:
            with st.expander("History"):
                df_h = get_user_history_dynamic(selected_user)
                if not df_h.empty:
                    for display_name, meta_key in FIELD_MAP.items():
                        df_h[display_name] = df_h['item_id'].apply(
                            lambda x: item_map.get(x, {}).get(meta_key, 'N/A')
                        )
                    st.dataframe(df_h, use_container_width=True)
        else:
            st.info("No History (Cold Start)")
            
        # --- Ground Truth Section ---
        gt_items = gt_clean.get(selected_user, set())
        st.subheader(f"🛒 Real Purchase ({len(gt_items)})")
        if gt_items:
            gt_rows = []
            for i in gt_items:
                row = {"item_id": i}
                for display_name, meta_key in FIELD_MAP.items():
                    row[display_name] = item_map.get(i, {}).get(meta_key, 'N/A')
                gt_rows.append(row)
            st.dataframe(pd.DataFrame(gt_rows), use_container_width=True)
        
        # --- Predictions Section ---
        c1, c2 = st.columns(2)
        with c1: display_recs(df_new, selected_user, item_map, gt_items, "🆕 New Items Rec")
        with c2: display_recs(df_all, selected_user, item_map, gt_items, "🌍 All Items Rec")

if __name__ == "__main__":
    main()