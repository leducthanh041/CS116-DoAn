import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import glob
from config.settings import CFG

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
PATH_SUB_NEW = "/datastore/uittogether2/LuuTru/tienptx/zDoAn/recsys_project/artifacts/submission_jan2025_new.csv"
PATH_SUB_ALL = "/datastore/uittogether2/LuuTru/tienptx/zDoAn/recsys_project/artifacts/submission_jan2025_all.csv"
PATH_GT = "/datastore/uittogether/LuuTru/Thanhld/CS116-DoAn/Phase-2/Pipeline-training2/Stage2/groundtruth.pkl"
PATH_ITEM_DIR = CFG.paths.raw_data_path 
PATH_TRX_DIR = CFG.paths.raw_data_path

# ==========================================
# HELPER: Đọc Parquet Pattern
# ==========================================
def read_parquet_pattern(base_path, keyword, columns=None):
    """
    Đọc file parquet theo pattern. 
    columns: List các cột cần đọc (giúp giảm RAM cực nhiều)
    """
    search_pattern = os.path.join(base_path, f"*{keyword}*.parquet")
    files = glob.glob(search_pattern)
    
    if not files: 
        print(f"Warning: No files found for {keyword} in {base_path}")
        return pd.DataFrame()
        
    dfs = []
    for f in files:
        try:
            # Chỉ đọc các cột cần thiết
            df = pd.read_parquet(f, columns=columns)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# ==========================================
# 1. LOAD STATIC DATA (Kết quả dự đoán & GT)
# ==========================================
@st.cache_data
def load_static_data():
    """Load các file kết quả tĩnh (CSV & Pickle)"""
    print(">>> Loading CSV Results...")
    
    # 1. Load Submission CSVs
    try:
        df_new = pd.read_csv(PATH_SUB_NEW)
        df_all = pd.read_csv(PATH_SUB_ALL)
        
        # Convert ID to string
        for df in [df_new, df_all]:
            df['customer_id'] = df['customer_id'].astype(str)
            df['item_id'] = df['item_id'].astype(str)
            
    except FileNotFoundError as e:
        st.error(f"Thiếu file CSV submission: {e}")
        st.stop()
    
    # 2. Load Ground Truth
    print(">>> Loading Ground Truth...")
    with open(PATH_GT, "rb") as f:
        gt_data = pickle.load(f)
        
    # Xử lý format GT
    if isinstance(gt_data, dict):
        for key in ['gt_test', 'test', 'groundtruth']:
            if key in gt_data:
                gt_data = gt_data[key]
                break
    
    # Chuẩn hóa GT thành dict {user_str: set(item_str)}
    gt_clean = {}
    for u, v in gt_data.items():
        if isinstance(v, (list, tuple, set, np.ndarray)):
            gt_clean[str(u)] = set(str(x) for x in v)
        else:
            gt_clean[str(u)] = {str(v)}
            
    return df_new, df_all, gt_clean

# ==========================================
# 2. LOAD METADATA (Item Info)
# ==========================================
@st.cache_data
def load_metadata():
    """Load thông tin Item"""
    print(">>> Loading Item Metadata...")
    
    # Chỉ đọc các cột cần thiết để tiết kiệm RAM
    cols_needed = ["item_id", "item_name", "category", "category_l1", "category_l2", "category_l3", "brand"]
    
    # Đọc thử 1 file để check cột nào tồn tại
    sample_file = glob.glob(os.path.join(PATH_ITEM_DIR, "*item_chunk*.parquet"))[0]
    sample_cols = pd.read_parquet(sample_file).columns.tolist()
    
    final_cols = [c for c in cols_needed if c in sample_cols]
    
    df_item = read_parquet_pattern(PATH_ITEM_DIR, "item_chunk", columns=final_cols)
    
    if df_item.empty:
        return {}

    # Map ID -> Info
    df_item['item_id'] = df_item['item_id'].astype(str)
    df_item = df_item.drop_duplicates("item_id")
    
    item_map = df_item.set_index("item_id").to_dict(orient="index")
    return item_map

# ==========================================
# 3. GET USER HISTORY (Dynamic Loading)
# ==========================================
def get_user_history_dynamic(user_id):
    """
    Quét file transaction để tìm lịch sử của user_id.
    Để tối ưu: Ta dùng pyarrow để filter ngay khi đọc (push-down predicate) nếu có thể,
    hoặc đọc từng chunk, filter rồi mới concat.
    """
    # Tìm các file transaction
    search_pattern = os.path.join(PATH_TRX_DIR, "*purchase_history_daily_chunk*.parquet")
    files = glob.glob(search_pattern)
    
    user_history_dfs = []
    
    # Các cột cần thiết
    cols = ['customer_id', 'item_id', 'created_date']
    
    for f in files:
        try:
            # Đọc file, chỉ lấy cột cần thiết
            df_chunk = pd.read_parquet(f, columns=cols)
            df_chunk['customer_id'] = df_chunk['customer_id'].astype(str)
            
            # Filter ngay lập tức
            df_user = df_chunk[df_chunk['customer_id'] == str(user_id)]
            
            if not df_user.empty:
                user_history_dfs.append(df_user)
                
        except Exception as e:
            continue
            
    if not user_history_dfs:
        return pd.DataFrame()
        
    df_hist = pd.concat(user_history_dfs, ignore_index=True)
    df_hist['item_id'] = df_hist['item_id'].astype(str)
    
    # Sort theo ngày
    if 'created_date' in df_hist.columns:
        df_hist = df_hist.sort_values('created_date', ascending=False)
        
    return df_hist

# ==========================================
# HELPER UI
# ==========================================
def highlight_hit(row, gt_items):
    color = '#d4edda' if str(row['item_id']) in gt_items else ''
    return [f'background-color: {color}' for _ in row]

def display_recs(df, user_id, item_map, gt_items, title, top_k=10):
    """Hiển thị bảng recommendation"""
    st.subheader(title)
    
    user_recs = df[df['customer_id'] == user_id].copy()
    
    if user_recs.empty:
        st.warning("No recommendations found.")
        return

    user_recs = user_recs.sort_values("pred_score", ascending=False).head(top_k)
    
    # Map info
    user_recs['Item Name'] = user_recs['item_id'].apply(lambda x: item_map.get(x, {}).get('item_name', 'Unknown'))
    user_recs['Category'] = user_recs['item_id'].apply(lambda x: item_map.get(x, {}).get('category', '-'))
    user_recs['Category l1'] = user_recs['item_id'].apply(lambda x: item_map.get(x, {}).get('category_l1', '-'))
    user_recs['Category l2'] = user_recs['item_id'].apply(lambda x: item_map.get(x, {}).get('category_l2', '-'))
    user_recs['Category l3'] = user_recs['item_id'].apply(lambda x: item_map.get(x, {}).get('category_l3', '-'))
    user_recs['Is Hit 🎯'] = user_recs['item_id'].apply(lambda x: '✅' if x in gt_items else '')
    
    cols = ['Is Hit 🎯', 'item_id', 'Item Name', 'Category', 'Category l1', 'Category l2', 'Category l3', 'pred_score']
    
    st.dataframe(
        user_recs[cols].style.apply(lambda x: highlight_hit(x, gt_items), axis=1),
        hide_index=True,
        use_container_width=True
    )
    
    hits = user_recs[user_recs['Is Hit 🎯'] == '✅'].shape[0]
    st.metric(f"Precision @ {top_k}", f"{hits}/{top_k}", delta=f"{hits/top_k:.1%}")

# ==========================================
# MAIN APP
# ==========================================
def main():
    st.set_page_config(page_title="RecSys Lite", layout="wide")
    st.title("📊 RecSys Result Viewer (Lite)")
    
    # 1. Load Resources
    with st.spinner("Loading Resources..."):
        df_new, df_all, gt_clean = load_static_data()
        item_map = load_metadata()
        
    all_users = df_new['customer_id'].unique()
    st.sidebar.success(f"Loaded Results for {len(all_users)} users")
    
    # 2. Select User
    user_mode = st.sidebar.radio("Select User Mode:", ["Random", "Input ID"])
    selected_user = None
    
    if user_mode == "Random":
        if st.sidebar.button("🎲 Pick Random User"):
            selected_user = np.random.choice(all_users)
            st.session_state['lite_user'] = selected_user
        elif 'lite_user' in st.session_state:
            selected_user = st.session_state['lite_user']
    else:
        selected_user = st.sidebar.text_input("Enter Customer ID:")
        
    top_k = st.sidebar.slider("Top K View", 5, 20, 10)

    # 3. Main View
    if selected_user:
        selected_user = str(selected_user).strip()
        st.markdown(f"## 👤 User: `{selected_user}`")
        
        # --- A. HISTORY (Lazy Load) ---
        with st.expander("📜 Purchase History", expanded=True):
            with st.spinner(f"Searching history for {selected_user}..."):
                df_hist = get_user_history_dynamic(selected_user)
            
            if not df_hist.empty:
                st.write(f"**Found {len(df_hist)} transactions**")
                
                # Map info cho history
                hist_display = []
                for _, row in df_hist.iterrows():
                    iid = str(row['item_id'])
                    info = item_map.get(iid, {})
                    hist_display.append({
                        "Date": row.get('created_date', 'N/A'),
                        "Item ID": iid,
                        "Name": info.get("item_name", "N/A"),
                        "Category": info.get("category", "N/A"),
                        "Category l1": info.get("category_l1", "N/A"),
                        "Category l2": info.get("category_l2", "N/A"),
                        "Category l3": info.get("category_l3", "N/A")
                    })
                
                st.dataframe(pd.DataFrame(hist_display), height=300, use_container_width=True)
            else:
                st.warning("⚠️ No history found (Cold Start or Data Missing)")

        st.divider()

        # --- B. RECOMMENDATION & GROUND TRUTH ---
        gt_items = gt_clean.get(selected_user, set())
        
        col1, col2, col3 = st.columns([1.2, 1.2, 0.8])
        
        with col1:
            display_recs(df_new, selected_user, item_map, gt_items, "🆕 New Items Recs", top_k)
            
        with col2:
            display_recs(df_all, selected_user, item_map, gt_items, "🌍 All Items Recs", top_k)
            
        with col3:
            st.subheader("🎯 Ground Truth")
            if gt_items:
                gt_list = []
                for iid in gt_items:
                    gt_list.append({
                        "Item ID": iid,
                        "Name": item_map.get(iid, {}).get("item_name", "N/A"),
                        "Category": item_map.get(iid, {}).get("category", "N/A"),
                        "Category l1": item_map.get(iid, {}).get("category_l1", "N/A"),
                        "Category l2": item_map.get(iid, {}).get("category_l2", "N/A"),
                        "Category l3": item_map.get(iid, {}).get("category_l3", "N/A"),
                        "In New?": "✅" if iid in df_new[df_new['customer_id']==selected_user]['item_id'].values else "",
                        "In All?": "✅" if iid in df_all[df_all['customer_id']==selected_user]['item_id'].values else ""
                    })
                st.dataframe(pd.DataFrame(gt_list), hide_index=True)
            else:
                st.info("User did not buy anything in validation.")

if __name__ == "__main__":
    main()