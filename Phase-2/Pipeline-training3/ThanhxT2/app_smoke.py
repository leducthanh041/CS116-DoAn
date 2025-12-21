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
PATH_SUB_NEW = get_path("artifacts/submission_jan2025_new.csv")
PATH_SUB_ALL = get_path("artifacts/submission_jan2025_all.csv")
PATH_GT = "/datastore/uittogether/LuuTru/Thanhld/CS116-DoAn/Phase-2/Pipeline-training3/Thanh/data/final_groundtruth.pkl"
PATH_MODEL_S1 = get_path("artifacts/stage1_model_base.pkl") # Cần file này để check warm users
PATH_ITEM_DIR = CFG.paths.raw_data_path 
PATH_TRX_DIR = CFG.paths.raw_data_path

# ==========================================
# HELPER: Đọc Parquet Pattern
# ==========================================
def read_parquet_pattern(base_path, keyword, columns=None):
    search_pattern = os.path.join(base_path, f"*{keyword}*.parquet")
    files = glob.glob(search_pattern)
    if not files: return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f, columns=columns)
            dfs.append(df)
        except Exception: continue
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# ==========================================
# LOAD DATA
# ==========================================
@st.cache_data
def load_static_data():
    """Load kết quả submission và Ground Truth (Hỗ trợ explode List/Array items)"""
    print(">>> Loading CSV Results...")
    try:
        # Load CSV kết quả dự đoán
        df_new = pd.read_csv(PATH_SUB_NEW, dtype={'customer_id': str, 'item_id': str})
        df_all = pd.read_csv(PATH_SUB_ALL, dtype={'customer_id': str, 'item_id': str})
    except FileNotFoundError as e:
        st.error(f"Thiếu file CSV submission: {e}")
        st.stop()
    
    print(">>> Loading Ground Truth...")
    with open(PATH_GT, "rb") as f:
        gt_data = pickle.load(f)
        
    gt_clean = {}

    # =========================================================
    # CASE 1: Dữ liệu là DataFrame (Đã fix logic Explode)
    # =========================================================
    if isinstance(gt_data, pd.DataFrame):
        print(f"   [INFO] Detected DataFrame GT with {len(gt_data)} rows.")
        u_col = 'customer_id'
        i_col = 'item_id'
        
        # 1. Kiểm tra cột
        if u_col not in gt_data.columns or i_col not in gt_data.columns:
            st.error(f"DataFrame GroundTruth thiếu cột {u_col} hoặc {i_col}")
            st.stop()

        # 2. [QUAN TRỌNG] Xử lý List/Array trong cột item_id (Explode)
        if not gt_data.empty:
            first_val = gt_data[i_col].iloc[0]
            # Nếu là list, array, set -> Bung ra thành từng dòng
            if isinstance(first_val, (list, np.ndarray, set)):
                print(f"   [INFO] Detected aggregated items ({type(first_val)}). Exploding...")
                gt_data = gt_data.explode(i_col)
        
        # 3. Lọc bỏ dòng rỗng sau khi explode
        gt_data = gt_data.dropna(subset=[u_col, i_col])

        # 4. Ép kiểu String & Xử lý số float (để tránh 123.0 != 123)
        gt_data[u_col] = gt_data[u_col].astype(str)
        try:
            # Convert float -> int -> str (vd: 123.0 -> 123 -> "123")
            gt_data[i_col] = gt_data[i_col].astype(float).astype(np.int64).astype(str)
        except:
            # Fallback nếu không phải số
            gt_data[i_col] = gt_data[i_col].astype(str)

        # 5. GroupBy User và gom Item thành Set
        gt_clean = gt_data.groupby(u_col)[i_col].apply(set).to_dict()

    # =========================================================
    # CASE 2: Dữ liệu là Dictionary (Legacy)
    # =========================================================
    elif isinstance(gt_data, dict):
        # Unwrap nếu bị lồng key
        for key in ['gt_test', 'test', 'groundtruth']:
            if key in gt_data:
                gt_data = gt_data[key]
                break
        
        # Chuẩn hóa ID
        for u, v in gt_data.items():
            user_str = str(u)
            if isinstance(v, (list, tuple, set, np.ndarray)):
                gt_clean[user_str] = set(str(x) for x in v)
            else:
                gt_clean[user_str] = {str(v)}
                
    print(f"   [SUCCESS] Loaded {len(gt_clean)} users in Ground Truth.")
    return df_new, df_all, gt_clean
@st.cache_resource
def load_train_user_set():
    """Load danh sách User trong tập Train để xác định Cold Start"""
    print(">>> Loading Stage 1 Model to identify Warm Users...")
    try:
        if os.path.exists(PATH_MODEL_S1):
            stage1 = joblib.load(PATH_MODEL_S1)
            # Lấy tập user đã biết trong quá trình train
            return set(stage1.user_id_to_index_.keys())
        else:
            st.warning("⚠️ Không tìm thấy model Stage 1. Không thể lọc chính xác Cold/Warm.")
            return set()
    except Exception as e:
        st.error(f"Lỗi load model Stage 1: {e}")
        return set()

@st.cache_data
def load_metadata():
    """Load thông tin Item"""
    cols_needed = ["item_id", "item_name", "category", "category_l1", "category_l2", "category_l3", "brand", "age_group_final"]
    sample_file = glob.glob(os.path.join(PATH_ITEM_DIR, "*item_chunk*.parquet"))[0]
    sample_cols = pd.read_parquet(sample_file).columns.tolist()
    final_cols = [c for c in cols_needed if c in sample_cols]
    
    df_item = read_parquet_pattern(PATH_ITEM_DIR, "item_chunk", columns=final_cols)
    if df_item.empty: return {}

    df_item['item_id'] = df_item['item_id'].astype(str)
    df_item = df_item.drop_duplicates("item_id")
    return df_item.set_index("item_id").to_dict(orient="index")

def get_user_history_dynamic(user_id):
    """Lazy load history"""
    search_pattern = os.path.join(PATH_TRX_DIR, "*purchase_history_daily_chunk*.parquet")
    files = glob.glob(search_pattern)
    user_history_dfs = []
    cols = ['customer_id', 'item_id', 'created_date']
    
    for f in files:
        try:
            df_chunk = pd.read_parquet(f, columns=cols)
            df_user = df_chunk[df_chunk['customer_id'].astype(str) == str(user_id)]
            if not df_user.empty: user_history_dfs.append(df_user)
        except: continue
            
    if not user_history_dfs: return pd.DataFrame()
    df_hist = pd.concat(user_history_dfs, ignore_index=True)
    df_hist['item_id'] = df_hist['item_id'].astype(str)
    if 'created_date' in df_hist.columns:
        df_hist = df_hist.sort_values('created_date', ascending=False)
    return df_hist

# ==========================================
# HELPER UI
# ==========================================
def highlight_hit(row, gt_items):
    return ['background-color: #d4edda' if str(row['item_id']) in gt_items else '' for _ in row]

def display_recs(df, user_id, item_map, gt_items, title, top_k=10):
    st.subheader(title)
    user_recs = df[df['customer_id'] == user_id].copy()
    
    if user_recs.empty:
        st.info("No recommendations generated.")
        return

    user_recs = user_recs.sort_values("pred_score", ascending=False).head(top_k)
    
    # Map info
    for col in ['item_name', 'category', 'category_l1', 'category_l2', 'category_l3']:
        key = 'Item Name' if col == 'item_name' else col.replace('_', ' ').capitalize()
        user_recs[key] = user_recs['item_id'].apply(lambda x: item_map.get(x, {}).get(col, 'Unknown'))
        
    user_recs['Is Hit 🎯'] = user_recs['item_id'].apply(lambda x: '✅' if x in gt_items else '')
    
    # Chọn cột hiển thị
    cols_show = ['Is Hit 🎯', 'item_id', 'Item Name', 'Category', 'Category l1', 'Category l2', 'Category l3', 'pred_score']
    
    st.dataframe(
        user_recs[cols_show].style.apply(lambda x: highlight_hit(x, gt_items), axis=1),
        hide_index=True,
        use_container_width=True
    )
    
    hits = user_recs[user_recs['Is Hit 🎯'] == '✅'].shape[0]
    st.metric(f"Precision @ {top_k}", f"{hits}/{top_k}", delta=f"{hits/top_k:.1%}")

# ==========================================
# MAIN APP
# ==========================================
def main():
    st.set_page_config(page_title="RecSys Cold Start Audit", layout="wide")
    st.title("❄️ RecSys Cold Start Auditor")
    
    with st.spinner("Loading Resources..."):
        df_new, df_all, gt_clean = load_static_data()
        item_map = load_metadata()
        train_user_set = load_train_user_set()
        
    # --- PHÂN LOẠI USER ---
    # Lấy danh sách user có trong tập test (từ file submission)
    test_users = set(df_new['customer_id'].unique())
    
    # Cold Users: Có trong Test nhưng KHÔNG có trong Train
    cold_users = sorted(list(test_users - train_user_set))
    # Warm Users: Có trong cả hai
    warm_users = sorted(list(test_users.intersection(train_user_set)))
    
    # --- SIDEBAR CONTROL ---
    st.sidebar.header("User Filter")
    
    filter_type = st.sidebar.radio(
        "Show Users:", 
        [f"Cold Start Only ({len(cold_users)})", f"Warm Users Only ({len(warm_users)})", "All Users"]
    )
    
    if "Cold" in filter_type:
        pool_users = cold_users
        st.sidebar.info("🥶 Đang xem các user MỚI (chưa từng mua gì trong tập Train).")
    elif "Warm" in filter_type:
        pool_users = warm_users
        st.sidebar.info("🔥 Đang xem các user CŨ (đã có lịch sử mua hàng).")
    else:
        pool_users = sorted(list(test_users))
        
    # Select User
    user_mode = st.sidebar.radio("Select Mode:", ["Random Picker", "Input ID"])
    selected_user = None
    
    if user_mode == "Random Picker":
        if st.sidebar.button("🎲 Pick Random User"):
            if pool_users:
                selected_user = np.random.choice(pool_users)
                st.session_state['curr_user'] = selected_user
            else:
                st.error("Danh sách user trống!")
        elif 'curr_user' in st.session_state:
            # Giữ user hiện tại nếu nó vẫn nằm trong pool, nếu không thì reset
            if st.session_state['curr_user'] in pool_users:
                selected_user = st.session_state['curr_user']
    else:
        selected_user = st.sidebar.text_input("Enter Customer ID:")

    top_k = st.sidebar.slider("Top K View", 5, 20, 10)

    # --- MAIN VIEW ---
    if selected_user:
        selected_user = str(selected_user).strip()
        
        # Check status
        is_cold = selected_user not in train_user_set
        status_icon = "🥶 COLD START" if is_cold else "🔥 WARM USER"
        
        st.markdown(f"## 👤 User: `{selected_user}` ({status_icon})")
        
        # --- A. HISTORY ---
        # Chỉ load history nếu là Warm User (vì Cold User chắc chắn trống, đỡ tốn time load)
        if not is_cold:
            with st.expander("📜 Purchase History (Train Period)", expanded=False):
                with st.spinner("Checking history..."):
                    df_hist = get_user_history_dynamic(selected_user)
                
                if not df_hist.empty:
                    st.write(f"Found {len(df_hist)} transactions.")
                    hist_display = []
                    for _, row in df_hist.iterrows():
                        iid = str(row['item_id'])
                        info = item_map.get(iid, {})
                        hist_display.append({
                            "Date": row.get('created_date'),
                            "Item ID": iid,
                            "Name": info.get("item_name", "N/A"),
                            "Category l1": info.get("category_l1", "N/A")
                        })
                    st.dataframe(pd.DataFrame(hist_display), use_container_width=True)
        else:
            st.info("ℹ️ User này không có lịch sử mua hàng trong tập Train (đúng tính chất Cold Start).")

        st.divider()

        # --- B. GROUND TRUTH (Test Period) ---
        # User thực sự mua gì trong tháng 1/2025
        gt_items = gt_clean.get(selected_user, set())
        
        st.subheader(f"🛒 Real Purchases (Jan 2025) - {len(gt_items)} items")
        if gt_items:
            gt_list = []
            for iid in gt_items:
                info = item_map.get(iid, {})
                gt_list.append({
                    "Item ID": iid,
                    "Name": info.get("item_name", "N/A"),
                    "Category": info.get("category", "N/A"),
                    "Category l1": info.get("category_l1", "N/A"),
                    "Category l2": info.get("category_l2", "N/A"),
                    "Category l3": info.get("category_l3", "N/A"),
                    "Price (Ref)": info.get("price", "N/A")
                })
            st.dataframe(pd.DataFrame(gt_list), use_container_width=True)
        else:
            st.warning("User này không mua gì trong tập Test (Không có trong Ground Truth).")

        st.divider()

        # --- C. PREDICTIONS ---
        col1, col2 = st.columns(2)
        
        with col1:
            display_recs(df_new, selected_user, item_map, gt_items, "🆕 New Items Strategy", top_k)
            
        with col2:
            display_recs(df_all, selected_user, item_map, gt_items, "🌍 All Items Strategy", top_k)

if __name__ == "__main__":
    main()