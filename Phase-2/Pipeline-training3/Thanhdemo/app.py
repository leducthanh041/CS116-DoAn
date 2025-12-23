# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import lightgbm as lgb
# from pathlib import Path
# from config.settings import CFG, get_path
# from src import ItemItemCFStage1

# # ==========================================
# # 1. CACHING & LOADING (Tối ưu tốc độ)
# # ==========================================
# @st.cache_resource
# def load_artifacts():
#     """Load model và config 1 lần duy nhất"""
#     print(">>> Loading Artifacts...")
    
#     # 1. Load Stage 1
#     # Thử load base model trước, nếu không có thì load model thường
#     path_s1 = get_path("artifacts/stage1_model_base.pkl")
#     if not Path(path_s1).exists():
#         path_s1 = get_path("artifacts/stage1_model.pkl")
    
#     stage1 = joblib.load(path_s1)
    
#     # 2. Load Stage 2
#     # Load model All Items hoặc New Items tùy bạn chọn default (ở đây load All Items)
#     path_s2 = get_path("artifacts/lgbm_model_all_item_rec.txt")
#     if not Path(path_s2).exists():
#         path_s2 = get_path("artifacts/lgbm_model.txt") # Fallback
        
#     lgbm = lgb.Booster(model_file=path_s2)
    
#     # 3. Load Features list
#     feats = joblib.load(get_path("artifacts/stage2_features.pkl"))
    
#     return stage1, lgbm, feats

# @st.cache_data
# def load_metadata():
#     """Load thông tin Item và User để hiển thị tên đẹp hơn"""
#     print(">>> Loading Metadata...")
#     # Load Item Metadata để map ID -> Tên sản phẩm
#     df_item = pd.read_parquet(f"{CFG.paths.raw_data_path}/item_chunk") # Hoặc path item của bạn
    
#     # Chỉ giữ cột cần thiết để nhẹ ram
#     cols = ["item_id"]
#     if "item_name" in df_item.columns: cols.append("item_name")
#     if "category_l1" in df_item.columns: cols.append("category_l1")
#     if "brand" in df_item.columns: cols.append("brand")
    
#     df_item = df_item[cols].drop_duplicates("item_id")
    
#     # Tạo dict map nhanh
#     item_map = df_item.set_index("item_id").to_dict(orient="index")
    
#     return df_item, item_map

# @st.cache_data
# def load_valid_data():
#     """Load tập Valid/Test để lấy Ground Truth"""
#     # Load data valid (Tháng 12 hoặc Test Tháng 1)
#     # Ở đây demo load file groundtruth.pkl nếu có, hoặc load raw valid
#     # Để đơn giản, ta load raw parquet valid
#     df_valid = pd.read_parquet(f"{CFG.paths.raw_data_path}/purchase_history_daily_chunk")
    
#     # Filter theo ngày Valid (Ví dụ T12/2024)
#     # Lưu ý: Cần chỉnh lại logic filter theo đúng config của bạn
#     df_valid['created_date'] = pd.to_datetime(df_valid['created_date'])
#     start_date = pd.to_datetime(CFG.data_split.hist_end_date) + pd.Timedelta(days=1)
#     end_date = start_date + pd.Timedelta(days=CFG.data_split.recent_days)
    
#     df_valid = df_valid[
#         (df_valid['created_date'] >= start_date) & 
#         (df_valid['created_date'] <= end_date)
#     ]
    
#     # Convert ID to string
#     df_valid['customer_id'] = df_valid['customer_id'].astype(str)
#     df_valid['item_id'] = df_valid['item_id'].astype(str)
    
#     return df_valid

# # ==========================================
# # 2. HELPER FUNCTIONS
# # ==========================================
# def get_item_info(item_id, item_map):
#     """Lấy thông tin item từ ID"""
#     info = item_map.get(str(item_id), {})
#     name = info.get("item_name", "Unknown")
#     cat = info.get("category_l1", "-")
#     return f"[{cat}] {name} ({item_id})"

# def highlight_hit(row, gt_items):
#     """Tô màu xanh nếu item nằm trong Ground Truth"""
#     return ['background-color: #d4edda' if str(row['item_id']) in gt_items else '' for _ in row]

# # ==========================================
# # 3. MAIN APP UI
# # ==========================================
# def main():
#     st.set_page_config(page_title="RecSys Monitor", layout="wide")
#     st.title("🛍️ Recommendation System Inspector")
    
#     # --- Sidebar: Controls ---
#     st.sidebar.header("Control Panel")
    
#     # Load Resources
#     with st.spinner("Loading System..."):
#         stage1_model, lgbm_model, feature_cols = load_artifacts()
#         df_item, item_map = load_metadata()
#         df_valid_trx = load_valid_data()
        
#     all_users = df_valid_trx['customer_id'].unique()
#     st.sidebar.success(f"Loaded {len(all_users)} users in Validation Set")
    
#     # Chọn User
#     user_mode = st.sidebar.radio("Select User Mode:", ["Random User", "Input ID"])
    
#     selected_user = None
#     if user_mode == "Random User":
#         if st.sidebar.button("🎲 Pick Random"):
#             selected_user = np.random.choice(all_users)
#             st.session_state['curr_user'] = selected_user
#         elif 'curr_user' in st.session_state:
#             selected_user = st.session_state['curr_user']
#     else:
#         selected_user = st.sidebar.text_input("Enter Customer ID:")

#     # Config Flow
#     st.sidebar.markdown("---")
#     st.sidebar.subheader("Inference Config")
#     allow_repeat = st.sidebar.checkbox("Allow Repeat Items?", value=True, help="Cho phép recommend lại đồ đã mua")
#     top_k_view = st.sidebar.slider("Top K Recommend", 5, 20, 10)

#     # --- Main Content ---
#     if selected_user:
#         st.markdown(f"### 👤 Analysis for User: `{selected_user}`")
        
#         col1, col2 = st.columns([1, 2])
        
#         # --- 1. History Info ---
#         with col1:
#             st.info(f"**History Window:** {CFG.data_split.hist_days} days before {CFG.data_split.hist_end_date}")
            
#             # Lấy history từ model Stage 1 (đã lưu trong object)
#             if selected_user in stage1_model.user_id_to_index_:
#                 u_idx = stage1_model.user_id_to_index_[selected_user]
#                 hist_indices = stage1_model.user_history_.get(u_idx, set())
#                 hist_items = [stage1_model.index_to_item_id_[i] for i in hist_indices]
                
#                 st.write(f"**Bought Items ({len(hist_items)}):**")
                
#                 # Tạo dataframe history để hiển thị đẹp
#                 hist_data = []
#                 for iid in hist_items:
#                     info = item_map.get(str(iid), {})
#                     hist_data.append({
#                         "Item ID": iid,
#                         "Name": info.get("item_name", "N/A"),
#                         "Category": info.get("category_l1", "N/A")
#                     })
#                 st.dataframe(pd.DataFrame(hist_data), height=300, hide_index=True)
#             else:
#                 st.warning("⚠️ This is a COLD START user (No history in Train set)")
#                 hist_items = []

#         # --- 2. Ground Truth & Prediction ---
#         with col2:
#             # Lấy Ground Truth (Thực tế mua trong Valid)
#             gt_items = set(df_valid_trx[df_valid_trx['customer_id'] == selected_user]['item_id'].tolist())
            
#             # --- PREDICT ---
#             # 1. Generate Candidates
#             candidates = stage1_model.recommend_candidates([selected_user], allow_repeat=allow_repeat)
            
#             if candidates.empty:
#                 st.error("No candidates generated!")
#             else:
#                 # 2. Rank with Stage 2
#                 # Đảm bảo cột features đúng thứ tự
#                 X_pred = candidates[feature_cols]
#                 candidates['pred_score'] = lgbm_model.predict(X_pred)
#                 candidates = candidates.sort_values('pred_score', ascending=False).head(top_k_view)
                
#                 # 3. Format hiển thị
#                 display_df = candidates.copy()
                
#                 # Map tên item
#                 display_df['Item Name'] = display_df['item_id'].apply(lambda x: item_map.get(str(x), {}).get('item_name', 'Unknown'))
#                 display_df['Category'] = display_df['item_id'].apply(lambda x: item_map.get(str(x), {}).get('category_l1', '-'))
                
#                 # Check Hit
#                 display_df['Is Hit 🎯'] = display_df['item_id'].apply(lambda x: '✅' if str(x) in gt_items else '')
                
#                 # Sắp xếp cột đẹp
#                 cols_show = ['Is Hit 🎯', 'item_id', 'Item Name', 'Category', 'pred_score', 'stage1_score'] 
#                 # Thêm các features quan trọng để debug
#                 cols_debug = ['sim_max', 'sim_avg', 'support_cnt', 'stage1_rank']
                
#                 st.subheader(f"🏆 Top {top_k_view} Recommendations")
#                 st.dataframe(
#                     display_df[cols_show + cols_debug].style.apply(
#                         lambda x: highlight_hit(x, gt_items), axis=1
#                     ),
#                     height=400,
#                     hide_index=True
#                 )
                
#                 # --- Metrics nhanh ---
#                 hits = display_df[display_df['Is Hit 🎯'] == '✅'].shape[0]
#                 st.metric("Precision @ This List", f"{hits}/{top_k_view}", delta=f"{hits/top_k_view:.1%}")

#         # --- 3. Ground Truth Detail ---
#         st.markdown("---")
#         st.subheader(f"🛒 Ground Truth (Items bought in Valid/Test Period)")
#         if len(gt_items) > 0:
#             gt_data = []
#             for iid in gt_items:
#                 info = item_map.get(str(iid), {})
#                 gt_data.append({
#                     "Item ID": iid,
#                     "Name": info.get("item_name", "N/A"),
#                     "Category": info.get("category_l1", "N/A"),
#                     "In Recs?": "✅" if iid in candidates['item_id'].values else "❌"
#                 })
#             st.dataframe(pd.DataFrame(gt_data), use_container_width=True)
#         else:
#             st.info("User did not buy anything in the validation period.")

# if __name__ == "__main__":
#     main()

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import os
import glob
from pathlib import Path
from config.settings import CFG, get_path
from src import ItemItemCFStage1

# ==========================================
# HELPER: Load parquet files by pattern
# ==========================================
def read_parquet_pattern(base_path, keyword):
    """
    Tìm và đọc tất cả các file parquet chứa 'keyword' trong tên file.
    Thay thế cho pd.read_parquet trực tiếp để tránh lỗi FileNotFoundError.
    """
    # Tạo pattern tìm kiếm: /path/to/data/*keyword*.parquet
    search_pattern = os.path.join(base_path, f"*{keyword}*.parquet")
    files = glob.glob(search_pattern)
    
    if not files:
        st.error(f"❌ Không tìm thấy file nào chứa từ khóa '{keyword}' tại: {base_path}")
        return pd.DataFrame() # Trả về DF rỗng để không crash app
    
    print(f">>> Found {len(files)} files matching '{keyword}'")
    
    # Đọc từng file và concat lại
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"Warning: Could not read {f}. Error: {e}")
            
    if not dfs:
        return pd.DataFrame()
        
    return pd.concat(dfs, ignore_index=True)

# ==========================================
# 1. CACHING & LOADING
# ==========================================
@st.cache_resource
def load_artifacts():
    """Load model và config 1 lần duy nhất"""
    print(">>> Loading Artifacts...")
    
    # 1. Load Stage 1
    path_s1 = get_path("artifacts/stage1_model_base.pkl")
    if not Path(path_s1).exists():
        path_s1 = get_path("artifacts/stage1_model.pkl")
    
    try:
        stage1 = joblib.load(path_s1)
    except FileNotFoundError:
        st.error("Chưa tìm thấy model Stage 1. Hãy chạy 'python main.py' trước!")
        st.stop()
    
    # 2. Load Stage 2
    path_s2 = get_path("artifacts/lgbm_model_all_item_rec.txt")
    if not Path(path_s2).exists():
        path_s2 = get_path("artifacts/lgbm_model.txt") 
        
    if not Path(path_s2).exists():
        st.error("Chưa tìm thấy model Stage 2 (LightGBM). Hãy chạy 'python main.py' trước!")
        st.stop()
        
    lgbm = lgb.Booster(model_file=path_s2)
    
    # 3. Load Features list
    path_feat = get_path("artifacts/stage2_features.pkl")
    if Path(path_feat).exists():
        feats = joblib.load(path_feat)
    else:
        st.error("Thiếu file features list. Hãy chạy lại main.py!")
        st.stop()
    
    return stage1, lgbm, feats

@st.cache_data
def load_metadata():
    """Load thông tin Item"""
    print(">>> Loading Metadata...")
    # SỬA LỖI: Dùng hàm helper để đọc file chunk
    df_item = read_parquet_pattern(CFG.paths.raw_data_path, "item_chunk")
    
    if df_item.empty:
        return pd.DataFrame(), {}

    # Chỉ giữ cột cần thiết để nhẹ ram
    cols = ["item_id"]
    if "item_name" in df_item.columns: cols.append("item_name")
    if "category" in df_item.columns: cols.append("category")
    if "category_l1" in df_item.columns: cols.append("category_l1")
    if "category_l2" in df_item.columns: cols.append("category_l2")
    if "category_l3" in df_item.columns: cols.append("category_l3")
    if "brand" in df_item.columns: cols.append("brand")
    
    # Convert ID sang string để map cho dễ
    df_item["item_id"] = df_item["item_id"].astype(str)
    df_item = df_item[cols].drop_duplicates("item_id")
    
    # Tạo dict map nhanh
    item_map = df_item.set_index("item_id").to_dict(orient="index")
    
    return df_item, item_map

@st.cache_data
def load_valid_data():
    """Load tập Valid Transactions"""
    print(">>> Loading Valid Transactions...")
    # SỬA LỖI: Dùng hàm helper để đọc file chunk
    df_valid = read_parquet_pattern(CFG.paths.raw_data_path, "purchase_history_daily_chunk")
    
    if df_valid.empty:
        return pd.DataFrame()

    # Filter theo ngày Valid (Ví dụ T12/2024)
    if 'created_date' in df_valid.columns:
        df_valid['created_date'] = pd.to_datetime(df_valid['created_date'])
        
        # Lấy khoảng valid từ config
        start_date = pd.to_datetime(CFG.data_split.hist_end_date) + pd.Timedelta(days=1)
        end_date = start_date + pd.Timedelta(days=CFG.data_split.recent_days)
        
        df_valid = df_valid[
            (df_valid['created_date'] >= start_date) & 
            (df_valid['created_date'] <= end_date)
        ]
    
    # Convert ID to string
    df_valid['customer_id'] = df_valid['customer_id'].astype(str)
    df_valid['item_id'] = df_valid['item_id'].astype(str)
    
    return df_valid

# ==========================================
# 2. HELPER FUNCTIONS UI
# ==========================================
def highlight_hit(row, gt_items):
    """Tô màu xanh nếu item nằm trong Ground Truth"""
    return ['background-color: #d4edda' if str(row['item_id']) in gt_items else '' for _ in row]

# ==========================================
# 3. MAIN APP UI
# ==========================================
def main():
    st.set_page_config(page_title="RecSys Monitor", layout="wide")
    st.title("🛍️ Recommendation System Inspector")
    
    # --- Sidebar: Controls ---
    st.sidebar.header("Control Panel")
    
    # Load Resources
    with st.spinner("Loading System & Data..."):
        stage1_model, lgbm_model, feature_cols = load_artifacts()
        df_item, item_map = load_metadata()
        df_valid_trx = load_valid_data()
        
    if df_valid_trx.empty:
        st.error("Không load được dữ liệu valid transaction!")
        st.stop()

    all_users = df_valid_trx['customer_id'].unique()
    st.sidebar.success(f"Loaded {len(all_users)} users in Validation Set")
    
    # Chọn User
    user_mode = st.sidebar.radio("Select User Mode:", ["Random User", "Input ID"])
    
    selected_user = None
    if user_mode == "Random User":
        if st.sidebar.button("🎲 Pick Random"):
            selected_user = np.random.choice(all_users)
            st.session_state['curr_user'] = selected_user
        elif 'curr_user' in st.session_state:
            selected_user = st.session_state['curr_user']
    else:
        selected_user = st.sidebar.text_input("Enter Customer ID:")

    # Config Flow
    st.sidebar.markdown("---")
    st.sidebar.subheader("Inference Config")
    allow_repeat = st.sidebar.checkbox("Allow Repeat Items?", value=True, help="Cho phép recommend lại đồ đã mua")
    top_k_view = st.sidebar.slider("Top K Recommend", 5, 20, 10)

    # --- Main Content ---
    if selected_user:
        selected_user = str(selected_user).strip() # Ensure string format
        st.markdown(f"### 👤 Analysis for User: `{selected_user}`")
        
        col1, col2 = st.columns([1, 2])
        
        # --- 1. History Info ---
        with col1:
            st.info(f"**History Window:** {CFG.data_split.hist_days} days before {CFG.data_split.hist_end_date}")
            
            # Lấy history từ model Stage 1
            # Lưu ý: user_id_to_index_ có thể key là int hoặc str tùy lúc train
            # Ta cần handle cả 2 trường hợp
            u_idx = None
            if selected_user in stage1_model.user_id_to_index_:
                u_idx = stage1_model.user_id_to_index_[selected_user]
            
            if u_idx is not None:
                hist_indices = stage1_model.user_history_.get(u_idx, set())
                # Map ngược từ index -> item_id (string)
                hist_items = []
                for i in hist_indices:
                    # Kiểm tra xem index_to_item_id_ trả về gì
                    raw_id = stage1_model.index_to_item_id_[i]
                    hist_items.append(str(raw_id))
                
                st.write(f"**Bought Items ({len(hist_items)}):**")
                
                # Tạo dataframe history
                hist_data = []
                for iid in hist_items:
                    info = item_map.get(iid, {})
                    hist_data.append({
                        "Item ID": iid,
                        "Name": info.get("item_name", "N/A"),
                        "Category": info.get("category", "N/A"),
                        "Category 1": info.get("category_l1", "N/A"),
                        "Category 2": info.get("category_l2", "N/A"),
                        "Category 3": info.get("category_l3", "N/A")
                    })
                
                if hist_data:
                    st.dataframe(pd.DataFrame(hist_data), height=300, hide_index=True)
                else:
                    st.write("(No details available for history items)")
            else:
                st.warning("⚠️ This is a COLD START user (No history in Train set)")
                hist_items = []

        # --- 2. Ground Truth & Prediction ---
        with col2:
            # Lấy Ground Truth (Thực tế mua trong Valid)
            user_valid_trx = df_valid_trx[df_valid_trx['customer_id'] == selected_user]
            gt_items = set(user_valid_trx['item_id'].tolist())
            
            # --- PREDICT ---
            # 1. Generate Candidates
            candidates = stage1_model.recommend_candidates([selected_user], allow_repeat=allow_repeat)
            
            if candidates.empty:
                st.error("No candidates generated!")
            else:
                # 2. Rank with Stage 2
                # Đảm bảo cột features đúng thứ tự
                # Nếu thiếu features nào trong candidates thì fill 0
                for col in feature_cols:
                    if col not in candidates.columns:
                        candidates[col] = 0
                
                X_pred = candidates[feature_cols]
                candidates['pred_score'] = lgbm_model.predict(X_pred)
                candidates = candidates.sort_values('pred_score', ascending=False).head(top_k_view)
                
                # 3. Format hiển thị
                display_df = candidates.copy()
                display_df['item_id'] = display_df['item_id'].astype(str)
                
                # Map tên item
                display_df['Item Name'] = display_df['item_id'].apply(lambda x: item_map.get(x, {}).get('item_name', 'Unknown'))
                display_df['Category'] = display_df['item_id'].apply(lambda x: item_map.get(x, {}).get('category', '-'))
                display_df['Category l1'] = display_df['item_id'].apply(lambda x: item_map.get(x, {}).get('category_l1', '-'))
                display_df['Category l2'] = display_df['item_id'].apply(lambda x: item_map.get(x, {}).get('category_l2', '-'))
                display_df['Category l3'] = display_df['item_id'].apply(lambda x: item_map.get(x, {}).get('category_l3', '-'))
                
                # Check Hit
                display_df['Is Hit 🎯'] = display_df['item_id'].apply(lambda x: '✅' if x in gt_items else '')
                
                # Sắp xếp cột đẹp
                cols_show = ['Is Hit 🎯', 'item_id', 'Item Name', 'Category', 'Category l1', 'Category l2', 'Category l3', 'pred_score'] 
                # Thêm các features quan trọng để debug
                cols_debug = ['stage1_score'] # Thêm các feature khác nếu có: 'sim_max', ...
                valid_cols = [c for c in cols_show + cols_debug if c in display_df.columns]
                
                st.subheader(f"🏆 Top {top_k_view} Recommendations")
                st.dataframe(
                    display_df[valid_cols].style.apply(
                        lambda x: highlight_hit(x, gt_items), axis=1
                    ),
                    height=400,
                    hide_index=True
                )
                
                # --- Metrics nhanh ---
                hits = display_df[display_df['Is Hit 🎯'] == '✅'].shape[0]
                if top_k_view > 0:
                    st.metric("Precision @ This List", f"{hits}/{top_k_view}", delta=f"{hits/top_k_view:.1%}")

        # --- 3. Ground Truth Detail ---
        st.markdown("---")
        st.subheader(f"🛒 Ground Truth (Items bought in Valid/Test Period)")
        if len(gt_items) > 0:
            gt_data = []
            for iid in gt_items:
                info = item_map.get(iid, {})
                gt_data.append({
                    "Item ID": iid,
                    "Name": info.get("item_name", "N/A"),
                    "Category": info.get("category", "N/A"),
                    "Category l1": info.get("category_l1", "N/A"),
                    "Category l2": info.get("category_l2", "N/A"),
                    "Category l3": info.get("category_l3", "N/A"),
                    "In Recs?": "✅" if iid in candidates['item_id'].values else "❌"
                })
            st.dataframe(pd.DataFrame(gt_data), use_container_width=True)
        else:
            st.info("User did not buy anything in the validation period.")

if __name__ == "__main__":
    main()