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
    """
    search_pattern = os.path.join(base_path, f"*{keyword}*.parquet")
    files = glob.glob(search_pattern)
    
    if not files:
        st.error(f"❌ Không tìm thấy file nào chứa từ khóa '{keyword}' tại: {base_path}")
        return pd.DataFrame()
    
    print(f">>> Found {len(files)} files matching '{keyword}'")
    
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
    print(">>> Loading Metadata...")
    df_item = read_parquet_pattern(CFG.paths.raw_data_path, "item_chunk")
    
    if df_item.empty:
        return pd.DataFrame(), {}

    # Chỉ giữ cột cần thiết
    cols = ["item_id"]
    for c in ["item_name", "category", "category_l1", "category_l2", "category_l3", "brand"]:
        if c in df_item.columns: cols.append(c)
    
    df_item["item_id"] = df_item["item_id"].astype(str)
    df_item = df_item[cols].drop_duplicates("item_id")
    
    item_map = df_item.set_index("item_id").to_dict(orient="index")
    
    return df_item, item_map

@st.cache_data
def load_valid_data():
    print(">>> Loading Valid Transactions...")
    df_valid = read_parquet_pattern(CFG.paths.raw_data_path, "purchase_history_daily_chunk")
    
    if df_valid.empty: return pd.DataFrame()

    if 'created_date' in df_valid.columns:
        df_valid['created_date'] = pd.to_datetime(df_valid['created_date'])
        start_date = pd.to_datetime(CFG.data_split.hist_end_date) + pd.Timedelta(days=1)
        end_date = start_date + pd.Timedelta(days=CFG.data_split.recent_days)
        
        df_valid = df_valid[
            (df_valid['created_date'] >= start_date) & 
            (df_valid['created_date'] <= end_date)
        ]
    
    df_valid['customer_id'] = df_valid['customer_id'].astype(str)
    df_valid['item_id'] = df_valid['item_id'].astype(str)
    
    return df_valid

# ==========================================
# 2. HELPER FUNCTIONS UI
# ==========================================
def highlight_hit(row, gt_items):
    return ['background-color: #d4edda' if str(row['item_id']) in gt_items else '' for _ in row]

def display_recommendation_table(df, item_map, gt_items, top_k, score_col, title):
    """Hàm hiển thị bảng recommendation generic"""
    # Copy và sort
    display_df = df.sort_values(score_col, ascending=False).head(top_k).copy()
    display_df['item_id'] = display_df['item_id'].astype(str)

    # Map thông tin
    display_df['Item Name'] = display_df['item_id'].apply(lambda x: item_map.get(x, {}).get('item_name', 'Unknown'))
    display_df['Category l1'] = display_df['item_id'].apply(lambda x: item_map.get(x, {}).get('category_l1', '-'))
    
    # Check Hit
    display_df['Is Hit 🎯'] = display_df['item_id'].apply(lambda x: '✅' if x in gt_items else '')

    # Chọn cột hiển thị
    cols_show = ['Is Hit 🎯', 'item_id', 'Item Name', 'Category l1', score_col]
    
    # Thêm cột debug nếu có
    if 'stage1_score' in display_df.columns and score_col != 'stage1_score':
        cols_show.append('stage1_score')
    
    valid_cols = [c for c in cols_show if c in display_df.columns]

    st.markdown(f"**{title}**")
    st.dataframe(
        display_df[valid_cols].style.apply(lambda x: highlight_hit(x, gt_items), axis=1),
        height=400,
        hide_index=True
    )
    
    # Metric nhanh
    hits = display_df[display_df['Is Hit 🎯'] == '✅'].shape[0]
    st.metric(f"Precision @ {top_k}", f"{hits}/{top_k}", delta=f"{hits/top_k:.1%}")


# ==========================================
# 3. MAIN APP UI
# ==========================================
def main():
    st.set_page_config(page_title="RecSys Monitor", layout="wide")
    st.title("🛍️ Recommendation System Inspector")
    
    # --- Sidebar ---
    st.sidebar.header("Control Panel")
    with st.spinner("Loading System & Data..."):
        stage1_model, lgbm_model, feature_cols = load_artifacts()
        df_item, item_map = load_metadata()
        df_valid_trx = load_valid_data()
        
    if df_valid_trx.empty:
        st.error("Không load được dữ liệu valid transaction!")
        st.stop()

    all_users = df_valid_trx['customer_id'].unique()
    st.sidebar.success(f"Loaded {len(all_users)} users")
    
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

    st.sidebar.markdown("---")
    st.sidebar.subheader("Config")
    allow_repeat = st.sidebar.checkbox("Allow Repeat Items?", value=True)
    top_k_view = st.sidebar.slider("Top K View", 5, 50, 10)

    # --- Main Content ---
    if selected_user:
        selected_user = str(selected_user).strip()
        st.markdown(f"### 👤 Analysis for User: `{selected_user}`")
        
        col1, col2 = st.columns([1, 2])
        
        # --- 1. History Info ---
        with col1:
            st.info("📜 History & Context")
            u_idx = None
            if selected_user in stage1_model.user_id_to_index_:
                u_idx = stage1_model.user_id_to_index_[selected_user]
            
            if u_idx is not None:
                hist_indices = stage1_model.user_history_.get(u_idx, set())
                hist_items = [str(stage1_model.index_to_item_id_[i]) for i in hist_indices]
                st.write(f"**Bought Items ({len(hist_items)}):**")
                
                hist_data = []
                for iid in hist_items:
                    info = item_map.get(iid, {})
                    hist_data.append({
                        "Item ID": iid,
                        "Name": info.get("item_name", "N/A"),
                        "Category l1": info.get("category_l1", "N/A")
                    })
                if hist_data:
                    st.dataframe(pd.DataFrame(hist_data), height=300, hide_index=True)
            else:
                st.warning("⚠️ COLD START USER")

        # --- 2. Ground Truth & Prediction ---
        with col2:
            # Ground Truth
            user_valid_trx = df_valid_trx[df_valid_trx['customer_id'] == selected_user]
            gt_items = set(user_valid_trx['item_id'].tolist())
            
            # --- PROCESS PREDICTION ---
            with st.spinner("Generating Recommendations..."):
                # 1. Candidate Gen (Stage 1)
                candidates = stage1_model.recommend_candidates([selected_user], allow_repeat=allow_repeat)
                
                if candidates.empty:
                    st.error("No candidates generated!")
                else:
                    # 2. Ranking (Stage 2)
                    for col in feature_cols:
                        if col not in candidates.columns: candidates[col] = 0
                    
                    X_pred = candidates[feature_cols]
                    candidates['pred_score'] = lgbm_model.predict(X_pred)
                    
                    # --- TÍNH TOÁN METRICS STAGE 1 ---
                    # Kiểm tra xem Stage 1 tìm được bao nhiêu Ground Truth
                    candidates_set = set(candidates['item_id'].astype(str).tolist())
                    hits_stage1 = len(candidates_set.intersection(gt_items))
                    total_gt = len(gt_items)
                    recall_s1 = hits_stage1 / total_gt if total_gt > 0 else 0
                    
                    st.success(f"**Stage 1 Retrieval Recall:** Found {hits_stage1}/{total_gt} items ({recall_s1:.1%}) in {len(candidates)} candidates.")

                    # --- HIỂN THỊ TABS ---
                    tab_final, tab_stage1, tab_gt = st.tabs(["🚀 Stage 2 (Final)", "🔍 Stage 1 (Retrieval)", "🛒 Ground Truth"])
                    
                    # Tab 1: Stage 2 (Sắp xếp theo pred_score)
                    with tab_final:
                        display_recommendation_table(
                            candidates, item_map, gt_items, top_k_view, 
                            score_col='pred_score', 
                            title=f"Final Recommendations (Top {top_k_view})"
                        )
                        
                    # Tab 2: Stage 1 (Sắp xếp theo stage1_score)
                    with tab_stage1:
                        st.caption("Đây là danh sách candidates thô từ Stage 1 chưa qua LightGBM rank lại.")
                        display_recommendation_table(
                            candidates, item_map, gt_items, top_k_view, 
                            score_col='stage1_score', 
                            title=f"Stage 1 Top Candidates (Top {top_k_view})"
                        )
                        
                    # Tab 3: Ground Truth Chi tiết
                    with tab_gt:
                        if len(gt_items) > 0:
                            gt_data = []
                            for iid in gt_items:
                                info = item_map.get(iid, {})
                                in_stage1 = "✅" if iid in candidates_set else "❌"
                                
                                # Check xem có trong top K final không
                                top_final = candidates.sort_values('pred_score', ascending=False).head(top_k_view)
                                in_top_final = "✅" if iid in top_final['item_id'].astype(str).values else "❌"

                                gt_data.append({
                                    "Item ID": iid,
                                    "Name": info.get("item_name", "N/A"),
                                    "Category l1": info.get("category_l1", "N/A"),
                                    "Found in S1?": in_stage1,
                                    "In Top Final?": in_top_final
                                })
                            st.dataframe(pd.DataFrame(gt_data), use_container_width=True)
                        else:
                            st.info("User did not buy anything in validation.")

if __name__ == "__main__":
    main()