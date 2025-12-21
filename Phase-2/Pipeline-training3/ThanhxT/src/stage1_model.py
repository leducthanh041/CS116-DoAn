import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfTransformer
from tqdm.auto import tqdm
from collections import Counter
import joblib
import json
import os
from config.settings import get_path

class ItemItemCFStage1(BaseEstimator):
    def __init__(self, config, df_item, df_user):
        self.cfg = config
        self.df_item = df_item
        self.df_user = df_user

        # --- Params ---
        self.n_neighbors = self.cfg.n_neighbors
        self.top_k = self.cfg.top_k_candidates
        self.weight_type = self.cfg.weight_type
        self.use_ui_recency = self.cfg.use_ui_recency
        self.ui_recency_lambda = self.cfg.ui_recency_lambda
        
        # Params cũ
        self.use_cat_l1 = getattr(self.cfg, 'use_cat_l1_pref', False)
        
        # --- Mappings & Storages ---
        self.user_id_to_index_ = {}
        self.user_index_to_id_ = {} 
        self.item_id_to_index_ = {}
        self.index_to_item_id_ = {}
        self.popular_item_indices_ = []
        self.user_history_ = {} 
        
        # Feature Storages
        self.item_metadata_ = {}   
        self.user_profiles_ = {}
        
        # [RESTORED] Khôi phục dictionary lưu ngày mua cuối
        self.user_item_last_date_ = {} 
        
        self.item_global_pop_30d_ = {}
        self.max_date_ = None
        
        # Cold Start Artifacts
        self.user_region_map_ = {}       
        self.region_trending_indices_ = {} 
        self.pregnancy_items_indices_ = [] 
        self.fashion_trending_indices_ = [] 
        
        # [REMOVED] Đã xóa phần load artifacts user_baby_dob và item_target_age

        # [NEW] ĐỊNH NGHĨA DANH SÁCH FEATURE (Đã thêm lại Recency)
        self.feature_names = [
            # 1. Ranking Info
            "stage1_score", "stage1_rank", "sim_max", "sim_avg", "support_cnt",
            # 2. Interaction Counters
            "item_hist_cnt", "brand_match_cnt", "cat2_match_cnt",
            # 3. Recency (Được thêm lại)
            "feat_days_since_item", "feat_days_since_cat",
            # 4. Price & Popularity
            "feat_price_ratio", "feat_log_price", "feat_pop_30d",
        ]

    def fit(self, df_train):
        print(f"[{self.__class__.__name__}] Starting fit process (With Recency)...")
        self.df_train = df_train.copy()
        
        # 1. Prepare Data & Matrix
        self._prepare_matrix_data()
        ui_matrix = self._build_csr_matrix()
        
        # 2. Build Features
        self._build_advanced_features_data()
        
        # 3. Cold Start Lists
        self._build_region_trending()
        self._build_pregnancy_list()
        self._build_fashion_trending()
        self._build_popularity()
        
        # 4. Train KNN Model
        print(f"[{self.__class__.__name__}] Training kNN model...")
        self.tfidf_ = TfidfTransformer(norm="l2", use_idf=True, sublinear_tf=True)
        ui_tfidf = self.tfidf_.fit_transform(ui_matrix)
        X_items = ui_tfidf.T 
        
        self.nn_model_ = NearestNeighbors(
            n_neighbors=self.n_neighbors + 1, 
            metric="cosine", algorithm="brute", n_jobs=-1
        )
        self.nn_model_.fit(X_items)
        
        # Pre-compute neighbors
        distances, indices = self.nn_model_.kneighbors(X_items, return_distance=True)
        self.item_neighbors_ = indices[:, 1:] 
        self.item_neighbor_sims_ = 1.0 - distances[:, 1:]
        
        print(f"[{self.__class__.__name__}] Fit complete.")
        
        # Lưu feature config
        self.save_feature_config()
        return self

    def save_feature_config(self):
        out_path = get_path("artifacts/feature_config.json")
        try:
            with open(out_path, "w") as f:
                json.dump(self.feature_names, f, indent=4)
            print(f"   -> [Auto-Config] Saved feature names to {out_path}")
        except Exception as e:
            print(f"   -> [Warning] Could not save feature config: {e}")

    def _prepare_matrix_data(self):
        df = self.df_train
        if "created_datetime" not in df.columns:
             df["created_datetime"] = pd.to_datetime(df["created_date"]) 
        cutoff_dt = df["created_datetime"].max()
        self.max_date_ = cutoff_dt 
        
        meta_cols = ["item_id", "category", "category_l1", "category_l2", "category_l3"]
        available_cols = [c for c in meta_cols if c in self.df_item.columns]
        
        df_item_small = self.df_item[available_cols].drop_duplicates("item_id")
        df = df.merge(df_item_small, on="item_id", how="left")
        
        for c in available_cols:
            if c != "item_id": df[c] = df[c].fillna("UNK")
        
        if "price" in df.columns: df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
        else: df["price"] = 0.0
        if "quantity" in df.columns: df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(1.0)
        else: df["quantity"] = 1.0
        df["spent_row"] = df["price"] * df["quantity"]

        ui = df.groupby(["customer_id", "item_id"], as_index=False).agg(
            freq_cnt=("item_id", "size"),
            last_purchase_dt=("created_datetime", "max")
        )
        
        if self.weight_type == "log_count": ui["base_weight"] = np.log1p(ui["freq_cnt"])
        else: ui["base_weight"] = ui["freq_cnt"]

        if self.use_ui_recency:
            days = (cutoff_dt - ui["last_purchase_dt"]).dt.days.fillna(999).clip(lower=0)
            ui["ui_recency_factor"] = np.exp(-self.ui_recency_lambda * days)
        else: ui["ui_recency_factor"] = 1.0
            
        ui["pref_l1"] = 0.0
        ui["value"] = (ui["base_weight"] * ui["ui_recency_factor"] * (1.0 + ui["pref_l1"])).astype(np.float32)
        self.ui_data = ui

    def _build_csr_matrix(self):
        user_cat = self.ui_data["customer_id"].astype("category")
        item_cat = self.ui_data["item_id"].astype("category")
        
        self.user_id_to_index_ = {uid: idx for idx, uid in enumerate(user_cat.cat.categories)}
        self.user_index_to_id_ = {idx: uid for uid, idx in self.user_id_to_index_.items()}
        self.item_id_to_index_ = {iid: idx for idx, iid in enumerate(item_cat.cat.categories)}
        self.index_to_item_id_ = {idx: iid for iid, idx in self.item_id_to_index_.items()}
        
        user_codes = user_cat.cat.codes.values
        item_codes = item_cat.cat.codes.values
        
        for u_idx, i_idx in zip(user_codes, item_codes):
            self.user_history_.setdefault(u_idx, set()).add(i_idx)
            
        return csr_matrix(
            (self.ui_data["value"].values, (user_codes, item_codes)),
            shape=(len(self.user_id_to_index_), len(self.item_id_to_index_))
        )

    def _build_advanced_features_data(self):
        print("   -> Building metadata & user profiles...")
        cols = ["item_id"]
        for c in ["brand", "brand_final", "category", "category_l1", "category_l2", "price"]:
            if c in self.df_item.columns: cols.append(c)
            
        df_meta = self.df_item[cols].drop_duplicates("item_id")
        if "price" in df_meta.columns:
            df_meta["price"] = pd.to_numeric(df_meta["price"], errors="coerce").fillna(0.0)
            
        item_prices_train = {}
        if "price" in self.df_train.columns:
            self.df_train["price"] = pd.to_numeric(self.df_train["price"], errors="coerce").fillna(0.0)
            item_prices_train = self.df_train.groupby("item_id")["price"].mean().to_dict()
        
        for _, row in df_meta.iterrows():
            iid = str(row["item_id"])
            if iid in self.item_id_to_index_:
                idx = self.item_id_to_index_[iid]
                p_meta = row.get("price", 0.0)
                p_final = item_prices_train.get(iid, p_meta)
                brand_val = row.get("brand_final", row.get("brand", "UNK"))
                
                self.item_metadata_[idx] = {
                    "brand": brand_val,
                    "cat": row.get("category", "UNK"),
                    "cat1": row.get("category_l1", "UNK"),
                    "cat2": row.get("category_l2", "UNK"),
                    "price": float(p_final)
                }

        start_date_pop = self.max_date_ - pd.Timedelta(days=30)
        df_recent = self.df_train[self.df_train["created_datetime"] >= start_date_pop]
        self.item_global_pop_30d_ = df_recent["item_id"].value_counts().to_dict()

        valid_u = self.df_train["customer_id"].isin(self.user_id_to_index_)
        df_valid = self.df_train[valid_u].copy()
        df_valid = df_valid.merge(df_meta, on="item_id", how="left")
        
        for c in ["brand", "category_l2"]:
            if c in df_valid.columns: df_valid[c] = df_valid[c].fillna("UNK")
            
        if "spent_row" not in df_valid.columns:
            p = df_valid["price"].astype(float).fillna(0.0) if "price" in df_valid.columns else 0.0
            q = df_valid["quantity"].astype(float).fillna(1.0) if "quantity" in df_valid.columns else 1.0
            df_valid["spent_row"] = p * q
        
        # [RESTORED] Tính toán ngày mua cuối của từng user-item để tính Recency
        self.user_item_last_date_ = df_valid.groupby(["customer_id", "item_id"])["created_datetime"].max().to_dict()
        
        for uid, group in tqdm(df_valid.groupby("customer_id"), desc="Building Profiles"):
            if uid not in self.user_id_to_index_: continue
            u_idx = self.user_id_to_index_[uid]
            item_indices = [self.item_id_to_index_[i] for i in group["item_id"] if i in self.item_id_to_index_]
            
            # [RESTORED] Tính stats cho Category L2 gồm cả ngày mua cuối
            cat_stats = group.groupby("category_l2").agg(
                last_date=("created_datetime", "max"),
                avg_spent=("spent_row", "mean")
            ).to_dict(orient="index")
            
            brand_col = "brand_final" if "brand_final" in group.columns else "brand"
            
            self.user_profiles_[u_idx] = {
                "item_counts": Counter(item_indices),
                "brand_counts": Counter(group[brand_col]),
                "cat2_counts": Counter(group["category_l2"]),
                "cat2_profile": cat_stats # Lưu đầy đủ profile (avg_spent + last_date)
            }

    def _build_region_trending(self):
        print("   -> Building Region-based Trending...")
        if "region" in self.df_user.columns:
            df_u_reg = self.df_user[["customer_id", "region"]].drop_duplicates("customer_id")
            df_u_reg["customer_id"] = df_u_reg["customer_id"].astype(str)
            self.user_region_map_ = df_u_reg.set_index("customer_id")["region"].fillna("UNK").to_dict()
        else:
            self.user_region_map_ = {}

        start_date_pop = self.max_date_ - pd.Timedelta(days=30)
        df_trend = self.df_train[self.df_train["created_datetime"] >= start_date_pop].copy()
        df_trend = df_trend[df_trend["item_id"].isin(self.item_id_to_index_)]
        
        if df_trend.empty: return

        df_trend["customer_id_str"] = df_trend["customer_id"].astype(str)
        df_trend["region"] = df_trend["customer_id_str"].map(self.user_region_map_).fillna("UNK")
        
        region_counts = df_trend.groupby(["region", "item_id"]).size().reset_index(name="count")
        region_counts = region_counts.sort_values(["region", "count"], ascending=[True, False])
        
        self.region_trending_indices_ = {}
        for region, group in region_counts.groupby("region"):
            top_items = group.head(100)["item_id"].tolist()
            top_indices = [self.item_id_to_index_[i] for i in top_items if i in self.item_id_to_index_]
            if top_indices:
                self.region_trending_indices_[region] = top_indices
                
        print(f"   -> Built trending for {len(self.region_trending_indices_)} regions.")

    def _build_pregnancy_list(self):
        print("   -> Building Pregnancy/Newborn Items List...")
        keywords = ["STEP 1", "NEWBORN", "SƠ SINH", "PREGNAN", "0-6M", "0-12M", "0-1Y", "NB"]
        mask = pd.Series(False, index=self.df_item.index)
        for col in ['age_group_final', 'description', 'category', 'category_l1', 'category_l2']:
            if col in self.df_item.columns:
                mask |= self.df_item[col].str.upper().str.contains('|'.join(keywords), na=False)
        items = self.df_item[mask]['item_id'].unique()
        valid_items = []
        for i in items:
            if i in self.item_id_to_index_ and self.item_global_pop_30d_.get(i, 0) > 0:
                valid_items.append(self.item_id_to_index_[i])
        valid_items.sort(key=lambda idx: self.item_global_pop_30d_.get(self.index_to_item_id_[idx], 0), reverse=True)
        self.pregnancy_items_indices_ = valid_items[:100]

    def _build_fashion_trending(self):
        print("   -> Building Fashion/Seasonal Items List...")
        fashion_keywords = ["THỜI TRANG", "QUẦN ÁO", "PHỤ KIỆN", "ĐỒ CHƠI", "GIFT", "ĐÔNG"]
        mask = pd.Series(False, index=self.df_item.index)
        for col in ['category', 'category_l1', 'category_l2']:
            if col in self.df_item.columns:
                mask |= self.df_item[col].str.upper().str.contains('|'.join(fashion_keywords), na=False)
        items = self.df_item[mask]['item_id'].unique()
        valid_items = []
        for i in items:
            if i in self.item_id_to_index_ and self.item_global_pop_30d_.get(i, 0) > 0:
                valid_items.append(self.item_id_to_index_[i])
        valid_items.sort(key=lambda idx: self.item_global_pop_30d_.get(self.index_to_item_id_[idx], 0), reverse=True)
        self.fashion_trending_indices_ = valid_items[:100]

    def _build_popularity(self):
        pop_counts = self.ui_data.groupby("item_id")["value"].sum()
        sorted_items = pop_counts.sort_values(ascending=False).index.tolist()
        self.popular_item_indices_ = []
        for iid in sorted_items:
            if iid in self.item_id_to_index_:
                self.popular_item_indices_.append(self.item_id_to_index_[iid])

    def get_user_history_dict(self):
        history_dict = {}
        for u_idx, item_indices in self.user_history_.items():
            user_str = str(self.user_index_to_id_[u_idx])
            items_str = set()
            for i_idx in item_indices:
                if i_idx in self.index_to_item_id_:
                    items_str.add(str(self.index_to_item_id_[i_idx]))
            history_dict[user_str] = items_str
        return history_dict

    def recommend_candidates(self, user_list, allow_repeat=True):
        rows = []
        pop_indices = self.popular_item_indices_[:self.top_k]
        
        # --- Helper: Add Row ---
        def add_row(u_id, idx, score, s_rank, s_sim_max, s_sim_avg, s_support):
            if idx not in self.index_to_item_id_: return
            
            item_id_str = self.index_to_item_id_[idx]
            c_meta = self.item_metadata_.get(idx, {"brand": "UNK", "cat2": "UNK", "price": 0})
            
            f_item_cnt = 0
            f_brand_cnt = 0
            f_cat2_cnt = 0
            f_days_item = 999
            f_days_cat = 999
            f_price_ratio = 1.0
            
            if u_id in self.user_id_to_index_:
                u_idx_local = self.user_id_to_index_[u_id]
                if u_idx_local in self.user_profiles_:
                    u_prof = self.user_profiles_[u_idx_local]
                    f_item_cnt = u_prof["item_counts"].get(idx, 0)
                    f_brand_cnt = u_prof["brand_counts"].get(c_meta["brand"], 0)
                    f_cat2_cnt = u_prof["cat2_counts"].get(c_meta["cat2"], 0)
                    
                    # [RESTORED] Tính Recency: Days since last purchase of this item
                    last_item_dt = self.user_item_last_date_.get((u_id, item_id_str), None)
                    if last_item_dt:
                        f_days_item = (self.max_date_ - last_item_dt).days
                    
                    # [RESTORED] Tính Recency & Price Ratio cho Category
                    cat_stat = u_prof["cat2_profile"].get(c_meta["cat2"], {})
                    last_cat_dt = cat_stat.get("last_date", None)
                    if last_cat_dt:
                        f_days_cat = (self.max_date_ - last_cat_dt).days
                    
                    avg_spent_cat = cat_stat.get("avg_spent", 0)
                    if avg_spent_cat > 0:
                        f_price_ratio = c_meta["price"] / avg_spent_cat

            f_pop_30d = self.item_global_pop_30d_.get(item_id_str, 0)
            
            rows.append({
                "customer_id": u_id, "item_id": item_id_str,
                "stage1_rank": s_rank, "stage1_score": score, 
                "sim_max": s_sim_max, "sim_avg": s_sim_avg, "support_cnt": s_support,
                
                # --- Features (With Recency Restored) ---
                "item_hist_cnt": f_item_cnt, 
                "brand_match_cnt": f_brand_cnt, 
                "cat2_match_cnt": f_cat2_cnt,
                "feat_days_since_item": f_days_item, 
                "feat_days_since_cat": f_days_cat,
                "feat_price_ratio": f_price_ratio, 
                "feat_pop_30d": f_pop_30d, 
                "feat_log_price": np.log1p(c_meta["price"]),
            })

        for user_id in tqdm(user_list, desc="Generating Candidates"):
            user_id_str = str(user_id)
            
            # Cold User
            if user_id not in self.user_id_to_index_:
                region = self.user_region_map_.get(user_id_str, "UNK")
                reg_pool = self.region_trending_indices_.get(region, [])
                final_indices = []
                
                if self.fashion_trending_indices_:
                    final_indices.extend(np.random.choice(self.fashion_trending_indices_, min(len(self.fashion_trending_indices_), 8), replace=False))
                if self.pregnancy_items_indices_:
                    final_indices.extend(np.random.choice(self.pregnancy_items_indices_, min(len(self.pregnancy_items_indices_), 6), replace=False))
                if reg_pool:
                    final_indices.extend(np.random.choice(reg_pool, min(len(reg_pool), 4), replace=False))
                
                needed = 20 - len(final_indices)
                if needed > 0:
                    candidates_pool = [i for i in pop_indices if i not in final_indices]
                    final_indices.extend(candidates_pool[:needed])
                
                final_indices = list(set(final_indices))[:20]
                for rank, idx in enumerate(final_indices, start=1):
                    add_row(user_id, idx, 0.0, rank, 0.0, 0.0, 0)
                continue
            
            # Warm User
            u_idx = self.user_id_to_index_[user_id]
            history_indices = self.user_history_.get(u_idx, set())
            
            if not history_indices:
                for rank, idx in enumerate(pop_indices[:20], start=1):
                    add_row(user_id, idx, 0.0, rank, 0.0, 0.0, 0)
                continue

            candidates = {}     
            max_sims = {}       
            support_counts = {} 
            
            for h_idx in history_indices:
                nbrs = self.item_neighbors_[h_idx]
                sims = self.item_neighbor_sims_[h_idx]
                for n_idx, sim in zip(nbrs, sims):
                    if (not allow_repeat) and (n_idx in history_indices): continue 
                    s = float(sim)
                    candidates[n_idx] = candidates.get(n_idx, 0.0) + s
                    max_sims[n_idx] = max(max_sims.get(n_idx, 0.0), s)
                    support_counts[n_idx] = support_counts.get(n_idx, 0) + 1
            
            sorted_cands = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:self.top_k]
            
            rank = 1
            added_indices = set()
            for c_idx, score in sorted_cands:
                add_row(user_id, c_idx, score, rank, max_sims[c_idx], score / support_counts[c_idx], support_counts[c_idx])
                added_indices.add(c_idx)
                rank += 1
            
            if len(sorted_cands) < self.top_k:
                for p_idx in self.popular_item_indices_:
                    if p_idx not in added_indices:
                        if (not allow_repeat) and (p_idx in history_indices): continue
                        add_row(user_id, p_idx, 0.0, rank, 0.0, 0.0, 0)
                        rank += 1
                        added_indices.add(p_idx)
                        if rank > self.top_k: break
                            
        return pd.DataFrame(rows)
    
    def get_stage2_features(self, candidates_df):
        return candidates_df