"""stage2_feature_label.py

Stage 2 (data prep): Build a Feature-Label dataset for top-k recommendation.

This module implements your `build_feature_label(...)` spec (HIST vs RECENT)
WITHOUT requiring `segment_name`.

It also supports joining additional features from external parquet files:

Item-level:
- price_segment
- top10_by_cat
- top10_by_cat_month (single file with columns: month, category_l1, item_id, total_sold, rank)

User-level:
- buy_segment
- luxury_level
- age_final
- brand_segment

Operational requirements:
- Print clear progress messages.
- Save intermediate outputs.

Notes
- This module does NOT train Stage 2; it prepares the training table.
- The "min_coo" feature is NOT implemented here because its definition/join keys
  were not specified.

"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List

import polars as pl
from tqdm.auto import tqdm


# =========================================================
# Logging
# =========================================================

def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_now_str()}] {msg}", flush=True)


# =========================================================
# Config
# =========================================================

@dataclass
class Stage2Params:
    # IO
    data_dir: str = "./preprocessed-dataset"
    new_feature_dir: str = "./new-feature"
    user_parquet_contains: str = "user_chunk"
    item_parquet_contains: str = "item_chunk"
    trx_parquet_contains: str = "purchase_history_daily_chunk"

    # core keys
    created_col: str = "created_date"
    user_col: str = "customer_id"
    item_col: str = "item_id"

    # time windows
    len_hist: int = 120
    len_recent: int = 28

    # optional reference datetime (ISO8601); if None, use max(created_col)
    ref_datetime: Optional[str] = None

    # saving
    out_dir: str = "./artifacts/stage2_feature_label"
    save_intermediate: bool = True
    compute_stats: bool = False

    # extra feature parquet paths (optional)
    price_segment_path: Optional[str] = None
    customer_behavior_path: Optional[str] = None
    customer_luxury_path: Optional[str] = None
    customer_age_features_path: Optional[str] = None
    brand_segment_path: Optional[str] = None
    top10_by_cat_path: Optional[str] = None
    top10_by_cat_month_path: Optional[str] = None

    # top10_by_cat_month join control
    top10_month_lag: int = 2          # e.g., Jan/2025 uses Nov/2024 if lag=2

    # Item attribute columns in items_lf (your current schema)
    # You can override these in JSON if needed.
    item_brand_col: str = "brand"
    item_age_bucket_col: str = "age_group_final"
    item_category_col: str = "category_l2"
    item_category_l1_col: str = "category_l1"
    item_target_user_group_col: str = "gender_target_final"


def load_params(path: str) -> Stage2Params:
    """Load params from JSON.

    Supports either flat keys or a nested `paths` block.

    Example:
      {
        "data_dir": "...",
        "new_feature_dir": "...",
        "paths": {
          "price_segment": "price_segment.parquet",
          ...
        }
      }
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    allowed = {fd.name for fd in fields(Stage2Params)}

    # Pull nested paths if present
    paths = raw.get("paths", {}) if isinstance(raw.get("paths", {}), dict) else {}

    # Map common path keys
    path_map = {
        "price_segment": "price_segment_path",
        "customer_behavior": "customer_behavior_path",
        "customer_luxury": "customer_luxury_path",
        "customer_age_features": "customer_age_features_path",
        "brand_segment": "brand_segment_path",
        "top10_by_cat": "top10_by_cat_path",
        "top10_by_cat_month": "top10_by_cat_month_path",
    }

    merged: Dict[str, Any] = {}
    for k, v in raw.items():
        if k in allowed:
            merged[k] = v

    for k, v in paths.items():
        if k in path_map:
            merged[path_map[k]] = v

    p = Stage2Params(**merged)

    # If relative paths are provided, resolve relative to new_feature_dir
    def _resolve(pth: Optional[str]) -> Optional[str]:
        if not pth:
            return None
        if os.path.isabs(pth):
            return pth
        # treat as relative filename under new_feature_dir
        return os.path.join(p.new_feature_dir, pth)

    p.price_segment_path = _resolve(p.price_segment_path)
    p.customer_behavior_path = _resolve(p.customer_behavior_path)
    p.customer_luxury_path = _resolve(p.customer_luxury_path)
    p.customer_age_features_path = _resolve(p.customer_age_features_path)
    p.brand_segment_path = _resolve(p.brand_segment_path)
    p.top10_by_cat_path = _resolve(p.top10_by_cat_path)
    p.top10_by_cat_month_path = _resolve(p.top10_by_cat_month_path)

    return p


def save_params(params: Stage2Params, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(params), f, ensure_ascii=False, indent=2, default=str)


# =========================================================
# Time helpers
# =========================================================

def _parse_ref_datetime(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s)
    except Exception:
        # fallback: allow "YYYY-MM-DD"
        return datetime.fromisoformat(s.strip() + " 00:00:00")


def compute_time_windows(ref_dt: datetime, len_hist: int, len_recent: int):
    """Return (begin_hist, end_hist, begin_recent, end_recent)."""
    end_recent = ref_dt
    begin_recent = ref_dt - timedelta(days=int(len_recent) - 1)

    end_hist = begin_recent - timedelta(days=1)
    begin_hist = end_hist - timedelta(days=int(len_hist) - 1)

    return begin_hist, end_hist, begin_recent, end_recent


def _shift_months(dt: datetime, months: int) -> datetime:
    """Shift datetime by N months (clamped day)."""
    y = dt.year
    m = dt.month + int(months)
    while m <= 0:
        y -= 1
        m += 12
    while m > 12:
        y += 1
        m -= 12

    import calendar

    last_day = calendar.monthrange(y, m)[1]
    d = min(dt.day, last_day)
    return dt.replace(year=y, month=m, day=d)


def _month_key(dt: datetime) -> str:
    return dt.strftime("%Y-%m")


# =========================================================
# Polars helpers
# =========================================================

def ensure_datetime_col(lf: pl.LazyFrame, col: str) -> pl.LazyFrame:
    cols = lf.columns
    if col not in cols:
        raise ValueError(f"LazyFrame must contain `{col}`. Available columns: {cols}")
    return lf.with_columns(pl.col(col).cast(pl.Datetime, strict=False))


def _maybe_scan_parquet(path: Optional[str]) -> Optional[pl.LazyFrame]:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parquet not found: {path}")
    return pl.scan_parquet(path)


def _glob_parquets(data_dir: str, contains: str) -> List[str]:
    files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".parquet") and (contains in os.path.basename(f))
    ]
    if not files:
        raise FileNotFoundError(f"Không tìm thấy parquet chứa '{contains}' trong: {data_dir}")
    return sorted(files)


def scan_user(params: Stage2Params) -> pl.LazyFrame:
    return pl.scan_parquet(_glob_parquets(params.data_dir, params.user_parquet_contains))


def scan_item(params: Stage2Params) -> pl.LazyFrame:
    return pl.scan_parquet(_glob_parquets(params.data_dir, params.item_parquet_contains))


def scan_transaction(params: Stage2Params) -> pl.LazyFrame:
    return pl.scan_parquet(_glob_parquets(params.data_dir, params.trx_parquet_contains))


def _sink_parquet(lf: pl.LazyFrame, path: str, msg: str, compute_stats: bool) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    log(f"{msg} -> {path}")

    if compute_stats:
        try:
            n = lf.select(pl.len()).collect(streaming=True).item()
            log(f"  rows = {n:,}")
        except Exception:
            log("  rows = (không tính được trong bước này)")

    # sink_parquet triggers collect internally; if Polars hits an unknown dtype
    # panic, we want the upstream graph to be fully typed.
    lf.sink_parquet(path)


def _require_cols(lf: pl.LazyFrame, cols: List[str], name: str) -> None:
    have = set(lf.columns)
    missing = [c for c in cols if c not in have]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}. Available: {lf.columns}")


def _build_item_attrs(items_lf: pl.LazyFrame, params: Stage2Params) -> pl.LazyFrame:
    ic = params.item_col

    # Ensure required base columns exist
    _require_cols(items_lf, [ic, params.item_category_l1_col], "items_lf")

    cols = items_lf.columns

    def _col_or_missing(c: str) -> pl.Expr:
        return pl.col(c) if c in cols else pl.lit("__UNK__")

    return items_lf.select([
        pl.col(ic),
        _col_or_missing(params.item_brand_col).cast(pl.Utf8, strict=False).alias(params.item_brand_col),
        _col_or_missing(params.item_age_bucket_col).cast(pl.Utf8, strict=False).alias(params.item_age_bucket_col),
        _col_or_missing(params.item_category_col).cast(pl.Utf8, strict=False).alias(params.item_category_col),
        _col_or_missing(params.item_category_l1_col).cast(pl.Utf8, strict=False).alias(params.item_category_l1_col),
        _col_or_missing(params.item_target_user_group_col).cast(pl.Utf8, strict=False).alias(params.item_target_user_group_col),
    ])


# =========================================================
# Core: build feature-label (NO segment_name)
# =========================================================

def build_feature_label(
    transactions_lf: pl.LazyFrame,
    items_lf: pl.LazyFrame,
    users_lf: pl.LazyFrame,  # kept for spec
    begin_hist: datetime,
    end_hist: datetime,
    begin_recent: datetime,
    end_recent: datetime,
    params: Stage2Params,
    candidate_pairs_lf: Optional[pl.LazyFrame] = None,
) -> pl.LazyFrame:
    """Create Feature-Label LazyFrame.

    Output includes:
      - customer_id, item_id
      - brand_counts, age_counts, category_counts
      - target_user_group_counts
      - time_since_last_purchase_in_B_category
      - Y

    Note: No segment_name and no segment_counts.
    """

    uc, ic, tc = params.user_col, params.item_col, params.created_col

    # 0) cleanup
    base_tx = transactions_lf.drop("segment_name_right") if "segment_name_right" in transactions_lf.columns else transactions_lf

    # Ensure datetime + provide typed bounds (avoid Polars unknown dtype)
    base_tx = ensure_datetime_col(base_tx, tc)
    bh = pl.lit(begin_hist, dtype=pl.Datetime)
    eh = pl.lit(end_hist, dtype=pl.Datetime)
    br = pl.lit(begin_recent, dtype=pl.Datetime)
    er = pl.lit(end_recent, dtype=pl.Datetime)

    # 1) hist / recent
    hist_lf = base_tx.filter(pl.col(tc).is_between(bh, eh, closed="both"))
    recent_lf = base_tx.filter(pl.col(tc).is_between(br, er, closed="both"))

    # 2) item attrs (mapped to your current schema via params)
    item_attrs = _build_item_attrs(items_lf, params)

    # 3) HIST enriched
    hist_enriched = hist_lf.join(item_attrs, on=ic, how="left")

    # 4) count features
    brand_counts = (
        hist_enriched
        .group_by([uc, params.item_brand_col])
        .agg(pl.len().alias("brand_counts"))
    )

    age_counts = (
        hist_enriched
        .group_by([uc, params.item_age_bucket_col])
        .agg(pl.len().alias("age_counts"))
    )

    category_counts = (
        hist_enriched
        .group_by([uc, params.item_category_col])
        .agg(pl.len().alias("category_counts"))
    )

    target_user_group_counts = (
        hist_enriched
        .group_by([uc, params.item_target_user_group_col])
        .agg(pl.len().alias("target_user_group_counts"))
    )

    # 5) last purchase per (user, category)
    last_cat_purchase = (
        hist_enriched
        .group_by([uc, params.item_category_col])
        .agg(pl.col(tc).max().alias("last_purchase_date"))
    )

    # 6) candidate pairs
    if candidate_pairs_lf is None:
        hist_pairs = hist_lf.select([uc, ic]).unique()
        recent_pairs = recent_lf.select([uc, ic]).unique()
        candidate_pairs = pl.concat([hist_pairs, recent_pairs]).unique()
    else:
        candidate_pairs = candidate_pairs_lf.select([uc, ic]).unique()
        recent_pairs = recent_lf.select([uc, ic]).unique()  # needed for labels

    # 7) candidate enriched
    candidate_enriched = candidate_pairs.join(item_attrs, on=ic, how="left")

    # 8) join all + compute recency feature (typed)
    end_hist_lit = pl.lit(end_hist, dtype=pl.Datetime)
    features = (
        candidate_enriched
        .join(brand_counts, on=[uc, params.item_brand_col], how="left")
        .join(age_counts, on=[uc, params.item_age_bucket_col], how="left")
        .join(category_counts, on=[uc, params.item_category_col], how="left")
        .join(target_user_group_counts, on=[uc, params.item_target_user_group_col], how="left")
        .join(last_cat_purchase, on=[uc, params.item_category_col], how="left")
        .with_columns([
            pl.col("brand_counts").fill_null(0).cast(pl.Int32),
            pl.col("age_counts").fill_null(0).cast(pl.Int32),
            pl.col("category_counts").fill_null(0).cast(pl.Int32),
            pl.col("target_user_group_counts").fill_null(0).cast(pl.Int32),
        ])
        .with_columns(
            (
                (
                    end_hist_lit
                    - pl.col("last_purchase_date").cast(pl.Datetime, strict=False)
                )
                .dt.total_days()
                .cast(pl.Int32)
            ).alias("time_since_last_purchase_in_B_category")
        )
        .with_columns(
            pl.col("time_since_last_purchase_in_B_category").fill_null(9999).cast(pl.Int32)
        )
    )

    # 9) labels from RECENT
    labels = recent_pairs.with_columns(pl.lit(1, dtype=pl.Int8).alias("Y"))

    feature_label_lf = (
        features
        .join(labels, on=[uc, ic], how="left")
        .with_columns(pl.col("Y").fill_null(0).cast(pl.Int8))
    )

    return feature_label_lf


# =========================================================
# Extra feature joins
# =========================================================

def _normalize_month_key(expr: pl.Expr) -> pl.Expr:
    """Normalize a `month` column to YYYY-MM (Utf8).

    Works for inputs like:
    - 2024-11-01
    - 2024-11
    - 202411
    """
    m = expr.cast(pl.Utf8, strict=False)

    # if contains '-', take first 7
    has_dash = m.str.contains("-")

    # if length == 6 (YYYYMM) -> YYYY-MM
    len6 = m.str.len_chars() == 6

    return (
        pl.when(has_dash)
        .then(m.str.slice(0, 7))
        .when(len6)
        .then(pl.concat_str([m.str.slice(0, 4), pl.lit("-"), m.str.slice(4, 2)]))
        .otherwise(m)
    )


def add_extra_features(feature_label_lf: pl.LazyFrame, params: Stage2Params, end_hist: datetime) -> pl.LazyFrame:
    uc, ic = params.user_col, params.item_col
    out = feature_label_lf

    # item: price_segment
    lf_price = _maybe_scan_parquet(params.price_segment_path)
    if lf_price is not None:
        cols = lf_price.columns
        if ic in cols and "price_segment" in cols:
            out = out.join(lf_price.select([ic, "price_segment"]).unique(), on=ic, how="left")
            out = out.with_columns(pl.col("price_segment").cast(pl.Utf8, strict=False).fill_null("__MISSING__"))
        else:
            log("WARN: price_segment_path provided but missing [item_id, price_segment].")

    # user: buy_segment
    lf_buy = _maybe_scan_parquet(params.customer_behavior_path)
    if lf_buy is not None:
        cols = lf_buy.columns
        if uc in cols and "buy_segment" in cols:
            out = out.join(lf_buy.select([uc, "buy_segment"]).unique(), on=uc, how="left")
            out = out.with_columns(pl.col("buy_segment").cast(pl.Utf8, strict=False).fill_null("__MISSING__"))
        else:
            log("WARN: customer_behavior_path missing [customer_id, buy_segment].")

    # user: luxury_level
    lf_lux = _maybe_scan_parquet(params.customer_luxury_path)
    if lf_lux is not None:
        cols = lf_lux.columns
        if uc in cols and "luxury_level" in cols:
            out = out.join(lf_lux.select([uc, "luxury_level"]).unique(), on=uc, how="left")
            out = out.with_columns(pl.col("luxury_level").cast(pl.Utf8, strict=False).fill_null("__MISSING__"))
        else:
            log("WARN: customer_luxury_path missing [customer_id, luxury_level].")

    # user: age_final
    lf_age = _maybe_scan_parquet(params.customer_age_features_path)
    if lf_age is not None:
        cols = lf_age.columns
        if uc in cols and "age_final" in cols:
            out = out.join(lf_age.select([uc, "age_final"]).unique(), on=uc, how="left")
            out = out.with_columns(pl.col("age_final").cast(pl.Float32, strict=False).fill_null(-1.0))
        else:
            log("WARN: customer_age_features_path missing [customer_id, age_final].")

    # user: brand_segment
    lf_bs = _maybe_scan_parquet(params.brand_segment_path)
    if lf_bs is not None:
        cols = lf_bs.columns
        if uc in cols and "brand_segment" in cols:
            out = out.join(lf_bs.select([uc, "brand_segment"]).unique(), on=uc, how="left")
            out = out.with_columns(pl.col("brand_segment").cast(pl.Utf8, strict=False).fill_null("__MISSING__"))
        else:
            log("WARN: brand_segment_path missing [customer_id, brand_segment].")

    # # item: top10_by_cat
    # lf_top = _maybe_scan_parquet(params.top10_by_cat_path)
    # if lf_top is not None:
    #     cols = lf_top.columns
    #     if ic in cols:
    #         if "top10_by_cat" in cols:
    #             out = out.join(lf_top.select([ic, "top10_by_cat"]).unique(), on=ic, how="left")
    #             out = out.with_columns(pl.col("top10_by_cat").cast(pl.Float32, strict=False).fill_null(0.0))
    #         elif "rank" in cols:
    #             out = out.join(lf_top.select([ic, "rank"]).unique(), on=ic, how="left")
    #             out = out.rename({"rank": "rank_top10_by_cat"})
    #             out = out.with_columns(pl.col("rank_top10_by_cat").cast(pl.Int32, strict=False).fill_null(9999))
    #         else:
    #             log("WARN: top10_by_cat_path has item_id but missing [top10_by_cat] or [rank].")
    #     else:
    #         log("WARN: top10_by_cat_path missing item_id column.")

    # item: top10_by_cat_month (single file: month, category_l1, item_id, total_sold, rank)
    lf_topm = _maybe_scan_parquet(params.top10_by_cat_month_path)
    if lf_topm is not None:
        cols = set(lf_topm.columns)
        need = {ic, params.item_category_l1_col, "month"}
        if not need.issubset(cols):
            log("WARN: top10_by_cat_month_path missing required columns [month, category_l1, item_id].")
        else:
            month_dt = _shift_months(end_hist, -int(params.top10_month_lag))
            mkey = _month_key(month_dt)

            # Ensure out has category_l1 (it should, from item_attrs join)
            if params.item_category_l1_col not in out.columns:
                log("WARN: feature_label table missing category_l1; join top10_by_cat_month by (item_id, month) only.")
                out = out.with_columns(pl.lit(mkey).alias("trend_month"))
                right = lf_topm.with_columns(_normalize_month_key(pl.col("month")).alias("month_key"))

                keep_cols = [ic, "month_key"]
                if "rank" in cols:
                    keep_cols.append("rank")
                if "total_sold" in cols:
                    keep_cols.append("total_sold")

                out = out.join(
                    right.select(keep_cols).unique(),
                    left_on=[ic, "trend_month"],
                    right_on=[ic, "month_key"],
                    how="left",
                )
            else:
                out = out.with_columns(pl.lit(mkey).alias("trend_month"))
                right = lf_topm.with_columns(_normalize_month_key(pl.col("month")).alias("month_key"))

                keep_cols = [ic, params.item_category_l1_col, "month_key"]
                if "rank" in cols:
                    keep_cols.append("rank")
                if "total_sold" in cols:
                    keep_cols.append("total_sold")

                out = out.join(
                    right.select(keep_cols).unique(),
                    left_on=[ic, params.item_category_l1_col, "trend_month"],
                    right_on=[ic, params.item_category_l1_col, "month_key"],
                    how="left",
                )

            # Standardize names + fill defaults
            if "rank" in out.columns:
                out = out.rename({"rank": "rank_top10_by_cat_month"})
            if "total_sold" in out.columns:
                out = out.rename({"total_sold": "total_sold_top10_by_cat_month"})

            if "rank_top10_by_cat_month" in out.columns:
                out = out.with_columns(pl.col("rank_top10_by_cat_month").cast(pl.Int32, strict=False).fill_null(9999))
            if "total_sold_top10_by_cat_month" in out.columns:
                out = out.with_columns(pl.col("total_sold_top10_by_cat_month").cast(pl.Float32, strict=False).fill_null(0.0))

    return out


# =========================================================
# Final selector
# =========================================================

def select_final_columns(feature_label_lf: pl.LazyFrame, params: Stage2Params) -> pl.LazyFrame:
    base_cols = [
        params.user_col,
        params.item_col,
        "brand_counts",
        "age_counts",
        "category_counts",
        "target_user_group_counts",
        "time_since_last_purchase_in_B_category",
        "Y",
    ]

    extra_cols = [
        # user/item features
        "price_segment",
        "buy_segment",
        "luxury_level",
        "age_final",
        "brand_segment",
        "top10_by_cat",
        "rank_top10_by_cat",
        # month trend
        "trend_month",
        "rank_top10_by_cat_month",
        "total_sold_top10_by_cat_month",
    ]

    cols = feature_label_lf.columns
    keep = [c for c in (base_cols + extra_cols) if c in cols]
    return feature_label_lf.select(keep)


# =========================================================
# Orchestrator
# =========================================================

def build_and_save_feature_label(
    transactions_lf: pl.LazyFrame,
    items_lf: pl.LazyFrame,
    users_lf: pl.LazyFrame,
    params: Stage2Params,
    candidate_pairs_lf: Optional[pl.LazyFrame] = None,
) -> str:
    os.makedirs(params.out_dir, exist_ok=True)

    pbar = tqdm(total=4, desc="Stage2: build feature-label", unit="step")

    # Ensure created datetime
    transactions_lf = ensure_datetime_col(transactions_lf, params.created_col)

    # Resolve ref datetime
    if params.ref_datetime:
        ref_dt = _parse_ref_datetime(params.ref_datetime)
        log(f"Using ref_datetime from config: {ref_dt}")
    else:
        log(f"ref_datetime not provided; computing max({params.created_col}) from transactions")
        ref_dt = (
            transactions_lf
            .select(pl.col(params.created_col).max().alias("ref_dt"))
            .collect(streaming=True)
            .item()
        )
        if not isinstance(ref_dt, datetime):
            ref_dt = datetime.fromisoformat(str(ref_dt))
        log(f"Computed ref_datetime = {ref_dt}")

    begin_hist, end_hist, begin_recent, end_recent = compute_time_windows(
        ref_dt=ref_dt,
        len_hist=params.len_hist,
        len_recent=params.len_recent,
    )

    log("Time windows:")
    log(f"  HIST  : {begin_hist} -> {end_hist} (len_hist={params.len_hist} days)")
    log(f"  RECENT: {begin_recent} -> {end_recent} (len_recent={params.len_recent} days)")

    save_params(params, os.path.join(params.out_dir, "stage2_params_used.json"))

    # Step 1: base feature-label
    log("[Step 1/4] Building base feature-label")
    fl_base = build_feature_label(
        transactions_lf=transactions_lf,
        items_lf=items_lf,
        users_lf=users_lf,
        begin_hist=begin_hist,
        end_hist=end_hist,
        begin_recent=begin_recent,
        end_recent=end_recent,
        params=params,
        candidate_pairs_lf=candidate_pairs_lf,
    )

    if params.save_intermediate:
        _sink_parquet(
            fl_base,
            os.path.join(params.out_dir, "feature_label_base.parquet"),
            msg="Saved base feature-label",
            compute_stats=params.compute_stats,
        )
    pbar.update(1)

    # Step 2: join extras
    log("[Step 2/4] Adding extra features")
    fl_extra = add_extra_features(fl_base, params=params, end_hist=end_hist)

    if params.save_intermediate:
        _sink_parquet(
            fl_extra,
            os.path.join(params.out_dir, "feature_label_with_extras.parquet"),
            msg="Saved feature-label with extras",
            compute_stats=params.compute_stats,
        )
    pbar.update(1)

    # Step 3: select final
    log("[Step 3/4] Selecting final columns")
    fl_final = select_final_columns(fl_extra, params=params)
    pbar.update(1)

    # Step 4: save final
    final_path = os.path.join(params.out_dir, "feature_label_final.parquet")
    log("[Step 4/4] Saving FINAL feature-label")
    _sink_parquet(fl_final, final_path, msg="Saved FINAL feature-label", compute_stats=params.compute_stats)

    pbar.update(1)
    pbar.close()

    log("DONE: feature-label created")
    return final_path


def build_feature_label_pipeline(params: Stage2Params, candidate_pairs_lf: Optional[pl.LazyFrame] = None) -> str:
    log("=== (0) Scan core tables (lazy) ===")
    lf_user = scan_user(params)
    lf_item = scan_item(params)
    lf_trx = scan_transaction(params)

    log(f"User columns (sample): {lf_user.columns[:15]}")
    log(f"Item columns (sample): {lf_item.columns[:15]}")
    log(f"Trx  columns (sample): {lf_trx.columns[:15]}")

    return build_and_save_feature_label(
        transactions_lf=lf_trx,
        items_lf=lf_item,
        users_lf=lf_user,
        params=params,
        candidate_pairs_lf=candidate_pairs_lf,
    )
