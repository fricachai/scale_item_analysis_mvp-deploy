# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
from scipy.stats import ttest_ind

def normalize_item_columns(df: pd.DataFrame):
    """
    正規化欄位名稱：偵測如 A11, B12 等格式並建立對照表。
    """
    ITEM_CODE_RE = re.compile(r"^[A-Za-z]\d{1,3}(_\d+)?$")
    mapping = {}
    new_cols = []
    for col in df.columns:
        s = str(col).strip()
        if ITEM_CODE_RE.match(s):
            new_cols.append(s)
        else:
            match = re.search(r"([A-Za-z]\d{1,3})", s)
            if match:
                code = match.group(1).upper()
                mapping[col] = code
                new_cols.append(code)
            else:
                new_cols.append(col)
    df_norm = df.copy()
    df_norm.columns = new_cols
    return df_norm, mapping

def calculate_cronbach_alpha(df: pd.DataFrame):
    df = df.dropna()
    if df.shape[1] <= 1: return 0.0
    item_vars = df.var(ddof=1)
    total_var = df.sum(axis=1).var(ddof=1)
    k = df.shape[1]
    if total_var == 0: return 0.0
    alpha = (k / (k - 1)) * (1 - item_vars.sum() / total_var)
    return alpha

def run_item_analysis(df_norm: pd.DataFrame):
    """
    極致修正版：對齊 JASP df=25 的分組邏輯。
    1. 採用「嚴格排序名次法」：確保高低分組人數固定為總數的前後 27%。
    2. 解決同分造成的樣本數跳動問題。
    3. CR值與 P值 四捨五入至小數第四位。
    """
    ITEM_CODE_RE = re.compile(r"^[A-Za-z]\d{1,3}(_\d+)?$")
    item_cols = [c for c in df_norm.columns if ITEM_CODE_RE.match(str(c))]
    if not item_cols: return pd.DataFrame()

    df_items = df_norm[item_cols].apply(pd.to_numeric, errors='coerce')
    unique_main_dims = sorted(list(set([c[0].upper() for c in item_cols])))
    
    group_map = {}
    for dim in unique_main_dims:
        dim_cols = [c for c in item_cols if c.startswith(dim)]
        # 計算大構面平均
        dim_mean = df_items[dim_cols].mean(axis=1, skipna=True)
        
        # --- 核心：嚴格名次切分邏輯 ---
        n_total = len(dim_mean)
        k = int(round(n_total * 0.27)) # 計算應取的人數 (例如 13 或 14 人)
        
        # 使用 mergesort 穩定排序，確保同分時位置固定
        ranked = dim_mean.sort_values(kind='mergesort')
        
        low_indices = ranked.head(k).index
        high_indices = ranked.tail(k).index
        
        labels = np.zeros(n_total)
        # 標記高低分組
        labels[dim_mean.index.isin(low_indices)] = 1 # 低分
        labels[dim_mean.index.isin(high_indices)] = 2 # 高
        group_map[dim] = labels

    results = []
    for col in item_cols:
        main_dim = col[0].upper()
        sub_dim = col[:2].upper()
        col_data = df_items[col]
        
        labels = group_map[main_dim]
        low_vals = col_data[labels == 1].dropna()
        high_vals = col_data[labels == 2].dropna()
        
        if len(low_vals) > 1 and len(high_vals) > 1:
            # 執行 Student's t-test (等變異)
            t_stat, p_val = ttest_ind(high_vals, low_vals, equal_var=True)
            cr_value = abs(t_stat)
            cr_p = p_val
        else:
            cr_value, cr_p = np.nan, np.nan

        sub_cols = [c for c in item_cols if c.startswith(sub_dim)]
        if len(sub_cols) > 1:
            sub_sum = df_items[sub_cols].sum(axis=1)
            corrected_sum = sub_sum - col_data.fillna(0)
            citc = col_data.corr(corrected_sum)
            alpha_del = calculate_cronbach_alpha(df_items[sub_cols].drop(columns=[col]))
            overall_alpha = calculate_cronbach_alpha(df_items[sub_cols])
        else:
            citc, alpha_del, overall_alpha = np.nan, np.nan, np.nan

        loading = col_data.corr(df_items[sub_cols].mean(axis=1))

        results.append({
            "構面": main_dim, "子構面": sub_dim, "題項": col,
            "平均數": round(col_data.mean(), 4),
            "標準差": round(col_data.std(), 4),
            "CITC": round(citc, 4) if not np.isnan(citc) else "—",
            "因素負荷量": round(loading, 4) if not np.isnan(loading) else "—",
            "刪除後 Cronbach α": round(alpha_del, 4) if not np.isnan(alpha_del) else "—",
            "該子構面整體 α": round(overall_alpha, 4) if not np.isnan(overall_alpha) else "—",
            "警示標記": "刪題α↑" if (not np.isnan(alpha_del) and alpha_del > overall_alpha) else "—",
            "決斷值(CR: Critical Ratio)": round(cr_value, 4) if not np.isnan(cr_value) else "",
            "CR_p值": f"{cr_p:.4f}" if not np.isnan(cr_p) else ""
        })
    return pd.DataFrame(results)
