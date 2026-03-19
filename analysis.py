# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
from scipy.stats import ttest_ind

# ✅ 支援欄名：
# - A11
# - A01.題目
# - A01 題目
# - A01題目（代碼後直接接中文字，也要能抓到）
CODE_RE_PLAIN = re.compile(r"^[A-Za-z]\d{2,3}$")
CODE_RE_FROM_TEXT = re.compile(r"^([A-Za-z]\d{2,3})(?:[\.、\s]|(?=[^\d]))")

def normalize_item_columns(df: pd.DataFrame):
    """
    將欄名正規化成純代碼（A01/A11/A101...），並回傳 mapping：原欄名 -> 代碼
    """
    mapping = {}
    used = set()
    new_cols = []

    for col in df.columns:
        s = str(col).strip()
        code = None

        if CODE_RE_PLAIN.match(s):
            code = s
        else:
            m = CODE_RE_FROM_TEXT.match(s)
            if m:
                code = m.group(1).upper()

        if code is None:
            new_cols.append(s)
            continue

        base = code
        k = 2
        while code in used:
            code = f"{base}_{k}"
            k += 1

        used.add(code)
        mapping[s] = code
        new_cols.append(code)

    df_norm = df.copy()
    df_norm.columns = new_cols
    return df_norm, mapping

def calculate_cronbach_alpha(df: pd.DataFrame):
    """
    計算 Cronbach's Alpha (處理遺漏值並計算信度)
    """
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
    核心修正：執行項目分析並對齊 JASP 獨立樣本 t 檢定邏輯
    1. 支援大構面基準：決斷值(CR)依「大構面」總平均進行排序分組。
    2. 嚴格切分：採用「排序位子」切分（前 27% 與 後 27%），確保人數與 JASP 同步。
    3. 綜合標記：整合平均數、CITC、負荷量、刪題α與 CR 值。
    """
    # 識別題項欄位
    ITEM_CODE_RE = re.compile(r"^[A-Za-z]\d{2,3}(_\d+)?$")
    item_cols = [c for c in df_norm.columns if ITEM_CODE_RE.match(str(c).strip())]
    
    if not item_cols:
        return pd.DataFrame()

    df_items = df_norm[item_cols].apply(pd.to_numeric, errors='coerce')
    
    # 取得大構面清單 (A, B, C...)
    unique_main_dims = sorted(list(set([c[0].upper() for c in item_cols])))
    
    # --- 預先計算每個「大構面」的高低分組標籤 ---
    group_map = {}
    for dim in unique_main_dims:
        dim_cols = [c for c in item_cols if c.startswith(dim)]
        # 計算大構面總平均 (如 A 構面下所有題項的平均)
        dim_mean = df_items[dim_cols].mean(axis=1, skipna=True)
        
        # 採用「嚴格排序名次法」：確保高低分組人數固定，解決同分跳動
        n_total = len(dim_mean)
        k = int(round(n_total * 0.27)) 
        
        # 使用 mergesort 穩定排序
        ranked = dim_mean.sort_values(kind='mergesort')
        
        low_indices = ranked.head(k).index
        high_indices = ranked.tail(k).index
        
        labels = np.zeros(n_total)
        # 透過 Index 標記，確保人數不多不少
        labels[dim_mean.index.isin(low_indices)] = 1 # 低分組
        labels[dim_mean.index.isin(high_indices)] = 2 # 高分組
        group_map[dim] = labels

    results = []
    for col in item_cols:
        # 子構面邏輯：取題號前兩碼 (如 A1, A2, B1)
        main_dim = col[0].upper()
        sub_dim = col[:2].upper()
        col_data = df_items[col]
        
        # 1. 決斷值 (CR) 計算
        labels = group_map[main_dim]
        low_vals = col_data[labels == 1].dropna()
        high_vals = col_data[labels == 2].dropna()
        
        if len(low_vals) > 1 and len(high_vals) > 1:
            t_stat, p_val = ttest_ind(high_vals, low_vals, equal_var=True)
            cr_value = abs(t_stat)
            cr_p = p_val
        else:
            cr_value, cr_p = np.nan, np.nan

        # 2. CITC 與 Alpha (以「子構面」為範圍)
        sub_cols = [c for c in item_cols if c.startswith(sub_dim)]
        if len(sub_cols) > 1:
            sub_sum = df_items[sub_cols].sum(axis=1)
            corrected_sum = sub_sum - col_data.fillna(0)
            citc_v = col_data.corr(corrected_sum)
            alpha_del = calculate_cronbach_alpha(df_items[sub_cols].drop(columns=[col]))
            overall_alpha = calculate_cronbach_alpha(df_items[sub_cols])
        else:
            citc_v, alpha_del, overall_alpha = np.nan, np.nan, np.nan

        # 3. 因素負荷量 (簡單相關法)
        loading_v = col_data.corr(df_items[sub_cols].mean(axis=1))

        # 4. 警示標記邏輯
        warn = []
        if np.isfinite(citc_v) and citc_v < 0.30:
            warn.append("CITC<.30")
        if np.isfinite(loading_v) and loading_v < 0.50:
            warn.append("loading<.50")
        if np.isfinite(overall_alpha) and np.isfinite(alpha_del) and alpha_del > overall_alpha:
            warn.append("刪題α↑")

        results.append({
            "構面": main_dim,
            "子構面": sub_dim,
            "題項": col,
            "平均數": round(col_data.mean(), 4),
            "標準差": round(col_data.std(), 4),
            "CITC": round(citc_v, 4) if not np.isnan(citc_v) else "—",
            "因素負荷量": round(loading_v, 4) if not np.isnan(loading_v) else "—",
            "刪除後 Cronbach α": round(alpha_del, 4) if not np.isnan(alpha_del) else "—",
            "該子構面整體 α": round(overall_alpha, 4) if not np.isnan(overall_alpha) else "—",
            "警示標記": "；".join(warn) if warn else "—",
            "決斷值(CR: Critical Ratio)": round(cr_value, 4) if not np.isnan(cr_value) else "",
            "CR_p值": f"{cr_p:.4f}" if not np.isnan(cr_p) else ""
        })

    return pd.DataFrame(results)
