# app.py
# -*- coding: utf-8 -*-
import io
import os
import re
import math
import traceback

import numpy as np
import pandas as pd
import streamlit as st
import statsmodels.api as sm
from scipy.stats import norm
from scipy.stats import pearsonr
from scipy.stats import ttest_ind
from scipy.stats import f as f_dist
from statsmodels.stats.stattools import durbin_watson

from analysis import run_item_analysis, normalize_item_columns


# ---- Optional GPT report (if gpt_report.py exists & has generate_gpt_report) ----
GPT_AVAILABLE = False
generate_gpt_report = None
try:
    from gpt_report import generate_gpt_report  # type: ignore
    GPT_AVAILABLE = callable(generate_gpt_report)
except Exception:
    GPT_AVAILABLE = False
    generate_gpt_report = None


# ---- Page ----
st.set_page_config(page_title="柴康偉 論文統計分析專業版(release 1.0) 2026.01.28 ", layout="wide")


import streamlit_authenticator as stauth
import inspect

def secrets_to_dict(x):
    if hasattr(x, "to_dict"):
        return secrets_to_dict(x.to_dict())
    if isinstance(x, dict):
        return {k: secrets_to_dict(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [secrets_to_dict(v) for v in x]
    return x

# ===== 1) 先把 secrets 轉成一般 dict =====
auth_config = secrets_to_dict(st.secrets["auth"])

# ===== 2) 先建立 authenticator（這行一定要在 safe_login 前面）=====
authenticator = stauth.Authenticate(
    auth_config["credentials"],
    auth_config["cookie_name"],
    auth_config["cookie_key"],
    auth_config["cookie_expiry_days"],
)

# ===== 3) 再登入（只呼叫一次，避免 duplicate form）=====
def safe_login(authenticator):
    fn = authenticator.login

    for call in [
        lambda: fn("登入系統", "main"),
        lambda: fn("main", "登入系統"),
        lambda: fn("登入系統"),
        lambda: fn(),
        lambda: fn(location="main"),
        lambda: fn("登入系統", location="main"),
    ]:
        try:
            return call()
        except (TypeError, ValueError):
            continue

    raise RuntimeError("streamlit-authenticator.login() 介面不相容：所有呼叫模式都失敗")


login_ret = safe_login(authenticator)

# ===== 4) 讀取結果（以 session_state 為主）=====
authentication_status = st.session_state.get("authentication_status", None)
name = st.session_state.get("name", "")
username = st.session_state.get("username", "")

if authentication_status is True:
    with st.sidebar:
        try:
            authenticator.logout("登出", "sidebar")
        except TypeError:
            authenticator.logout("登出")
        st.caption(f"登入者：{name} ({username})")
elif authentication_status is False:
    st.error("帳號或密碼錯誤")
    st.stop()
else:
    st.warning("請先登入")
    st.stop()


st.title("📊 柴康偉 論文統計分析專業版(release 1.0) 2026.01.28")


# ---- Helpers ----
def read_csv_safely(uploaded_file) -> pd.DataFrame:
    """
    Robust CSV loader for Streamlit UploadedFile.
    Tries common encodings and handles BOM.
    """
    if uploaded_file is None:
        raise ValueError("尚未上傳 CSV 檔案。")

    raw = uploaded_file.getvalue()
    if raw is None or len(raw) == 0:
        raise ValueError("上傳的檔案是空的（0 bytes）。請確認 CSV 內容是否存在。")

    encodings = ["utf-8-sig", "utf-8", "cp950", "big5", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            bio = io.BytesIO(raw)
            return pd.read_csv(bio, encoding=enc)
        except Exception as e:
            last_err = e

    raise ValueError(f"讀取 CSV 失敗（已嘗試 {encodings}）。最後錯誤：{repr(last_err)}")


def safe_show_exception(e: Exception):
    st.error("發生錯誤（safe）")
    st.code(repr(e))
    with st.expander("Traceback（除錯用）"):
        st.code(traceback.format_exc())


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Excel-friendly: UTF-8 with BOM
    """
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


# =========================================================================
# ✅ 修正點 1：放寬正則表達式，允許只有一碼數字的題號 (例如 D1, E4)
# =========================================================================
ITEM_CODE_RE = re.compile(r"^[A-Za-z]\d{1,3}(_\d+)?$")


def _find_item_cols(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        s = str(c).strip()
        if ITEM_CODE_RE.match(s):
            cols.append(s)
    return cols


def _dim_letter(code: str) -> str | None:
    m = re.match(r"^([A-Za-z])", str(code))
    return m.group(1).upper() if m else None


def build_dim_means_per_row(df_norm: pd.DataFrame) -> pd.DataFrame:
    item_cols_all = _find_item_cols(df_norm)
    if not item_cols_all:
        return pd.DataFrame()

    dims = sorted({d for d in (_dim_letter(c) for c in item_cols_all) if d is not None})

    df_item = df_norm[item_cols_all].apply(pd.to_numeric, errors="coerce")

    out = pd.DataFrame(index=df_norm.index)
    for d in dims:
        cols_d = [c for c in item_cols_all if _dim_letter(c) == d]
        mean_series = df_item[cols_d].mean(axis=1, skipna=True)
        out[d] = mean_series  # float
    return out


# =========================
# Formatting helpers (四位小數 + 顯著星號)
# =========================
def _sig_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _p_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _format_no_leading_zero(x: float, ndigits: int = 4) -> str:
    if pd.isna(x):
        return ""
    s = f"{x:.{ndigits}f}"
    if s.startswith("0."):
        return s[1:]
    if s.startswith("-0."):
        return "-" + s[2:]
    return s


def _std_beta(params: pd.Series, X: pd.DataFrame, y: pd.Series) -> dict:
    sd_y = y.std(ddof=1)
    sd_x = X.std(ddof=1)
    out = {}
    for v in X.columns:
        if sd_y == 0 or pd.isna(sd_y) or sd_x[v] == 0 or pd.isna(sd_x[v]):
            out[v] = np.nan
        else:
            out[v] = float(params[v]) * float(sd_x[v] / sd_y)
    return out


def _fmt_beta(beta: float, p: float) -> str:
    if pd.isna(beta):
        return ""
    stars = _sig_stars(p)
    return f"{beta:.4f}{stars}"


def _fmt_t(t: float) -> str:
    if pd.isna(t):
        return ""
    return f"{t:.4f}"


# ===== Regression table =====
def build_regression_table(df: pd.DataFrame, iv_vars: list[str], dv_var: str):
    if not iv_vars or not dv_var:
        raise ValueError("請先設定自變數與依變數。")

    cols = iv_vars + [dv_var]
    d = df[cols].apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    if d.empty:
        raise ValueError("可用資料為空（IV/DV 可能有空值或非數值）。")

    y = d[dv_var].astype(float)
    X = d[iv_vars].astype(float)
    Xc = sm.add_constant(X, has_constant="add")

    model = sm.OLS(y, Xc).fit()

    params = model.params
    tvals = model.tvalues
    pvals = model.pvalues

    sd_y = y.std(ddof=1)
    sd_x = X.std(ddof=1)

    beta_std = {}
    for v in iv_vars:
        if sd_y == 0 or pd.isna(sd_y) or sd_x[v] == 0 or pd.isna(sd_x[v]):
            beta_std[v] = np.nan
        else:
            beta_std[v] = params[v] * (sd_x[v] / sd_y)

    rows = []
    rows.append(
        {
            "自變項": "（常數）",
            "未標準化係數 β估計值": f"{params['const']:.4f}",
            "標準化係數 Beta": "—",
            "t": f"{tvals['const']:.4f}{_sig_stars(pvals['const'])}",
            "顯著性": f"{pvals['const']:.4f}",
        }
    )

    for v in iv_vars:
        rows.append(
            {
                "自變項": v,
                "未標準化係數 β估計值": f"{params[v]:.4f}",
                "標準化係數 Beta": ("" if pd.isna(beta_std[v]) else f"{beta_std[v]:.4f}"),
                "t": f"{tvals[v]:.4f}{_sig_stars(pvals[v])}",
                "顯著性": f"{pvals[v]:.4f}",
            }
        )

    table_df = pd.DataFrame(rows)

    summary = {
        "F": float(model.fvalue) if model.fvalue is not None else np.nan,
        "P(F)": float(model.f_pvalue) if model.f_pvalue is not None else np.nan,
        "R2": float(model.rsquared),
        "Adj_R2": float(model.rsquared_adj),
        "N": int(model.nobs),
    }
    return table_df, summary


# ===== Mediation analysis (IV -> M -> DV) =====
def _to_num_df(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")


def _fit_ols(y: pd.Series, X: pd.DataFrame):
    Xc = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, Xc).fit()


def build_mediation_results(
    df: pd.DataFrame,
    iv: str,
    med: str,
    dv: str,
    n_boot: int = 2000,
    seed: int = 42,
):
    d = _to_num_df(df, [iv, med, dv])
    if d.empty:
        raise ValueError("可用資料為空（IV/M/DV 可能有空值或非數值）。")

    # a path
    m_a = _fit_ols(d[med], d[[iv]])
    a = float(m_a.params[iv])
    se_a = float(m_a.bse[iv])
    p_a = float(m_a.pvalues[iv])

    # c path (total)
    m_c = _fit_ols(d[dv], d[[iv]])
    c = float(m_c.params[iv])
    se_c = float(m_c.bse[iv])
    p_c = float(m_c.pvalues[iv])

    # b and c' path
    m_bc = _fit_ols(d[dv], d[[iv, med]])
    b = float(m_bc.params[med])
    se_b = float(m_bc.bse[med])
    p_b = float(m_bc.pvalues[med])

    c_prime = float(m_bc.params[iv])
    se_cprime = float(m_bc.bse[iv])
    p_cprime = float(m_bc.pvalues[iv])

    indirect = a * b

    # Sobel test (normal approximation)
    sobel_se = math.sqrt((b * b * se_a * se_a) + (a * a * se_b * se_b))
    sobel_z = (indirect / sobel_se) if sobel_se != 0 else float("nan")

    if np.isfinite(sobel_z):
        sobel_p = float(2 * (1 - norm.cdf(abs(sobel_z))))
    else:
        sobel_p = float("nan")

    # Bootstrap CI for indirect
    rng = np.random.default_rng(seed)
    n = len(d)
    inds = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        ds = d.iloc[idx]
        try:
            ma = _fit_ols(ds[med], ds[[iv]])
            mbc = _fit_ols(ds[dv], ds[[iv, med]])
            inds.append(float(ma.params[iv]) * float(mbc.params[med]))
        except Exception:
            continue

    if len(inds) >= 20:
        ci_low, ci_high = np.percentile(inds, [2.5, 97.5])
    else:
        ci_low, ci_high = (np.nan, np.nan)

    paths_df = pd.DataFrame(
        [
            {"路徑": "a (IV→M)", "係數": a, "SE": se_a, "t": float(m_a.tvalues[iv]), "p": p_a},
            {"路徑": "c (IV→DV total)", "係數": c, "SE": se_c, "t": float(m_c.tvalues[iv]), "p": p_c},
            {"路徑": "b (M→DV | IV)", "係數": b, "SE": se_b, "t": float(m_bc.tvalues[med]), "p": p_b},
            {"路徑": "c' (IV→DV direct | M)", "係數": c_prime, "SE": se_cprime, "t": float(m_bc.tvalues[iv]), "p": p_cprime},
        ]
    )

    effects_df = pd.DataFrame(
        [
            {
                "效果": "Indirect (a*b)",
                "值": indirect,
                "Sobel z": sobel_z,
                "Sobel p": sobel_p,
                "Boot CI 2.5%": ci_low,
                "Boot CI 97.5%": ci_high,
            }
        ]
    )

    summary = {
        "N": int(n),
        "indirect": float(indirect),
        "sobel_z": float(sobel_z) if np.isfinite(sobel_z) else np.nan,
        "sobel_p": float(sobel_p) if np.isfinite(sobel_p) else np.nan,
        "ci_low": float(ci_low) if np.isfinite(ci_low) else np.nan,
        "ci_high": float(ci_high) if np.isfinite(ci_high) else np.nan,
        "boot_used": int(len(inds)),
    }
    return paths_df, effects_df, summary


def build_mediation_paper_table(df: pd.DataFrame, iv: str, med: str, dv: str):
    d = df[[iv, med, dv]].apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    if d.empty:
        raise ValueError("可用資料為空（IV/M/DV 可能有空值或非數值）。")

    y2 = d[med].astype(float)
    X2 = d[[iv]].astype(float)
    m2 = _fit_ols(y2, X2)
    beta2 = _std_beta(m2.params, X2, y2)

    y1 = d[dv].astype(float)
    X1 = d[[iv]].astype(float)
    m1 = _fit_ols(y1, X1)
    beta1 = _std_beta(m1.params, X1, y1)

    y3 = d[dv].astype(float)
    X3 = d[[iv, med]].astype(float)
    m3 = _fit_ols(y3, X3)
    beta3 = _std_beta(m3.params, X3, y3)

    col_c2_beta = f"{med}（條件二）β值"
    col_c2_t = f"{med}（條件二）t值"
    col_c1_beta = f"{dv}（條件一）β值"
    col_c1_t = f"{dv}（條件一）t值"
    col_c3_beta = f"{dv}（條件三）β值"
    col_c3_t = f"{dv}（條件三）t值"

    rows = []
    rows.append({
        "自變項": iv,
        col_c2_beta: _fmt_beta(beta2.get(iv, np.nan), float(m2.pvalues.get(iv, np.nan))),
        col_c2_t: _fmt_t(float(m2.tvalues.get(iv, np.nan))),
        col_c1_beta: _fmt_beta(beta1.get(iv, np.nan), float(m1.pvalues.get(iv, np.nan))),
        col_c1_t: _fmt_t(float(m1.tvalues.get(iv, np.nan))),
        col_c3_beta: _fmt_beta(beta3.get(iv, np.nan), float(m3.pvalues.get(iv, np.nan))),
        col_c3_t: _fmt_t(float(m3.tvalues.get(iv, np.nan))),
    })

    rows.append({
        "自變項": med,
        col_c2_beta: "", col_c2_t: "", col_c1_beta: "", col_c1_t: "",
        col_c3_beta: _fmt_beta(beta3.get(med, np.nan), float(m3.pvalues.get(med, np.nan))),
        col_c3_t: _fmt_t(float(m3.tvalues.get(med, np.nan))),
    })

    rows.append({
        "自變項": "R²",
        col_c2_beta: f"{float(m2.rsquared):.4f}", col_c2_t: "",
        col_c1_beta: f"{float(m1.rsquared):.4f}", col_c1_t: "",
        col_c3_beta: f"{float(m3.rsquared):.4f}", col_c3_t: "",
    })

    rows.append({
        "自變項": "ΔR²",
        col_c2_beta: f"{float(m2.rsquared_adj):.4f}", col_c2_t: "",
        col_c1_beta: f"{float(m1.rsquared_adj):.4f}", col_c1_t: "",
        col_c3_beta: f"{float(m3.rsquared_adj):.4f}", col_c3_t: "",
    })

    rows.append({
        "自變項": "F",
        col_c2_beta: f"{float(m2.fvalue):.4f}{_sig_stars(float(m2.f_pvalue))}", col_c2_t: "",
        col_c1_beta: f"{float(m1.fvalue):.4f}{_sig_stars(float(m1.f_pvalue))}", col_c1_t: "",
        col_c3_beta: f"{float(m3.fvalue):.4f}{_sig_stars(float(m3.f_pvalue))}", col_c3_t: "",
    })

    rows.append({
        "自變項": "D-W",
        col_c2_beta: f"{float(durbin_watson(m2.resid)):.4f}", col_c2_t: "",
        col_c1_beta: f"{float(durbin_watson(m1.resid)):.4f}", col_c1_t: "",
        col_c3_beta: f"{float(durbin_watson(m3.resid)):.4f}", col_c3_t: "",
    })

    table_df = pd.DataFrame(rows)
    meta = {"N": int(m3.nobs), "cond1": m1, "cond2": m2, "cond3": m3}
    return table_df, meta


def build_moderation_paper_table(df: pd.DataFrame, iv: str, mod: str, dv: str):
    d = df[[iv, mod, dv]].apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    if d.empty:
        raise ValueError("可用資料為空（IV/MOD/DV 可能有空值或非數值）。")

    inter_name = f"{iv}×{mod}"
    d[inter_name] = d[iv] * d[mod]

    y1 = d[dv].astype(float)
    X1 = d[[iv]].astype(float)
    m1 = _fit_ols(y1, X1)
    beta1 = _std_beta(m1.params, X1, y1)

    y2 = d[dv].astype(float)
    X2 = d[[iv, mod]].astype(float)
    m2 = _fit_ols(y2, X2)
    beta2 = _std_beta(m2.params, X2, y2)

    y3 = d[dv].astype(float)
    X3 = d[[iv, mod, inter_name]].astype(float)
    m3 = _fit_ols(y3, X3)
    beta3 = _std_beta(m3.params, X3, y3)

    col_m1_beta = f"{dv}（模型一）β值"
    col_m1_t = f"{dv}（模型一）t值"
    col_m2_beta = f"{dv}（模型二）β值"
    col_m2_t = f"{dv}（模型二）t值"
    col_m3_beta = f"{dv}（模型三）β值"
    col_m3_t = f"{dv}（模型三）t值"

    r2_1 = float(m1.rsquared)
    r2_2 = float(m2.rsquared)
    r2_3 = float(m3.rsquared)

    dr2_2 = r2_2 - r2_1
    dr2_3 = r2_3 - r2_2

    rows = []
    rows.append({
        "自變項": iv,
        col_m1_beta: _fmt_beta(beta1.get(iv, np.nan), float(m1.pvalues.get(iv, np.nan))),
        col_m1_t: _fmt_t(float(m1.tvalues.get(iv, np.nan))),
        col_m2_beta: _fmt_beta(beta2.get(iv, np.nan), float(m2.pvalues.get(iv, np.nan))),
        col_m2_t: _fmt_t(float(m2.tvalues.get(iv, np.nan))),
        col_m3_beta: _fmt_beta(beta3.get(iv, np.nan), float(m3.pvalues.get(iv, np.nan))),
        col_m3_t: _fmt_t(float(m3.tvalues.get(iv, np.nan))),
    })

    rows.append({
        "自變項": mod,
        col_m1_beta: "", col_m1_t: "",
        col_m2_beta: _fmt_beta(beta2.get(mod, np.nan), float(m2.pvalues.get(mod, np.nan))),
        col_m2_t: _fmt_t(float(m2.tvalues.get(mod, np.nan))),
        col_m3_beta: _fmt_beta(beta3.get(mod, np.nan), float(m3.pvalues.get(mod, np.nan))),
        col_m3_t: _fmt_t(float(m3.tvalues.get(mod, np.nan))),
    })

    rows.append({
        "自變項": f"{iv}*{mod}",
        col_m1_beta: "", col_m1_t: "", col_m2_beta: "", col_m2_t: "",
        col_m3_beta: _fmt_beta(beta3.get(inter_name, np.nan), float(m3.pvalues.get(inter_name, np.nan))),
        col_m3_t: _fmt_t(float(m3.tvalues.get(inter_name, np.nan))),
    })

    rows.append({
        "自變項": "R²",
        col_m1_beta: f"{r2_1:.4f}", col_m1_t: "",
        col_m2_beta: f"{r2_2:.4f}", col_m2_t: "",
        col_m3_beta: f"{r2_3:.4f}", col_m3_t: "",
    })

    rows.append({
        "自變項": "ΔR²",
        col_m1_beta: "", col_m1_t: "",
        col_m2_beta: f"{dr2_2:.4f}", col_m2_t: "",
        col_m3_beta: f"{dr2_3:.4f}", col_m3_t: "",
    })

    rows.append({
        "自變項": "F",
        col_m1_beta: f"{float(m1.fvalue):.4f}{_sig_stars(float(m1.f_pvalue))}", col_m1_t: "",
        col_m2_beta: f"{float(m2.fvalue):.4f}{_sig_stars(float(m2.f_pvalue))}", col_m2_t: "",
        col_m3_beta: f"{float(m3.fvalue):.4f}{_sig_stars(float(m3.f_pvalue))}", col_m3_t: "",
    })

    table_df = pd.DataFrame(rows)
    meta = {"N": int(m3.nobs), "interaction_col": inter_name}
    return table_df, meta


def build_discriminant_validity_table(df_norm: pd.DataFrame, item_df: pd.DataFrame):
    import re
    from scipy.stats import pearsonr

    sub_alpha = (
        item_df.groupby("子構面")["該子構面整體 α"]
        .first()
        .dropna()
        .to_dict()
    )

    sub_dims = sorted(sub_alpha.keys())
    sub_scores = {}

    for sd in sub_dims:
        cols = [c for c in df_norm.columns if isinstance(c, str) and re.match(rf"^{sd}\d+", c)]
        if cols:
            sub_scores[sd] = (
                df_norm[cols]
                .apply(pd.to_numeric, errors="coerce")
                .mean(axis=1)
            )

    score_df = pd.DataFrame(sub_scores)
    mat = pd.DataFrame("", index=sub_dims, columns=sub_dims)

    for i, r in enumerate(sub_dims):
        for j, c in enumerate(sub_dims):
            if i == j:
                try:
                    mat.loc[r, c] = f"{float(sub_alpha[r]):.4f}"
                except Exception:
                    mat.loc[r, c] = str(sub_alpha[r])
            elif i > j:
                valid_pair = score_df[[r, c]].dropna()
                
                if len(valid_pair) > 2:
                    r_val, p_val = pearsonr(valid_pair[r], valid_pair[c])
                    star = "**" if p_val < 0.01 else ""
                    mat.loc[r, c] = f"{r_val:.4f}{star}"
                else:
                    mat.loc[r, c] = "N/A"
            else:
                mat.loc[r, c] = ""

    return mat


# =========================================================================
# ✅ 修正點 2：子構面代碼判斷邏輯優化，正確捕捉 D1, E4 這類單一題號
# =========================================================================
def _subdim_code(item_code: str) -> str:
    """
    子構面代碼邏輯：
    - 若為 A11 -> A1 (字母+第一碼數字)
    - 若為 D1, E4 -> D, E (無子構面，直接回傳字母)
    """
    s = str(item_code).strip()
    m = re.match(r"^([A-Za-z])(\d+)(?:_(\d+))?$", s)
    if m:
        letter = m.group(1).upper()
        digits = m.group(2)
        # 若數字只有 1 碼，或者明確屬於無子構面的 D, E，直接回傳字母
        if len(digits) == 1 or letter in ['D', 'E']:
            return letter
        else:
            return letter + digits[0]
            
    # 保底機制
    return s[:2].upper() if len(s) >= 2 else s.upper()


def _item_sort_key(code: str):
    s = str(code).strip()
    m = re.match(r"^([A-Za-z])(\d+)(?:_(\d+))?$", s)
    if not m:
        return (s, 10**9, 10**9)
    letter = m.group(1).upper()
    num = int(m.group(2))
    suf = int(m.group(3)) if m.group(3) is not None else 0
    return (letter, num, suf)


def build_item_status_table(df_raw: pd.DataFrame, df_norm: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    item_cols = _find_item_cols(df_norm)
    if not item_cols:
        return pd.DataFrame()

    inv_map = {}
    if isinstance(mapping, dict) and mapping:
        for k, v in mapping.items():
            vv = str(v).strip()
            if vv not in inv_map:
                inv_map[vv] = str(k)

    rows = []
    for code in item_cols:
        x = pd.to_numeric(df_norm[code], errors="coerce")
        mean_v = float(x.mean()) if x.notna().any() else np.nan
        std_v = float(x.std(ddof=1)) if x.notna().sum() >= 2 else np.nan

        sub = _subdim_code(code)
        q_text = inv_map.get(str(code).strip(), str(code).strip())

        rows.append({
            "題號": str(code).strip(),
            "構面": sub,
            "問項": q_text,
            "平均數": mean_v,
            "標準差": std_v,
        })

    out = pd.DataFrame(rows)

    sub_mean_map = (
        out.groupby("構面")["平均數"]
        .mean()
        .to_dict()
    )
    out["構面平均"] = out["構面"].map(sub_mean_map)

    out["構面排序"] = (
        out.groupby("構面")["平均數"]
        .rank(method="dense", ascending=False)
        .astype("Int64")
    )

    out = out.sort_values(by="題號", key=lambda s: s.map(_item_sort_key)).reset_index(drop=True)

    out["平均數"] = out["平均數"].map(lambda v: f"{v:.4f}" if np.isfinite(v) else "")
    out["標準差"] = out["標準差"].map(lambda v: f"{v:.4f}" if np.isfinite(v) else "")
    out["構面平均"] = out["構面平均"].map(lambda v: f"{v:.4f}" if np.isfinite(v) else "")
    out["構面排序"] = out["構面排序"].astype(str).replace({"<NA>": ""})

    return out


def _find_profile_cols(df_raw: pd.DataFrame, df_norm: pd.DataFrame) -> list[str]:
    item_cols_norm = set(_find_item_cols(df_norm))

    profile_cols = []
    for c in df_raw.columns:
        s = str(c).strip()
        if ITEM_CODE_RE.match(s):
            continue
        if s in item_cols_norm:
            continue
        profile_cols.append(c)

    return profile_cols

def _find_profile_cols_left_of_items(df_raw: pd.DataFrame, mapping: dict) -> list[str]:
    cols = list(df_raw.columns)

    if not isinstance(mapping, dict) or not mapping:
        out = [c for c in cols if not str(c).strip().startswith("Unnamed")]
        return out

    raw_item_cols = [k for k in mapping.keys() if k in df_raw.columns]
    if not raw_item_cols:
        out = [c for c in cols if not str(c).strip().startswith("Unnamed")]
        return out

    idxs = [cols.index(k) for k in raw_item_cols]
    first_item_idx = min(idxs)

    profile_cols = cols[:first_item_idx]
    profile_cols = [c for c in profile_cols if not str(c).strip().startswith("Unnamed")]
    return profile_cols


def build_sample_profile_table(df_raw: pd.DataFrame, mapping: dict) -> tuple[pd.DataFrame, int]:
    profile_cols = _find_profile_cols_left_of_items(df_raw, mapping=mapping)
    N = int(len(df_raw))

    rows = []
    for col in profile_cols:
        s = df_raw[col].copy()
        s = s.astype(str).str.strip()
        s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
        s = s.dropna()

        if s.empty:
            continue

        vc = s.value_counts(dropna=True)
        total = int(vc.sum())
        if total == 0:
            continue

        cum = 0.0
        first = True
        for cat, cnt in vc.items():
            pct = (int(cnt) / total) * 100.0
            cum += pct

            rows.append({
                "項目": str(col) if first else "",
                "類別": str(cat),
                "樣本數": int(cnt),
                "百分比(%)": round(pct, 1),
                "累積百分比(%)": round(cum, 1),
            })
            first = False

    out = pd.DataFrame(rows, columns=["項目", "類別", "樣本數", "百分比(%)", "累積百分比(%)"])
    return out, N

def build_independent_ttest_table(
    df: pd.DataFrame,
    group_col: str,
    dv_cols: list[str],
):
    d = df[[group_col] + dv_cols].copy()

    d[group_col] = d[group_col].astype(str).str.strip()
    d = d.replace({group_col: {"": np.nan, "nan": np.nan, "None": np.nan}})
    d = d.dropna(subset=[group_col])

    groups = [g for g in d[group_col].dropna().unique().tolist() if str(g).strip() != ""]
    if len(groups) != 2:
        raise ValueError(f"此欄位需剛好兩組才可做獨立樣本t檢定（目前={len(groups)}組）。")

    g1, g2 = groups[0], groups[1]

    rows = []
    for v in dv_cols:
        x1 = pd.to_numeric(d.loc[d[group_col] == g1, v], errors="coerce").dropna()
        x2 = pd.to_numeric(d.loc[d[group_col] == g2, v], errors="coerce").dropna()

        if len(x1) < 2 or len(x2) < 2:
            tval, pval = (np.nan, np.nan)
            m1 = float(x1.mean()) if len(x1) else np.nan
            m2 = float(x2.mean()) if len(x2) else np.nan
        else:
            ttest = ttest_ind(x1, x2, equal_var=True, nan_policy="omit")
            tval, pval = float(ttest.statistic), float(ttest.pvalue)
            m1, m2 = float(x1.mean()), float(x2.mean())

        t_star = _p_stars(pval) if np.isfinite(pval) else ""

        rows.append(
            {
                "變項": v,
                str(g1): f"{m1:.4f}" if np.isfinite(m1) else "",
                str(g2): f"{m2:.4f}" if np.isfinite(m2) else "",
                "t值": (_format_no_leading_zero(tval, 4) + t_star) if np.isfinite(tval) else "",
                "P值": _format_no_leading_zero(pval, 4) if np.isfinite(pval) else "",
            }
        )

    out = pd.DataFrame(rows)
    meta = {"group1": str(g1), "group2": str(g2)}
    return out, meta


def _fmt_f_with_stars(F: float, p: float) -> str:
    if not np.isfinite(F):
        return ""
    return f"{F:.4f}{_p_stars(p)}"


def _scheffe_posthoc_pairs(
    group_means: dict[str, float],
    group_ns: dict[str, int],
    msw: float,
    dfb: int,
    dfw: int,
    alpha: float = 0.05,
) -> str:
    if dfw <= 0 or dfb <= 0 or (not np.isfinite(msw)) or msw <= 0:
        return "—"

    labels = list(group_means.keys())
    k = len(labels)
    if k < 3:
        return "—"

    idx_map = {lab: i + 1 for i, lab in enumerate(labels)}

    sig_pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            li, lj = labels[i], labels[j]
            mi, mj = group_means[li], group_means[lj]
            ni, nj = group_ns[li], group_ns[lj]

            if ni < 2 or nj < 2:
                continue

            diff = mi - mj
            denom = msw * (1.0 / ni + 1.0 / nj)
            if denom <= 0:
                continue

            F_pair = (diff * diff) / denom / dfb
            p_pair = float(f_dist.sf(F_pair, dfb, dfw))

            if p_pair < alpha:
                if mi > mj:
                    sig_pairs.append((idx_map[li], idx_map[lj]))
                elif mj > mi:
                    sig_pairs.append((idx_map[lj], idx_map[li]))

    if not sig_pairs:
        return "—"

    win_to_losers: dict[int, set[int]] = {}
    for w, l in sig_pairs:
        win_to_losers.setdefault(w, set()).add(l)

    parts = []
    for w in sorted(win_to_losers.keys(), reverse=True):
        losers = sorted(win_to_losers[w])
        parts.append(f"{w}>{','.join(map(str, losers))}")

    return "；".join(parts)


def build_oneway_anova_table(
    df: pd.DataFrame,
    group_col: str,
    dv_cols: list[str],
):
    d = df[[group_col] + dv_cols].copy()

    d[group_col] = d[group_col].astype(str).str.strip()
    d = d.replace({group_col: {"": np.nan, "nan": np.nan, "None": np.nan}})
    d = d.dropna(subset=[group_col])

    groups = [g for g in d[group_col].dropna().unique().tolist() if str(g).strip() != ""]
    if len(groups) < 3:
        raise ValueError(f"此欄位需至少 3 組才可做單因子變異數分析（目前={len(groups)}組）。")

    rows_comp = []
    rows_anova = []

    for v in dv_cols:
        xs = []
        ns = {}
        means = {}

        for g in groups:
            xg = pd.to_numeric(d.loc[d[group_col] == g, v], errors="coerce").dropna()
            xs.append(xg.values)
            ns[str(g)] = int(len(xg))
            means[str(g)] = float(xg.mean()) if len(xg) else np.nan

        valid_groups = [g for g in groups if ns[str(g)] >= 2]
        k = len(groups)
        N = sum(ns[str(g)] for g in groups)

        if N <= k or len(valid_groups) < 3:
            Fv, pv = (np.nan, np.nan)
            msw = np.nan
            dfb, dfw = (k - 1, N - k)
            ssb, ssw = (np.nan, np.nan)
        else:
            all_vals = []
            for g in groups:
                xg = pd.to_numeric(d.loc[d[group_col] == g, v], errors="coerce").dropna().values
                all_vals.append(xg)
            all_concat = np.concatenate([a for a in all_vals if len(a) > 0])
            grand_mean = float(np.mean(all_concat))

            ssb = 0.0
            ssw = 0.0
            for g in groups:
                xg = pd.to_numeric(d.loc[d[group_col] == g, v], errors="coerce").dropna().values
                if len(xg) == 0:
                    continue
                mg = float(np.mean(xg))
                ssb += len(xg) * (mg - grand_mean) ** 2
                ssw += float(np.sum((xg - mg) ** 2))

            dfb = k - 1
            dfw = N - k
            msb = ssb / dfb if dfb > 0 else np.nan
            msw = ssw / dfw if dfw > 0 else np.nan

            if np.isfinite(msb) and np.isfinite(msw) and msw > 0:
                Fv = msb / msw
                pv = float(f_dist.sf(Fv, dfb, dfw))
            else:
                Fv, pv = (np.nan, np.nan)

        # Scheffe Post-Hoc
        scheffe_txt = _scheffe_posthoc_pairs(
            group_means=means,
            group_ns=ns,
            msw=msw,
            dfb=dfb,
            dfw=dfw,
            alpha=0.05,
        )

        # --- 建立原本的「差異比較表」資料 ---
        row = {"變項": v}
        for g in groups:
            m = means.get(str(g), np.nan)
            row[str(g)] = f"{m:.4f}" if np.isfinite(m) else ""

        row["F值"] = _fmt_f_with_stars(float(Fv), float(pv)) if np.isfinite(Fv) else ""
        row["P值"] = _format_no_leading_zero(float(pv), 4) if np.isfinite(pv) else ""
        row["Scheffe法"] = scheffe_txt

        rows_comp.append(row)

        # --- 建立新增的「單因子變異數分析表」資料 ---
        if np.isfinite(ssb) and np.isfinite(ssw):
            p_star_str = _p_stars(pv)
            rows_anova.append({
                "變項": v,
                "變異來源": "組間",
                "平方和": f"{ssb:.3f}",
                "自由度": dfb,
                "均方": f"{msb:.3f}",
                "F 值": f"{Fv:.3f}",
                "P 值": f"{pv:.3f}{p_star_str}" if pv < 0.05 else f"{pv:.3f}"
            })
            rows_anova.append({
                "變項": "",
                "變異來源": "組內",
                "平方和": f"{ssw:.3f}",
                "自由度": dfw,
                "均方": f"{msw:.3f}",
                "F 值": "",
                "P 值": ""
            })
            rows_anova.append({
                "變項": "",
                "變異來源": "總計",
                "平方和": f"{(ssb+ssw):.3f}",
                "自由度": dfb + dfw,
                "均方": "",
                "F 值": "",
                "P 值": ""
            })

    out_comp = pd.DataFrame(rows_comp)
    out_anova = pd.DataFrame(rows_anova)

    code_map = {i + 1: str(g) for i, g in enumerate(groups)}
    meta = {"groups": groups, "code_map": code_map}
    
    return out_anova, out_comp, meta


# ---- Sidebar ----
with st.sidebar:
    st.header("設定")
    st.caption("1) 上傳 CSV → 2) 產出 Item Analysis → 3) 下載結果（CSV）")

    uploaded_file = st.file_uploader("上傳 CSV", type=["csv"])

    st.divider()
    st.subheader("GPT 論文報告生成（可選）")

    gpt_on = st.toggle("啟用 GPT 報告", value=False, help="需要 OpenAI API Key 與可用額度（quota）。")

    model_options = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "gpt-4.1"]
    model_pick = st.selectbox("選擇 GPT 模型", options=model_options, index=0)
    model_custom = st.text_input("或自行輸入模型名稱（選填）", value="", placeholder="例如：gpt-4o-mini")
    model_name = (model_custom.strip() or model_pick).strip()

    api_key = st.text_input("OpenAI API Key（以 sk- 開頭）", type="password", value="")
    st.caption("建議用環境變數也可：先在系統設定 OPENAI_API_KEY，再留空此欄。")

    st.divider()
    st.subheader("子構面規則（你指定）")
    st.write("子構面只取題項代碼的**前兩碼**：例如 A01→A0、A11→A1、A105→A1。若為 D, E 則獨立計算。")


# ---- Main ----
if uploaded_file is None:
    st.info("請先在左側上傳 CSV 檔案。")
    st.stop()

try:
    df_raw = read_csv_safely(uploaded_file)
except Exception as e:
    safe_show_exception(e)
    st.stop()

df_norm, mapping = normalize_item_columns(df_raw)

st.subheader("原始資料預覽（前 5 列）")
st.dataframe(df_raw.head(), width="stretch")

with st.expander("欄名正規化對照（原始欄名 → 題項代碼）"):
    if mapping:
        map_df = pd.DataFrame([{"原始欄名": k, "題項代碼": v} for k, v in mapping.items()])
        st.dataframe(map_df, width="stretch")
    else:
        st.write("未偵測到可正規化的題項欄名（請確認欄名格式）。")

st.subheader("📈 Item Analysis 結果")

try:
    result_df = run_item_analysis(df_norm)
    st.success("Item analysis completed.")
    st.dataframe(result_df, width="stretch", height=520)

    st.download_button(
        "下載 Item Analysis 結果 CSV",
        data=df_to_csv_bytes(result_df),
        file_name="item_analysis_results.csv",
        mime="text/csv",
    )

    df_dim_means_row = build_dim_means_per_row(df_norm)
    if df_dim_means_row.empty:
        st.warning("找不到題項代碼欄位，無法產生構面平均（A/B/C...）。")
        st.stop()

    df_raw_plus_dimmeans = df_norm.copy()
    for c in df_dim_means_row.columns:
        df_raw_plus_dimmeans[c] = df_dim_means_row[c]

    dim_cols = list(df_dim_means_row.columns)

    st.divider()
    st.subheader("📊 區別效度分析表")

    try:
        disc_df = build_discriminant_validity_table(df_norm, result_df)

        st.dataframe(disc_df, width="stretch")
        st.caption("註：對角線為各子構面之 Cronbach’s α；左下三角為子構面間之皮爾森相關係數（** P<0.01）。")

        st.download_button(
            "下載 區別效度分析表 CSV",
            data=df_to_csv_bytes(disc_df),
            file_name="discriminant_validity_table.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error("區別效度分析失敗（safe）")
        safe_show_exception(e)

    st.divider()
    st.subheader("📋 構面現況分析表")

    try:
        item_status_df = build_item_status_table(df_raw=df_raw, df_norm=df_norm, mapping=mapping)

        if item_status_df.empty:
            st.info("找不到題項代碼欄位，無法產生構面現況分析表。")
        else:
            st.dataframe(item_status_df, width="stretch", height=520)

            st.download_button(
                "下載 構面現況分析表 CSV",
                data=df_to_csv_bytes(item_status_df),
                file_name="item_status_table.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error("構面現況分析表失敗（safe）")
        safe_show_exception(e)

    st.divider()

    try:
        sample_profile_df, N_profile = build_sample_profile_table(df_raw=df_raw, mapping=mapping)

        st.subheader(f"樣本基本資料分析表(N={N_profile})")

        if sample_profile_df.empty:
            st.info("找不到可用的基本資料欄位（A11 左側欄位）或資料皆為空。")
        else:
            st.dataframe(sample_profile_df, width="stretch", height=520)

            st.download_button(
                "下載 樣本基本資料分析表 CSV",
                data=df_to_csv_bytes(sample_profile_df),
                file_name="sample_profile_table.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error("樣本基本資料分析表失敗（safe）")
        safe_show_exception(e)  


    st.divider()
    st.subheader("📊 獨立樣本 t 檢定（基本資料分組）")

    try:
        profile_cols = _find_profile_cols(df_raw, df_norm)

        if not profile_cols:
            st.info("未偵測到可用的個人基本資料欄位（非題項欄位）。")
        else:
            picked_profiles = st.multiselect(
                "請勾選要進行獨立樣本t檢定的個人基本資料欄位（可複選；每個欄位需剛好兩組）",
                options=profile_cols,
                default=[],
            )

            if picked_profiles:
                dim_cols_for_t = list(df_dim_means_row.columns)

                df_for_t = df_raw.copy()
                for c in df_dim_means_row.columns:
                    df_for_t[c] = df_dim_means_row[c]

                for gc in picked_profiles:
                    st.markdown(f"### {gc} 獨立樣本t檢定表")

                    try:
                        t_table, meta = build_independent_ttest_table(
                            df_for_t,
                            group_col=gc,
                            dv_cols=dim_cols_for_t,
                        )

                        st.dataframe(t_table, width="stretch")
                        st.caption("註：* P<0.05，** P<0.01，*** P<0.001")

                        st.download_button(
                            f"下載 {gc} t檢定表 CSV",
                            data=df_to_csv_bytes(t_table),
                            file_name=f"ttest_{str(gc).strip()}.csv",
                            mime="text/csv",
                        )

                    except Exception as e:
                        st.error(f"【{gc}】無法產生 t 檢定表：{repr(e)}")

            else:
                st.info("請先勾選至少一個基本資料欄位。")

    except Exception as e:
        st.error("t 檢定區塊失敗（safe）")
        safe_show_exception(e)

    st.divider()
    st.subheader("📊 單因子變異數分析（基本資料分組）")

    try:
        profile_cols2 = _find_profile_cols(df_raw, df_norm)

        if not profile_cols2:
            st.info("未偵測到可用的個人基本資料欄位（非題項欄位）。")
        else:
            picked_profiles_anova = st.multiselect(
                "請勾選要進行單因子變異數分析的個人基本資料欄位（需至少三組）",
                options=profile_cols2,
                default=[],
                key="anova_profiles",
            )

            if picked_profiles_anova:
                dim_cols_for_a = list(df_dim_means_row.columns)

                df_for_a = df_raw.copy()
                for c in df_dim_means_row.columns:
                    df_for_a[c] = df_dim_means_row[c]

                for gc in picked_profiles_anova:
                    try:
                        anova_summary_table, a_table, meta = build_oneway_anova_table(
                            df_for_a,
                            group_col=gc,
                            dv_cols=dim_cols_for_a,
                        )

                        # 新增：單因子變異數分析表
                        st.markdown(f"### {gc} 單因子變異數分析表")
                        st.dataframe(anova_summary_table, width="stretch")

                        # 修改：差異比較表
                        st.markdown(f"### {gc} 差異比較表")
                        st.dataframe(a_table, width="stretch")
                        st.caption("註：* P<0.05，** P<0.01，*** P<0.001")

                        code_map = meta.get("code_map", {})
                        if code_map:
                            mapping_txt = "；".join([f"{k}={v}" for k, v in code_map.items()])
                            st.caption(f"Scheffe法組別代碼：{mapping_txt}")

                        st.download_button(
                            f"下載 {gc} 單因子變異數分析表 CSV",
                            data=df_to_csv_bytes(anova_summary_table),
                            file_name=f"anova_summary_{str(gc).strip()}.csv",
                            mime="text/csv",
                        )

                        st.download_button(
                            f"下載 {gc} 差異比較表 CSV",
                            data=df_to_csv_bytes(a_table),
                            file_name=f"anova_comparison_{str(gc).strip()}.csv",
                            mime="text/csv",
                        )

                    except Exception as e:
                        st.error(f"【{gc}】無法產生單因子變異數分析表：{repr(e)}")

            else:
                st.info("請先勾選至少一個基本資料欄位。")

    except Exception as e:
        st.error("ANOVA 區塊失敗（safe）")
        safe_show_exception(e)

    st.divider()
    st.subheader("📌 研究變數設定（自變數 / 依變數）")

    iv_vars = st.multiselect("① 勾選自變數（可複選）", options=dim_cols, default=[])

    dv_var = st.selectbox("② 選擇依變數（單一）", options=[""] + dim_cols, index=0)

    if dv_var and dv_var in iv_vars:
        st.error("⚠️ 依變數不可同時被選為自變數，請重新設定。")

    elif iv_vars and dv_var:
        st.success(f"研究模型：IV = {iv_vars} → DV = {dv_var}")

        df_research = df_raw_plus_dimmeans[iv_vars + [dv_var]].copy()
        st.dataframe(df_research, width="stretch")

        st.download_button(
            "下載 研究用資料 CSV（IV + DV）",
            data=df_to_csv_bytes(df_research),
            file_name="research_dataset_IV_DV.csv",
            mime="text/csv",
        )

        st.divider()
        st.subheader("📊 迴歸分析表（論文格式）")

        if st.button("執行迴歸分析", type="primary"):
            try:
                reg_table, reg_sum = build_regression_table(df_research, iv_vars, dv_var)

                st.dataframe(reg_table, width="stretch")
                st.markdown(
                    f"**F={reg_sum['F']:.4f}，P={reg_sum['P(F)']:.4f}，"
                    f"R²={reg_sum['R2']:.4f}，Adj R²={reg_sum['Adj_R2']:.4f}，"
                    f"N={reg_sum['N']}**"
                )

            except Exception as e:
                st.error("迴歸分析失敗（safe）")
                safe_show_exception(e)

    else:
        st.info("請先選擇至少一個自變數與一個依變數。")

except Exception as e:
    st.error("Item Analysis 主流程失敗（safe）")
    safe_show_exception(e)
    st.stop()


st.divider()
st.subheader("🧩 中介分析設定")

dim_cols_all = dim_cols

col1, col2, col3 = st.columns(3)

with col1:
    iv_m = st.selectbox("① 自變數(IV)", options=[""] + dim_cols_all, index=0, key="med_iv")

with col2:
    med_options = [""] + [c for c in dim_cols_all if c != iv_m]
    med_m = st.selectbox("② 中介變數(Me)", options=med_options, index=0, key="med_m")

with col3:
    dv_options = [""] + [c for c in dim_cols_all if c not in {iv_m, med_m}]
    dv_m = st.selectbox("③ 依變數(DV)", options=dv_options, index=0, key="med_dv")

chosen = [x for x in [iv_m, med_m, dv_m] if x]

if len(chosen) != len(set(chosen)):
    st.error("⚠️ IV / Me / DV 不可重複，A、B、C、D… 每個只能出現在一個角色中。")

elif iv_m and med_m and dv_m:
    st.success(f"中介模型：{iv_m} → {med_m} → {dv_m}")

    st.markdown("### 研究用資料表（僅保留 IV / Me / DV）")
    df_mediation = df_raw_plus_dimmeans[[iv_m, med_m, dv_m]].copy()
    st.dataframe(df_mediation, width="stretch")

    st.download_button(
        "下載 中介分析研究用資料 CSV（IV + Me + DV）",
        data=df_to_csv_bytes(df_mediation),
        file_name=f"mediation_dataset_{iv_m}_{med_m}_{dv_m}.csv",
        mime="text/csv",
    )

    st.markdown("### 中介分析")

    n_boot = st.number_input("Bootstrap 次數（建議 2000）", min_value=200, max_value=20000, value=2000, step=200)

    if st.button("執行中介分析", type="primary", key="run_mediation"):
        try:
            paper_table, meta = build_mediation_paper_table(df_raw_plus_dimmeans, iv=iv_m, med=med_m, dv=dv_m)

            st.markdown(f"### 中介變數({med_m}) 對 自變數({iv_m}) 與 依變數({dv_m})之中介分析表")
            st.dataframe(paper_table, width="stretch")

            st.caption("註：* P<0.05，** P<0.01，*** P<0.001；ΔR² 為調整後 R²（Adj R²）；D-W 為 Durbin–Watson。")

            tag = f"{iv_m}_to_{med_m}_to_{dv_m}".replace(" ", "")
            st.download_button(
                "下載 中介分析表 CSV",
                data=df_to_csv_bytes(paper_table),
                file_name=f"mediation_table_{tag}.csv",
                mime="text/csv",
            )

            st.markdown(f"**N={meta['N']}**")

        except Exception as e:
            st.error("中介分析失敗（safe）")
            safe_show_exception(e)

else:
    st.info("請依序選擇 IV / M / DV（且三者不可重複）後，才會顯示中介分析資料與結果。")


st.divider()
st.subheader("🧩 干擾分析設定")

col1, col2, col3 = st.columns(3)

with col1:
    iv_w = st.selectbox("① 自變數(IV)", options=[""] + dim_cols, index=0, key="mod_iv")

mod_options = [""] + [c for c in dim_cols if c != iv_w]
with col2:
    w_var = st.selectbox("② 干擾變數(Mo)", options=mod_options, index=0, key="mod_w")

dv_options2 = [""] + [c for c in dim_cols if c not in {iv_w, w_var}]
with col3:
    dv_w = st.selectbox("③ 依變數(DV)", options=dv_options2, index=0, key="mod_dv")

chosen2 = [x for x in [iv_w, w_var, dv_w] if x]
if len(chosen2) != len(set(chosen2)):
    st.error("⚠️ IV / W / DV 不可重複，A、B、C、D… 每個只能出現在一個角色中。")
else:
    if iv_w and w_var and dv_w:
        st.success(f"干擾模型：{iv_w} → {dv_w}（W={w_var}）")

        st.markdown("### 研究用資料表（僅保留 IV / Mo / DV）")
        df_moderation = df_raw_plus_dimmeans[[iv_w, w_var, dv_w]].copy()
        st.dataframe(df_moderation, width="stretch")

        st.download_button(
            "下載 干擾分析研究用資料 CSV（IV + Mo + DV）",
            data=df_to_csv_bytes(df_moderation),
            file_name=f"moderation_dataset_{iv_w}_{w_var}_{dv_w}.csv",
            mime="text/csv",
        )

        run_mod = st.button("執行干擾分析", type="primary", key="run_moderation")

        if run_mod:
            try:
                mod_table, mod_meta = build_moderation_paper_table(df_raw_plus_dimmeans, iv=iv_w, mod=w_var, dv=dv_w)

                st.markdown(f"### 干擾變數({w_var}) 對 自變數({iv_w}) 與 依變數 ({dv_w}) 之干擾分析表")
                st.dataframe(mod_table, width="stretch")
                st.caption("註：* P<0.05，** P<0.01，*** P<0.001；ΔR² 為 R² 變化量（R² change）。")

                tag2 = f"{iv_w}_x_{w_var}_to_{dv_w}".replace(" ", "")
                st.download_button(
                    "下載 干擾分析表 CSV",
                    data=df_to_csv_bytes(mod_table),
                    file_name=f"moderation_table_{tag2}.csv",
                    mime="text/csv",
                )

                st.markdown(f"**N={mod_meta['N']}**")

            except Exception as e:
                st.error("干擾分析失敗（safe）")
                safe_show_exception(e)

    else:
        st.info("請依序選擇 IV / Mo / DV（且三者不可重複）後，才會顯示干擾分析資料與結果。")


# ---- GPT report (optional) ----
st.divider()
st.subheader("📝 GPT 論文報告生成（文字）")

if not gpt_on:
    st.info("你目前未啟用 GPT 報告。若要生成論文文字，請在左側打開「啟用 GPT 報告」。")
    st.stop()

if not GPT_AVAILABLE:
    st.warning("找不到可用的 generate_gpt_report（請確認 gpt_report.py 中有定義 generate_gpt_report）。")
    st.stop()

key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
if not key:
    st.warning("尚未提供 OpenAI API Key。請在左側輸入，或設定環境變數 OPENAI_API_KEY。")
    st.stop()

gen = st.button("生成 GPT 報告（文字）", type="primary")

if gen:
    try:
        report = generate_gpt_report(result_df, model=model_name, api_key=key)

        paper_text = None
        if isinstance(report, dict):
            paper_text = report.get("paper_text") or report.get("text") or report.get("output")
        elif isinstance(report, str):
            paper_text = report

        if not paper_text:
            st.warning("GPT 回傳內容為空，請檢查 gpt_report.py 的回傳格式。")
        else:
            st.success("GPT 報告生成完成。")
            st.text_area("GPT 論文報告（可複製）", value=paper_text, height=420)

            st.download_button(
                "下載 GPT 報告 TXT",
                data=paper_text.encode("utf-8"),
                file_name="gpt_paper_report.txt",
                mime="text/plain",
            )

    except Exception as e:
        msg = repr(e)
        if "insufficient_quota" in msg or "You exceeded your current quota" in msg:
            st.error("GPT report failed：你的 OpenAI API 帳號目前沒有可用額度（insufficient_quota）。")
            st.caption("解法：到 OpenAI 平台 Billing/Credits 加值後再試。")
        else:
            st.error("GPT report failed. See error details below (safe).")
            safe_show_exception(e)
