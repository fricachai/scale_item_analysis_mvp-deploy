def run_item_analysis(df_norm: pd.DataFrame):
    # ... 前段代碼保持不變 ...
    
    group_map = {}
    for dim in unique_main_dims:
        dim_cols = [c for c in item_cols if c.startswith(dim)]
        dim_mean = df_items[dim_cols].mean(axis=1, skipna=True)
        
        # --- 核心修正：對齊人頭數邏輯 ---
        n_valid = len(dim_mean.dropna())
        # JASP/SPSS 常用邏輯：取總人數的 27% 並取整數
        k = int(round(n_valid * 0.27)) 
        
        # 依平均數排序
        sorted_series = dim_mean.sort_values()
        
        # 取得邊界分數
        # 低分組臨界：排名第 k 名的分數
        low_bound = sorted_series.iloc[k-1]
        # 高分組臨界：倒數第 k 名的分數
        high_bound = sorted_series.iloc[-k]
        
        # 標記
        labels = np.zeros(len(dim_mean))
        labels[dim_mean <= low_bound] = 1 # 低
        labels[dim_mean >= high_bound] = 2 # 高
        group_map[dim] = labels

    # ... 後續 t 檢定與 round(..., 4) 保持不變 ...
