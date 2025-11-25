from __future__ import annotations
from typing import Dict
import pandas as pd
from .data_manager import DataManager
from .cross_asset import generate_cross_asset_predictions
from .models import LightGBMModel

def train_ndx_predictor(config: Dict, dm_source: DataManager) -> pd.DataFrame:
    """训练美股(NDX)预测模型"""
    print("开始训练美股预测模型...")
    
    # 准备美股数据
    if 'log_ret' not in dm_source.get_target_df().columns:
        dm_source.generate_target_features_base()
    
    # 使用LightGBM预测美股
    model = LightGBMModel(
        name="NDX_Predictor",
        config=config
    )
    
    target_df = dm_source.get_target_df().copy()
    dates = target_df.index.to_list()
    label_col = "label"
    valid_len = config["valid_len"]
    start_idx = config["start_index_for_backtest"]
    
    results = []
    
    for i in range(start_idx, len(dates) - 1):
        current_date = dates[i]
        train_end = i - valid_len - 1
        valid_start = i - valid_len
        valid_end = i - 1

        # 应用训练窗口限制
        train_window = config.get("train_window", None)
        if train_window is not None and train_end + 1 > train_window:
            train_start = max(0, train_end - train_window + 1)
            train_df = target_df.iloc[train_start : train_end + 1]
        else:
            train_df = target_df.iloc[: train_end + 1]
            
        valid_df = target_df.iloc[valid_start : valid_end + 1]
        test_row = target_df.iloc[i : i + 1]

        # 准备特征数据
        X_train = train_df[config["feature_cols_tree_base"]].values
        y_train = train_df[label_col].values
        X_valid = valid_df[config["feature_cols_tree_base"]].values
        y_valid = valid_df[label_col].values

        # 定期重训练
        if (i - start_idx) % config["retrain_interval"] == 0:
            model.fit(X_train, y_train, X_valid, y_valid)

        # 预测
        X_today = test_row[config["feature_cols_tree_base"]].values
        prob = model.predict_proba(X_today)[0]
        
        final_label = int(prob > 0.5)
        actual = int(test_row[label_col].iloc[0])
        
        results.append({
            "date": current_date,
            "prob": prob,
            "pred": final_label,
            "actual": actual,
            "ret_1d_ahead": float(test_row["ret_1d_ahead"].iloc[0]),
            "model": "NDX_Predictor"
        })

    return pd.DataFrame(results).set_index("date")

def generate_ndx_predictions(config: Dict, dm_source: DataManager, dm_target: DataManager) -> pd.Series:
    """生成美股预测特征"""
    ndx_results = train_ndx_predictor(config, dm_source)
    
    # 将对齐的预测结果作为特征返回
    target_dates = dm_target.get_target_df().index
    aligned_probs = ndx_results['prob'].reindex(target_dates).ffill().dropna()
    
    return aligned_probs