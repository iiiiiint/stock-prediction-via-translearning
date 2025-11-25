from __future__ import annotations
from typing import Dict
import pandas as pd
from .data_manager import DataManager
from .models import LSTMModel

def train_lstm_single(config: Dict, dm: DataManager) -> pd.DataFrame:
    """训练单独的LSTM模型"""
    print("开始训练LSTM模型...")
    
    # 确保特征已生成
    if 'log_ret' not in dm.get_target_df().columns:
        dm.generate_target_features_base()
    
    models = {
        "LSTM": LSTMModel(
            name="LSTM_Single",
            input_shape=(config["seq_len"], len(config["feature_cols_deep_base"])),
            config=config
        )
    }
    
    return _run_single_model_experiment(config, dm, models, "LSTM")

def _run_single_model_experiment(config, dm, models, model_name):
    """运行单模型实验"""
    target_df = dm.get_target_df().copy()
    dates = target_df.index.to_list()
    label_col = "label"
    seq_len = config["seq_len"]
    valid_len = config["valid_len"]
    start_idx = config["start_index_for_backtest"]
    
    results = []
    
    for i in range(start_idx, len(dates) - 1):
        current_date = dates[i]
        train_end = i - valid_len - 1
        valid_start = i - valid_len
        valid_end = i - 1

        if train_end < seq_len:
            continue

        # 应用训练窗口限制
        train_window = config.get("train_window", None)
        if train_window is not None and train_end + 1 > train_window:
            train_start = max(0, train_end - train_window + 1)
            train_df = target_df.iloc[train_start : train_end + 1]
        else:
            train_df = target_df.iloc[: train_end + 1]
            
        valid_df = target_df.iloc[valid_start : valid_end + 1]
        test_row = target_df.iloc[i : i + 1]

        # 准备LSTM序列数据
        X_train, y_train, _ = dm.build_sequence_data(
            train_df, config["feature_cols_deep_base"], label_col, seq_len
        )
        X_valid, y_valid, _ = dm.build_sequence_data(
            valid_df, config["feature_cols_deep_base"], label_col, seq_len
        )

        # 定期重训练
        if (i - start_idx) % config["retrain_interval"] == 0:
            for model in models.values():
                model.fit(X_train, y_train, X_valid, y_valid)

        # 预测
        hist_df = target_df.iloc[: i + 1]
        X_hist, _, _ = dm.build_sequence_data(
            hist_df, config["feature_cols_deep_base"], label_col, seq_len
        )
        
        if len(X_hist) > 0:
            prob = list(models.values())[0].predict_proba(X_hist[-1:])[0]
            
            final_label = int(prob > 0.5)
            actual = int(test_row[label_col].iloc[0])
            
            results.append({
                "date": current_date,
                "prob": prob,
                "pred": final_label,
                "actual": actual,
                "ret_1d_ahead": float(test_row["ret_1d_ahead"].iloc[0]),
                "model": model_name
            })

    return pd.DataFrame(results).set_index("date")