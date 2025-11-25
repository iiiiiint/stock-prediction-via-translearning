from __future__ import annotations

from typing import Any, Dict


def get_default_config() -> Dict[str, Any]:
    """
    返回主流程所需的所有默认配置。
    如需自定义，可在调用 main 之前覆盖字典内的键。
    """
    return {
        # ================== 数据 & 特征 ==================
        "data_dir": "data/raw",
        "data_sources": {
            "target": {
                "filename": "targetdata.xlsx",  # 请替换为您的实际Excel文件名
                "date_col": "date",
                "file_type": "excel",  # 支持Excel文件
                "column_mapping": {},  # 您的列名已经是英文open/high/low/close/volume，不需要映射
            },
            "US_NDX": {
                "filename": "sourcedata.xlsx",  # 请替换为您的实际Excel文件名
                "file_type": "excel",  # 支持Excel文件
                "date_col": "date",
                "column_mapping": {},
            },
        },
        "seq_len": 30,
        "data_time_range": "all",  # 数据时间范围：1y=1年，6m=6个月，3m=3个月，all=全部数据
        "feature_cols_deep_base": ["log_ret", "ma5", "ma10", "ma20"],
        "feature_cols_tree_base": [
            "log_ret", "ma5", "ma10", "ma20",
            "ret_mean_30", "ret_std_30", "ret_skew_30", "ret_kurt_30",
            "price_rank_30", "atr_14", "volatility_30",
            "volume_ratio_30", "volume_std_30", "volume_skew_30",
            "momentum_10", "momentum_20", "rsi_14",
            "ret_1d", "ret_2d", "ret_3d",
            "volume_change_1d", "volume_change_2d", "volume_change_3d",
            "close_ratio_1d", "close_ratio_2d", "close_ratio_3d",
            "close_vs_ma5", "close_vs_ma20", "ma5_vs_ma20"
        ],
        "feature_cols_deep_full": [
            "log_ret",
            "ma5",
            "ma10",
            "ma20",
            "US_NDX_pred",
        ],
        "feature_cols_tree_full": [
            "log_ret",
            "ma5",
            "ma10",
            "ma20",
            "US_NDX_pred",
        ],

        # ================== 跨市场配置 ==================
        "use_cross_ndx": True,
        "cross_asset_sources": {
            "US_NDX": {
                "min_history": 80,
                "feature_name": "US_NDX_pred",
            }
        },

        # ================== 回测窗口 & 训练 ==================
        "meta_window": 10,  # 减小元学习窗口
        "meta_eta": 3.0,
        "start_index_for_backtest": 50,  # 减少回测起始点
        "retrain_interval": 10,  # 增加重训练间隔
        "train_window": 180,  # 训练窗口：只使用最近N天的数据训练
        "lstm_epochs": 20,  # 增加训练轮次配合早停
        "lstm_lr": 1e-3,
        "lstm_hidden_size": 32,  
        "lstm_batch_size": 32,  
        "lstm_dropout": 0.1,
        "lstm_weight_decay": 1e-4,  # 权重衰减
        "lstm_grad_clip": 1.0,      # 梯度裁剪
        "lstm_patience": 5,         # 早停耐心值
        "lgbm_params": {
            "n_estimators": 2000,
            "learning_rate": 0.01,
            "num_leaves": 31,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "min_child_samples": 20,
            "early_stopping_rounds": 100,
        },

        # ================== 迁移学习配置 ==================
        "use_transfer_learning": True,
        "transfer_hidden_size": 32,  # 减小隐藏层大小
        "transfer_dropout": 0.2,
        "transfer_epochs": 3,  # 减少微调轮数
        "transfer_lr": 1e-4,
        "transfer_batch_size": 16,  # 减小批次大小
        "transfer_pretrain_epochs": 2,  # 减少预训练轮数
        "transfer_model_path": "artifacts/transfer_model.pt",
        
        # ================== LLM 开关 ==================
        "use_llm_as_expert": True,

        # ================== LLM 滚动窗配置 ==================
        "llm_hist_window": 120,
        "llm_max_tokens": 4096,
        "llm_model_name": "gpt-4.1-mini",
        "llm_temperature": 0.0,
        "llm_api_key": None,  # 若已设置环境变量 OPENAI_API_KEY，可保持 None
    }
