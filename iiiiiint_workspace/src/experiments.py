from __future__ import annotations

from typing import Dict

import os
import sys
import numpy as np
import pandas as pd

# Ensure project root is on sys.path so absolute 'src.*' imports work when running
# this file as a script (e.g. `python src/experiments.py`) instead of as a package.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import get_default_config
from src.cross_asset import generate_cross_asset_predictions
from src.data_manager import DataManager
from src.meta import MetaLearner
from src.models import BasePredictor, LSTMModel, LLMPredictor, LightGBMModel
from src.services import RollingLLMService
from src.transfer_learning import TransferLSTMPredictor, prepare_transfer_data

def run_experiment(
    config: Dict,
    dm: DataManager,
    experiment_name: str,
    llm_service: RollingLLMService | None = None,
) -> pd.DataFrame:
    target_df = dm.get_target_df().copy()
    dates = target_df.index.to_list()

    if experiment_name in ["baseline_single_lstm", "baseline_ensemble_no_transfer"]:
        feature_cols_deep = config["feature_cols_deep_base"]
        feature_cols_tree = config["feature_cols_tree_base"]
        use_ensemble = experiment_name == "baseline_ensemble_no_transfer"
        use_llm = False
    elif experiment_name == "baseline_single_lgbm":
        feature_cols_deep = []
        feature_cols_tree = config["feature_cols_tree_base"]
        use_ensemble = False
        use_llm = False
    elif experiment_name == "transfer_full_model":
        feature_cols_deep = config["feature_cols_deep_full"]
        feature_cols_tree = config["feature_cols_tree_full"]
        use_ensemble = True
        use_llm = config.get("use_llm_as_expert", True)
    else:
        raise ValueError(f"Unknown experiment_name: {experiment_name}")

    label_col = "label"
    seq_len = config["seq_len"]
    valid_len = config["valid_len"]
    start_idx = config["start_index_for_backtest"]

    models: Dict[str, object] = {}
    if feature_cols_deep:
        input_shape = (seq_len, len(feature_cols_deep))
        models["LSTM_Main"] = LSTMModel(
            name=f"LSTM_{experiment_name}",
            input_shape=input_shape,
            config=config,
        )

    if feature_cols_tree:
        models["LGBM_Main"] = LightGBMModel(
            name=f"LGBM_{experiment_name}",
            config=config,
        )

    if use_llm and llm_service is not None:
        models["LLM_Expert"] = LLMPredictor(
            name="LLM_Expert",
            llm_service=llm_service,
        )

    meta_learner = MetaLearner(
        window_size=config["meta_window"],
        eta=config["meta_eta"],
        min_samples=5,
    )

    results = []
    current_weights = None

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
            train_start = train_end - train_window + 1
            train_df = target_df.iloc[train_start : train_end + 1]
        else:
            train_df = target_df.iloc[: train_end + 1]
            
        valid_df = target_df.iloc[valid_start : valid_end + 1]
        test_row = target_df.iloc[i : i + 1]
        
        # 打印训练窗口信息（调试用）
        if i == start_idx:
            print(f"训练窗口: {len(train_df)} 天, 验证窗口: {len(valid_df)} 天")

        # 深度模型序列
        if "LSTM_Main" in models:
            X_train_deep, y_train_deep, _ = dm.build_sequence_data(
                train_df, feature_cols_deep, label_col, seq_len
            )
            X_valid_deep, y_valid_deep, idx_valid_deep = dm.build_sequence_data(
                valid_df, feature_cols_deep, label_col, seq_len
            )
            if len(X_valid_deep) == 0:
                X_valid_deep = None
                y_valid_deep = None
                idx_valid_deep = []
        else:
            X_train_deep = y_train_deep = None
            X_valid_deep = y_valid_deep = None
            idx_valid_deep = []

        # 树模型特征
        if "LGBM_Main" in models:
            X_train_tree = train_df[feature_cols_tree].values
            y_train_tree = train_df[label_col].values
            X_valid_tree = valid_df[feature_cols_tree].values
            y_valid_tree = valid_df[label_col].values
            idx_valid_tree = valid_df.index
        else:
            X_train_tree = y_train_tree = None
            X_valid_tree = y_valid_tree = None
            idx_valid_tree = []

        if (i - start_idx) % config["retrain_interval"] == 0:
            if "LSTM_Main" in models:
                models["LSTM_Main"].fit(
                    X_train_deep,
                    y_train_deep,
                    X_valid=X_valid_deep,
                    y_valid=y_valid_deep,
                )
            if "LGBM_Main" in models:
                models["LGBM_Main"].fit(
                    X_train_tree,
                    y_train_tree,
                    X_valid=X_valid_tree,
                    y_valid=y_valid_tree,
                )

        if use_ensemble and len(models) > 1:
            recent_preds = pd.DataFrame(index=valid_df.index)

            if "LSTM_Main" in models and X_valid_deep is not None:
                probs = models["LSTM_Main"].predict_proba(X_valid_deep)
                recent_preds["LSTM_Main"] = pd.Series(probs, index=idx_valid_deep)

            if "LGBM_Main" in models and X_valid_tree is not None and len(X_valid_tree) > 0:
                probs = models["LGBM_Main"].predict_proba(X_valid_tree)
                recent_preds["LGBM_Main"] = pd.Series(probs, index=idx_valid_tree)

            if "LLM_Expert" in models:
                dates_valid = np.array(valid_df.index).reshape(-1, 1)
                probs = models["LLM_Expert"].predict_proba(dates_valid)
                recent_preds["LLM_Expert"] = pd.Series(probs, index=valid_df.index)

            recent_preds = recent_preds.tail(meta_learner.window_size)
            recent_true = target_df[label_col].loc[recent_preds.index]

            current_weights = meta_learner.update_weights(recent_preds, recent_true)

        if not current_weights:
            if len(models) == 1:
                m = list(models.keys())[0]
                current_weights = {m: 1.0}
            else:
                current_weights = {m: 1.0 / len(models) for m in models.keys()}

        preds_today = {}

        if "LSTM_Main" in models:
            hist_df = target_df.iloc[: i + 1]
            X_hist, _, _ = dm.build_sequence_data(hist_df, feature_cols_deep, label_col, seq_len)
            if len(X_hist) > 0:
                preds_today["LSTM_Main"] = float(models["LSTM_Main"].predict_proba(X_hist[-1:])[0])

        if "LGBM_Main" in models:
            X_today = test_row[feature_cols_tree].values
            preds_today["LGBM_Main"] = float(models["LGBM_Main"].predict_proba(X_today)[0])

        if "LLM_Expert" in models:
            preds_today["LLM_Expert"] = float(
                models["LLM_Expert"].predict_proba(np.array([current_date]).reshape(-1, 1))[0]
            )

        final_prob = 0.0
        for m_name, prob in preds_today.items():
            weight = current_weights.get(m_name, 0.0)
            final_prob += weight * prob

        final_label = int(final_prob > 0.5)
        actual = int(test_row[label_col].iloc[0])

        results.append(
            {
                "date": current_date,
                "prob": final_prob,
                "pred": final_label,
                "actual": actual,
                "date": current_date,  # 确保包含date字段
                "ret_1d_ahead": float(test_row["ret_1d_ahead"].iloc[0]),
                "weights": current_weights.copy(),
            }
        )

    results_df = pd.DataFrame(results)
    if not results_df.empty and 'date' in results_df.columns:
        results_df = results_df.set_index("date")
    return results_df


def evaluate_and_compare(all_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    for name, df in all_results.items():
        if df.empty:
            continue

        temp = df.copy()
        temp["position"] = np.where(temp["prob"] >= 0.5, 1, -1)
        temp["strategy_ret"] = temp["position"] * temp["ret_1d_ahead"]

        accuracy = (temp["pred"] == temp["actual"]).mean()
        avg_prob = temp["prob"].mean()
        cum_return = temp["strategy_ret"].cumsum().iloc[-1]
        hit_ratio_up = temp.loc[temp["position"] == 1, "actual"].mean()

        records.append(
            {
                "experiment": name,
                "accuracy": accuracy,
                "avg_prob": avg_prob,
                "cum_return": cum_return,
                "hit_ratio_when_long": hit_ratio_up,
                "num_trades": len(temp),
            }
        )

    metrics = pd.DataFrame(records).set_index("experiment")
    return metrics


def train_lstm_only(config: Dict, dm: DataManager) -> pd.DataFrame:
    """单独训练和验证LSTM模型"""
    models = {
        "LSTM_Main": LSTMModel(
            name="LSTM_Single",
            input_shape=(config["seq_len"], len(config["feature_cols_deep_base"])),
            config=config
        )
    }
    return _run_single_model_experiment(config, dm, models, "lstm_only")

def train_lgbm_only(config: Dict, dm: DataManager) -> pd.DataFrame:
    """单独训练和验证LightGBM模型"""
    models = {
        "LGBM_Main": LightGBMModel(
            name="LGBM_Single",
            config=config
        )
    }
    return _run_single_model_experiment(config, dm, models, "lgbm_only")

def train_llm_only(config: Dict, dm: DataManager) -> pd.DataFrame:
    """单独验证LLM模型"""
    llm_service = RollingLLMService(config, dm)
    models = {
        "LLM_Expert": LLMPredictor(
            name="LLM_Expert",
            llm_service=llm_service
        )
    }
    return _run_single_model_experiment(config, dm, models, "llm_only")

def train_transfer_learning(config: Dict, dm_target: DataManager, dm_source: DataManager) -> pd.DataFrame:
    """训练迁移学习模型"""
    print("开始迁移学习训练...")
    
    # 准备迁移学习数据
    source_features, X_target, y_target = prepare_transfer_data(dm_source, dm_target, config)
    
    # 创建迁移学习模型
    input_shape = (config["seq_len"], len(config["feature_cols_deep_base"]))
    transfer_model = TransferLSTMPredictor(
        name="Transfer_LSTM",
        input_shape=input_shape,
        config=config
    )
    
    # 预训练编码器
    transfer_model.pretrain_encoder(source_features)
    
    # 微调整个模型
    results = _run_transfer_experiment(config, dm_target, transfer_model, X_target, y_target)
    
    return results

def _run_transfer_experiment(config, dm, transfer_model, X_full, y_full):
    """运行迁移学习实验"""
    target_df = dm.get_target_df().copy()
    dates = target_df.index.to_list()
    valid_len = config["valid_len"]
    start_idx = config["start_index_for_backtest"]
    
    results = []
    
    for i in range(start_idx, len(dates) - 1):
        current_date = dates[i]
        train_end = i - valid_len - 1
        valid_start = i - valid_len
        valid_end = i - 1
        
        if train_end < config["seq_len"]:
            continue
        
        # 应用训练窗口限制
        train_window = config.get("train_window", None)
        if train_window is not None and train_end + 1 > train_window:
            train_start = max(0, train_end - train_window + 1)
            X_train = X_full[train_start:train_end + 1] if train_end + 1 < len(X_full) else X_full
            y_train = y_full[train_start:train_end + 1] if train_end + 1 < len(y_full) else y_full
        else:
            X_train = X_full[:train_end + 1] if train_end + 1 < len(X_full) else X_full
            y_train = y_full[:train_end + 1] if train_end + 1 < len(y_full) else y_full
            
        X_valid = X_full[valid_start:valid_end + 1] if valid_end + 1 < len(X_full) else None
        y_valid = y_full[valid_start:valid_end + 1] if valid_end + 1 < len(y_full) else None
        
        # 定期重训练
        if (i - start_idx) % config["retrain_interval"] == 0:
            transfer_model.fit(X_train, y_train, X_valid, y_valid)
        
        # 预测
        if len(X_full) > i:
            prob = transfer_model.predict_proba(X_full[i:i+1])[0]
            
            test_row = target_df.iloc[i : i + 1]
            final_label = int(prob > 0.5)
            actual = int(test_row["label"].iloc[0])
            
            results.append({
                "date": current_date,
                "prob": prob,
                "pred": final_label,
                "actual": actual,
                "ret_1d_ahead": float(test_row["ret_1d_ahead"].iloc[0]),
                "model": "Transfer_LSTM"
            })
    
    return pd.DataFrame(results).set_index("date")

def _run_single_model_experiment(
    config: Dict, 
    dm: DataManager,
    models: Dict[str, BasePredictor],
    experiment_name: str
) -> pd.DataFrame:
    """单模型实验的通用流程"""
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

        # 准备训练和验证数据
        X_train, y_train, X_valid, y_valid = None, None, None, None
        
        # 为LSTM准备序列数据
        if "LSTM_Main" in models:
            feature_cols = config["feature_cols_deep_base"]
            X_train, y_train, _ = dm.build_sequence_data(
                train_df, feature_cols, label_col, seq_len
            )
            X_valid, y_valid, _ = dm.build_sequence_data(
                valid_df, feature_cols, label_col, seq_len
            )
        
        # 为LightGBM准备特征数据
        elif "LGBM_Main" in models:
            feature_cols = config["feature_cols_tree_base"]
            X_train = train_df[feature_cols].values
            y_train = train_df[label_col].values
            X_valid = valid_df[feature_cols].values
            y_valid = valid_df[label_col].values
        
        # LLM不需要训练数据，只需要验证日期
        elif "LLM_Expert" in models:
            X_train = np.array(valid_df.index).reshape(-1, 1) if len(valid_df) > 0 else None
            X_valid = np.array(valid_df.index).reshape(-1, 1) if len(valid_df) > 0 else None

        # 定期重训练（仅对有训练能力的模型）
        if (i - start_idx) % config["retrain_interval"] == 0:
            for model_name, model in models.items():
                if model_name != "LLM_Expert":  # LLM不需要训练
                    model.fit(X_train, y_train, X_valid, y_valid)

        # 在当前交易日进行预测
        preds_today = {}
        for model_name, model in models.items():
            if model_name == "LSTM_Main":
                hist_df = target_df.iloc[: i + 1]
                X_hist, _, _ = dm.build_sequence_data(
                    hist_df, config["feature_cols_deep_base"], label_col, seq_len
                )
                if len(X_hist) > 0:
                    preds_today[model_name] = float(model.predict_proba(X_hist[-1:])[0])
            
            elif model_name == "LGBM_Main":
                X_today = test_row[config["feature_cols_tree_base"]].values
                preds_today[model_name] = float(model.predict_proba(X_today)[0])
            
            elif model_name == "LLM_Expert":
                preds_today[model_name] = float(
                    model.predict_proba(np.array([current_date]).reshape(-1, 1))[0]
                )

        # 记录结果
        final_prob = next(iter(preds_today.values())) if preds_today else 0.5
        final_label = int(final_prob >= 0.5)
        actual = int(test_row[label_col].iloc[0])

        results.append({
            "date": current_date,
            "prob": final_prob,
            "pred": final_label,
            "actual": actual,
            "ret_1d_ahead": float(test_row["ret_1d_ahead"].iloc[0]),
            "model": list(models.keys())[0]
        })

    return pd.DataFrame(results).set_index("date")

def main_pipeline_with_comparison(config: Dict, single_model: str = None):
    """支持单独模型训练的主流程
    Args:
        single_model: None表示完整流程，可选"lstm"/"lgbm"/"llm"
    """
    dm = DataManager(config)
    dm.load_all_raw_data()
    dm.align_merge_target()
    dm.generate_target_features_base()
    
    # 单模型训练模式
    if single_model:
        if single_model.lower() == "lstm":
            results = train_lstm_only(config, dm)
            experiment_name = "LSTM_单独训练"
        elif single_model.lower() == "lgbm":
            results = train_lgbm_only(config, dm)
            experiment_name = "LightGBM_单独训练"
        elif single_model.lower() == "llm":
            results = train_llm_only(config, dm)
            experiment_name = "LLM_单独验证"
        else:
            raise ValueError(f"未知的单模型类型: {single_model}")
        
        # 评估单个模型
        metrics = evaluate_and_compare({experiment_name: results})
        print(f"\n===== {experiment_name} 指标 =====")
        print(metrics)
        return metrics
    
    # 完整多模型比较模式
    dm_base = dm
    cross_features = {}
    if config.get("use_cross_ndx", False):
        source_cfg = config.get("cross_asset_sources", {}).get("US_NDX", {})
        feature_series = generate_cross_asset_predictions(
            dm=dm_base,
            source_key="US_NDX",
            feature_name=source_cfg.get("feature_name", "US_NDX_pred"),
            min_history=source_cfg.get("min_history", 80),
        )
        cross_features["US_NDX_pred"] = feature_series

    dm_full = dm_base.clone()
    if cross_features:
        dm_full.merge_cross_asset_features(cross_features)

    llm_service = RollingLLMService(config=config, dm_for_prompt=dm_full) if config.get("use_llm_as_expert", False) else None

    experiments = {
        "baseline_single_lstm": run_experiment(config, dm_base, "baseline_single_lstm", llm_service=None),
        "baseline_single_lgbm": run_experiment(config, dm_base, "baseline_single_lgbm", llm_service=None),
        "baseline_ensemble_no_transfer": run_experiment(
            config, dm_base, "baseline_ensemble_no_transfer", llm_service=None
        ),
        "transfer_full_model": run_experiment(config, dm_full, "transfer_full_model", llm_service=llm_service),
    }

    metrics = evaluate_and_compare(experiments)
    print("\n===== 各方案关键指标 =====")
    print(metrics)
    return metrics
