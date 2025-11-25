from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .data_manager import DataManager

def _prepare_cross_asset_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()

    frame["log_ret"] = np.log(frame["close"] / frame["close"].shift(1))
    frame["ma5"] = frame["close"].rolling(5, min_periods=5).mean()
    frame["ma10"] = frame["close"].rolling(10, min_periods=10).mean()
    frame["ma20"] = frame["close"].rolling(20, min_periods=20).mean()
    frame["ret_1d_ahead"] = frame["close"].shift(-1) / frame["close"] - 1.0
    frame["label"] = (frame["ret_1d_ahead"] > 0).astype(int)

    frame = frame.dropna()
    return frame

def generate_cross_asset_predictions(
    dm: DataManager,
    source_key: str,
    feature_name: str,
    min_history: int = 80,
) -> pd.Series:
    """
    使用滚动 Logistic Regression，对 source_key 市场生成“下一交易日上涨概率”。
    - 在日期 t，只使用 t 之前的数据拟合模型，再预测 t 的 label（对应 t+1 方向）。
    - 最终对齐到 target 的交易日索引。
    """
    if source_key not in dm.raw_data:
        raise ValueError(f"cross asset source '{source_key}' not found in raw_data")

    df = _prepare_cross_asset_frame(dm.raw_data[source_key])
    if len(df) <= min_history:
        raise ValueError(f"not enough data for {source_key}; need > {min_history}")

    feature_cols = ["log_ret", "ma5", "ma10", "ma20"]
    probs: Dict[pd.Timestamp, float] = {}
    model = LogisticRegression(class_weight="balanced", max_iter=200)

    last_prob = 0.5
    for idx in range(min_history, len(df)):
        train_df = df.iloc[:idx]
        if train_df["label"].nunique() < 2:
            probs[df.index[idx]] = last_prob
            continue

        model.fit(train_df[feature_cols], train_df["label"])
        test_X = df[feature_cols].iloc[idx : idx + 1]
        prob = float(model.predict_proba(test_X)[0, 1])
        prob = float(np.clip(prob, 0.01, 0.99))
        probs[df.index[idx]] = prob
        last_prob = prob

    series = pd.Series(probs, name=feature_name).sort_index()
    aligned = series.reindex(dm.get_target_df().index)
    return aligned
