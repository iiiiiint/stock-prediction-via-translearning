from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


class MetaLearner:
    def __init__(self, window_size: int = 20, eta: float = 5.0, min_samples: int = 5):
        self.window_size = window_size
        self.eta = eta
        self.min_samples = min_samples

    @staticmethod
    def bce_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def update_weights(self, recent_preds: pd.DataFrame, recent_true: pd.Series) -> Dict[str, float]:
        if recent_preds.empty:
            return {}

        recent_true = recent_true.reindex(recent_preds.index)
        losses = {}
        for model in recent_preds.columns:
            series = recent_preds[model].dropna()
            if len(series) < self.min_samples:
                continue
            y = recent_true.loc[series.index].values
            p = series.values
            loss = self.bce_loss(y, p).mean()
            losses[model] = loss

        if not losses:
            model_names = recent_preds.columns.tolist()
            return {m: 1.0 / len(model_names) for m in model_names}

        names = list(losses.keys())
        loss_vals = np.array([losses[m] for m in names])
        weights_raw = np.exp(-self.eta * loss_vals)
        weights = weights_raw / weights_raw.sum()
        return {m: w for m, w in zip(names, weights)}
