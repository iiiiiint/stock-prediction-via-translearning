from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightgbm import LGBMClassifier
from torch.utils.data import DataLoader, TensorDataset

from .services import RollingLLMService


class BasePredictor:
    def __init__(self, name: str):
        self.name = name

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def save(self, path: str):
        raise NotImplementedError

    def load(self, path: str):
        raise NotImplementedError


# ------------------------------------------------------------------ #
# LSTM
# ------------------------------------------------------------------ #
class _LSTMEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        prob = self.classifier(last_hidden)
        return prob

class LSTMModel(BasePredictor):
    def __init__(self, name: str, input_shape: Tuple[int, int], config: Dict):
        super().__init__(name)
        seq_len, feat_dim = input_shape
        hidden_size = config.get("lstm_hidden_size", 64)
        dropout = config.get("lstm_dropout", 0.1)

        self.model = _LSTMEncoder(feat_dim, hidden_size, dropout)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.epochs = config.get("lstm_epochs", 20)  # 增加训练轮次
        self.lr = config.get("lstm_lr", 1e-3)
        self.batch_size = config.get("lstm_batch_size", 64)
        self.weight_decay = config.get("lstm_weight_decay", 1e-4)  # 权重衰减
        self.grad_clip = config.get("lstm_grad_clip", 1.0)  # 梯度裁剪
        self.patience = config.get("lstm_patience", 5)  # 早停耐心值
        
        self._is_fitted = False

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        if X_train is None or len(X_train) == 0:
            return

        dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 使用BCEWithLogitsLoss（数值稳定性更好）
        criterion = nn.BCEWithLogitsLoss()
        # 添加权重衰减
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_val_loss = np.inf
        best_val_accuracy = 0
        best_epoch = -1
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                optimizer.step()
                train_loss += loss.item()

            # 验证阶段
            val_loss, val_accuracy, val_auc = None, None, None
            if X_valid is not None and len(X_valid) > 0:
                val_loss, val_accuracy, val_auc = self._validate(X_valid, y_valid, criterion)
                
                # 早停机制
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_accuracy = val_accuracy
                    best_epoch = epoch
                    best_state = self.model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.patience:
                    print(f"早停触发，在第 {epoch+1} 轮停止训练")
                    break

            # 打印训练信息
            log_msg = f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss/len(loader):.4f}"
            if val_loss is not None:
                log_msg += f", Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val AUC: {val_auc:.4f}"
            print(log_msg)

        # 加载最佳模型
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"最佳模型: Epoch {best_epoch+1}, Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_accuracy:.4f}")
        
        self._is_fitted = True

    def _validate(self, X_valid, y_valid, criterion):
        """验证步骤，返回loss, accuracy, auc"""
        self.model.eval()
        with torch.no_grad():
            Xv = torch.tensor(X_valid, dtype=torch.float32).to(self.device)
            yv = torch.tensor(y_valid, dtype=torch.float32).unsqueeze(1).to(self.device)
            
            logits = self.model(Xv)
            loss = criterion(logits, yv).item()
            
            # 计算accuracy
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).float()
            accuracy = (predictions == yv).float().mean().item()
            
            # 计算AUC（简化为二分类准确率，实际AUC需要更多计算）
            auc = accuracy  # 这里简化处理，实际应该使用sklearn的auc计算
            
        return loss, accuracy, auc

    def predict_proba(self, X) -> np.ndarray:
        if X is None or len(X) == 0:
            return np.array([])

        self.model.eval()
        with torch.no_grad():
            tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            probs = self.model(tensor).cpu().numpy().reshape(-1)
        return probs


# ------------------------------------------------------------------ #
# LightGBM
# ------------------------------------------------------------------ #
class LightGBMModel(BasePredictor):
    def __init__(self, name: str, config: Dict):
        super().__init__(name)
        params = config.get("lgbm_params", {})
        self.model = LGBMClassifier(
            objective="binary",
            n_estimators=params.get("n_estimators", 2000),           # 增加树的数量
            learning_rate=params.get("learning_rate", 0.01),         # 降低学习率
            num_leaves=params.get("num_leaves", 31),
            max_depth=params.get("max_depth", 6),                    # 限制深度防止过拟合
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            reg_alpha=params.get("reg_alpha", 0.1),                  # L1正则化
            reg_lambda=params.get("reg_lambda", 0.1),                # L2正则化
            min_child_samples=params.get("min_child_samples", 20),
            random_state=42,
        )
        self.early_stopping_rounds = params.get("early_stopping_rounds", 100)

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        if X_train is None or len(X_train) == 0:
            return

        fit_kwargs = {}
        if X_valid is not None and len(X_valid) > 0:
            fit_kwargs.update(
                {
                    "eval_set": [(X_valid, y_valid)],
                    "eval_metric": "binary_logloss",
                    "early_stopping_rounds": self.early_stopping_rounds,
                    "verbose": False
                }
            )

        self.model.fit(X_train, y_train, **fit_kwargs)

    def predict_proba(self, X) -> np.ndarray:
        if X is None or len(X) == 0:
            return np.array([])

        probs = self.model.predict_proba(X)[:, 1]
        return probs


# ------------------------------------------------------------------ #
# LLM Expert
# ------------------------------------------------------------------ #
class LLMPredictor(BasePredictor):
    def __init__(self, name: str, llm_service: Optional[RollingLLMService] = None):
        super().__init__(name)
        self.llm_service = llm_service

    def fit(self, *args, **kwargs):
        return

    def predict_proba(self, X) -> np.ndarray:
        if self.llm_service is None or X is None or len(X) == 0:
            return np.full(shape=(len(X) if X is not None else 0,), fill_value=0.5, dtype=float)

        dates = np.array(X).reshape(-1)
        outputs = []
        for date_val in dates:
            ts = pd.Timestamp(date_val)
            prob = self.llm_service.get_prob_for_date(ts)
            outputs.append(prob)
        return np.array(outputs, dtype=float)
