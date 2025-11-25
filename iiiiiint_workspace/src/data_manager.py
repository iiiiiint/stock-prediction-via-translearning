from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class DataManager:
    """
    管理原始数据、目标对齐与基础特征生成。
    """

    def __init__(self, config: Dict):
        self.config = config
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.target_df: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    # 数据加载与对齐
    # ------------------------------------------------------------------ #
    def load_all_raw_data(self) -> None:
        data_dir = Path(self.config.get("data_dir", "data/raw"))
        data_sources = self.config.get("data_sources", {})

        for key, meta in data_sources.items():
            filename = meta.get("filename")
            explicit_path = meta.get("path")
            path = Path(explicit_path) if explicit_path else data_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"data source '{key}' missing at {path}")

            date_col = meta.get("date_col", "date")
            file_type = meta.get("file_type", "csv")  # 默认为csv

            # 根据文件类型读取数据
            if file_type.lower() == "excel":
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)

            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col).sort_index()
            
            # 按时间范围截断数据
            time_range = self.config.get("data_time_range", "all")
            if time_range != "all" and len(df) > 0:
                latest_date = df.index.max()
                if time_range == "1y":  # 1年
                    cutoff_date = latest_date - pd.DateOffset(years=1)
                elif time_range == "6m":  # 6个月
                    cutoff_date = latest_date - pd.DateOffset(months=6)
                elif time_range == "3m":  # 3个月
                    cutoff_date = latest_date - pd.DateOffset(months=3)
                else:
                    cutoff_date = latest_date - pd.DateOffset(years=1)  # 默认1年
                
                df = df[df.index >= cutoff_date]
                print(f"截断数据到最近 {time_range}，剩余 {len(df)} 个交易日")

            col_map = meta.get("column_mapping") or {}
            if col_map:
                df = df.rename(columns=col_map)

            required_cols = {"open", "high", "low", "close", "volume"}
            if not required_cols.issubset(df.columns):
                missing = required_cols - set(df.columns)
                raise ValueError(f"data '{key}' missing columns: {missing}")

            self.raw_data[key] = df

    def align_merge_target(self) -> None:
        if "target" not in self.raw_data:
            raise ValueError("target data not loaded. Check config['data_sources']['target'].'")

        target_df = self.raw_data["target"].copy().sort_index()
        self.target_df = target_df

    def generate_target_features_base(self) -> None:
        if self.target_df is None:
            raise ValueError("call align_merge_target() before generating features")

        df = self.target_df.copy()

        df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
        df["ma5"] = df["close"].rolling(5, min_periods=5).mean()
        df["ma10"] = df["close"].rolling(10, min_periods=10).mean()
        df["ma20"] = df["close"].rolling(20, min_periods=20).mean()

        df["ret_1d_ahead"] = df["close"].shift(-1) / df["close"] - 1.0
        df["label"] = (df["ret_1d_ahead"] > 0).astype(int)

        df = df.dropna()
        self.target_df = df

    def generate_advanced_lgbm_features(self, window=30) -> None:
        """为LightGBM生成高级统计特征"""
        if self.target_df is None:
            raise ValueError("target_df not ready")
        
        df = self.target_df.copy()
        
        # 1. 收益率统计特征
        df['ret_mean_30'] = df['log_ret'].rolling(window).mean()
        df['ret_std_30'] = df['log_ret'].rolling(window).std()
        df['ret_skew_30'] = df['log_ret'].rolling(window).skew()
        df['ret_kurt_30'] = df['log_ret'].rolling(window).kurt()
        
        # 2. 价格位置特征
        df['price_rank_30'] = (df['close'] - df['low'].rolling(window).min()) / \
                            (df['high'].rolling(window).max() - df['low'].rolling(window).min() + 1e-8)
        
        # 3. 波动率特征
        df['atr_14'] = (df['high'] - df['low']).rolling(14).mean()
        df['volatility_30'] = df['close'].pct_change().rolling(window).std()
        
        # 4. 成交量特征
        df['volume_ratio_30'] = df['volume'] / (df['volume'].rolling(window).mean() + 1e-8)
        df['volume_std_30'] = df['volume'].rolling(window).std()
        df['volume_skew_30'] = df['volume'].rolling(window).skew()
        
        # 5. 动量特征
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # 6. RSI相对强弱指标
        gains = df['close'].diff().where(lambda x: x > 0, 0)
        losses = -df['close'].diff().where(lambda x: x < 0, 0)
        avg_gain = gains.rolling(14).mean()
        avg_loss = losses.rolling(14).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # 7. 近期特征（保留1-3天）
        for i in range(1, 4):
            df[f'ret_{i}d'] = df['close'].pct_change(i)
            df[f'volume_change_{i}d'] = df['volume'].pct_change(i)
            df[f'close_ratio_{i}d'] = df['close'] / df['close'].shift(i)
        
        # 8. 价格与移动平均线关系
        df['close_vs_ma5'] = df['close'] / df['ma5'] - 1
        df['close_vs_ma20'] = df['close'] / df['ma20'] - 1
        df['ma5_vs_ma20'] = df['ma5'] / df['ma20'] - 1
        
        df = df.dropna()
        self.target_df = df

    def merge_cross_asset_features(self, feature_map: Dict[str, pd.Series]) -> None:
        if self.target_df is None:
            raise ValueError("target_df not initialized")

        df = self.target_df.copy()
        for feat_name, series in feature_map.items():
            aligned = series.reindex(df.index)
            df[feat_name] = aligned
        df = df.dropna()
        self.target_df = df

    def get_target_df(self) -> pd.DataFrame:
        if self.target_df is None:
            raise ValueError("target_df not ready. Run align_merge_target & feature generation.")
        return self.target_df

    def clone(self) -> "DataManager":
        cloned = DataManager(self.config)
        cloned.raw_data = {k: v.copy() for k, v in self.raw_data.items()}
        cloned.target_df = self.target_df.copy() if self.target_df is not None else None
        return cloned

    # ------------------------------------------------------------------ #
    # 序列化特征
    # ------------------------------------------------------------------ #
    def build_sequence_data(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str,
        seq_len: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
        if df.empty:
            return np.empty((0, seq_len, len(feature_cols))), np.empty((0,)), []

        X_list, y_list, idx_list = [], [], []
        for idx in range(seq_len - 1, len(df)):
            window = df.iloc[idx - seq_len + 1 : idx + 1]
            X_list.append(window[feature_cols].values)
            y_list.append(window[label_col].iloc[-1])
            idx_list.append(window.index[-1])

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        return X, y, idx_list
