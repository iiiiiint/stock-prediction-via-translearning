from __future__ import annotations
from typing import Dict, List
import pandas as pd
import numpy as np
from .meta import MetaLearner
from .data_manager import DataManager

class EnsembleModel:
    """加权混合模型集成器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.meta_learner = MetaLearner(
            window_size=config["meta_window"],
            eta=config["meta_eta"],
            min_samples=5,
        )
        self.model_results = {}
    
    def add_model_results(self, model_name: str, results: pd.DataFrame):
        """添加单个模型的结果"""
        self.model_results[model_name] = results
    
    def run_ensemble(self) -> pd.DataFrame:
        """运行集成预测"""
        if not self.model_results:
            raise ValueError("没有添加模型结果")
        
        # 对齐所有模型的结果日期
        aligned_results = self._align_results()
        
        final_results = []
        dates = aligned_results.index.unique()
        
        for i, current_date in enumerate(dates):
            if i < self.config["meta_window"]:
                continue
                
            # 收集近期预测用于元学习
            recent_preds = {}
            recent_true = []
            
            for model_name, df in aligned_results.groupby(level=0):
                if current_date in df.index:
                    model_df = df.loc[:current_date]
                    recent_data = model_df.tail(self.config["meta_window"])
                    
                    if not recent_data.empty:
                        recent_preds[model_name] = recent_data['prob']
                        if len(recent_true) == 0:
                            recent_true = recent_data['actual']
            
            if not recent_preds:
                continue
                
            # 更新权重
            recent_preds_df = pd.DataFrame(recent_preds)
            weights = self.meta_learner.update_weights(recent_preds_df, recent_true)
            
            # 当前预测
            current_preds = {}
            for model_name in self.model_results.keys():
                if current_date in aligned_results.index:
                    model_pred = aligned_results.loc[current_date, model_name]['prob']
                    current_preds[model_name] = model_pred
            
            # 加权集成
            final_prob = 0.0
            for model_name, prob in current_preds.items():
                weight = weights.get(model_name, 1.0/len(current_preds))
                final_prob += weight * prob
            
            # 记录结果
            final_label = int(final_prob > 0.5)
            actual = aligned_results.loc[current_date, list(self.model_results.keys())[0]]['actual']
            ret_1d = aligned_results.loc[current_date, list(self.model_results.keys())[0]]['ret_1d_ahead']
            
            final_results.append({
                "date": current_date,
                "prob": final_prob,
                "pred": final_label,
                "actual": actual,
                "ret_1d_ahead": ret_1d,
                "weights": weights
            })
        
        return pd.DataFrame(final_results).set_index("date")
    
    def _align_results(self) -> pd.DataFrame:
        """对齐所有模型的结果"""
        aligned_dfs = []
        for model_name, results in self.model_results.items():
            df = results.copy()
            df['model'] = model_name
            aligned_dfs.append(df.reset_index())
        
        # 合并所有结果
        full_df = pd.concat(aligned_dfs, ignore_index=True)
        
        # 创建多级索引 (date, model)
        aligned = full_df.set_index(['date', 'model']).sort_index()
        
        return aligned

def evaluate_ensemble_performance(results: pd.DataFrame) -> Dict:
    """评估集成模型表现"""
    if results.empty:
        return {}
    
    temp = results.copy()
    temp["position"] = np.where(temp["prob"] >= 0.5, 1, -1)
    temp["strategy_ret"] = temp["position"] * temp["ret_1d_ahead"]
    
    metrics = {
        "accuracy": (temp["pred"] == temp["actual"]).mean(),
        "avg_return": temp["strategy_ret"].mean(),
        "total_return": temp["strategy_ret"].sum(),
        "win_rate": (temp["strategy_ret"] > 0).mean(),
        "num_trades": len(temp)
    }
    
    return metrics