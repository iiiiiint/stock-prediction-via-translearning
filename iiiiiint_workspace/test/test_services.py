import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime
from src.services import RollingLLMService

class TestDataManager:
    def get_target_df(self):
        dates = pd.date_range(start="2025-01-01", periods=150)
        df = pd.DataFrame(
            {
                "close": np.random.uniform(100, 200, size=150),
                "log_ret": np.random.normal(0, 0.01, size=150),
                "ma5": np.random.uniform(100, 200, size=150),
                "ma10": np.random.uniform(100, 200, size=150),
                "ma20": np.random.uniform(100, 200, size=150),
            },
            index=dates
        )
        return df

class TestRollingLLMService:
    def setup_method(self):
        self.config = {
            "llm_hist_window": 120,
            "llm_model_name": "gpt-4.1-mini",
            "llm_temperature": 0.0,
            "llm_max_tokens": 1024
        }
        self.dm = TestDataManager()
        self.service = RollingLLMService(self.config, self.dm)

    def test_get_prob_for_date_cache_hit(self):
        """测试缓存命中情况"""
        test_date = pd.Timestamp("2025-01-10")
        self.service.cache[test_date] = 0.7
        assert self.service.get_prob_for_date(test_date) == 0.7

    def test_get_prob_for_date_not_in_index(self):
        """测试日期不在索引中的情况"""
        test_date = pd.Timestamp("2024-12-31")  # 不在测试数据范围内
        assert self.service.get_prob_for_date(test_date) == 0.5
        assert test_date in self.service.cache

    def test_get_prob_for_date_normal_case(self):
        """测试正常情况下的概率计算"""
        test_date = pd.Timestamp("2025-01-15")
        with patch.object(self.service, '_call_llm_and_parse_prob', return_value=0.6):
            prob = self.service.get_prob_for_date(test_date)
            assert 0 <= prob <= 1
            assert test_date in self.service.cache

    def test_get_prob_for_date_beginning_of_series(self):
        """测试序列开始时的边界情况"""
        test_date = pd.Timestamp("2025-01-01")
        with patch.object(self.service, '_call_llm_and_parse_prob', return_value=0.55):
            prob = self.service.get_prob_for_date(test_date)
            assert 0 <= prob <= 1
            assert test_date in self.service.cache

    def test_get_prob_for_date_end_of_series(self):
        """测试序列结束时的边界情况"""
        test_date = pd.Timestamp("2025-05-30")  # 测试数据的最后日期
        with patch.object(self.service, '_call_llm_and_parse_prob', return_value=0.65):
            prob = self.service.get_prob_for_date(test_date)
            assert 0 <= prob <= 1
            assert test_date in self.service.cache

    def test_get_prob_for_date_with_fallback(self):
        """测试LLM不可用时的回退逻辑"""
        test_date = pd.Timestamp("2025-02-01")
        original_client = self.service.client
        self.service.client = None  # 模拟LLM不可用
        prob = self.service.get_prob_for_date(test_date)
        assert 0.05 <= prob <= 0.95  # 回退概率应在限定范围内
        assert test_date in self.service.cache
        self.service.client = original_client