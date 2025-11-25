from __future__ import annotations

import os
import re
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    from openai import OpenAI   
except ImportError:  # 允许环境未安装 openai
    OpenAI = None


class RollingLLMService:
    """
    严格滚动窗生成 LLM 预测概率。
    """

    def __init__(self, config: Dict, dm_for_prompt):
        self.config = config
        self.dm_for_prompt = dm_for_prompt
        self.cache: Dict[pd.Timestamp, float] = {}

        self.hist_window = int(config.get("llm_hist_window", 120))
        self.model_name = config.get("llm_model_name", "gpt-4.1-mini")
        self.temperature = config.get("llm_temperature", 0.0)
        self.max_tokens = int(config.get("llm_max_tokens", 1024))

        api_key = config.get("llm_api_key") or os.getenv("OPENAI_API_KEY")
        client_instance = OpenAI(api_key=api_key) if (api_key and OpenAI is not None) else None
        self.client = client_instance

    def get_prob_for_date(self, current_date: pd.Timestamp) -> float:
        if current_date in self.cache:
            return self.cache[current_date]

        df = self.dm_for_prompt.get_target_df()
        if current_date not in df.index:
            self.cache[current_date] = 0.5
            return 0.5

        pos = df.index.get_loc(current_date)
        start_idx = max(0, pos - self.hist_window + 1)
        hist_df = df.iloc[start_idx : pos + 1]

        prompt = self._build_prompt_from_history(hist_df)
        prob = self._call_llm_and_parse_prob(prompt, hist_df)
        self.cache[current_date] = prob
        return prob

    def _build_prompt_from_history(self, hist_df: pd.DataFrame) -> str:
        lines = [
            "你是一个量化交易助手。以下是截至今日的历史行情，请估计下一交易日（T+1）上涨概率（0-1之间）。",
            "",
        ]
        tail_df = hist_df.tail(min(30, len(hist_df)))
        for idx, row in tail_df.iterrows():
            lines.append(
                f"{idx.date()} 收盘: {row['close']:.2f}, 日对数收益: {row['log_ret']:.4f}, MA5: {row['ma5']:.2f}, MA10: {row['ma10']:.2f}, MA20: {row['ma20']:.2f}"
            )

        lines.append("")
        lines.append("请只输出一个 0 到 1 之间的小数，表示下一交易日上涨概率。")
        return "\n".join(lines)

    def _call_llm_and_parse_prob(self, prompt: str, hist_df: pd.DataFrame) -> float:
        if self.client is None:
            return self._fallback_prob(hist_df)

        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
            temperature=self.temperature,
            max_output_tokens=64,
        )

        text_chunks = []
        for item in response.output:
            for content in item.content:
                if hasattr(content, "text"):
                    text_chunks.append(content.text)
        raw_text = " ".join(text_chunks).strip()

        match = re.search(r"([01]?\.\d+|\d+)", raw_text)
        if not match:
            return self._fallback_prob(hist_df)

        prob = float(match.group(1))
        prob = float(np.clip(prob, 0.01, 0.99))
        return prob

    @staticmethod
    def _fallback_prob(hist_df: pd.DataFrame) -> float:
        recent = hist_df["log_ret"].tail(20).fillna(0.0)
        momentum = recent.sum()
        prob = 0.5 + 0.25 * np.tanh(momentum * 10)
        return float(np.clip(prob, 0.05, 0.95))
