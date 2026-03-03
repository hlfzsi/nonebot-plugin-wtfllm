"""MMR 选择算法单元测试"""

import numpy as np
import pytest

from nonebot_plugin_wtfllm.topic.clustering.mmr import mmr_select


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n == 0, 1, n)
    return v / n


class TestMMRSelect:

    def test_empty_candidates(self):
        candidates = np.zeros((0, 8), dtype=np.float64)
        query = np.ones(8, dtype=np.float64)
        assert mmr_select(candidates, query, k=3) == []

    def test_k_larger_than_n(self):
        """k > N 时应返回所有 N 个索引"""
        candidates = _norm(np.random.randn(3, 8))
        query = _norm(np.random.randn(8))
        result = mmr_select(candidates, query, k=10)
        assert len(result) == 3
        assert len(set(result)) == 3  # 无重复

    def test_k_equals_one(self):
        """k=1 应返回与 query 最相似的候选"""
        query = _norm(np.array([1.0, 0.0, 0.0, 0.0]))
        candidates = _norm(np.array([
            [0.9, 0.1, 0.0, 0.0],  # 最接近 query
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]))
        result = mmr_select(candidates, query, k=1, lambda_param=1.0)
        assert result == [0]

    def test_no_duplicates(self):
        """结果不应包含重复索引"""
        np.random.seed(42)
        candidates = _norm(np.random.randn(20, 16))
        query = _norm(np.random.randn(16))
        result = mmr_select(candidates, query, k=10)
        assert len(result) == 10
        assert len(set(result)) == 10

    def test_lambda_1_pure_relevance(self):
        """lambda=1 时退化为纯相关性排序"""
        query = _norm(np.array([1.0, 0.0]))
        candidates = _norm(np.array([
            [0.5, 0.866],   # cos ≈ 0.5
            [0.95, 0.312],  # cos ≈ 0.95
            [0.3, 0.954],   # cos ≈ 0.3
        ]))
        result = mmr_select(candidates, query, k=3, lambda_param=1.0)
        # 应按相似度递减: 1 (0.95), 0 (0.5), 2 (0.3)
        assert result[0] == 1

    def test_lambda_0_pure_diversity(self):
        """lambda=0 时应最大化多样性"""
        query = _norm(np.array([1.0, 0.0, 0.0]))
        candidates = _norm(np.array([
            [1.0, 0.0, 0.0],  # 与 query 完全相同
            [0.99, 0.1, 0.0],  # 与 candidate[0] 极相似
            [0.0, 0.0, 1.0],  # 与前两者正交
        ]))
        result = mmr_select(candidates, query, k=3, lambda_param=0.0)
        # 第一个选任意，后续应避免与已选重复
        # 结果应包含 index 2（正交向量）而不是仅选相似的
        assert 2 in result

    def test_balanced_selection(self):
        """lambda=0.5 应在相关性和多样性间取平衡"""
        np.random.seed(123)
        # 用更高维、更接近的向量模拟实际场景
        dim = 32
        base = _norm(np.random.randn(1, dim))
        # 所有候选都与 query 有一定相似度，但彼此间有差异
        candidates = _norm(base + np.random.randn(9, dim) * 0.5)
        query = base.flatten()

        result = mmr_select(candidates, query, k=5, lambda_param=0.5)
        assert len(result) == 5
        assert len(set(result)) == 5

        # 第一个应是最接近 query 的
        sims = candidates @ query
        assert result[0] == int(np.argmax(sims))

    def test_identical_candidates(self):
        """所有候选相同时应正常返回 k 个不同索引"""
        v = _norm(np.array([[1.0, 0.0, 0.0]]))
        candidates = np.tile(v, (5, 1))
        query = _norm(np.array([1.0, 0.0, 0.0]))
        result = mmr_select(candidates, query, k=3)
        assert len(result) == 3
        assert len(set(result)) == 3
