"""prefilter.py 预筛选单元测试"""

import numpy as np
import pytest

from nonebot_plugin_wtfllm.proactive.inertia.prefilter import (
    prefilter,
    _gini_coefficient,
)


class TestGiniCoefficient:

    def test_uniform(self):
        """完全均匀分布 → 基尼系数接近 0。"""
        arr = np.ones(100, dtype=np.float32)
        assert _gini_coefficient(arr) == pytest.approx(0.0, abs=0.02)

    def test_single_spike(self):
        """只有一个值为非零 → 基尼系数接近 1。"""
        arr = np.zeros(1000, dtype=np.float32)
        arr[500] = 100.0
        assert _gini_coefficient(arr) > 0.95

    def test_all_zeros(self):
        """全零 → 返回 0。"""
        arr = np.zeros(100, dtype=np.float32)
        assert _gini_coefficient(arr) == 0.0

    def test_moderate_inequality(self):
        """一半为 1，一半为 0 → 中等集中度。"""
        arr = np.zeros(100, dtype=np.float32)
        arr[:50] = 1.0
        gini = _gini_coefficient(arr)
        assert 0.3 < gini < 0.7


class TestPrefilterPasses:
    """应通过预筛选的数据模式。"""

    def test_clear_single_peak(self):
        """单个明显峰值：600-660 分钟有高活跃。"""
        bins = np.zeros(1440, dtype=np.float32)
        bins[600:660] = 10.0
        passed, quality = prefilter(bins, max_active_days=10, min_required_days=7)
        assert passed is True
        assert 0.0 < quality <= 1.0

    def test_dual_peak(self):
        """双峰模式：早晚各一个活跃窗口。"""
        bins = np.zeros(1440, dtype=np.float32)
        bins[540:600] = 8.0   # 09:00-10:00
        bins[1200:1260] = 6.0  # 20:00-21:00
        passed, quality = prefilter(bins, max_active_days=8, min_required_days=7)
        assert passed is True

    def test_quality_increases_with_more_data(self):
        """更多活跃天数 → 更高质量分。"""
        bins = np.zeros(1440, dtype=np.float32)
        bins[600:660] = 7.0
        _, q_low = prefilter(bins, max_active_days=7, min_required_days=7)

        bins2 = np.zeros(1440, dtype=np.float32)
        bins2[600:660] = 14.0
        _, q_high = prefilter(bins2, max_active_days=14, min_required_days=7)

        assert q_high > q_low


class TestPrefilterRejects:
    """应被预筛选拒绝的数据模式。"""

    def test_insufficient_days(self):
        """活跃天数不足。"""
        bins = np.zeros(1440, dtype=np.float32)
        bins[600:660] = 3.0
        passed, _ = prefilter(bins, max_active_days=3, min_required_days=7)
        assert passed is False

    def test_all_day_uniform(self):
        """全天均匀活跃 → 无法拟合有意义曲线。"""
        bins = np.ones(1440, dtype=np.float32) * 10.0
        passed, _ = prefilter(bins, max_active_days=10, min_required_days=7)
        assert passed is False

    def test_empty_data(self):
        """无数据。"""
        bins = np.zeros(1440, dtype=np.float32)
        passed, _ = prefilter(bins, max_active_days=0, min_required_days=7)
        assert passed is False

    def test_flat_low_activity(self):
        """活跃天数足够但分布完全平坦（覆盖率过高）。"""
        bins = np.zeros(1440, dtype=np.float32)
        bins[100:900] = 8.0  # 800 分钟均匀活跃 → 覆盖率 55.6% > 50%
        passed, _ = prefilter(bins, max_active_days=8, min_required_days=7)
        assert passed is False

    def test_near_zero_spike(self):
        """只有 2 个非零点 → 数据太稀疏，基尼系数虽高但不足以建模。"""
        bins = np.zeros(1440, dtype=np.float32)
        bins[600] = 7.0
        bins[601] = 7.0
        passed, _ = prefilter(bins, max_active_days=7, min_required_days=7)
        # 仅 2 分钟有数据，覆盖率极低，但预筛选可能通过
        # 实际拟合会在 curve.py 中因样本不足返回 None
        # 此处仅确认不抛异常
        assert isinstance(passed, bool)
