"""curve.py LightGBM 曲线拟合单元测试"""

import numpy as np
import pytest

from nonebot_plugin_wtfllm.proactive.inertia._types import SessionKey
from nonebot_plugin_wtfllm.proactive.inertia.curve import (
    fit_activity_curve,
    _build_features,
    _compute_r_squared,
)


class TestBuildFeatures:
    def test_shape(self):
        minutes = np.arange(10, dtype=np.int32)
        features = _build_features(minutes)
        assert features.shape == (10, 12)  # 6 谐波 × 2 (sin + cos)

    def test_full_1440(self):
        minutes = np.arange(1440, dtype=np.int32)
        features = _build_features(minutes)
        assert features.shape == (1440, 12)

    def test_periodic(self):
        """minute 0 和 minute 1440 应该有相同的特征（周期性）。"""
        m0 = _build_features(np.array([0], dtype=np.int32))
        m1440 = _build_features(np.array([1440], dtype=np.int32))
        np.testing.assert_allclose(m0, m1440, atol=1e-10)


class TestComputeRSquared:
    def test_perfect_fit(self):
        y = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        assert _compute_r_squared(y, y.astype(np.float64)) == pytest.approx(1.0)

    def test_zero_variance(self):
        """常数目标 + 完美预测 → R²=1。"""
        y = np.full(10, 5.0, dtype=np.float32)
        assert _compute_r_squared(y, y.astype(np.float64)) == 1.0

    def test_poor_fit(self):
        y_true = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        y_pred = np.array([5, 4, 3, 2, 1], dtype=np.float64)
        r2 = _compute_r_squared(y_true, y_pred)
        assert r2 < 0  # 反向预测 → 负 R²


class TestFitActivityCurve:
    _SESSION = SessionKey(group_id="g1", sender="u1")

    def test_single_peak(self):
        """单峰用户应产出合理曲线。"""
        bins = np.zeros(1440, dtype=np.float32)
        bins[540:600] = 10.0  # 09:00-10:00 活跃

        curve = fit_activity_curve(self._SESSION, bins, data_quality=0.8)
        assert curve is not None
        assert curve.predicted_median.shape == (1440,)
        assert curve.session == self._SESSION
        assert curve.data_quality == 0.8

    def test_lower_le_median_le_upper(self):
        """预测区间：lower ≤ median ≤ upper 在所有分钟上成立。"""
        bins = np.zeros(1440, dtype=np.float32)
        bins[600:660] = 8.0
        bins[1200:1260] = 5.0

        curve = fit_activity_curve(self._SESSION, bins, data_quality=0.7)
        assert curve is not None
        assert np.all(curve.predicted_lower <= curve.predicted_median)
        assert np.all(curve.predicted_median <= curve.predicted_upper)

    def test_non_negative_predictions(self):
        """所有预测值 ≥ 0。"""
        bins = np.zeros(1440, dtype=np.float32)
        bins[300:360] = 12.0

        curve = fit_activity_curve(self._SESSION, bins, data_quality=0.9)
        assert curve is not None
        assert np.all(curve.predicted_lower >= 0)
        assert np.all(curve.predicted_median >= 0)
        assert np.all(curve.predicted_upper >= 0)

    def test_insufficient_data_returns_none(self):
        """非零点不足 3 个 → 返回 None。"""
        bins = np.zeros(1440, dtype=np.float32)
        bins[600] = 5.0
        bins[601] = 3.0

        curve = fit_activity_curve(self._SESSION, bins, data_quality=0.5)
        assert curve is None

    def test_peak_higher_than_quiet(self):
        """活跃区间的中位预测应高于安静区间。"""
        bins = np.zeros(1440, dtype=np.float32)
        bins[540:600] = 15.0  # 明显峰值

        curve = fit_activity_curve(self._SESSION, bins, data_quality=0.8)
        assert curve is not None
        peak_median = float(curve.predicted_median[540:600].mean())
        quiet_median = float(curve.predicted_median[0:60].mean())
        assert peak_median > quiet_median

    def test_r_squared_in_valid_range(self):
        """R² 对有结构数据应为有限值。"""
        bins = np.zeros(1440, dtype=np.float32)
        bins[600:660] = 10.0

        curve = fit_activity_curve(self._SESSION, bins, data_quality=0.8)
        assert curve is not None
        assert isinstance(curve.r_squared, float)
        assert not np.isnan(curve.r_squared)

    def test_quantile_width(self):
        """更宽的分位数应产出更宽的预测区间。"""
        bins = np.zeros(1440, dtype=np.float32)
        bins[600:660] = 10.0

        narrow = fit_activity_curve(
            self._SESSION,
            bins,
            data_quality=0.8,
            quantile_lower=0.1,
            quantile_upper=0.9,
        )
        wide = fit_activity_curve(
            self._SESSION,
            bins,
            data_quality=0.8,
            quantile_lower=0.01,
            quantile_upper=0.99,
        )
        assert narrow is not None and wide is not None

        narrow_width = float((narrow.predicted_upper - narrow.predicted_lower).mean())
        wide_width = float((wide.predicted_upper - wide.predicted_lower).mean())
        assert wide_width >= narrow_width
