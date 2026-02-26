"""detect.py 突兀点检测单元测试"""

import numpy as np
import pytest

from nonebot_plugin_wtfllm.proactive.inertia._types import (
    ActivityCurve,
    AnomalyPoint,
    AnomalyType,
    SessionKey,
)
from nonebot_plugin_wtfllm.proactive.inertia.detect import (
    detect_anomalies,
    _ts_to_minute_of_day,
    _format_time,
)


_SESSION = SessionKey(group_id="g1", sender="u1")
_UTC_OFFSET = 8 * 3600  # UTC+8


def _make_curve(
    peak_range: tuple[int, int] = (540, 660),
    peak_value: float = 10.0,
) -> ActivityCurve:
    """创建一个在 peak_range 有明显峰值的测试曲线。"""
    median = np.zeros(1440, dtype=np.float32)
    lower = np.zeros(1440, dtype=np.float32)
    upper = np.full(1440, 0.3, dtype=np.float32)  # 安静时段上界低

    start, end = peak_range
    median[start:end] = peak_value
    lower[start:end] = peak_value * 0.5
    upper[start:end] = peak_value * 1.5

    return ActivityCurve(
        session=_SESSION,
        predicted_median=median,
        predicted_lower=lower,
        predicted_upper=upper,
        r_squared=0.85,
        data_quality=0.8,
    )


def _make_ts(minute_of_day: int) -> int:
    """构造一个落在指定 minute_of_day 的 UTC 时间戳（考虑 UTC+8 偏移）。"""
    # 以 86400 为一天基准，减去 utc_offset 使得 local minute 落在 minute_of_day
    base = 1740000000  # 任意基准时间戳
    local_midnight = (base + _UTC_OFFSET) - ((base + _UTC_OFFSET) % 86400)
    return local_midnight + minute_of_day * 60 - _UTC_OFFSET


class TestTsToMinuteOfDay:

    def test_midnight(self):
        ts = _make_ts(0)
        assert _ts_to_minute_of_day(ts, _UTC_OFFSET) == 0

    def test_noon(self):
        ts = _make_ts(720)
        assert _ts_to_minute_of_day(ts, _UTC_OFFSET) == 720

    def test_end_of_day(self):
        ts = _make_ts(1439)
        assert _ts_to_minute_of_day(ts, _UTC_OFFSET) == 1439


class TestFormatTime:

    def test_midnight(self):
        assert _format_time(0) == "00:00"

    def test_noon(self):
        assert _format_time(720) == "12:00"

    def test_arbitrary(self):
        assert _format_time(635) == "10:35"


class TestDetectPositiveAnomaly:
    """正向异常：用户在安静时段出现。"""

    def test_activity_in_quiet_zone(self):
        curve = _make_curve(peak_range=(540, 660))
        # 在凌晨 2:00（minute=120）出现活动 → 安静区
        ts = _make_ts(120)
        anomalies = detect_anomalies(
            curve, [ts], _UTC_OFFSET, activity_flags=[True]
        )
        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == AnomalyType.UNEXPECTED_ACTIVITY
        assert anomalies[0].deviation_score > 0

    def test_activity_in_peak_not_anomaly(self):
        curve = _make_curve(peak_range=(540, 660))
        # 在活跃区 10:00（minute=600）出现 → 不应报异常
        ts = _make_ts(600)
        anomalies = detect_anomalies(
            curve, [ts], _UTC_OFFSET, activity_flags=[True]
        )
        assert len(anomalies) == 0

    def test_reason_contains_time(self):
        curve = _make_curve(peak_range=(540, 660))
        ts = _make_ts(120)
        anomalies = detect_anomalies(
            curve, [ts], _UTC_OFFSET, activity_flags=[True]
        )
        assert "02:00" in anomalies[0].reason


class TestDetectNegativeAnomaly:
    """负向异常：用户在活跃时段缺席。"""

    def test_absence_in_peak_zone(self):
        curve = _make_curve(peak_range=(540, 660), peak_value=10.0)
        # 在活跃区 10:00（minute=600）无活动 → 缺席异常
        ts = _make_ts(600)
        anomalies = detect_anomalies(
            curve, [ts], _UTC_OFFSET, activity_flags=[False]
        )
        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == AnomalyType.UNEXPECTED_ABSENCE
        assert anomalies[0].deviation_score > 0

    def test_absence_in_quiet_not_anomaly(self):
        curve = _make_curve(peak_range=(540, 660))
        # 在安静区 2:00 无活动 → 正常
        ts = _make_ts(120)
        anomalies = detect_anomalies(
            curve, [ts], _UTC_OFFSET, activity_flags=[False]
        )
        assert len(anomalies) == 0


class TestDetectMixed:
    """同时检测正向和负向异常。"""

    def test_mixed_flags(self):
        curve = _make_curve(peak_range=(540, 660), peak_value=10.0)
        ts_quiet = _make_ts(120)   # 安静区有活动 → 正向异常
        ts_peak = _make_ts(600)    # 活跃区无活动 → 负向异常
        ts_normal = _make_ts(550)  # 活跃区有活动 → 正常

        anomalies = detect_anomalies(
            curve,
            [ts_quiet, ts_peak, ts_normal],
            _UTC_OFFSET,
            activity_flags=[True, False, True],
        )
        # 应检出 2 个异常（安静区出现 + 活跃区缺席），正常点不报
        assert len(anomalies) == 2
        types = {a.anomaly_type for a in anomalies}
        assert AnomalyType.UNEXPECTED_ACTIVITY in types
        assert AnomalyType.UNEXPECTED_ABSENCE in types

    def test_sorted_by_deviation(self):
        """结果按 deviation_score 降序排列。"""
        curve = _make_curve(peak_range=(540, 660), peak_value=10.0)
        ts1 = _make_ts(120)
        ts2 = _make_ts(600)
        anomalies = detect_anomalies(
            curve, [ts1, ts2], _UTC_OFFSET,
            activity_flags=[True, False],
        )
        if len(anomalies) >= 2:
            assert anomalies[0].deviation_score >= anomalies[1].deviation_score


class TestDetectEdgeCases:

    def test_empty_timestamps(self):
        curve = _make_curve()
        assert detect_anomalies(curve, [], _UTC_OFFSET) == []

    def test_none_flags_means_all_active(self):
        """activity_flags=None → 仅检测正向异常。"""
        curve = _make_curve(peak_range=(540, 660), peak_value=10.0)
        ts = _make_ts(600)
        # activity_flags 为 None，即全部视为有活动
        # minute=600 在活跃区上界高 → 不是正向异常
        anomalies = detect_anomalies(curve, [ts], _UTC_OFFSET)
        assert len(anomalies) == 0

    def test_length_mismatch_raises(self):
        curve = _make_curve()
        with pytest.raises(ValueError, match="长度不一致"):
            detect_anomalies(
                curve, [1000000], _UTC_OFFSET, activity_flags=[True, False]
            )

    def test_reason_is_nonempty(self):
        curve = _make_curve()
        ts = _make_ts(120)
        anomalies = detect_anomalies(
            curve, [ts], _UTC_OFFSET, activity_flags=[True]
        )
        if anomalies:
            assert len(anomalies[0].reason) > 0

    def test_expected_range_tuple(self):
        curve = _make_curve()
        ts = _make_ts(120)
        anomalies = detect_anomalies(
            curve, [ts], _UTC_OFFSET, activity_flags=[True]
        )
        if anomalies:
            lo, hi = anomalies[0].expected_range
            assert lo <= hi


class TestFalseNegativeBias:
    """验证系统偏向假阴性（保守检测）。"""

    def test_borderline_not_flagged(self):
        """上界刚好等于阈值 → 不报正向异常。"""
        median = np.zeros(1440, dtype=np.float32)
        lower = np.zeros(1440, dtype=np.float32)
        # 上界恰好 = 0.5（默认阈值）
        upper = np.full(1440, 0.5, dtype=np.float32)

        curve = ActivityCurve(
            session=_SESSION,
            predicted_median=median,
            predicted_lower=lower,
            predicted_upper=upper,
            r_squared=0.8,
            data_quality=0.7,
        )
        ts = _make_ts(120)
        anomalies = detect_anomalies(
            curve, [ts], _UTC_OFFSET, activity_flags=[True]
        )
        assert len(anomalies) == 0  # 边界值不应报

    def test_borderline_absence_not_flagged(self):
        """下界刚好低于活跃阈值 → 不报负向异常。"""
        median = np.full(1440, 1.0, dtype=np.float32)
        lower = np.full(1440, 1.4, dtype=np.float32)  # < 1.5 默认阈值
        upper = np.full(1440, 3.0, dtype=np.float32)

        curve = ActivityCurve(
            session=_SESSION,
            predicted_median=median,
            predicted_lower=lower,
            predicted_upper=upper,
            r_squared=0.8,
            data_quality=0.7,
        )
        ts = _make_ts(600)
        anomalies = detect_anomalies(
            curve, [ts], _UTC_OFFSET, activity_flags=[False]
        )
        assert len(anomalies) == 0
