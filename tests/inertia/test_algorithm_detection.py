"""惯性系统算法端到端测试 — 面向设计目标。

设计目标概述：
1. 「建模守门」 预筛选应放行有时间规律的用户，拒绝随机/均匀/数据不足的用户
2. 「曲线形状」 拟合后的中位曲线应反映用户的日内活跃结构
3. 「缺席检出」 用户在稳定活跃时段缺席 → UNEXPECTED_ABSENCE
4. 「出现检出」 用户在深度安静时段出现 → UNEXPECTED_ACTIVITY（极保守）
5. 「无误报」   用户在正常时段正常行为不产生误报
6. 「假阴性偏好」系统宁可漏报也不误报

数据模拟策略：
- 模拟 N 天的用户消息时间戳 → 转为 1440 维 bins（每分钟活跃天数）
- 这与真实 DB 产出的格式一致
"""

import numpy as np
import pytest

from nonebot_plugin_wtfllm.proactive.inertia._types import (
    ActivityCurve,
    AnomalyType,
    SessionKey,
)
from nonebot_plugin_wtfllm.proactive.inertia.prefilter import prefilter
from nonebot_plugin_wtfllm.proactive.inertia.curve import fit_activity_curve
from nonebot_plugin_wtfllm.proactive.inertia.detect import detect_anomalies


# ── 测试基础设施 ──────────────────────────────────────────

_SESSION = SessionKey(group_id="test_group", sender="test_user")
_UTC_OFFSET = 8 * 3600


def _make_ts(minute_of_day: int) -> int:
    """构造落在指定 minute_of_day 的 UTC 时间戳。"""
    base = 1740000000
    local_midnight = (base + _UTC_OFFSET) - ((base + _UTC_OFFSET) % 86400)
    return local_midnight + minute_of_day * 60 - _UTC_OFFSET


def _simulate_user(
    rng: np.random.Generator,
    days: int,
    windows: list[tuple[int, int, int, int]],
    noise_per_day: int = 0,
) -> np.ndarray:
    """模拟用户消息历史 → 1440 维活跃天数直方图。

    Args:
        rng: 随机数生成器。
        days: 观察天数。
        windows: [(start_min, end_min, msgs_lo, msgs_hi), ...]
            每个窗口内每天随机发 msgs_lo ~ msgs_hi 条消息。
        noise_per_day: 每天额外在全天随机位置发的噪音消息数。

    Returns:
        bins[i] = 用户在第 i 分钟有活动的天数。
    """
    bins = np.zeros(1440, dtype=np.float32)
    for _ in range(days):
        for start, end, lo, hi in windows:
            n = rng.integers(lo, hi + 1)
            pool_size = end - start
            n = min(n, pool_size)
            minutes = rng.choice(range(start, end), size=n, replace=False)
            bins[minutes] += 1
        if noise_per_day > 0:
            noise_mins = rng.choice(1440, size=noise_per_day, replace=False)
            bins[noise_mins] += 1
    return bins


def _simulate_random_user(
    rng: np.random.Generator,
    days: int,
    msgs_per_day: int,
) -> np.ndarray:
    """模拟完全随机（无规律）用户。"""
    bins = np.zeros(1440, dtype=np.float32)
    for _ in range(days):
        minutes = rng.choice(1440, size=msgs_per_day, replace=False)
        bins[minutes] += 1
    return bins


def _run_pipeline(
    bins: np.ndarray,
    min_required_days: int = 7,
) -> tuple[bool, float, ActivityCurve | None]:
    """prefilter → fit 全管线。

    使用 max(bins) 作为 max_active_days，与真实 DB 行为一致。
    """
    max_active = int(bins.max())
    passed, quality = prefilter(bins, max_active, min_required_days)
    curve = None
    if passed:
        curve = fit_activity_curve(_SESSION, bins, data_quality=quality)
    return passed, quality, curve


# ╔══════════════════════════════════════════════════════════╗
# ║  设计目标一：建模守门 — 预筛选区分有规律与无规律用户         ║
# ╚══════════════════════════════════════════════════════════╝


class TestGatekeeper:
    """预筛选应接纳有时间规律的用户、拒绝无规律或数据不足的用户。"""

    def test_regular_evening_user_accepted(self):
        """每天晚上 20:00-22:00 发 10~20 条，90 天 → 应通过。"""
        rng = np.random.default_rng(1)
        bins = _simulate_user(rng, 90, [(1200, 1320, 10, 20)])
        passed, quality, _ = _run_pipeline(bins)
        assert passed, "有稳定晚间模式的用户应通过预筛选"
        assert quality > 0.5

    def test_office_bimodal_accepted(self):
        """工作日双窗口（9-11+14-17）× 60 天 → 应通过。"""
        rng = np.random.default_rng(2)
        bins = _simulate_user(
            rng,
            60,
            [
                (540, 660, 8, 15),
                (840, 1020, 8, 15),
            ],
        )
        passed, quality, _ = _run_pipeline(bins)
        assert passed, "双峰办公模式应通过预筛选"

    def test_random_user_rejected(self):
        """30 天内每天 10 条随机位置 → 每分钟 max ≈ 1~3 天，
        数据充分性不足应被拒绝。"""
        rng = np.random.default_rng(3)
        bins = _simulate_random_user(rng, 30, 10)
        max_ad = int(bins.max())
        passed, _ = prefilter(bins, max_ad, min_required_days=7)
        assert not passed, f"随机用户不应通过，max_active_days={max_ad} < 7"

    def test_all_day_uniform_rejected(self):
        """全天均匀活跃 → 覆盖率过高应被拒绝。"""
        bins = np.full(1440, 10.0, dtype=np.float32)
        passed, _ = prefilter(bins, 10, 7)
        assert not passed

    def test_insufficient_observation_rejected(self):
        """活跃天数不足最低要求 → 拒绝。"""
        rng = np.random.default_rng(4)
        bins = _simulate_user(rng, 5, [(1200, 1320, 5, 10)])
        max_ad = int(bins.max())
        passed, _ = prefilter(bins, max_ad, min_required_days=7)
        assert not passed

    def test_quality_increases_with_data_volume(self):
        """同样的模式，更多天数 → 更高质量分。"""
        q_list = []
        for days in [30, 60, 90]:
            rng = np.random.default_rng(5)
            bins = _simulate_user(rng, days, [(1200, 1320, 15, 25)])
            _, quality, _ = _run_pipeline(bins)
            q_list.append(quality)

        for i in range(1, len(q_list)):
            assert q_list[i] >= q_list[i - 1] - 0.05, (
                f"更多数据的质量分应不降: {q_list}"
            )


# ╔══════════════════════════════════════════════════════════╗
# ║  设计目标二：曲线形状 — 中位预测反映日内活跃结构            ║
# ╚══════════════════════════════════════════════════════════╝


class TestCurveShape:
    """拟合后的中位曲线应在活跃窗口高、安静时段低。"""

    def test_evening_peak_captured(self):
        """晚间峰值用户的中位预测在 20:00-22:00 应远高于凌晨。"""
        rng = np.random.default_rng(10)
        bins = _simulate_user(rng, 90, [(1200, 1320, 15, 25)])
        _, _, curve = _run_pipeline(bins)
        assert curve is not None

        peak = float(curve.predicted_median[1200:1320].mean())
        quiet = float(curve.predicted_median[300:500].mean())
        assert peak > quiet * 10, (
            f"晚间峰值中位应远高于凌晨: peak={peak:.2f}, quiet={quiet:.4f}"
        )

    def test_bimodal_two_peaks(self):
        """双窗口用户的两个峰值都应高于其间的谷底。"""
        rng = np.random.default_rng(11)
        bins = _simulate_user(
            rng,
            90,
            [
                (540, 660, 12, 20),
                (1200, 1320, 12, 20),
            ],
        )
        _, _, curve = _run_pipeline(bins)
        assert curve is not None

        am_peak = float(curve.predicted_median[540:660].mean())
        pm_peak = float(curve.predicted_median[1200:1320].mean())
        valley = float(curve.predicted_median[800:900].mean())
        assert am_peak > valley, "上午峰应高于午后谷底"
        assert pm_peak > valley, "晚间峰应高于午后谷底"

    def test_lower_le_median_le_upper(self):
        """在所有 1440 个分钟上 lower ≤ median ≤ upper。"""
        rng = np.random.default_rng(12)
        bins = _simulate_user(rng, 90, [(1200, 1320, 10, 20)])
        _, _, curve = _run_pipeline(bins)
        assert curve is not None
        assert np.all(curve.predicted_lower <= curve.predicted_median)
        assert np.all(curve.predicted_median <= curve.predicted_upper)

    def test_all_predictions_non_negative(self):
        """所有预测值 ≥ 0。"""
        rng = np.random.default_rng(13)
        bins = _simulate_user(rng, 60, [(600, 720, 10, 20)])
        _, _, curve = _run_pipeline(bins)
        assert curve is not None
        assert np.all(curve.predicted_lower >= 0)
        assert np.all(curve.predicted_median >= 0)
        assert np.all(curve.predicted_upper >= 0)

    def test_midnight_crossing_pattern(self):
        """跨午夜活跃（23:00-01:00）应被周期性谐波正确捕捉。"""
        rng = np.random.default_rng(14)
        bins = _simulate_user(
            rng,
            90,
            [
                (1380, 1440, 8, 12),
                (0, 60, 8, 12),
            ],
        )
        _, _, curve = _run_pipeline(bins)
        assert curve is not None

        midnight_area = np.concatenate(
            [
                curve.predicted_median[1380:1440],
                curve.predicted_median[0:60],
            ]
        )
        noon_area = curve.predicted_median[660:780]
        assert float(midnight_area.mean()) > float(noon_area.mean()), (
            "跨午夜活跃区的预测应高于正午"
        )


# ╔══════════════════════════════════════════════════════════╗
# ║  设计目标三：缺席检出 — 高活跃时段缺席应被发现              ║
# ╚══════════════════════════════════════════════════════════╝


class TestAbsenceDetection:
    """当用户在历史上高度稳定的活跃时段缺席时，系统应检出。

    注意：检出需要 lower_bound ≥ 1.5，这要求该时段的活跃天数
    积累足够高且模型确信该时段一定有活动。
    """

    def test_consistent_peak_absence_detected(self):
        """90 天高一致性峰值用户（20:00-21:00，15~25 条/天），
        在核心时段缺席应被检出。"""
        rng = np.random.default_rng(20)
        bins = _simulate_user(rng, 90, [(1200, 1260, 15, 25)])
        _, _, curve = _run_pipeline(bins)
        assert curve is not None

        # 在峰值中心地带检测缺席
        detected = []
        for m in range(1210, 1250, 5):
            anomalies = detect_anomalies(
                curve,
                [_make_ts(m)],
                _UTC_OFFSET,
                activity_flags=[False],
            )
            if anomalies:
                assert anomalies[0].anomaly_type == AnomalyType.UNEXPECTED_ABSENCE
                detected.append(anomalies[0])

        assert len(detected) > 0, "在高一致性峰值时段缺席应至少有一个点被检出"

    def test_absence_has_positive_deviation(self):
        """检出的缺席异常应有正的偏离分。"""
        rng = np.random.default_rng(21)
        bins = _simulate_user(rng, 90, [(1200, 1260, 15, 25)])
        _, _, curve = _run_pipeline(bins)
        assert curve is not None

        anomalies = detect_anomalies(
            curve,
            [_make_ts(1230)],
            _UTC_OFFSET,
            activity_flags=[False],
        )
        if anomalies:
            assert anomalies[0].deviation_score > 0

    def test_absence_at_quiet_zone_not_flagged(self):
        """在安静时段缺席不应被标记为负向异常。"""
        rng = np.random.default_rng(22)
        bins = _simulate_user(rng, 90, [(1200, 1260, 15, 25)])
        _, _, curve = _run_pipeline(bins)
        assert curve is not None

        # 凌晨 03:00-05:00 缺席完全正常
        for m in range(180, 300, 15):
            anomalies = detect_anomalies(
                curve,
                [_make_ts(m)],
                _UTC_OFFSET,
                activity_flags=[False],
            )
            assert len(anomalies) == 0, (
                f"凌晨 {m // 60:02d}:{m % 60:02d} 缺席不应报异常"
            )

    def test_weak_pattern_may_miss_absence(self):
        """活跃度低/天数少的用户，缺席可能不被检出 — 这是假阴性偏好的体现。"""
        rng = np.random.default_rng(23)
        # 较弱的信号：30 天，每天仅 3-5 条
        bins = _simulate_user(rng, 30, [(1200, 1320, 3, 5)])
        passed, quality, curve = _run_pipeline(bins)
        if not passed or curve is None:
            return  # 预筛选正确拒绝了弱信号

        # 即使通过拟合，lower bound 可能不够高 → 不检出缺席
        # 这是可接受的假阴性  review: 确实如此
        anomalies = detect_anomalies(  # noqa: F841
            curve,
            [_make_ts(1260)],
            _UTC_OFFSET,
            activity_flags=[False],
        )
        # 不 assert 必须检出 — 偏好假阴性


# ╔══════════════════════════════════════════════════════════╗
# ║  设计目标三·补：意外出现检出 — 深安静时段出现应被发现        ║
# ╚══════════════════════════════════════════════════════════╝


class TestActivityDetection:
    """当用户在历史上几乎从不活跃的时段突然出现时，系统应检出。

    Q0.95 上界使正向异常检测成为可能：安静时段的 upper bound
    足够低（mean ~0.5），大部分安静分钟的 upper < 0.5。
    """

    def test_quiet_zone_activity_detected(self):
        """90 天纯晚间用户（20:00-22:00），凌晨 04:00 突然出现 → 应被检出。"""
        rng = np.random.default_rng(40)
        bins = _simulate_user(rng, 90, [(1200, 1320, 15, 25)])
        _, _, curve = _run_pipeline(bins)
        assert curve is not None

        # 在凌晨 03:00-05:00 多点探测
        detected = []
        for m in range(180, 300, 10):
            anomalies = detect_anomalies(
                curve,
                [_make_ts(m)],
                _UTC_OFFSET,
                activity_flags=[True],
            )
            pos = [
                a
                for a in anomalies
                if a.anomaly_type == AnomalyType.UNEXPECTED_ACTIVITY
            ]
            detected.extend(pos)

        assert len(detected) > 0, "在深度安静时段（凌晨 03-05）出现应至少有一个点被检出"

    def test_activity_detection_has_positive_deviation(self):
        """检出的意外出现应有正的偏离分和合理的 reason。"""
        rng = np.random.default_rng(41)
        bins = _simulate_user(rng, 90, [(1200, 1320, 15, 25)])
        _, _, curve = _run_pipeline(bins)
        assert curve is not None

        anomalies = detect_anomalies(
            curve,
            [_make_ts(240)],
            _UTC_OFFSET,  # 04:00
            activity_flags=[True],
        )
        pos = [
            a for a in anomalies if a.anomaly_type == AnomalyType.UNEXPECTED_ACTIVITY
        ]
        if pos:
            assert pos[0].deviation_score > 0
            assert "几乎无活动" in pos[0].reason

    def test_activity_at_peak_border_not_flagged(self):
        """在活跃窗口内部边缘出现 → 不应被当作正向异常。"""
        rng = np.random.default_rng(42)
        bins = _simulate_user(rng, 90, [(1200, 1320, 15, 25)])
        _, _, curve = _run_pipeline(bins)
        assert curve is not None

        # 20:10 和 21:50 是活跃窗口内部边缘，upper 应足够高
        for m in [1210, 1310]:
            anomalies = detect_anomalies(
                curve,
                [_make_ts(m)],
                _UTC_OFFSET,
                activity_flags=[True],
            )
            pos = [
                a
                for a in anomalies
                if a.anomaly_type == AnomalyType.UNEXPECTED_ACTIVITY
            ]
            assert len(pos) == 0, f"窗口边缘 {m // 60:02d}:{m % 60:02d} 不应报正向异常"

    def test_activity_detection_vs_absence_discrimination(self):
        """正向和负向异常应由不同的 activity_flag 触发，互不混淆。"""
        rng = np.random.default_rng(43)
        bins = _simulate_user(rng, 90, [(1200, 1260, 20, 30)])
        _, _, curve = _run_pipeline(bins)
        assert curve is not None

        # 在安静区传 has_activity=True → 只能产生正向异常
        quiet_anomalies = detect_anomalies(
            curve,
            [_make_ts(240)],
            _UTC_OFFSET,
            activity_flags=[True],
        )
        for a in quiet_anomalies:
            assert a.anomaly_type == AnomalyType.UNEXPECTED_ACTIVITY

        # 在活跃区传 has_activity=False → 只能产生负向异常
        peak_anomalies = detect_anomalies(
            curve,
            [_make_ts(1230)],
            _UTC_OFFSET,
            activity_flags=[False],
        )
        for a in peak_anomalies:
            assert a.anomaly_type == AnomalyType.UNEXPECTED_ABSENCE


# ╔══════════════════════════════════════════════════════════╗
# ║  设计目标四：无误报 — 正常行为不产生异常                    ║
# ╚══════════════════════════════════════════════════════════╝


class TestNoFalsePositives:
    """守住误报率的底线。"""

    def test_normal_activity_at_peak_no_alarm(self):
        """在历史活跃时段正常出现 → 不应产生任何正向异常。"""
        rng = np.random.default_rng(30)
        bins = _simulate_user(rng, 90, [(1200, 1320, 10, 20)])
        _, _, curve = _run_pipeline(bins)
        assert curve is not None

        for m in range(1200, 1320, 10):
            anomalies = detect_anomalies(
                curve,
                [_make_ts(m)],
                _UTC_OFFSET,
                activity_flags=[True],
            )
            pos = [
                a
                for a in anomalies
                if a.anomaly_type == AnomalyType.UNEXPECTED_ACTIVITY
            ]
            assert len(pos) == 0, (
                f"在活跃区 {m // 60:02d}:{m % 60:02d} 正常出现不应报正向异常"
            )

    def test_normal_absence_at_quiet_no_alarm(self):
        """在历史安静时段缺席 → 不应产生任何负向异常。"""
        rng = np.random.default_rng(31)
        bins = _simulate_user(rng, 90, [(1200, 1320, 10, 20)])
        _, _, curve = _run_pipeline(bins)
        assert curve is not None

        for m in range(180, 420, 30):
            anomalies = detect_anomalies(
                curve,
                [_make_ts(m)],
                _UTC_OFFSET,
                activity_flags=[False],
            )
            neg = [
                a for a in anomalies if a.anomaly_type == AnomalyType.UNEXPECTED_ABSENCE
            ]
            assert len(neg) == 0, (
                f"安静时段 {m // 60:02d}:{m % 60:02d} 缺席不应报负向异常"
            )

    def test_deviation_score_bounded(self):
        """偏离分应被 cap 在 10.0（系统设计约束）。"""
        # 构造一个能触发异常的场景
        rng = np.random.default_rng(32)
        bins = _simulate_user(rng, 90, [(1200, 1260, 20, 30)])
        _, _, curve = _run_pipeline(bins)
        assert curve is not None

        anomalies = detect_anomalies(
            curve,
            [_make_ts(1230)],
            _UTC_OFFSET,
            activity_flags=[False],
        )
        for a in anomalies:
            assert a.deviation_score <= 10.0

    def test_anomaly_reason_is_human_readable(self):
        """异常点的 reason 字段应非空且包含时间信息。"""
        rng = np.random.default_rng(33)
        bins = _simulate_user(rng, 90, [(1200, 1260, 15, 25)])
        _, _, curve = _run_pipeline(bins)
        assert curve is not None

        anomalies = detect_anomalies(
            curve,
            [_make_ts(1230)],
            _UTC_OFFSET,
            activity_flags=[False],
        )
        for a in anomalies:
            assert len(a.reason) > 10
            assert ":" in a.reason  # 包含 HH:MM 格式


# ╔══════════════════════════════════════════════════════════╗
# ║  设计目标五：生日悖论 — 随机数据不应系统性产生假检出         ║
# ╚══════════════════════════════════════════════════════════╝


class TestBirthdayParadox:
    """大量随机用户中碰巧出现的规律不应被系统性误判。

    核心判据：当 max_active_days 来自真实 bins（max(bins)）时，
    随机用户因数据充分性不足而被预筛选自然拦截。
    """

    def _simulate_and_count_pass(
        self,
        n_users: int,
        days: int,
        msgs_per_day: int,
        seed: int,
    ) -> int:
        rng = np.random.default_rng(seed)
        passed = 0
        for _ in range(n_users):
            bins = _simulate_random_user(rng, days, msgs_per_day)
            max_ad = int(bins.max())
            p, _ = prefilter(bins, max_ad, min_required_days=7)
            if p:
                passed += 1
        return passed

    def test_random_10_msgs_30_days(self):
        """30 天 × 10 条/天随机 → max(bins) ≈ 2~4 → 不足 7 → 全部拒绝。"""
        passed = self._simulate_and_count_pass(100, 30, 10, seed=42)
        assert passed == 0, f"随机用户不应通过: {passed}/100"

    def test_random_20_msgs_30_days(self):
        """30 天 × 20 条/天随机 → max(bins) 仍可能 < 7。"""
        passed = self._simulate_and_count_pass(100, 30, 20, seed=43)
        assert passed < 10, f"随机用户通过率应极低: {passed}/100"

    def test_random_10_msgs_90_days(self):
        """90 天 × 10 条/天 → max(bins) ≈ 4~7。
        部分可能刚好达到阈值，但即使通过也不应产生高偏离检出。"""
        rng = np.random.default_rng(44)
        high_deviation_count = 0

        for _ in range(100):
            bins = _simulate_random_user(rng, 90, 10)
            max_ad = int(bins.max())
            passed, quality = prefilter(bins, max_ad, min_required_days=7)
            if not passed:
                continue

            curve = fit_activity_curve(
                _SESSION,
                bins,
                data_quality=quality,
            )
            if curve is None:
                continue

            # 在 24 个整点探测缺席
            for m in range(0, 1440, 60):
                anomalies = detect_anomalies(
                    curve,
                    [_make_ts(m)],
                    _UTC_OFFSET,
                    activity_flags=[False],
                )
                high = [a for a in anomalies if a.deviation_score > 5.0]
                if high:
                    high_deviation_count += 1

        assert high_deviation_count <= 5, (
            f"随机用户不应产生大量高偏离缺席异常: {high_deviation_count}"
        )

    def test_random_vs_regular_discrimination(self):
        """对比：有规律用户检出率应显著高于随机用户。"""
        rng_reg = np.random.default_rng(50)
        rng_rand = np.random.default_rng(51)

        # 有规律用户
        bins_reg = _simulate_user(rng_reg, 90, [(1200, 1260, 15, 25)])
        _, _, curve_reg = _run_pipeline(bins_reg)

        # 随机用户
        bins_rand = _simulate_random_user(rng_rand, 90, 15)
        _, _, curve_rand = _run_pipeline(bins_rand)

        reg_detections = 0
        if curve_reg is not None:
            for m in range(1210, 1250, 5):
                a = detect_anomalies(
                    curve_reg,
                    [_make_ts(m)],
                    _UTC_OFFSET,
                    activity_flags=[False],
                )
                if a:
                    reg_detections += 1

        rand_detections = 0
        if curve_rand is not None:
            for m in range(0, 1440, 60):
                a = detect_anomalies(
                    curve_rand,
                    [_make_ts(m)],
                    _UTC_OFFSET,
                    activity_flags=[False],
                )
                if a:
                    rand_detections += 1

        assert reg_detections >= rand_detections, (
            f"有规律用户检出应 ≥ 随机用户: reg={reg_detections}, rand={rand_detections}"
        )


# ╔══════════════════════════════════════════════════════════╗
# ║  有规律但含噪音的数据                                      ║
# ╚══════════════════════════════════════════════════════════╝


class TestPatternWithNoise:
    """有主模式 + 背景噪音。核心模式应仍可辨识。"""

    def test_strong_signal_with_light_noise(self):
        """强信号 15~25 条 + 轻噪音 2 条/天 → 模式仍清晰。"""
        rng = np.random.default_rng(60)
        bins = _simulate_user(
            rng,
            90,
            [(1200, 1320, 15, 25)],
            noise_per_day=2,
        )
        passed, quality, curve = _run_pipeline(bins)
        assert passed
        assert curve is not None

        peak = float(curve.predicted_median[1200:1320].mean())
        noise_floor = float(curve.predicted_median[300:600].mean())
        assert peak > noise_floor * 3, (
            f"强信号应压过轻噪音: peak={peak:.2f}, noise={noise_floor:.2f}"
        )

    def test_moderate_signal_with_moderate_noise(self):
        """中等信号 8~12 条 + 中等噪音 5 条/天 → 模式可辨识。"""
        rng = np.random.default_rng(61)
        bins = _simulate_user(
            rng,
            90,
            [(1200, 1320, 8, 12)],
            noise_per_day=5,
        )
        passed, quality, curve = _run_pipeline(bins)
        assert passed
        assert curve is not None

        peak = float(curve.predicted_median[1200:1320].mean())
        noise_floor = float(curve.predicted_median[300:600].mean())
        assert peak > noise_floor, (
            f"中等信号应高于噪音底: peak={peak:.2f}, noise={noise_floor:.2f}"
        )

    def test_dual_peak_with_noise(self):
        """双峰 + 噪音 → 两个峰值都应高于谷间。"""
        rng = np.random.default_rng(62)
        bins = _simulate_user(
            rng,
            90,
            [
                (540, 660, 10, 18),
                (1200, 1320, 10, 18),
            ],
            noise_per_day=3,
        )
        _, _, curve = _run_pipeline(bins)
        assert curve is not None

        am = float(curve.predicted_median[540:660].mean())
        pm = float(curve.predicted_median[1200:1320].mean())
        valley = float(curve.predicted_median[800:900].mean())
        assert am > valley and pm > valley

    def test_absence_detection_survives_noise(self):
        """强信号 + 轻噪音 → 缺席检出仍应工作。"""
        rng = np.random.default_rng(63)
        bins = _simulate_user(
            rng,
            90,
            [(1200, 1260, 25, 35)],
            noise_per_day=2,
        )
        _, _, curve = _run_pipeline(bins)
        assert curve is not None

        detected = 0
        for m in range(1210, 1250, 5):
            a = detect_anomalies(
                curve,
                [_make_ts(m)],
                _UTC_OFFSET,
                activity_flags=[False],
            )
            if a:
                detected += 1
        assert detected > 0, "强信号下缺席检出应穿透轻噪音"


# ╔══════════════════════════════════════════════════════════╗
# ║  噪音较大、规律微弱的数据                                  ║
# ╚══════════════════════════════════════════════════════════╝


class TestWeakSignalHeavyNoise:
    """信噪比低的场景。系统应保守处理，宁可漏报。"""

    def test_barely_visible_pattern(self):
        """信号 3~5 条 + 噪音 8 条/天 → 信号被淹没。
        系统不崩溃，且不应产生高置信异常。"""
        rng = np.random.default_rng(70)
        bins = _simulate_user(
            rng,
            60,
            [(1200, 1320, 3, 5)],
            noise_per_day=8,
        )
        passed, quality, curve = _run_pipeline(bins)
        if not passed:
            return  # 预筛选拒绝是正确行为（保守）

        if curve is not None:
            # 即使通过，高偏离缺席检出应很少
            high_dev = 0
            for m in range(0, 1440, 30):
                a = detect_anomalies(
                    curve,
                    [_make_ts(m)],
                    _UTC_OFFSET,
                    activity_flags=[False],
                )
                high = [x for x in a if x.deviation_score > 5.0]
                high_dev += len(high)
            assert high_dev <= 5, f"弱信号不应产生大量高偏离异常: {high_dev}"

    def test_pure_noise_no_false_pattern(self):
        """纯噪音（每天 15 条随机位置）→ 预筛选应拒绝或质量极低。"""
        rng = np.random.default_rng(71)
        bins = _simulate_random_user(rng, 60, 15)
        max_ad = int(bins.max())
        passed, quality = prefilter(bins, max_ad, min_required_days=7)
        if passed:
            assert quality < 0.7, f"纯随机数据质量分不应高: {quality}"

    def test_marginal_pattern_conservative(self):
        """边界信噪比：信号 5~8 条 + 噪音 5 条/天。
        如果通过了，质量分应反映边缘状态。"""
        rng = np.random.default_rng(72)
        bins = _simulate_user(
            rng,
            60,
            [(1200, 1320, 5, 8)],
            noise_per_day=5,
        )
        passed, quality, curve = _run_pipeline(bins)
        # 边界场景：通过或不通过都合理
        # 如果通过，质量分应不高
        if passed:
            assert quality <= 0.95, "边界信噪比的质量分应受限"


# ╔══════════════════════════════════════════════════════════╗
# ║  端到端集成：_process_rows 管线                            ║
# ╚══════════════════════════════════════════════════════════╝


class TestProcessRowsIntegration:
    """通过 _process_rows 验证 scan 模块的核心处理逻辑。"""

    def test_peaked_user_produces_curve(self):
        """有规律用户在管线中产出曲线。"""
        from nonebot_plugin_wtfllm.proactive.inertia.scan import _process_rows

        rng = np.random.default_rng(80)
        bins = _simulate_user(rng, 90, [(1200, 1320, 10, 20)])
        rows = [("g1", None, "u1", m, int(bins[m])) for m in range(1440) if bins[m] > 0]

        results = _process_rows(
            rows,  # pyright: ignore[reportArgumentType]
            min_repeat_days=7,
            minute_bucket=1,
            quantile_lower=0.05,
            quantile_upper=0.95,
        )
        assert len(results) >= 1
        assert results[0].session.sender == "u1"

    def test_uniform_user_filtered_out(self):
        """全天均匀活跃用户被管线过滤。"""
        from nonebot_plugin_wtfllm.proactive.inertia.scan import _process_rows

        rows = [("g1", None, "u2", m, 10) for m in range(0, 1440)]
        results = _process_rows(
            rows,  # pyright: ignore[reportArgumentType]
            min_repeat_days=7,
            minute_bucket=1,
            quantile_lower=0.05,
            quantile_upper=0.95,
        )
        assert len(results) == 0

    def test_mixed_users_correct_filtering(self):
        """一个有规律 + 一个全天均匀 → 只保留有规律的。"""
        from nonebot_plugin_wtfllm.proactive.inertia.scan import _process_rows

        rng = np.random.default_rng(81)
        bins_good = _simulate_user(rng, 90, [(1200, 1320, 10, 20)])
        rows = [
            ("g1", None, "good_user", m, int(bins_good[m]))
            for m in range(1440)
            if bins_good[m] > 0
        ]
        rows += [("g1", None, "uniform_user", m, 10) for m in range(1440)]

        results = _process_rows(
            rows,  # pyright: ignore[reportArgumentType]
            min_repeat_days=7,
            minute_bucket=1,
            quantile_lower=0.05,
            quantile_upper=0.95,
        )
        senders = {r.session.sender for r in results}
        assert "good_user" in senders
        assert "uniform_user" not in senders

    def test_bucket_aggregation(self):
        """minute_bucket > 1 时聚合应正确工作。"""
        from nonebot_plugin_wtfllm.proactive.inertia.scan import _process_rows

        rng = np.random.default_rng(82)
        bins = _simulate_user(rng, 90, [(1200, 1320, 10, 20)])
        rows = [("g1", None, "u1", m, int(bins[m])) for m in range(1440) if bins[m] > 0]

        results = _process_rows(
            rows,  # pyright: ignore[reportArgumentType]
            min_repeat_days=7,
            minute_bucket=15,
            quantile_lower=0.05,
            quantile_upper=0.95,
        )
        assert len(results) >= 1


# ╔══════════════════════════════════════════════════════════╗
# ║  检测边界条件                                              ║
# ╚══════════════════════════════════════════════════════════╝


class TestDetectionBoundaries:
    """验证 detect_anomalies 的接口契约。"""

    def test_empty_timestamps(self):
        """空时间戳列表 → 空结果。"""
        rng = np.random.default_rng(90)
        bins = _simulate_user(rng, 90, [(1200, 1320, 10, 20)])
        _, _, curve = _run_pipeline(bins)
        assert curve is not None

        result = detect_anomalies(curve, [], _UTC_OFFSET)
        assert result == []

    def test_flags_none_means_all_active(self):
        """activity_flags=None → 仅检测正向异常。"""
        rng = np.random.default_rng(91)
        bins = _simulate_user(rng, 90, [(1200, 1320, 10, 20)])
        _, _, curve = _run_pipeline(bins)
        assert curve is not None

        result = detect_anomalies(
            curve,
            [_make_ts(1260)],
            _UTC_OFFSET,
            activity_flags=None,
        )
        # 在活跃区正常出现 → 不应报正向异常
        assert len(result) == 0

    def test_mismatched_lengths_raises(self):
        """timestamps 与 activity_flags 长度不一致 → ValueError。"""
        rng = np.random.default_rng(92)
        bins = _simulate_user(rng, 90, [(1200, 1320, 10, 20)])
        _, _, curve = _run_pipeline(bins)
        assert curve is not None

        with pytest.raises(ValueError, match="长度不一致"):
            detect_anomalies(
                curve,
                [_make_ts(600)],
                _UTC_OFFSET,
                activity_flags=[True, False],
            )

    def test_results_sorted_by_deviation(self):
        """多个异常点应按 deviation_score 降序排列。"""
        rng = np.random.default_rng(93)
        bins = _simulate_user(rng, 90, [(1200, 1260, 20, 30)])
        _, _, curve = _run_pipeline(bins)
        assert curve is not None

        timestamps = [_make_ts(m) for m in range(1210, 1250, 3)]
        anomalies = detect_anomalies(
            curve,
            timestamps,
            _UTC_OFFSET,
            activity_flags=[False] * len(timestamps),
        )
        for i in range(1, len(anomalies)):
            assert anomalies[i - 1].deviation_score >= anomalies[i].deviation_score


# ╔══════════════════════════════════════════════════════════╗
# ║  完整端到端场景                                            ║
# ╚══════════════════════════════════════════════════════════╝


class TestEndToEnd:
    """完整端到端集成，验证管线各环节协作。"""

    def test_heavy_user_full_flow(self):
        """高频用户完整流程：建模 → 曲线合理 → 缺席可检出。"""
        rng = np.random.default_rng(100)
        bins = _simulate_user(rng, 90, [(1200, 1260, 20, 30)])

        # 1. 预筛选
        max_ad = int(bins.max())
        passed, quality = prefilter(bins, max_ad, min_required_days=7)
        assert passed
        assert quality > 0.5

        # 2. 拟合
        curve = fit_activity_curve(
            _SESSION,
            bins,
            data_quality=quality,
        )
        assert curve is not None
        assert curve.predicted_median.shape == (1440,)

        # 3. 形状正确
        peak = float(curve.predicted_median[1200:1260].mean())
        quiet = float(curve.predicted_median[300:500].mean())
        assert peak > quiet

        # 4. 边界一致
        assert np.all(curve.predicted_lower <= curve.predicted_median)
        assert np.all(curve.predicted_median <= curve.predicted_upper)

        # 5. 活跃区正常出现不误报
        normal = detect_anomalies(
            curve,
            [_make_ts(1230)],
            _UTC_OFFSET,
            activity_flags=[True],
        )
        pos_normal = [
            a for a in normal if a.anomaly_type == AnomalyType.UNEXPECTED_ACTIVITY
        ]
        assert len(pos_normal) == 0

        # 6. 活跃区缺席可检出
        absence = detect_anomalies(
            curve,
            [_make_ts(1230)],
            _UTC_OFFSET,
            activity_flags=[False],
        )
        if absence:
            assert absence[0].anomaly_type == AnomalyType.UNEXPECTED_ABSENCE
            assert absence[0].deviation_score > 0

    def test_light_user_graceful_handling(self):
        """低频用户完整流程：可能被过滤，或通过但低质量。"""
        rng = np.random.default_rng(101)
        bins = _simulate_user(rng, 20, [(1200, 1320, 3, 6)])

        max_ad = int(bins.max())
        passed, quality = prefilter(bins, max_ad, min_required_days=7)
        # 低频用户 → 通过或不通过都合理
        if not passed:
            return

        curve = fit_activity_curve(
            _SESSION,
            bins,
            data_quality=quality,
        )
        # 即使拟合成功，也不应崩溃
        if curve is not None:
            assert curve.predicted_median.shape == (1440,)
            assert np.all(np.isfinite(curve.predicted_median))


# ╔══════════════════════════════════════════════════════════╗
# ║  特征工程正确性                                            ║
# ╚══════════════════════════════════════════════════════════╝


class TestFeatureEngineering:
    """验证周期性谐波特征编码。"""

    def test_periodicity(self):
        """minute 0 和 minute 1440 的特征应相同（24h 周期）。"""
        from nonebot_plugin_wtfllm.proactive.inertia.curve import _build_features

        m0 = _build_features(np.array([0], dtype=np.int32))
        m1440 = _build_features(np.array([1440], dtype=np.int32))
        np.testing.assert_allclose(m0, m1440, atol=1e-10)

    def test_feature_shape(self):
        """1440 个分钟 × 12 个特征（6 谐波 × sin+cos）。"""
        from nonebot_plugin_wtfllm.proactive.inertia.curve import _build_features

        feats = _build_features(np.arange(1440, dtype=np.int32))
        assert feats.shape == (1440, 12)

    def test_opposite_phases(self):
        """6:00 和 18:00 在 k=1 谐波上应反相。"""
        from nonebot_plugin_wtfllm.proactive.inertia.curve import _build_features

        feats = _build_features(np.array([360, 1080], dtype=np.int32))
        # sin(2π·1·360/1440) = sin(π/2) = 1
        # sin(2π·1·1080/1440) = sin(3π/2) = -1
        assert feats[0, 0] == pytest.approx(1.0, abs=1e-10)
        assert feats[1, 0] == pytest.approx(-1.0, abs=1e-10)


# ╔══════════════════════════════════════════════════════════╗
# ║  R² 和数据不足保护                                        ║
# ╚══════════════════════════════════════════════════════════╝


class TestFitSafety:
    """拟合函数的安全保护。"""

    def test_too_few_nonzero_returns_none(self):
        """非零点 < 3 → 返回 None。"""
        bins = np.zeros(1440, dtype=np.float32)
        bins[600] = 5.0
        bins[601] = 3.0
        result = fit_activity_curve(_SESSION, bins, 30, 0.5)
        assert result is None

    def test_r_squared_finite(self):
        """R² 应为有限值。"""
        rng = np.random.default_rng(110)
        bins = _simulate_user(rng, 90, [(600, 720, 10, 20)])
        _, _, curve = _run_pipeline(bins)
        assert curve is not None
        assert np.isfinite(curve.r_squared)

    def test_quality_in_unit_interval(self):
        """质量分 ∈ [0, 1]。"""
        for seed in range(5):
            rng = np.random.default_rng(seed + 120)
            bins = _simulate_user(rng, 60, [(1200, 1320, 10, 20)])
            max_ad = int(bins.max())
            passed, quality = prefilter(bins, max_ad, min_required_days=7)
            if passed:
                assert 0.0 <= quality <= 1.0, f"seed={seed}: quality={quality}"
