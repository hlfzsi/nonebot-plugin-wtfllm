"""时间戳与活跃曲线比较，识别异常偏离。"""

import numpy as np

from ._types import ActivityCurve, AnomalyPoint, AnomalyType


_QUIET_UPPER_QUANTILE = 0.20
"""正向异常的自适应阈值分位数：
upper <= quantile(upper, q) 视为"安静时段"。"""

_ACTIVE_LOWER_QUANTILE = 0.85
"""负向异常的自适应阈值分位数：
lower >= quantile(lower, q) 视为"应活跃时段"。"""

_MIN_QUIET_THRESHOLD = 0.35
"""正向异常的最小安静阈值，避免阈值过低导致完全不检出。"""

_MIN_ACTIVE_THRESHOLD = 0.30
"""负向异常的最小活跃阈值，避免全局低活跃用户出现大量误报。"""

_MAX_GAP_MINUTES = 30
"""连续确认时允许的最大时间间隔（分钟）。"""


def _ts_to_minute_of_day(timestamp: int, utc_offset: int) -> int:
    """将 UTC 时间戳转换为本地 minute_of_day (0-1439)。"""
    local_ts = timestamp + utc_offset
    return (local_ts % 86400) // 60


def _format_time(minute_of_day: int) -> str:
    """将 minute_of_day 格式化为 HH:MM。"""
    return f"{minute_of_day // 60:02d}:{minute_of_day % 60:02d}"


def detect_anomalies(
    curve: ActivityCurve,
    timestamps: list[int],
    utc_offset: int,
    activity_flags: list[bool] | None = None,
    *,
    quiet_upper_quantile: float = _QUIET_UPPER_QUANTILE,
    active_lower_quantile: float = _ACTIVE_LOWER_QUANTILE,
    min_quiet_threshold: float = _MIN_QUIET_THRESHOLD,
    min_active_threshold: float = _MIN_ACTIVE_THRESHOLD,
    min_consecutive: int = 1,
    max_gap_minutes: int = _MAX_GAP_MINUTES,
) -> list[AnomalyPoint]:
    """检测外部时间点相对于活跃曲线的突兀点。

    Args:
        curve: 拟合好的用户活跃曲线。
        timestamps: UTC 时间戳列表，待检测的时间点。
        utc_offset: 本地时区偏移（秒）。
        activity_flags: 与 timestamps 等长的布尔列表。
            - True = 该时间戳有实际活动 → 可检测正向异常。
            - False = 该时间戳无活动 → 可检测负向异常。
            - None → 全部视为 True（仅检测正向异常）。
        quiet_upper_quantile: 正向异常安静阈值分位数。
        active_lower_quantile: 负向异常活跃阈值分位数。
        min_quiet_threshold: 安静阈值最小值。
        min_active_threshold: 活跃阈值最小值。
        min_consecutive: 连续触发次数阈值，>1 时只保留连续片段中的点。
        max_gap_minutes: 连续触发判定的最大间隔（分钟）。

    Returns:
        检测到的异常点列表，按 deviation_score 降序排列。
    """
    if not timestamps:
        return []

    if activity_flags is None:
        activity_flags = [True] * len(timestamps)

    if len(timestamps) != len(activity_flags):
        raise ValueError(
            f"timestamps ({len(timestamps)}) 与 "
            f"activity_flags ({len(activity_flags)}) 长度不一致"
        )

    quiet_upper_quantile = float(quiet_upper_quantile)
    active_lower_quantile = float(active_lower_quantile)
    min_quiet_threshold = float(min_quiet_threshold)
    min_active_threshold = float(min_active_threshold)

    quiet_upper_threshold = max(
        float(np.quantile(curve.predicted_upper, quiet_upper_quantile)),
        min_quiet_threshold,
    )
    active_lower_threshold = float(
        np.quantile(curve.predicted_lower, active_lower_quantile)
    )
    active_lower_threshold = max(active_lower_threshold, min_active_threshold)

    candidates: list[tuple[int, AnomalyPoint]] = []

    for ts, has_activity in zip(timestamps, activity_flags):
        minute = _ts_to_minute_of_day(ts, utc_offset)
        time_str = _format_time(minute)

        median_val = float(curve.predicted_median[minute])
        lower_val = float(curve.predicted_lower[minute])
        upper_val = float(curve.predicted_upper[minute])

        if has_activity:
            if upper_val < quiet_upper_threshold:
                band_width = max(upper_val - lower_val, 0.01)
                deviation = (quiet_upper_threshold - upper_val) / band_width
                deviation = min(deviation, 10.0)
                if deviation <= 0:
                    continue

                candidates.append(
                    (
                        ts,
                        AnomalyPoint(
                            minute_of_day=minute,
                            anomaly_type=AnomalyType.UNEXPECTED_ACTIVITY,
                            deviation_score=round(deviation, 4),
                            expected_median=round(median_val, 4),
                            expected_range=(round(lower_val, 4), round(upper_val, 4)),
                            reason=(
                                f"用户在 {time_str} 发言，"
                                f"但该时段处于个体历史低活跃分位区"
                                f"（阈值≤{quiet_upper_threshold:.2f}，"
                                f"预期区间: [{lower_val:.2f}, {upper_val:.2f}]）"
                            ),
                        ),
                    )
                )
        else:
            if lower_val > active_lower_threshold:
                band_width = max(upper_val - lower_val, 0.01)
                deviation = (lower_val - active_lower_threshold) / band_width
                deviation = min(deviation, 10.0)
                if deviation <= 0:
                    continue

                candidates.append(
                    (
                        ts,
                        AnomalyPoint(
                            minute_of_day=minute,
                            anomaly_type=AnomalyType.UNEXPECTED_ABSENCE,
                            deviation_score=round(deviation, 4),
                            expected_median=round(median_val, 4),
                            expected_range=(round(lower_val, 4), round(upper_val, 4)),
                            reason=(
                                f"用户在 {time_str} 无活动，"
                                f"但该时段处于个体历史高活跃分位区"
                                f"（阈值≥{active_lower_threshold:.2f}，"
                                f"预期区间: [{lower_val:.2f}, {upper_val:.2f}]）"
                            ),
                        ),
                    )
                )

    if min_consecutive <= 1:
        anomalies = [point for _, point in candidates]
        anomalies.sort(key=lambda a: a.deviation_score, reverse=True)
        return anomalies

    candidates.sort(key=lambda item: item[0])
    selected: list[AnomalyPoint] = []
    run: list[tuple[int, AnomalyPoint]] = []

    def flush_run() -> None:
        if len(run) >= min_consecutive:
            selected.extend(point for _, point in run)

    for ts, point in candidates:
        if not run:
            run.append((ts, point))
            continue

        prev_ts, prev_point = run[-1]
        gap_min = (ts - prev_ts) / 60
        if (
            point.anomaly_type == prev_point.anomaly_type
            and gap_min <= max_gap_minutes
        ):
            run.append((ts, point))
        else:
            flush_run()
            run = [(ts, point)]

    flush_run()
    selected.sort(key=lambda a: a.deviation_score, reverse=True)
    return selected
