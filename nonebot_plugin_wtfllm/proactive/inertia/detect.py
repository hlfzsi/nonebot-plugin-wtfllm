"""时间戳与活跃曲线比较，识别异常偏离。"""

from ._types import ActivityCurve, AnomalyPoint, AnomalyType


_UPPER_QUIET_THRESHOLD = 0.5
"""正向异常：上界预测值低于此阈值时，该时段才被视为"安静"。
仅当用户在安静时段出现时才报告正向异常。"""

_LOWER_ACTIVE_THRESHOLD = 1.5
"""负向异常：下界预测值高于此阈值时，该时段才被视为"应活跃"。
仅当用户在应活跃时段缺席时才报告负向异常。"""


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
    upper_quiet_threshold: float = _UPPER_QUIET_THRESHOLD,
    lower_active_threshold: float = _LOWER_ACTIVE_THRESHOLD,
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
        upper_quiet_threshold: 正向异常的上界安静阈值。
        lower_active_threshold: 负向异常的下界活跃阈值。

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

    anomalies: list[AnomalyPoint] = []

    for ts, has_activity in zip(timestamps, activity_flags):
        minute = _ts_to_minute_of_day(ts, utc_offset)
        time_str = _format_time(minute)

        median_val = float(curve.predicted_median[minute])
        lower_val = float(curve.predicted_lower[minute])
        upper_val = float(curve.predicted_upper[minute])

        if has_activity:
            if upper_val < upper_quiet_threshold:
                band_width = max(upper_val, 0.01)
                deviation = (upper_quiet_threshold - upper_val) / band_width
                deviation = min(deviation, 10.0)

                anomalies.append(
                    AnomalyPoint(
                        minute_of_day=minute,
                        anomaly_type=AnomalyType.UNEXPECTED_ACTIVITY,
                        deviation_score=round(deviation, 4),
                        expected_median=round(median_val, 4),
                        expected_range=(round(lower_val, 4), round(upper_val, 4)),
                        reason=(
                            f"用户在 {time_str} 发言，"
                            f"但历史数据显示该时段几乎无活动"
                            f"（预期活跃度: {median_val:.2f}，"
                            f"区间: [{lower_val:.2f}, {upper_val:.2f}]）"
                        ),
                    )
                )
        else:
            if lower_val >= lower_active_threshold:
                deviation = (
                    lower_val - lower_active_threshold
                ) / lower_active_threshold
                deviation = min(deviation, 10.0)

                anomalies.append(
                    AnomalyPoint(
                        minute_of_day=minute,
                        anomaly_type=AnomalyType.UNEXPECTED_ABSENCE,
                        deviation_score=round(deviation, 4),
                        expected_median=round(median_val, 4),
                        expected_range=(round(lower_val, 4), round(upper_val, 4)),
                        reason=(
                            f"用户在 {time_str} 无活动，"
                            f"但历史数据显示该时段通常活跃"
                            f"（预期活跃度: {median_val:.2f}，"
                            f"区间: [{lower_val:.2f}, {upper_val:.2f}]）"
                        ),
                    )
                )

    anomalies.sort(key=lambda a: a.deviation_score, reverse=True)
    return anomalies
