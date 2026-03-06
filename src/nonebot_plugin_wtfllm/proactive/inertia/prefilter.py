"""用户数据预筛选：在 LightGBM 拟合前排除不适合建模的用户。"""

import numpy as np
from numpy.typing import NDArray


_ACTIVE_COVERAGE_CEIL = 0.50
"""活跃分钟占全天比例上限。超过此值视为全天均匀活跃，无法拟合有意义的曲线。"""

_PEAK_RATIO_FLOOR = 2.0
"""峰值 / 非零中位数 的最低倍数。低于此值说明没有可辨识的活跃高峰。"""

_GINI_FLOOR = 0.30
"""基尼系数下限。低于此值说明分布过于均匀，缺乏可学习的结构。"""


def _gini_coefficient(arr: NDArray[np.float32]) -> float:
    """计算一维非负数组的基尼系数 (0=完全均匀, 1=完全集中)。"""
    total = float(arr.sum())
    if total == 0:
        return 0.0
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * (index * sorted_arr).sum() / total - (n + 1)) / n)


def prefilter(
    bins: NDArray[np.float32],
    max_active_days: int,
    min_required_days: int,
    *,
    active_coverage_ceil: float = _ACTIVE_COVERAGE_CEIL,
    peak_ratio_floor: float = _PEAK_RATIO_FLOOR,
    gini_floor: float = _GINI_FLOOR,
) -> tuple[bool, float]:
    """判定用户数据是否适合进行回归曲线拟合。

    Args:
        bins: 1440 维活跃天数数组。
        max_active_days: 该用户最大活跃天数。
        min_required_days: 最低所需活跃天数（来自配置）。
        active_coverage_ceil: 活跃覆盖率上限。
        peak_ratio_floor: 峰值突出度下限。
        gini_floor: 基尼系数下限。

    Returns:
        (通过与否, 数据质量分 [0, 1])。
        质量分由充分性、峰值突出度和集中度三项加权混合。
    """
    if max_active_days < min_required_days:
        return False, 0.0

    nonzero_count = int(np.count_nonzero(bins))
    coverage = nonzero_count / 1440
    if coverage > active_coverage_ceil:
        return False, 0.0

    if nonzero_count == 0:
        return False, 0.0

    overall_mean = float(bins.mean())
    peak = float(bins.max())
    peak_ratio = peak / overall_mean if overall_mean > 0 else peak
    if peak_ratio < peak_ratio_floor:
        return False, 0.0

    gini = _gini_coefficient(bins)
    if gini < gini_floor:
        return False, 0.0

    sufficiency_score = min(max_active_days / (min_required_days * 2), 1.0)
    peak_score = min((peak_ratio - peak_ratio_floor) / peak_ratio_floor, 1.0)
    gini_score = min((gini - gini_floor) / (1.0 - gini_floor), 1.0)

    quality = sufficiency_score * 0.35 + peak_score * 0.30 + gini_score * 0.35
    return True, round(quality, 4)
