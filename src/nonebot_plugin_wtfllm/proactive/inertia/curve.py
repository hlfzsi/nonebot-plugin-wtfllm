"""LightGBM 分位数回归曲线拟合。"""

import numpy as np
from numpy.typing import NDArray
import lightgbm as lgb

from ._types import ActivityCurve, SessionKey

# k ∈ {1, 2, 3, 4, 6, 12} 对应 24h / 12h / 8h / 6h / 4h / 2h 周期
_HARMONICS: tuple[int, ...] = (1, 2, 3, 4, 6, 12)

_LGB_BASE_PARAMS: dict[str, object] = {
    "num_leaves": 15,
    "min_data_in_leaf": 12,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "num_threads": 1,
    "verbosity": -1,
}

_NUM_BOOST_ROUND = 240
_MIN_NONZERO_POINTS = 6
_ZERO_SAMPLE_RATIO = 4
_MAX_ZERO_SAMPLES = 1200

_FULL_FEATURES: NDArray[np.float64] | None = None


def _build_features(minutes: NDArray[np.int32]) -> NDArray[np.float64]:
    """将 minute_of_day 编码为周期性谐波特征。

    对每个谐波阶数 k，生成:
        sin(2π·k·m / 1440),  cos(2π·k·m / 1440)

    Returns:
        shape = (len(minutes), 2 * len(_HARMONICS)) 的特征矩阵。
    """
    cols: list[NDArray[np.float64]] = []
    for k in _HARMONICS:
        angle = 2.0 * np.pi * k * minutes / 1440.0
        cols.append(np.sin(angle))
        cols.append(np.cos(angle))
    return np.column_stack(cols)


def _get_full_features() -> NDArray[np.float64]:
    """获取 1440 个分钟的完整特征矩阵"""
    global _FULL_FEATURES
    if _FULL_FEATURES is None:
        _FULL_FEATURES = _build_features(np.arange(1440, dtype=np.int32))
    return _FULL_FEATURES


def _train_quantile_booster(
    X: NDArray[np.float64],
    y: NDArray[np.float32],
    alpha: float,
) -> lgb.Booster:
    """训练单个分位数回归模型。"""
    params = {
        **_LGB_BASE_PARAMS,
        "objective": "quantile",
        "alpha": alpha,
    }
    dataset = lgb.Dataset(X, label=y, free_raw_data=False)
    return lgb.train(
        params,
        dataset,
        num_boost_round=_NUM_BOOST_ROUND,
    )


def _compute_r_squared(
    y_true: NDArray[np.float32],
    y_pred: NDArray[np.float64],
) -> float:
    """计算 R²"""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def fit_activity_curve(
    session: SessionKey,
    bins: NDArray[np.float32],
    data_quality: float,
    quantile_lower: float = 0.05,
    quantile_upper: float = 0.95,
) -> ActivityCurve | None:
    """拟合用户活跃曲线。

    Args:
        session: 用户会话标识。
        bins: 1440 维活跃天数数组。
        data_quality: 来自预筛选的质量分。
        quantile_lower: 下界分位数（越小越保守，降低假阳性）。
        quantile_upper: 上界分位数。

    Returns:
        拟合后的 ActivityCurve，或 None（训练数据不足时）。
    """

    nonzero_mask = bins > 0
    nonzero_indices = np.where(nonzero_mask)[0].astype(np.int32)

    if len(nonzero_indices) < _MIN_NONZERO_POINTS:
        return None

    X_train = _build_features(nonzero_indices)
    y_train = bins[nonzero_mask]

    zero_indices = np.where(~nonzero_mask)[0].astype(np.int32)
    if len(zero_indices) > 0:
        n_zero_sample = min(
            len(nonzero_indices) * _ZERO_SAMPLE_RATIO,
            len(zero_indices),
            _MAX_ZERO_SAMPLES,
        )
        rng = np.random.default_rng(42)
        sampled_zeros = rng.choice(zero_indices, size=n_zero_sample, replace=False)
        X_zeros = _build_features(sampled_zeros)
        y_zeros = np.zeros(n_zero_sample, dtype=np.float32)
        X_train = np.vstack([X_train, X_zeros])
        y_train = np.concatenate([y_train, y_zeros])

    booster_lower = _train_quantile_booster(X_train, y_train, quantile_lower)
    booster_median = _train_quantile_booster(X_train, y_train, 0.5)
    booster_upper = _train_quantile_booster(X_train, y_train, quantile_upper)

    X_full = _get_full_features()
    pred_lower = np.maximum(np.asarray(booster_lower.predict(X_full)), 0).astype(
        np.float32
    )
    pred_median = np.maximum(np.asarray(booster_median.predict(X_full)), 0).astype(
        np.float32
    )
    pred_upper = np.maximum(np.asarray(booster_upper.predict(X_full)), 0).astype(
        np.float32
    )

    pred_lower = np.minimum(pred_lower, pred_median)
    pred_upper = np.maximum(pred_upper, pred_median)

    r_squared = _compute_r_squared(bins, pred_median.astype(np.float64))

    return ActivityCurve(
        session=session,
        predicted_median=pred_median,
        predicted_lower=pred_lower,
        predicted_upper=pred_upper,
        r_squared=round(r_squared, 4),
        data_quality=data_quality,
    )
