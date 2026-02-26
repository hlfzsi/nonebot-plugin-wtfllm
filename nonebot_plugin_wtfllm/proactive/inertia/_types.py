from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray


class AnomalyType(StrEnum):
    """异常类型。"""

    UNEXPECTED_ACTIVITY = "unexpected_activity"
    """用户在历史低活跃时段出现发言。"""

    UNEXPECTED_ABSENCE = "unexpected_absence"
    """用户在历史高活跃时段缺席。"""


@dataclass(slots=True, frozen=True)
class SessionKey:
    """会话标识：群聊为 (group_id, sender)，私聊为 (user_id,)"""

    group_id: str | None = None
    user_id: str | None = None
    sender: str = ""

    @property
    def is_group(self) -> bool:
        return self.group_id is not None

    @property
    def group_user_key(self) -> tuple[str | None, str]:
        """(group_id, target_user_id)。私聊时 group_id 为 None"""
        if self.user_id is not None:
            return (None, self.user_id)
        return (self.group_id, self.sender)

    @property
    def target_id(self) -> str:
        if self.user_id is not None:
            return self.user_id
        elif self.group_id is not None:
            return self.group_id
        raise ValueError("Invalid SessionKey: both group_id and user_id are None")

    def __hash__(self) -> int:
        return hash((self.group_id, self.user_id, self.sender))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SessionKey):
            return NotImplemented
        return (
            self.group_id == other.group_id
            and self.user_id == other.user_id
            and self.sender == other.sender
        )

    def __repr__(self) -> str:
        if self.group_id is not None:
            return f"GroupSession(group_id={self.group_id}, sender={self.sender})"
        else:
            return f"PrivateSession(user_id={self.user_id})"


@dataclass(slots=True)
class ActivityCurve:
    """LightGBM 分位数回归拟合的用户活跃曲线。

    每个数组长度为 1440，索引 = minute_of_day (0-1439)。
    """

    session: SessionKey
    predicted_median: NDArray[np.float32]
    """中位预测曲线 (q=0.5)。"""
    predicted_lower: NDArray[np.float32]
    """下界预测曲线 (低分位数，如 q=0.05)。"""
    predicted_upper: NDArray[np.float32]
    """上界预测曲线 (高分位数，如 q=0.95)。"""
    r_squared: float
    """拟合优度 R²，基于中位曲线计算。"""
    data_quality: float
    """预筛选数据质量分 [0, 1]。"""

    def __post_init__(self) -> None:
        for name in ("predicted_median", "predicted_lower", "predicted_upper"):
            arr = getattr(self, name)
            if arr.shape != (1440,):
                raise ValueError(f"{name} 长度必须为 1440，实际为 {arr.shape}")

    def __repr__(self) -> str:
        return (
            f"ActivityCurve(session={self.session}, "
            f"R²={self.r_squared:.3f}, quality={self.data_quality:.3f})"
        )


@dataclass(slots=True)
class AnomalyPoint:
    """相对于活跃曲线的异常偏离点。"""

    minute_of_day: int
    """检测到异常的时刻 (0-1439)。"""
    anomaly_type: AnomalyType
    """异常类型。"""
    deviation_score: float
    """偏离程度，>0 表示在预测区间之外。值越大越异常。"""
    expected_median: float
    """该时刻的预期中位活跃度。"""
    expected_range: tuple[float, float]
    """预测区间 (lower, upper)。"""
    reason: str
    """人类可读的原因描述。"""
