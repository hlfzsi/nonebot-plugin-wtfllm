from dataclasses import dataclass
from enum import StrEnum


class State(StrEnum):
    """会话热度状态。"""

    IDLE = "idle"
    """空闲"""

    ACTIVE = "active"
    """活跃"""

    INACTIVE = "inactive"
    """冷却"""


@dataclass(frozen=True, slots=True)
class MachineConfig:
    """状态机参数配置。"""

    half_life: float = 300.0
    """消息频率半衰期（秒）。控制 msg_ema 衰减速度。"""

    activate_threshold: float = 2.0
    """IDLE/INACTIVE → ACTIVE 的热度阈值（人均消息数）。"""

    deactivate_threshold: float = 0.5
    """ACTIVE → INACTIVE 的热度阈值（人均消息数）。
    必须小于 activate_threshold 以形成迟滞。"""

    idle_timeout: float = 30.0
    """INACTIVE → IDLE 的超时时间（秒）。"""

    velocity_alpha: float = 0.3
    """热度变化速度的 EMA 平滑系数。越大越偏向最新瞬时速度。"""

    base_increment: float = 1.0
    """每条消息的基础热度增量。"""

    participant_decay_threshold: float = 0.1
    """参与人判活的最低衰减权重。低于此值视为已离场。
    0.1 ≈ 3.32 个半衰期。"""

    def __post_init__(self) -> None:
        if self.half_life <= 0:
            raise ValueError("half_life must be positive")
        if self.deactivate_threshold >= self.activate_threshold:
            raise ValueError(
                "deactivate_threshold must be less than activate_threshold"
            )


@dataclass(frozen=True, slots=True)
class HeatSnapshot:
    """状态机某一时刻的只读快照。"""

    state: State
    heat: float
    """当前热度值（人均消息率）。"""
    msg_ema: float
    """衰减后的消息累计值。"""
    n_participants: int
    """当前活跃参与人数。"""
    velocity: float
    """热度变化速度（heat/s）。"""
    last_update: float
    """上次更新的时间戳。"""
    state_entered_at: float
    """进入当前状态的时间戳。"""


@dataclass(frozen=True, slots=True)
class TransitionEvent:
    """状态转移事件。"""

    prev_state: State
    new_state: State
    snapshot: HeatSnapshot
    """转移发生时的快照。"""
    timestamp: float
    """转移发生的时间戳。"""
