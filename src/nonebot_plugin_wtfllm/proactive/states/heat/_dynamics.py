import math
from dataclasses import dataclass, field

from ._types import MachineConfig


@dataclass(slots=True)
class HeatContext:
    """热度模型的可变运行态。"""

    msg_ema: float = 0.0
    last_update: float = 0.0
    last_heat: float = 0.0
    velocity: float = 0.0
    participants: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class HeatMetrics:
    """某一时刻的派生热度指标。"""

    heat: float
    msg_ema: float
    n_participants: int
    velocity: float
    last_update: float


class HeatDynamics:
    """热度数值演化器。"""

    __slots__ = ("_cfg",)

    def __init__(self, config: MachineConfig) -> None:
        """初始化热度数值演化器。

        Args:
            config: 热度计算相关配置。
        """
        self._cfg = config

    def ingest_message(
        self,
        context: HeatContext,
        timestamp: float,
        sender_id: str,
        increment: float,
    ) -> HeatMetrics:
        """处理一条新消息并返回更新后的热度指标。

        Args:
            context: 可变运行态。
            timestamp: 消息时间戳。
            sender_id: 发送者标识。
            increment: 本次消息带来的热度增量。

        Returns:
            更新后的热度指标。
        """
        context.msg_ema = self._decayed_ema(context, timestamp)
        context.msg_ema += increment
        context.participants[sender_id] = timestamp
        return self._finalize(context, timestamp)

    def advance_time(
        self,
        context: HeatContext,
        timestamp: float,
    ) -> HeatMetrics:
        """推进时间并返回自然衰减后的热度指标。

        Args:
            context: 可变运行态。
            timestamp: 目标时间戳。

        Returns:
            推进后的热度指标。
        """
        context.msg_ema = self._decayed_ema(context, timestamp)
        return self._finalize(context, timestamp)

    def measure(self, context: HeatContext, timestamp: float) -> HeatMetrics:
        """在不给运行态落盘的前提下测量某时刻的热度指标。

        Args:
            context: 可变运行态。
            timestamp: 观测时间戳。

        Returns:
            该时刻的热度指标。
        """
        msg_ema = self._decayed_ema(context, timestamp)
        n_active = self._count_active_participants(context.participants, timestamp)
        heat = msg_ema / max(1, n_active)
        velocity = self._measure_velocity(context, heat, timestamp)
        return HeatMetrics(
            heat=heat,
            msg_ema=msg_ema,
            n_participants=n_active,
            velocity=velocity,
            last_update=context.last_update,
        )

    def predict_deactivation_time(
        self,
        context: HeatContext,
        now: float,
        deactivate_threshold: float,
        epsilon: float,
    ) -> float:
        """预测 ACTIVE 状态下何时会降到去激活阈值。

        Args:
            context: 可变运行态。
            now: 当前时间戳。
            deactivate_threshold: 去激活阈值。
            epsilon: 浮点比较容差。

        Returns:
            预计触达阈值的时间戳。
        """
        n_active = self._count_active_participants(context.participants, now)
        target_ema = deactivate_threshold * max(1, n_active)
        current_ema = self._decayed_ema(context, now)
        if current_ema <= target_ema + epsilon:
            return now
        delay = self._cfg.half_life * math.log2(current_ema / target_ema)
        return now + delay

    def _finalize(self, context: HeatContext, timestamp: float) -> HeatMetrics:
        """完成一次更新周期并写回派生字段。

        Args:
            context: 可变运行态。
            timestamp: 当前时间戳。

        Returns:
            最终热度指标。
        """
        self._prune_participants(context.participants, timestamp)
        n_active = self._count_active_participants(context.participants, timestamp)
        heat = context.msg_ema / max(1, n_active)
        velocity = self._update_velocity(context, heat, timestamp)
        context.last_update = timestamp
        return HeatMetrics(
            heat=heat,
            msg_ema=context.msg_ema,
            n_participants=n_active,
            velocity=velocity,
            last_update=context.last_update,
        )

    def _decay_factor(self, dt: float) -> float:
        """计算给定时长对应的指数衰减系数。

        Args:
            dt: 时间间隔，单位为秒。

        Returns:
            对应的衰减系数。
        """
        if dt <= 0:
            return 1.0
        return 2.0 ** (-dt / self._cfg.half_life)

    def _decayed_ema(self, context: HeatContext, timestamp: float) -> float:
        """计算某时刻的衰减后消息 EMA。

        Args:
            context: 可变运行态。
            timestamp: 目标时间戳。

        Returns:
            衰减后的消息 EMA。
        """
        dt = timestamp - context.last_update
        return context.msg_ema * self._decay_factor(dt)

    def _count_active_participants(
        self,
        participants: dict[str, float],
        timestamp: float,
    ) -> int:
        """统计在指定时刻仍被视为活跃的参与人数。

        Args:
            participants: 参与者最后活跃时间表。
            timestamp: 统计时间戳。

        Returns:
            活跃参与人数。
        """
        threshold = self._cfg.participant_decay_threshold
        count = 0
        for last_seen in participants.values():
            weight = self._decay_factor(timestamp - last_seen)
            if weight >= threshold:
                count += 1
        return count

    def _prune_participants(
        self,
        participants: dict[str, float],
        timestamp: float,
    ) -> None:
        """移除过于久远的参与者记录。

        Args:
            participants: 参与者最后活跃时间表。
            timestamp: 当前时间戳。
        """
        cutoff = timestamp - 5.0 * self._cfg.half_life
        stale = [sid for sid, seen_at in participants.items() if seen_at < cutoff]
        for sid in stale:
            del participants[sid]

    def _update_velocity(
        self,
        context: HeatContext,
        heat: float,
        timestamp: float,
    ) -> float:
        """基于瞬时变化更新热度速度的 EMA。

        Args:
            context: 可变运行态。
            heat: 当前热度值。
            timestamp: 当前时间戳。

        Returns:
            更新后的速度值。
        """
        dt = timestamp - context.last_update if context.last_update > 0 else 0.0
        if dt <= 0:
            instant_velocity = heat - context.last_heat
        else:
            instant_velocity = (heat - context.last_heat) / dt

        alpha = self._cfg.velocity_alpha
        context.velocity = alpha * instant_velocity + (1.0 - alpha) * context.velocity
        context.last_heat = heat
        return context.velocity

    def _measure_velocity(
        self,
        context: HeatContext,
        heat: float,
        timestamp: float,
    ) -> float:
        """测量指定时刻的即时热度趋势，不修改上下文。"""
        dt = timestamp - context.last_update
        if dt <= 0:
            return context.velocity
        return (heat - context.last_heat) / dt
