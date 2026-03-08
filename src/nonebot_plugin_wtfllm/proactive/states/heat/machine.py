"""热度状态机核心：状态编排与事件生成。

热度数值演化由 HeatDynamics 负责，这里只保留状态转移决策。
"""

from ._dynamics import HeatContext, HeatDynamics, HeatMetrics
from ._types import HeatSnapshot, MachineConfig, State, TransitionEvent


_THRESHOLD_EPSILON = 1e-9


class HeatMachine:
    """单会话热度状态机。"""

    __slots__ = (
        "_cfg",
        "_dynamics",
        "_context",
        "_state",
        "_state_entered_at",
    )

    def __init__(self, config: "MachineConfig | None" = None) -> None:
        """初始化单会话热度状态机。

        Args:
            config: 状态机配置；为空时使用默认配置。
        """
        self._cfg = config or MachineConfig()
        self._dynamics = HeatDynamics(self._cfg)
        self._context = HeatContext()
        self._state: State = State.IDLE
        self._state_entered_at: float = 0.0

    def feed(
        self,
        timestamp: float,
        sender_id: str,
        increment: float = 1.0,
    ) -> TransitionEvent | None:
        """喂入一条消息并尝试触发状态转移。

        Args:
            timestamp: 消息时间戳。
            sender_id: 发送者标识。
            increment: 消息带来的热度增量。

        Returns:
            若发生状态转移则返回事件，否则返回 None。
        """
        metrics = self._dynamics.ingest_message(
            self._context,
            timestamp,
            sender_id,
            increment,
        )
        return self._evaluate_transition(metrics, timestamp)

    def tick(self, timestamp: float) -> TransitionEvent | None:
        """推进时间并处理纯时间驱动的状态变化。

        Args:
            timestamp: 当前时间戳。

        Returns:
            若发生状态转移则返回事件，否则返回 None。
        """
        metrics = self._dynamics.advance_time(self._context, timestamp)
        event = self._evaluate_transition(
            metrics,
            timestamp,
            allow_activation=False,
        )

        if event is None:
            return self._maybe_transition_to_idle(metrics, timestamp)

        return event

    def peek(self, timestamp: float) -> HeatSnapshot:
        """查看指定时刻的状态快照而不修改内部状态。

        Args:
            timestamp: 观测时间戳。

        Returns:
            该时刻的只读快照。
        """
        metrics = self._dynamics.measure(self._context, timestamp)
        return self._build_snapshot(metrics, self._state, self._state_entered_at)

    def predict_transition_time(self, now: float) -> float | None:
        """预测下一次可能发生状态转移的时间。

        Args:
            now: 当前时间戳。

        Returns:
            下一次转移时间；若当前无可预测转移则返回 None。
        """
        if self._state is State.ACTIVE:
            return self._dynamics.predict_deactivation_time(
                self._context,
                now,
                self._cfg.deactivate_threshold,
                _THRESHOLD_EPSILON,
            )

        if self._state is State.INACTIVE:
            return self._state_entered_at + self._cfg.idle_timeout

        return None

    @property
    def state(self) -> State:
        """返回当前状态。"""
        return self._state

    def _evaluate_transition(
        self,
        metrics: HeatMetrics,
        timestamp: float,
        *,
        allow_activation: bool = True,
    ) -> TransitionEvent | None:
        """根据当前指标判断是否需要进行状态转移。

        Args:
            metrics: 当前热度指标。
            timestamp: 当前时间戳。
            allow_activation: 是否允许进入 ACTIVE。

        Returns:
            若发生状态转移则返回事件，否则返回 None。
        """
        cfg = self._cfg
        heat = metrics.heat

        if (
            allow_activation
            and self._state is State.IDLE
            and heat >= cfg.activate_threshold - _THRESHOLD_EPSILON
        ):
            return self._do_transition(State.ACTIVE, metrics, timestamp)

        if (
            self._state is State.ACTIVE
            and heat <= cfg.deactivate_threshold + _THRESHOLD_EPSILON
        ):
            return self._do_transition(State.INACTIVE, metrics, timestamp)

        if (
            allow_activation
            and self._state is State.INACTIVE
            and heat >= cfg.activate_threshold - _THRESHOLD_EPSILON
        ):
            return self._do_transition(State.ACTIVE, metrics, timestamp)

        return None

    def _maybe_transition_to_idle(
        self,
        metrics: HeatMetrics,
        timestamp: float,
    ) -> TransitionEvent | None:
        """在冷却超时后尝试从 INACTIVE 进入 IDLE。

        Args:
            metrics: 当前热度指标。
            timestamp: 当前时间戳。

        Returns:
            若发生状态转移则返回事件，否则返回 None。
        """
        if (
            self._state is State.INACTIVE
            and timestamp - self._state_entered_at >= self._cfg.idle_timeout
        ):
            return self._do_transition(State.IDLE, metrics, timestamp)

        return None

    def _do_transition(
        self,
        new_state: State,
        metrics: HeatMetrics,
        timestamp: float,
    ) -> TransitionEvent:
        """执行一次状态切换并构造转移事件。

        Args:
            new_state: 目标状态。
            metrics: 当前热度指标。
            timestamp: 转移时间戳。

        Returns:
            已构造好的状态转移事件。
        """
        prev = self._state
        self._state = new_state
        self._state_entered_at = timestamp

        snapshot = self._build_snapshot(
            metrics,
            new_state,
            state_entered_at=timestamp,
        )
        return TransitionEvent(
            prev_state=prev,
            new_state=new_state,
            snapshot=snapshot,
            timestamp=timestamp,
        )

    def _build_snapshot(
        self,
        metrics: HeatMetrics,
        state: State,
        state_entered_at: float,
    ) -> HeatSnapshot:
        """将当前指标封装为只读快照。

        Args:
            metrics: 当前热度指标。
            state: 快照对应状态。
            state_entered_at: 进入该状态的时间戳。

        Returns:
            构造出的热度快照。
        """
        return HeatSnapshot(
            state=state,
            heat=metrics.heat,
            msg_ema=metrics.msg_ema,
            n_participants=metrics.n_participants,
            velocity=metrics.velocity,
            last_update=metrics.last_update,
            state_entered_at=state_entered_at,
        )
