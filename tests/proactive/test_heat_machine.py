"""HeatMachine 纯同步单元测试。

验证核心热度计算、状态转移、迟滞效应、预测唤醒等逻辑。
不涉及 asyncio 或 I/O。
"""

import math

import pytest

from nonebot_plugin_wtfllm.proactive.states.heat._types import (
    HeatSnapshot,
    MachineConfig,
    State,
    TransitionEvent,
)
from nonebot_plugin_wtfllm.proactive.states.heat.machine import HeatMachine


@pytest.fixture
def cfg() -> MachineConfig:
    return MachineConfig(
        half_life=300.0,
        activate_threshold=2.0,
        deactivate_threshold=0.5,
        idle_timeout=1800.0,
    )


@pytest.fixture
def m(cfg: MachineConfig) -> HeatMachine:
    return HeatMachine(cfg)


class TestInitialState:
    def test_starts_idle(self, m: HeatMachine) -> None:
        assert m.state is State.IDLE

    def test_peek_returns_zero_snapshot(self, m: HeatMachine) -> None:
        snap = m.peek(1000.0)
        assert snap.state is State.IDLE
        assert snap.heat == 0.0
        assert snap.msg_ema == 0.0
        assert snap.n_participants == 0


class TestAntiNoise:
    """单条消息不应触发 ACTIVE（抗噪）。"""

    def test_single_message_stays_idle(self, m: HeatMachine) -> None:
        event = m.feed(1000.0, "user_a", 1.0)
        # heat = 1.0 / max(1, 1) = 1.0 < activate(2.0)
        assert event is None
        assert m.state is State.IDLE

    def test_single_message_from_different_senders_stays_idle(
        self, m: HeatMachine
    ) -> None:
        """3 人各说 1 句 → heat = 3/3 = 1.0 → 不触发。"""
        m.feed(1000.0, "user_a", 1.0)
        m.feed(1000.1, "user_b", 1.0)
        event = m.feed(1000.2, "user_c", 1.0)
        assert event is None
        assert m.state is State.IDLE


class TestActivation:
    """连续消息触发 ACTIVE。"""

    def test_two_messages_same_sender_activates(self, m: HeatMachine) -> None:
        """1 人 2 条 → heat = 2/1 = 2.0 ≥ activate。"""
        m.feed(1000.0, "user_a", 1.0)
        event = m.feed(1000.0, "user_a", 1.0)
        assert event is not None
        assert event.prev_state is State.IDLE
        assert event.new_state is State.ACTIVE
        assert m.state is State.ACTIVE

    def test_two_senders_five_messages_activates(self, m: HeatMachine) -> None:
        """2 人 5 条 → heat = 5/2 = 2.5 ≥ activate。"""
        t = 1000.0
        for i in range(5):
            sender = "user_a" if i % 2 == 0 else "user_b"
            event = m.feed(t + i * 0.1, sender, 1.0)
        assert m.state is State.ACTIVE

    def test_ten_senders_ten_messages_stays_idle(self, m: HeatMachine) -> None:
        """10 人各 1 条 → heat = 10/10 = 1.0 → 不触发。"""
        for i in range(10):
            m.feed(1000.0 + i * 0.1, f"user_{i}", 1.0)
        assert m.state is State.IDLE

    def test_ten_senders_thirty_messages_activates(self, m: HeatMachine) -> None:
        """10 人 共 30 条 → heat = 30/10 = 3.0 ≥ activate。"""
        t = 1000.0
        for i in range(30):
            sender = f"user_{i % 10}"
            m.feed(t + i * 0.1, sender, 1.0)
        assert m.state is State.ACTIVE


class TestDecayAndDeactivation:
    """时间推进引发衰减 → INACTIVE。"""

    def _activate(self, m: HeatMachine, t: float = 1000.0) -> None:
        """辅助：快速激活到 ACTIVE 状态（同一时刻 2 条消息 → ema=2.0）。"""
        m.feed(t, "user_a", 1.0)
        m.feed(t, "user_a", 1.0)
        assert m.state is State.ACTIVE

    def test_decay_triggers_inactive(self, m: HeatMachine, cfg: MachineConfig) -> None:
        self._activate(m)
        # 足够时间让 msg_ema 衰减到 deactivate_threshold
        # msg_ema=2, 需要 2 × 2^(-t/300) / 1 ≤ 0.5
        # → 2^(-t/300) ≤ 0.25 → t ≥ 600s (2 个半衰期)
        event = m.tick(1000.0 + 700.0)
        assert event is not None
        assert event.new_state is State.INACTIVE
        assert m.state is State.INACTIVE

    def test_tick_inactive_to_idle(self, m: HeatMachine, cfg: MachineConfig) -> None:
        self._activate(m)
        # 衰减到 INACTIVE
        m.tick(1000.0 + 700.0)
        assert m.state is State.INACTIVE
        # idle_timeout (1800s) 后 → IDLE
        event = m.tick(1000.0 + 700.0 + 1800.0)
        assert event is not None
        assert event.new_state is State.IDLE
        assert m.state is State.IDLE

    def test_inactive_reactivates_on_new_messages(
        self, m: HeatMachine
    ) -> None:
        self._activate(m)
        m.tick(1000.0 + 700.0)
        assert m.state is State.INACTIVE
        # 等待足够久让旧参与人衰减下线（weight < 0.1）
        # user_a 在 t=1000, 需要 dt > 300*log2(10) ≈ 997s
        t = 2000.0
        m.feed(t, "user_b", 1.0)
        event = m.feed(t, "user_b", 1.0)
        assert event is not None
        assert event.new_state is State.ACTIVE

    def test_tick_does_not_spontaneously_reactivate(self) -> None:
        """无新消息时，tick 不应仅因参与人数衰减而重新激活。"""
        machine = HeatMachine(
            MachineConfig(
                half_life=300.0,
                activate_threshold=0.7,
                deactivate_threshold=0.3,
                idle_timeout=1800.0,
            )
        )

        machine.feed(0.0, "user_b", 1.0)
        machine.feed(0.0, "user_c", 1.0)
        machine.feed(0.0, "user_d", 1.0)
        machine.feed(100.0, "user_a", 4.0)

        assert machine.state is State.ACTIVE

        inactive_event = None
        for t in range(101, 1501):
            event = machine.tick(float(t))
            if event is not None and event.new_state is State.INACTIVE:
                inactive_event = event
                break

        assert inactive_event is not None
        assert machine.state is State.INACTIVE

        for t in range(int(inactive_event.timestamp) + 1, 1501):
            event = machine.tick(float(t))
            assert not (
                event is not None
                and event.prev_state is State.INACTIVE
                and event.new_state is State.ACTIVE
            )

        assert machine.state is not State.ACTIVE


class TestHysteresis:
    """迟滞效应：deactivate < heat < activate 区间不触发跳变。"""

    def test_no_flapping_in_hysteresis_band(self, m: HeatMachine) -> None:
        """heat 在 0.5 < h < 2.0 区间时不应来回跳变。"""
        # 先激活
        m.feed(1000.0, "user_a", 1.0)
        m.feed(1000.0, "user_a", 1.0)
        assert m.state is State.ACTIVE

        # 让热度衰减到迟滞区间内（如 heat ≈ 1.0）
        # msg_ema=2, 衰减到 1.0 需 1 个半衰期 (300s)
        m.tick(1300.0)
        # heat ≈ 1.0, 在迟滞区间内 → 仍然 ACTIVE
        snap = m.peek(1300.0)
        assert snap.heat > 0.5  # 还在去激活阈值以上
        assert m.state is State.ACTIVE


class TestPredictTransitionTime:
    """predict_transition_time 精度验证。"""

    def test_active_predicts_deactivation(
        self, m: HeatMachine, cfg: MachineConfig
    ) -> None:
        m.feed(1000.0, "user_a", 1.0)
        m.feed(1000.0, "user_a", 1.0)
        assert m.state is State.ACTIVE

        predicted = m.predict_transition_time(1000.0)
        assert predicted is not None

        # 在预测时间 tick 一下，应该触发 INACTIVE
        event = m.tick(predicted + 0.1)
        assert event is not None
        assert event.new_state is State.INACTIVE

    def test_active_deactivates_at_predicted_time(
        self, m: HeatMachine
    ) -> None:
        m.feed(1000.0, "user_a", 1.0)
        m.feed(1000.0, "user_a", 1.0)

        predicted = m.predict_transition_time(1000.0)
        assert predicted is not None

        event = m.tick(predicted)
        assert event is not None
        assert event.new_state is State.INACTIVE

    def test_inactive_predicts_idle_timeout(
        self, m: HeatMachine, cfg: MachineConfig
    ) -> None:
        m.feed(1000.0, "user_a", 1.0)
        m.feed(1000.0, "user_a", 1.0)
        m.tick(1000.0 + 700.0)  # → INACTIVE
        assert m.state is State.INACTIVE

        predicted = m.predict_transition_time(1000.0 + 700.0)
        assert predicted is not None

        event = m.tick(predicted)
        assert event is not None
        assert event.new_state is State.IDLE

    def test_idle_predicts_none(self, m: HeatMachine) -> None:
        assert m.predict_transition_time(1000.0) is None

    def test_prediction_accuracy(
        self, m: HeatMachine, cfg: MachineConfig
    ) -> None:
        """预测时间 ±1s 内应确实触发转移。"""
        m.feed(1000.0, "user_a", 1.0)
        m.feed(1000.0, "user_a", 1.0)
        predicted = m.predict_transition_time(1000.0)
        assert predicted is not None

        # 预测前 1 秒不应触发
        event_before = m.tick(predicted - 1.0)
        assert event_before is None
        assert m.state is State.ACTIVE


class TestPeekIsReadOnly:
    """peek 不修改任何内部状态。"""

    def test_peek_does_not_mutate(self, m: HeatMachine) -> None:
        m.feed(1000.0, "user_a", 1.0)
        snap1 = m.peek(2000.0)
        snap2 = m.peek(2000.0)
        assert snap1 == snap2
        # 状态不会因 peek 改变
        assert m.state is State.IDLE

    def test_peek_at_future_shows_decay(self, m: HeatMachine) -> None:
        m.feed(1000.0, "user_a", 1.0)
        snap_now = m.peek(1000.0)
        snap_later = m.peek(1300.0)  # 1 个半衰期后
        assert snap_later.msg_ema < snap_now.msg_ema
        assert snap_later.msg_ema == pytest.approx(snap_now.msg_ema * 0.5, rel=1e-6)


class TestCausalOrder:
    """因果顺序：先衰减再叠加。"""

    def test_decay_then_accumulate(self, m: HeatMachine) -> None:
        """在一个半衰期后再 feed，msg_ema 应为 衰减后的值 + increment。"""
        m.feed(1000.0, "user_a", 1.0)
        # 1 个半衰期后 feed
        m.feed(1300.0, "user_a", 1.0)
        snap = m.peek(1300.0)
        # 应为 1.0 * 0.5 + 1.0 = 1.5
        assert snap.msg_ema == pytest.approx(1.5, rel=1e-6)


class TestVelocity:
    """速度方向正确性。"""

    def test_velocity_positive_on_feed(self, m: HeatMachine) -> None:
        m.feed(1000.0, "user_a", 1.0)
        m.feed(1001.0, "user_a", 1.0)
        snap = m.peek(1001.0)
        assert snap.velocity > 0

    def test_velocity_negative_on_decay(self, m: HeatMachine) -> None:
        m.feed(1000.0, "user_a", 1.0)
        m.feed(1000.0, "user_a", 1.0)
        # 多次 tick 让 velocity EMA 收敛到负方向
        for t in range(1010, 1310, 10):
            m.tick(float(t))
        snap = m.peek(1310.0)
        assert snap.velocity < 0

    def test_peek_future_velocity_reflects_decay_without_tick(
        self, m: HeatMachine
    ) -> None:
        m.feed(1000.0, "user_a", 1.0)
        snap = m.peek(1300.0)
        assert snap.velocity < 0


class TestParticipantPruning:
    """参与人过期清理。"""

    def test_stale_participants_pruned(self, m: HeatMachine, cfg: MachineConfig) -> None:
        m.feed(1000.0, "user_a", 1.0)
        # 超过 5 × half_life = 1500s 后该参与人应被清理
        far_future = 1000.0 + 5.0 * cfg.half_life + 1.0
        m.feed(far_future, "user_b", 1.0)
        snap = m.peek(far_future)
        # user_a 已过期，只剩 user_b
        assert snap.n_participants == 1


class TestMultiplierEffect:
    """外部 multiplier 影响增量。"""

    def test_low_multiplier_needs_more_messages(self, m: HeatMachine) -> None:
        """multiplier=0.5 时需要更多消息才能激活。"""
        m.feed(1000.0, "user_a", 0.5)
        m.feed(1000.0, "user_a", 0.5)
        # heat = 1.0 / 1 = 1.0 < 2.0
        assert m.state is State.IDLE

        m.feed(1000.0, "user_a", 0.5)
        m.feed(1000.0, "user_a", 0.5)
        # heat = 2.0 / 1 = 2.0 ≥ 2.0
        assert m.state is State.ACTIVE


class TestMachineConfigValidation:
    def test_negative_half_life_raises(self) -> None:
        with pytest.raises(ValueError, match="half_life"):
            MachineConfig(half_life=-1.0)

    def test_deactivate_ge_activate_raises(self) -> None:
        with pytest.raises(ValueError, match="deactivate_threshold"):
            MachineConfig(activate_threshold=2.0, deactivate_threshold=2.0)

    def test_default_config_is_valid(self) -> None:
        cfg = MachineConfig()
        assert cfg.half_life > 0
        assert cfg.deactivate_threshold < cfg.activate_threshold
