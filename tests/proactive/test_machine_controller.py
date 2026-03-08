"""MachineController 异步单元测试。

验证 MemoryItem → HeatMachine 转译、定时器调度、回调触发。
"""

import asyncio
from time import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from nonebot_plugin_wtfllm.proactive.states.heat._types import (
    MachineConfig,
    State,
    TransitionEvent,
)
from nonebot_plugin_wtfllm.proactive.states.heat.controller import MachineController


def _make_item(
    sender: str,
    created_at: float | None = None,
    agent_id: str = "agent_1",
) -> MagicMock:
    """构造最小 MemoryItem mock。"""
    item = MagicMock()
    item.sender = sender
    item.created_at = created_at if created_at is not None else time()
    item.agent_id = agent_id
    item.is_from_agent = sender == agent_id
    return item


@pytest.fixture
def cfg() -> MachineConfig:
    return MachineConfig(
        half_life=300.0,
        activate_threshold=2.0,
        deactivate_threshold=0.5,
        idle_timeout=5.0,  # 短超时便于测试
    )


class TestControllerFeed:
    @pytest.mark.asyncio
    async def test_feed_triggers_callback_on_activation(
        self, cfg: MachineConfig
    ) -> None:
        callback = AsyncMock()
        ctrl = MachineController(cfg, on_transition=callback)

        try:
            now = time()
            item1 = _make_item("user_a", now)
            item2 = _make_item("user_a", now)
            ctrl.feed(item1)
            ctrl.feed(item2)

            # 让 event loop 处理 create_task
            await asyncio.sleep(0.01)

            assert callback.call_count == 1
            event: TransitionEvent = callback.call_args[0][0]
            assert event.prev_state is State.IDLE
            assert event.new_state is State.ACTIVE
        finally:
            ctrl.dispose()

    @pytest.mark.asyncio
    async def test_agent_messages_are_skipped(self, cfg: MachineConfig) -> None:
        callback = AsyncMock()
        ctrl = MachineController(cfg, on_transition=callback)

        try:
            item = _make_item("agent_1", agent_id="agent_1")
            ctrl.feed(item)
            ctrl.feed(item)
            ctrl.feed(item)

            await asyncio.sleep(0.01)
            callback.assert_not_called()
        finally:
            ctrl.dispose()

    @pytest.mark.asyncio
    async def test_multiplier_affects_activation(
        self, cfg: MachineConfig
    ) -> None:
        callback = AsyncMock()
        ctrl = MachineController(cfg, on_transition=callback)

        try:
            now = time()
            item1 = _make_item("user_a", now)
            item2 = _make_item("user_a", now)
            # 低 multiplier → 不够激活
            ctrl.feed(item1, multiplier=0.5)
            ctrl.feed(item2, multiplier=0.5)

            await asyncio.sleep(0.01)
            callback.assert_not_called()
            assert ctrl.get_state() is State.IDLE
        finally:
            ctrl.dispose()


class TestControllerTimer:
    @pytest.mark.asyncio
    async def test_timer_fires_deactivation(self) -> None:
        """使用极短半衰期验证定时器自动触发 INACTIVE。"""
        short_cfg = MachineConfig(
            half_life=0.05,  # 50ms 半衰期
            activate_threshold=2.0,
            deactivate_threshold=0.5,
            idle_timeout=0.2,
        )
        callback = AsyncMock()
        ctrl = MachineController(short_cfg, on_transition=callback)

        try:
            now = time()
            item1 = _make_item("user_a", now)
            item2 = _make_item("user_a", now)
            ctrl.feed(item1)
            ctrl.feed(item2)

            await asyncio.sleep(0.01)
            assert callback.call_count == 1
            assert ctrl.get_state() is State.ACTIVE

            # 等定时器触发衰减 → INACTIVE
            await asyncio.sleep(0.5)
            assert callback.call_count >= 2
            # 检查最后一个回调是否是 INACTIVE
            last_event = callback.call_args_list[-1][0][0]
            assert last_event.new_state in (State.INACTIVE, State.IDLE)
        finally:
            ctrl.dispose()

    @pytest.mark.asyncio
    async def test_dispose_cancels_timer(self, cfg: MachineConfig) -> None:
        callback = AsyncMock()
        ctrl = MachineController(cfg, on_transition=callback)

        now = time()
        item1 = _make_item("user_a", now)
        item2 = _make_item("user_a", now)
        ctrl.feed(item1)
        ctrl.feed(item2)

        await asyncio.sleep(0.01)
        ctrl.dispose()
        # timer_handle 应已取消
        assert ctrl._timer_handle is None


class TestControllerPeek:
    @pytest.mark.asyncio
    async def test_peek_returns_snapshot(self, cfg: MachineConfig) -> None:
        callback = AsyncMock()
        ctrl = MachineController(cfg, on_transition=callback)

        try:
            item = _make_item("user_a")
            ctrl.feed(item)
            snap = ctrl.peek()
            assert snap.state is State.IDLE
            assert snap.msg_ema > 0
        finally:
            ctrl.dispose()
