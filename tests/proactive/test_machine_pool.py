"""MachinePool 集成测试。

验证多会话隔离、全局回调分发、IDLE 自动回收。
"""

import asyncio
from collections.abc import Iterator
from time import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from nonebot_plugin_wtfllm.proactive.states.heat._types import (
    MachineConfig,
    State,
)
from nonebot_plugin_wtfllm.proactive.states.heat.pool import MachinePool
from nonebot_plugin_wtfllm.proactive.states.heat.utils import SessionKey


def _make_item(
    sender: str,
    created_at: float | None = None,
    agent_id: str = "agent_1",
    group_id: str | None = None,
) -> MagicMock:
    item = MagicMock()
    item.sender = sender
    item.created_at = created_at if created_at is not None else time()
    item.agent_id = agent_id
    item.group_id = group_id
    item.is_from_agent = sender == agent_id
    return item


@pytest.fixture
def cfg() -> MachineConfig:
    return MachineConfig(
        half_life=300.0,
        activate_threshold=2.0,
        deactivate_threshold=0.5,
        idle_timeout=5.0,
    )


@pytest.fixture
def pool(cfg: MachineConfig) -> Iterator[MachinePool]:
    p = MachinePool(cfg, max_sessions=128)
    yield p
    p.dispose()


class TestPoolFeed:
    @pytest.mark.asyncio
    async def test_feed_creates_controller_and_triggers_callback(
        self, pool: MachinePool
    ) -> None:
        callback = AsyncMock()
        pool.on_transition(callback)

        key = SessionKey(agent_id="a1", user_id="u1", group_id="g1")
        now = time()
        pool.feed(_make_item("u1", now, agent_id="a1", group_id="g1"))
        pool.feed(_make_item("u1", now, agent_id="a1", group_id="g1"))

        await asyncio.sleep(0.05)

        assert callback.call_count == 1
        call_key, event = callback.call_args[0]
        assert call_key == key
        assert event.new_state is State.ACTIVE

    @pytest.mark.asyncio
    async def test_get_state_returns_correct_state(
        self, pool: MachinePool
    ) -> None:
        assert pool.get_state("a1", "u1", "g1") is State.IDLE

        now = time()
        pool.feed(_make_item("u1", now, agent_id="a1", group_id="g1"))
        pool.feed(_make_item("u1", now, agent_id="a1", group_id="g1"))

        await asyncio.sleep(0.01)
        assert pool.get_state("a1", "u1", "g1") is State.ACTIVE


class TestMultiSessionIsolation:
    @pytest.mark.asyncio
    async def test_different_sessions_are_independent(
        self, pool: MachinePool
    ) -> None:
        callback = AsyncMock()
        pool.on_transition(callback)

        now = time()
        # 只激活 g1
        pool.feed(_make_item("u1", now, agent_id="a1", group_id="g1"))
        pool.feed(_make_item("u1", now, agent_id="a1", group_id="g1"))

        # g2 只收到 1 条
        pool.feed(_make_item("u2", now, agent_id="a1", group_id="g2"))

        await asyncio.sleep(0.05)

        assert pool.get_state("a1", "u1", "g1") is State.ACTIVE
        assert pool.get_state("a1", "u2", "g2") is State.IDLE

    @pytest.mark.asyncio
    async def test_private_and_group_sessions_isolated(
        self, pool: MachinePool
    ) -> None:
        now = time()
        pool.feed(_make_item("u1", now, agent_id="a1", group_id="g1"))
        pool.feed(_make_item("u1", now, agent_id="a1", group_id="g1"))

        await asyncio.sleep(0.01)

        assert pool.get_state("a1", "u1", "g1") is State.ACTIVE
        assert pool.get_state("a1", "u1", None) is State.IDLE


class TestIdleReclamation:
    @pytest.mark.asyncio
    async def test_idle_transition_removes_controller(self) -> None:
        """INACTIVE → IDLE 后 Controller 应从池中移除。"""
        short_cfg = MachineConfig(
            half_life=0.03,
            activate_threshold=2.0,
            deactivate_threshold=0.5,
            idle_timeout=0.1,
        )
        pool = MachinePool(short_cfg, max_sessions=128)
        callback = AsyncMock()
        pool.on_transition(callback)

        key = SessionKey(agent_id="a1", user_id="u1", group_id="g1")

        try:
            now = time()
            pool.feed(_make_item("u1", now, agent_id="a1", group_id="g1"))
            pool.feed(_make_item("u1", now, agent_id="a1", group_id="g1"))
            await asyncio.sleep(0.01)
            assert pool.get_state("a1", "u1", "g1") is State.ACTIVE

            # 等待 ACTIVE → INACTIVE → IDLE
            await asyncio.sleep(1.0)

            assert pool.get_state("a1", "u1", "g1") is State.IDLE
            # Controller 应已被移除
            assert key.available_id not in pool._controllers
        finally:
            pool.dispose()


class TestPoolSnapshot:
    @pytest.mark.asyncio
    async def test_snapshot_for_unknown_session(self, pool: MachinePool) -> None:
        snap = pool.get_snapshot("a1", "unknown", None)
        assert snap.state is State.IDLE
        assert snap.heat == 0.0

    @pytest.mark.asyncio
    async def test_snapshot_reflects_feed(self, pool: MachinePool) -> None:
        pool.feed(_make_item("u1", agent_id="a1", group_id="g1"))
        snap = pool.get_snapshot("a1", "u1", "g1")
        assert snap.msg_ema > 0
        assert snap.n_participants == 1


class TestPoolDispose:
    @pytest.mark.asyncio
    async def test_dispose_clears_all(self, pool: MachinePool) -> None:
        now = time()
        pool.feed(_make_item("u1", now, agent_id="a1", group_id="g1"))
        pool.feed(_make_item("u1", now, agent_id="a1", group_id="g1"))

        await asyncio.sleep(0.01)
        pool.dispose()
        assert len(pool._controllers) == 0


class TestMultipleCallbacks:
    @pytest.mark.asyncio
    async def test_all_callbacks_receive_events(
        self, pool: MachinePool
    ) -> None:
        cb1 = AsyncMock()
        cb2 = AsyncMock()
        pool.on_transition(cb1)
        pool.on_transition(cb2)

        now = time()
        pool.feed(_make_item("u1", now, agent_id="a1", group_id="g1"))
        pool.feed(_make_item("u1", now, agent_id="a1", group_id="g1"))

        await asyncio.sleep(0.05)

        assert cb1.call_count == 1
        assert cb2.call_count == 1
