import asyncio
from time import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from nonebot_plugin_wtfllm.proactive.states.heat import MachineConfig, MachinePool, State
from nonebot_plugin_wtfllm.proactive.states.silence import (
    is_silence_transition,
    silence_observer,
)
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


class TestSilenceTransitionPredicate:
    @pytest.mark.asyncio
    async def test_decorator_ignores_non_silence_transition(self) -> None:
        callback = AsyncMock()
        cfg = MachineConfig()
        pool = MachinePool(cfg)
        observer = silence_observer(pool)(callback)
        events: list[tuple[SessionKey, object]] = []

        async def record(key: SessionKey, event: object) -> None:
            events.append((key, event))

        pool.on_transition(record)
        try:
            now = time()
            pool.feed(_make_item("u1", now, agent_id="a1", group_id="g1"))
            pool.feed(_make_item("u1", now, agent_id="a1", group_id="g1"))
            await asyncio.sleep(0.05)
            assert events

            key, event = events[0]
            assert is_silence_transition(event) is False
            await observer(key, event)
            callback.assert_not_awaited()
        finally:
            pool.dispose()


class TestObserveSilence:
    @pytest.mark.asyncio
    async def test_decorator_triggers_on_silence_transition(self) -> None:
        callback = AsyncMock()
        cfg = MachineConfig(
            half_life=0.03,
            activate_threshold=2.0,
            deactivate_threshold=0.5,
            idle_timeout=0.1,
        )
        pool = MachinePool(cfg, max_sessions=64)
        observer = silence_observer(pool)(callback)
        events: list[tuple[SessionKey, object]] = []

        async def record(key: SessionKey, event: object) -> None:
            events.append((key, event))

        pool.on_transition(record)
        try:
            now = time()
            pool.feed(_make_item("u1", now, agent_id="a1", group_id="g1"))
            pool.feed(_make_item("u1", now, agent_id="a1", group_id="g1"))
            await asyncio.sleep(1.0)

            assert events
            key, event = events[-1]
            assert is_silence_transition(event) is True
            callback.assert_awaited_once_with(key, event)

            callback.reset_mock()
            await observer(key, event)
            callback.assert_awaited_once_with(key, event)
        finally:
            pool.dispose()

    @pytest.mark.asyncio
    async def test_silence_observer_registers_to_pool(self) -> None:
        cfg = MachineConfig(
            half_life=0.03,
            activate_threshold=2.0,
            deactivate_threshold=0.5,
            idle_timeout=0.1,
        )
        pool = MachinePool(cfg, max_sessions=64)
        callback = AsyncMock()
        silence_observer(pool)(callback)

        try:
            now = time()
            pool.feed(_make_item("u1", now, agent_id="a1", group_id="g1"))
            pool.feed(_make_item("u1", now, agent_id="a1", group_id="g1"))

            await asyncio.sleep(1.0)

            callback.assert_awaited_once()
            key, event = callback.await_args.args
            assert key == SessionKey(agent_id="a1", user_id="u1", group_id="g1")
            assert event.prev_state is State.INACTIVE
            assert event.new_state is State.IDLE
            assert is_silence_transition(event) is True
        finally:
            pool.dispose()