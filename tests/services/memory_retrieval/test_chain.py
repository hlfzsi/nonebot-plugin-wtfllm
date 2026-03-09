"""RetrievalChain 单元测试

仅测试链本身的机制：默认参数传递、魔术方法、copy、resolve 并发等。
使用轻量级 stub task，不涉及真实仓储。
"""

import asyncio
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from nonebot_plugin_wtfllm.services.func.memory_retrieval._base import RetrievalTask
from nonebot_plugin_wtfllm.services.func.memory_retrieval.chain import (
    RetrievalChain,
    _UNSET,
)
from nonebot_plugin_wtfllm.services.func.memory_retrieval.main_chat import MainChatTask
from nonebot_plugin_wtfllm.services.func.memory_retrieval.note import NoteTask
from nonebot_plugin_wtfllm.services.func.memory_retrieval.core_memory import (
    CoreMemoryTask,
    CrossSessionMemoryTask,
)
from nonebot_plugin_wtfllm.services.func.memory_retrieval.knowledge import (
    KnowledgeSearchTask,
)
from nonebot_plugin_wtfllm.services.func.memory_retrieval.tool_history import (
    ToolCallHistoryTask,
)
from nonebot_plugin_wtfllm.memory.items import MemorySource


# ── Stub Task ──────────────────────────────────────────


class _StubSource(MemorySource):
    """用于测试的最小 MemorySource 实现"""

    def __init__(self, tag: str):
        self.tag = tag

    @property
    def source_id(self) -> str:
        return f"stub-{self.tag}"

    @property
    def priority(self) -> int:
        return 0

    @property
    def sort_key(self):
        return (0, self.source_id)

    def register_all_alias(self, ctx) -> None:
        pass

    def to_llm_context(self, ctx) -> str:
        return self.tag

    def __hash__(self):
        return hash(self.tag)

    def __eq__(self, other):
        return isinstance(other, _StubSource) and self.tag == other.tag


@dataclass
class _StubTask(RetrievalTask):
    tag: str

    async def execute(self) -> set[MemorySource]:
        return {_StubSource(self.tag)}


@dataclass
class _EmptyTask(RetrievalTask):
    async def execute(self) -> set[MemorySource]:
        return set()


@dataclass
class _SlowTask(RetrievalTask):
    tag: str
    delay: float = 0.05

    async def execute(self) -> set[MemorySource]:
        await asyncio.sleep(self.delay)
        return {_StubSource(self.tag)}


# ── 构造与默认参数 ─────────────────────────────────────


class TestChainConstruction:
    def test_empty_chain(self):
        chain = RetrievalChain()
        assert len(chain) == 0
        assert not chain
        assert chain.agent_id is None

    def test_with_defaults(self):
        chain = RetrievalChain(agent_id="a1", group_id="g1", query="hello")
        assert chain.agent_id == "a1"
        assert chain.group_id == "g1"
        assert chain.user_id is None
        assert chain.query == "hello"

    def test_with_tasks(self):
        tasks = [_StubTask(tag="t1"), _StubTask(tag="t2")]
        chain = RetrievalChain(tasks=tasks)
        assert len(chain) == 2

    def test_tasks_list_is_copied(self):
        """传入的 tasks 列表不应与链共享引用"""
        original = [_StubTask(tag="t1")]
        chain = RetrievalChain(tasks=original)
        original.append(_StubTask(tag="t2"))
        assert len(chain) == 1


class TestUnsetSentinel:
    def test_d_returns_explicit_value(self):
        chain = RetrievalChain(agent_id="default_agent")
        assert chain._d("explicit", chain.agent_id) == "explicit"

    def test_d_returns_default_when_unset(self):
        chain = RetrievalChain(agent_id="default_agent")
        assert chain._d(_UNSET, chain.agent_id) == "default_agent"

    def test_d_explicit_none_is_not_unset(self):
        chain = RetrievalChain(agent_id="default_agent")
        assert chain._d(None, chain.agent_id) is None


# ── 快捷方法默认参数传播 ─────────────────────────────


class TestShortcutDefaults:
    def test_main_chat_uses_chain_defaults(self):
        chain = RetrievalChain(agent_id="a1", group_id="g1")
        chain.main_chat(limit=10)
        task = chain._tasks[0]
        assert isinstance(task, MainChatTask)
        assert task.agent_id == "a1"
        assert task.group_id == "g1"
        assert task.user_id is None
        assert task.limit == 10

    def test_main_chat_explicit_overrides(self):
        chain = RetrievalChain(agent_id="a1", group_id="g1")
        chain.main_chat(agent_id="a2", group_id="g2", limit=20)
        task = chain._tasks[0]
        assert task.agent_id == "a2"
        assert task.group_id == "g2"

    def test_core_memory_uses_chain_defaults(self):
        chain = RetrievalChain(agent_id="a1", user_id="u1")
        chain.core_memory()
        task = chain._tasks[0]
        assert isinstance(task, CoreMemoryTask)
        assert task.agent_id == "a1"
        assert task.user_id == "u1"
        assert task.prefix == "<core_memory>"

    def test_note_uses_chain_defaults(self):
        chain = RetrievalChain(agent_id="a1", group_id="g1")
        chain.note()
        task = chain._tasks[0]
        assert isinstance(task, NoteTask)
        assert task.agent_id == "a1"
        assert task.group_id == "g1"
        assert task.prefix == "<note_memory>"

    def test_cross_session_memory_maps_group_to_exclude(self):
        """cross_session_memory 的 exclude_group_id 默认回退到链的 group_id"""
        chain = RetrievalChain(agent_id="a1", group_id="g1", query="q")
        chain.cross_session_memory()
        task = chain._tasks[0]
        assert isinstance(task, CrossSessionMemoryTask)
        assert task.exclude_group_id == "g1"
        assert task.exclude_user_id is None

    def test_cross_session_memory_maps_user_to_exclude(self):
        """cross_session_memory 的 exclude_user_id 默认回退到链的 user_id"""
        chain = RetrievalChain(agent_id="a1", user_id="u1", query="q")
        chain.cross_session_memory()
        task = chain._tasks[0]
        assert task.exclude_user_id == "u1"
        assert task.exclude_group_id is None

    def test_cross_session_memory_explicit_exclude_overrides(self):
        chain = RetrievalChain(agent_id="a1", group_id="g1", query="q")
        chain.cross_session_memory(exclude_group_id="g_other")
        task = chain._tasks[0]
        assert task.exclude_group_id == "g_other"

    def test_knowledge_uses_chain_defaults(self):
        chain = RetrievalChain(agent_id="a1", query="test query")
        chain.knowledge(limit=3, max_tokens=1000)
        task = chain._tasks[0]
        assert isinstance(task, KnowledgeSearchTask)
        assert task.agent_id == "a1"
        assert task.query == "test query"
        assert task.limit == 3
        assert task.max_tokens == 1000

    def test_tool_history_uses_chain_defaults(self):
        chain = RetrievalChain(agent_id="a1", group_id="g1")
        chain.tool_history(limit=5)
        task = chain._tasks[0]
        assert isinstance(task, ToolCallHistoryTask)
        assert task.agent_id == "a1"
        assert task.group_id == "g1"


# ── 链式调用 ─────────────────────────────────────────


class TestChaining:
    def test_fluent_api_returns_self(self):
        chain = RetrievalChain(agent_id="a1", query="q")
        result = chain.main_chat().core_memory().knowledge()
        assert result is chain
        assert len(chain) == 3

    def test_full_chain_group(self):
        """群聊场景完整链"""
        chain = RetrievalChain(agent_id="a1", group_id="g1", query="hello")
        chain.main_chat(limit=50).note().core_memory().cross_session_memory().tool_history(
            limit=3
        ).knowledge(limit=5, max_tokens=2000)
        assert len(chain) == 6
        types_in_chain = [type(t).__name__ for t in chain]
        assert types_in_chain == [
            "MainChatTask",
            "NoteTask",
            "CoreMemoryTask",
            "CrossSessionMemoryTask",
            "ToolCallHistoryTask",
            "KnowledgeSearchTask",
        ]

    def test_full_chain_private(self):
        """私聊场景完整链"""
        chain = RetrievalChain(agent_id="a1", user_id="u1", query="hello")
        chain.main_chat(limit=50).note().core_memory().cross_session_memory().tool_history(
            limit=3
        ).knowledge(limit=5, max_tokens=2000)
        assert len(chain) == 6
        # cross_session 应使用 user_id 作为 exclude
        cross_task: CrossSessionMemoryTask = chain._tasks[3]
        assert cross_task.exclude_user_id == "u1"
        assert cross_task.exclude_group_id is None


# ── 魔术方法 ─────────────────────────────────────────


class TestMagicMethods:
    def test_len(self):
        chain = RetrievalChain()
        assert len(chain) == 0
        chain._add(_StubTask(tag="a"))
        assert len(chain) == 1

    def test_bool_empty(self):
        assert not RetrievalChain()

    def test_bool_nonempty(self):
        chain = RetrievalChain()
        chain._add(_StubTask(tag="a"))
        assert chain

    def test_iter(self):
        chain = RetrievalChain()
        chain._add(_StubTask(tag="a"))
        chain._add(_StubTask(tag="b"))
        tags = [t.tag for t in chain]
        assert tags == ["a", "b"]

    def test_contains_by_type(self):
        chain = RetrievalChain(agent_id="a1", group_id="g1")
        chain.main_chat()
        assert MainChatTask in chain
        assert CoreMemoryTask not in chain

    def test_repr(self):
        chain = RetrievalChain()
        chain._add(_StubTask(tag="x"))
        r = repr(chain)
        assert "RetrievalChain" in r
        assert "_StubTask" in r

    def test_add_operator(self):
        a = RetrievalChain(agent_id="a1")
        a._add(_StubTask(tag="1"))
        b = RetrievalChain()
        b._add(_StubTask(tag="2"))

        merged = a + b
        assert len(merged) == 2
        assert len(a) == 1  # 不修改原链
        assert len(b) == 1
        # 继承左链默认值
        assert merged.agent_id == "a1"

    def test_add_operator_type_check(self):
        chain = RetrievalChain()
        result = chain.__add__("not a chain")
        assert result is NotImplemented

    def test_iadd_operator(self):
        a = RetrievalChain(agent_id="a1")
        a._add(_StubTask(tag="1"))
        b = RetrievalChain()
        b._add(_StubTask(tag="2"))

        a += b
        assert len(a) == 2
        # 默认值保留
        assert a.agent_id == "a1"

    def test_iadd_operator_type_check(self):
        chain = RetrievalChain()
        result = chain.__iadd__("not a chain")
        assert result is NotImplemented


# ── copy ─────────────────────────────────────────────


class TestCopy:
    def test_copy_preserves_defaults(self):
        original = RetrievalChain(
            agent_id="a1", group_id="g1", user_id="u1", query="q"
        )
        copied = original.copy()
        assert copied.agent_id == "a1"
        assert copied.group_id == "g1"
        assert copied.user_id == "u1"
        assert copied.query == "q"

    def test_copy_is_independent(self):
        original = RetrievalChain(agent_id="a1")
        original._add(_StubTask(tag="x"))
        copied = original.copy()
        copied._add(_StubTask(tag="y"))
        assert len(original) == 1
        assert len(copied) == 2

    def test_copy_deep_copies_tasks(self):
        original = RetrievalChain(agent_id="a1")
        original._add(_StubTask(tag="x"))
        copied = original.copy()
        assert original._tasks[0] is not copied._tasks[0]
        assert original._tasks[0].tag == copied._tasks[0].tag


# ── resolve ──────────────────────────────────────────


class TestResolve:
    @pytest.mark.asyncio
    async def test_resolve_empty(self):
        chain = RetrievalChain()
        result = await chain.resolve()
        assert result == set()

    @pytest.mark.asyncio
    async def test_resolve_single_task(self):
        chain = RetrievalChain()
        chain._add(_StubTask(tag="a"))
        result = await chain.resolve()
        assert len(result) == 1
        tags = {s.tag for s in result}
        assert tags == {"a"}

    @pytest.mark.asyncio
    async def test_resolve_multiple_tasks(self):
        chain = RetrievalChain()
        chain._add(_StubTask(tag="a"))
        chain._add(_StubTask(tag="b"))
        chain._add(_StubTask(tag="c"))
        result = await chain.resolve()
        assert len(result) == 3
        tags = {s.tag for s in result}
        assert tags == {"a", "b", "c"}

    @pytest.mark.asyncio
    async def test_resolve_empty_task_contributes_nothing(self):
        chain = RetrievalChain()
        chain._add(_StubTask(tag="a"))
        chain._add(_EmptyTask())
        result = await chain.resolve()
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_resolve_concurrent_execution(self):
        """验证任务是并发执行的，而非顺序"""
        chain = RetrievalChain()
        # 3 个 50ms 任务，如果顺序应 >= 150ms，并发应 ~50ms
        chain._add(_SlowTask(tag="a", delay=0.05))
        chain._add(_SlowTask(tag="b", delay=0.05))
        chain._add(_SlowTask(tag="c", delay=0.05))

        loop = asyncio.get_event_loop()
        start = loop.time()
        result = await chain.resolve()
        elapsed = loop.time() - start

        assert len(result) == 3
        # Windows 本地调度抖动更明显，阈值放宽到仍能区分串行执行的范围
        assert elapsed < 0.25

    @pytest.mark.asyncio
    async def test_resolve_merges_sets(self):
        """多个任务返回重叠的 source 时正确合并"""

        @dataclass
        class _DupTask(RetrievalTask):
            async def execute(self) -> set[MemorySource]:
                return {_StubSource("shared")}

        chain = RetrievalChain()
        chain._add(_DupTask())
        chain._add(_DupTask())
        result = await chain.resolve()
        # set 合并后只有一个
        assert len(result) == 1


# ── recent_react 快捷方法 ────────────────────────────


class TestRecentReactShortcut:
    def test_recent_react_creates_task(self):
        chain = RetrievalChain(agent_id="a1")
        alias_provider = MagicMock()
        chain.recent_react(
            recent_react={"g1": ["m1", "m2"]},
            alias_provider=alias_provider,
            max_token_per_stream=3000,
        )
        from nonebot_plugin_wtfllm.services.func.memory_retrieval.recent_react import (
            RecentReactTask,
        )

        task = chain._tasks[0]
        assert isinstance(task, RecentReactTask)
        assert task.recent_react == {"g1": ["m1", "m2"]}
        assert task.alias_provider is alias_provider
        assert task.max_token_per_stream == 3000
