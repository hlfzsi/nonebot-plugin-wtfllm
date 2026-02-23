"""abilities/core_memory_compressor.py 单元测试

覆盖:
- schedule_compress  调度逻辑与防重入
- _do_compress       压缩流程与边界条件
- _compress_memories 解析 LLM 输出、related_entities 过滤
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

MODULE = "nonebot_plugin_wtfllm.abilities.core_memory_compressor"

# ---------------------------------------------------------------------------
# 导入被测模块（conftest 已完成 nonebot mock，可安全导入）
# ---------------------------------------------------------------------------
from nonebot_plugin_wtfllm.abilities.core_memory_compressor import (
    schedule_compress,
    _do_compress,
    _compress_memories,
    _compressing_sessions,
)
import nonebot_plugin_wtfllm.abilities.core_memory_compressor as _mod


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_memory(
    storage_id: str = "cm_1",
    content: str = "some memory",
    token_count: int = 50,
    updated_at: int = 1000,
    related_entities: list | None = None,
    agent_id: str = "agent1",
    group_id: str | None = "g1",
    user_id: str | None = None,
) -> MagicMock:
    """构造一个类似 CoreMemory 的 mock 对象"""
    mem = MagicMock()
    mem.storage_id = storage_id
    mem.content = content
    mem.token_count = token_count
    mem.updated_at = updated_at
    mem.related_entities = related_entities if related_entities is not None else []
    mem.agent_id = agent_id
    mem.group_id = group_id
    mem.user_id = user_id
    return mem


# ===========================================================================
# TestScheduleCompress
# ===========================================================================


class TestScheduleCompress:
    """schedule_compress 调度逻辑"""

    @pytest.fixture(autouse=True)
    def _clean_sessions(self):
        """每个测试前后清空 _compressing_sessions"""
        _compressing_sessions.clear()
        yield
        _compressing_sessions.clear()

    def test_schedules_task(self):
        """首次调用应创建 asyncio task 并将 session_key 加入集合"""
        mock_task = MagicMock()
        with patch(f"{MODULE}.asyncio.create_task", return_value=mock_task) as ct:
            schedule_compress("agent1", "g1", None)

        ct.assert_called_once()
        # 关闭未 await 的协程，避免 RuntimeWarning
        ct.call_args[0][0].close()
        mock_task.add_done_callback.assert_called_once()
        assert ("agent1", "g1", None) in _compressing_sessions

    def test_skips_if_already_compressing(self):
        """session_key 已在集合中时应跳过，不创建新 task"""
        _compressing_sessions.add(("agent1", "g1", None))

        with patch(f"{MODULE}.asyncio.create_task") as ct:
            schedule_compress("agent1", "g1", None)

        ct.assert_not_called()


# ===========================================================================
# TestDoCompress
# ===========================================================================


class TestDoCompress:
    """_do_compress 后台压缩流程"""

    @pytest.fixture(autouse=True)
    def _clean_globals(self):
        """每个测试前后清空全局状态"""
        _compressing_sessions.clear()
        _mod._compress_agent = None
        yield
        _compressing_sessions.clear()
        _mod._compress_agent = None

    async def test_under_limit_returns_early(self):
        """总 token 数 <= max_tokens 时应直接返回，不调用压缩"""
        memories = [
            _make_memory(storage_id="cm_1", token_count=50, updated_at=1000),
            _make_memory(storage_id="cm_2", token_count=50, updated_at=2000),
        ]
        # 总计 100，max_tokens 默认 2048，不会超限

        with (
            patch(
                f"{MODULE}.core_memory_repo.get_by_session",
                new_callable=AsyncMock,
                return_value=memories,
            ),
            patch(f"{MODULE}._compress_memories", new_callable=AsyncMock) as cm,
        ):
            await _do_compress("agent1", "g1", None)

        cm.assert_not_called()

    async def test_only_one_memory_to_compress_returns_early(self):
        """超限但仅有 1 条可压缩时应直接返回"""
        # 制造场景：总 token 超限，但只有一条旧记忆可压缩
        # max_tokens=2048, compress_ratio=0.6 => target=1228
        # 需要 total > 2048
        memories = [
            _make_memory(storage_id="cm_1", token_count=900, updated_at=1000),
            _make_memory(storage_id="cm_2", token_count=1200, updated_at=2000),
        ]
        # total=2100 > 2048, target=1228
        # sorted by updated_at: cm_1(900), cm_2(1200)
        # 遍历: cm_1 => compress_tokens=900, total-900=1200 <= 1228 => break
        # to_compress=[cm_1] => len < 2 => return

        with (
            patch(
                f"{MODULE}.core_memory_repo.get_by_session",
                new_callable=AsyncMock,
                return_value=memories,
            ),
            patch(f"{MODULE}._compress_memories", new_callable=AsyncMock) as cm,
        ):
            await _do_compress("agent1", "g1", None)

        cm.assert_not_called()

    async def test_compresses_and_saves(self):
        """超限且有 >= 2 条可压缩时应调用 _compress_memories、delete、save"""
        memories = [
            _make_memory(storage_id="cm_1", token_count=800, updated_at=1000),
            _make_memory(storage_id="cm_2", token_count=800, updated_at=2000),
            _make_memory(storage_id="cm_3", token_count=800, updated_at=3000),
        ]
        # total=2400 > 2048, target=1228
        # sorted: cm_1, cm_2, cm_3
        # cm_1: compress=800,  remaining=1600 > 1228 => add
        # cm_2: compress=1600, remaining=800  <= 1228 => break
        # to_compress = [cm_1, cm_2] (len=2 >= 2)

        compressed_result = [_make_memory(storage_id="cm_new", token_count=100)]

        mock_delete = AsyncMock()
        mock_save = AsyncMock()

        with (
            patch(
                f"{MODULE}.core_memory_repo.get_by_session",
                new_callable=AsyncMock,
                return_value=memories,
            ),
            patch(
                f"{MODULE}._compress_memories",
                new_callable=AsyncMock,
                return_value=compressed_result,
            ) as cm,
            patch(
                f"{MODULE}.core_memory_repo.delete_by_storage_ids",
                mock_delete,
            ),
            patch(
                f"{MODULE}.core_memory_repo.save_many_core_memories",
                mock_save,
            ),
        ):
            await _do_compress("agent1", "g1", None)

        cm.assert_called_once()
        # 被压缩的是 cm_1, cm_2
        compressed_ids = cm.call_args[0][0]
        assert [m.storage_id for m in compressed_ids] == ["cm_1", "cm_2"]

        mock_delete.assert_awaited_once_with(["cm_1", "cm_2"])
        mock_save.assert_awaited_once_with(compressed_result)

    async def test_handles_exception(self):
        """get_by_session 抛出 ValueError 时不应崩溃"""
        with patch(
            f"{MODULE}.core_memory_repo.get_by_session",
            new_callable=AsyncMock,
            side_effect=ValueError("test error"),
        ):
            # 不应抛出异常
            await _do_compress("agent1", "g1", None)


# ===========================================================================
# TestCompressMemories
# ===========================================================================


class TestCompressMemories:
    """_compress_memories LLM 输出解析 & entity 过滤"""

    @pytest.fixture(autouse=True)
    def _clean_agent(self):
        """每个测试前后重置全局 _compress_agent"""
        _mod._compress_agent = None
        yield
        _mod._compress_agent = None

    def _patch_agent(self, output_text: str):
        """返回一个 context manager，将 _get_compress_agent 替换为返回
        output_text 的 mock agent"""
        mock_result = MagicMock()
        mock_result.output = output_text

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)

        return patch(
            f"{MODULE}._get_compress_agent",
            return_value=mock_agent,
        )

    async def test_parses_agent_output(self):
        """LLM 返回多行文本时应拆分为多个 CoreMemory 对象"""
        from nonebot_plugin_wtfllm.memory.items.core_memory import CoreMemory

        source = [
            _make_memory(
                storage_id="cm_1",
                content="old memory 1",
                related_entities=[],
            ),
            _make_memory(
                storage_id="cm_2",
                content="old memory 2",
                related_entities=[],
            ),
        ]

        with self._patch_agent("line one\nline two\n"):
            result = await _compress_memories(source, "agent1", "g1", None)

        assert len(result) == 2
        assert isinstance(result[0], CoreMemory)
        assert isinstance(result[1], CoreMemory)
        assert result[0].content == "line one"
        assert result[1].content == "line two"
        # 验证元数据
        assert result[0].agent_id == "agent1"
        assert result[0].group_id == "g1"
        assert result[0].user_id is None
        assert result[0].source == "compression"

    async def test_preserves_known_entities(self):
        """压缩结果中仅保留源记忆已有的 entity_id，过滤未知实体"""
        source = [
            _make_memory(
                storage_id="cm_1",
                content="{{user_123}} likes cats",
                related_entities=["user_123"],
            ),
            _make_memory(
                storage_id="cm_2",
                content="{{user_456}} likes dogs",
                related_entities=["user_456"],
            ),
        ]

        # LLM 输出同时包含已知 user_123 和未知 unknown
        with self._patch_agent("{{user_123}} and {{unknown}} are friends"):
            result = await _compress_memories(source, "agent1", "g1", None)

        assert len(result) == 1
        assert "user_123" in result[0].related_entities
        assert "unknown" not in result[0].related_entities
