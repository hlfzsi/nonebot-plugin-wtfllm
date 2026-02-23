"""llm/tools/tool_group/core_memory.py 单元测试"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from nonebot_plugin_wtfllm.llm.tools.tool_group.core_memory import (
    append_core_memory as _append_wrapped,
    update_core_memory as _update_wrapped,
    delete_core_memory as _delete_wrapped,
)

import nonebot_plugin_wtfllm.llm.deps as _deps

_append = _append_wrapped.__wrapped__
_update = _update_wrapped.__wrapped__
_delete = _delete_wrapped.__wrapped__

AgentDeps = _deps.AgentDeps
IDs = _deps.IDs


def _make_ctx(group_id=None):
    from nonebot_plugin_wtfllm.memory import MemoryContextBuilder

    mock_ctx_builder = MagicMock(spec=MemoryContextBuilder)
    mock_ctx_builder.ctx = MagicMock()
    deps = AgentDeps(
        ids=IDs(user_id="u1", group_id=group_id, agent_id="a1"),
        context=mock_ctx_builder,
        active_tool_groups={"CoreMemory"},
    )
    ctx = MagicMock()
    ctx.deps = deps
    return ctx


MODULE = "nonebot_plugin_wtfllm.llm.tools.tool_group.core_memory"


class TestAppendCoreMemory:
    @pytest.mark.asyncio
    @patch(f"{MODULE}.schedule_compress")
    @patch(f"{MODULE}.core_memory_repo")
    @patch(f"{MODULE}.count_tokens", return_value=10)
    async def test_append_private(self, mock_tokens, mock_repo, mock_compress):
        mock_repo.save_core_memory = AsyncMock()
        ctx = _make_ctx()
        result = await _append(ctx, content="用户喜欢猫")
        assert "已记录" in result
        assert "10" in result
        mock_repo.save_core_memory.assert_called_once()
        mock_compress.assert_called_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.schedule_compress")
    @patch(f"{MODULE}.core_memory_repo")
    @patch(f"{MODULE}.count_tokens", return_value=5)
    async def test_append_group(self, mock_tokens, mock_repo, mock_compress):
        mock_repo.save_core_memory = AsyncMock()
        ctx = _make_ctx(group_id="g1")
        result = await _append(ctx, content="群规")
        assert "已记录" in result
        mock_compress.assert_called_once_with(agent_id="a1", group_id="g1", user_id=None)


class TestUpdateCoreMemory:
    @pytest.mark.asyncio
    @patch(f"{MODULE}.schedule_compress")
    @patch(f"{MODULE}.core_memory_repo")
    @patch(f"{MODULE}.count_tokens", return_value=8)
    async def test_update_found(self, mock_tokens, mock_repo, mock_compress):
        mock_repo.save_core_memory = AsyncMock()
        ctx = _make_ctx()
        mock_memory = MagicMock()
        mock_memory.storage_id = "cm_1"
        ctx.deps.context.resolve_core_memory_ref = MagicMock(return_value=mock_memory)

        result = await _update(ctx, memory_ref="CM:1", new_content="更新内容")
        assert "已更新" in result
        assert mock_memory.content == "更新内容"

    @pytest.mark.asyncio
    async def test_update_not_found(self):
        ctx = _make_ctx()
        ctx.deps.context.resolve_core_memory_ref = MagicMock(return_value=None)
        result = await _update(ctx, memory_ref="CM:99", new_content="x")
        assert "错误" in result
        assert "CM:99" in result


class TestDeleteCoreMemory:
    @pytest.mark.asyncio
    @patch(f"{MODULE}.core_memory_repo")
    async def test_delete_found(self, mock_repo):
        mock_repo.delete_by_storage_ids = AsyncMock()
        ctx = _make_ctx()
        mock_memory = MagicMock()
        mock_memory.storage_id = "cm_del"
        ctx.deps.context.resolve_core_memory_ref = MagicMock(return_value=mock_memory)

        result = await _delete(ctx, memory_ref="CM:1")
        assert "已删除" in result
        mock_repo.delete_by_storage_ids.assert_called_once_with(["cm_del"])

    @pytest.mark.asyncio
    async def test_delete_not_found(self):
        ctx = _make_ctx()
        ctx.deps.context.resolve_core_memory_ref = MagicMock(return_value=None)
        result = await _delete(ctx, memory_ref="CM:404")
        assert "错误" in result
