"""llm/tools/tool_group/core_memory.py 单元测试"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nonebot_plugin_wtfllm.llm.tools.tool_group.core_memory import (
    memory_crud as _memory_crud_wrapped,
    MemoryCrudRequest,
    AppendMemoryRequest,
    UpdateMemoryRequest,
    DeleteMemoryRequest,
    CoreMemoryAppendPayload,
    CoreMemoryUpdatePayload,
    NoteAppendPayload,
    NoteUpdatePayload,
)

import nonebot_plugin_wtfllm.llm.deps as _deps

_memory_crud = _memory_crud_wrapped.__wrapped__

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
        request = MemoryCrudRequest(
            action=AppendMemoryRequest(
                payload=CoreMemoryAppendPayload(content="用户喜欢猫")
            )
        )
        result = await _memory_crud(ctx, request=request)
        assert "已记录" in result
        mock_repo.save_core_memory.assert_called_once()
        mock_compress.assert_called_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.schedule_compress")
    @patch(f"{MODULE}.core_memory_repo")
    @patch(f"{MODULE}.count_tokens", return_value=5)
    async def test_append_group(self, mock_tokens, mock_repo, mock_compress):
        mock_repo.save_core_memory = AsyncMock()
        ctx = _make_ctx(group_id="g1")
        request = MemoryCrudRequest(
            action=AppendMemoryRequest(payload=CoreMemoryAppendPayload(content="群规"))
        )
        result = await _memory_crud(ctx, request=request)
        assert "已记录" in result
        mock_compress.assert_called_once_with(agent_id="a1", group_id="g1", user_id=None)


class TestAppendNote:
    @pytest.mark.asyncio
    @patch(f"{MODULE}.note_memory_repo")
    @patch(f"{MODULE}.count_tokens", return_value=8)
    @patch(f"{MODULE}.time.time", return_value=1_700_000_000)
    async def test_append_note(self, mock_time, mock_tokens, mock_repo):
        mock_repo.save_note = AsyncMock()
        ctx = _make_ctx()
        request = MemoryCrudRequest(
            action=AppendMemoryRequest(
                payload=NoteAppendPayload(
                    content="30分钟后提醒用户处理作业",
                    duration_minutes=30,
                )
            )
        )
        result = await _memory_crud(ctx, request=request)
        assert "短期备忘已记录" in result
        mock_repo.save_note.assert_called_once()
        saved_note = mock_repo.save_note.await_args.args[0]
        assert saved_note.expires_at == 1_700_001_800


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

        request = MemoryCrudRequest(
            action=UpdateMemoryRequest(
                target_ref="CM:1",
                payload=CoreMemoryUpdatePayload(new_content="更新内容"),
            )
        )
        result = await _memory_crud(ctx, request=request)
        assert "已更新" in result
        assert mock_memory.content == "更新内容"

    @pytest.mark.asyncio
    async def test_update_not_found(self):
        ctx = _make_ctx()
        ctx.deps.context.resolve_core_memory_ref = MagicMock(return_value=None)
        request = MemoryCrudRequest(
            action=UpdateMemoryRequest(
                target_ref="CM:99",
                payload=CoreMemoryUpdatePayload(new_content="x"),
            )
        )
        result = await _memory_crud(ctx, request=request)
        assert "错误" in result
        assert "CM:99" in result

    @pytest.mark.asyncio
    @patch(f"{MODULE}.note_memory_repo")
    @patch(f"{MODULE}.count_tokens", return_value=7)
    @patch(f"{MODULE}.time.time", return_value=1_700_000_000)
    async def test_update_note_found(self, mock_time, mock_tokens, mock_repo):
        mock_repo.save_note = AsyncMock()
        ctx = _make_ctx()
        mock_note = MagicMock()
        mock_note.storage_id = "nt_1"
        mock_note.token_count = 4
        ctx.deps.context.resolve_note_ref = MagicMock(return_value=mock_note)

        request = MemoryCrudRequest(
            action=UpdateMemoryRequest(
                target_ref="NT:1",
                payload=NoteUpdatePayload(
                    new_content="新的备忘内容", new_duration_minutes=45
                ),
            )
        )
        result = await _memory_crud(ctx, request=request)
        assert "短期备忘已更新" in result
        mock_repo.save_note.assert_called_once()
        assert mock_note.expires_at == 1_700_002_700


class TestDeleteCoreMemory:
    @pytest.mark.asyncio
    @patch(f"{MODULE}.core_memory_repo")
    async def test_delete_found(self, mock_repo):
        mock_repo.delete_by_storage_ids = AsyncMock()
        ctx = _make_ctx()
        mock_memory = MagicMock()
        mock_memory.storage_id = "cm_del"
        ctx.deps.context.resolve_core_memory_ref = MagicMock(return_value=mock_memory)

        request = MemoryCrudRequest(
            action=DeleteMemoryRequest(memory_kind="core_memory", target_ref="CM:1")
        )
        result = await _memory_crud(ctx, request=request)
        assert "已删除" in result
        mock_repo.delete_by_storage_ids.assert_called_once_with(["cm_del"])

    @pytest.mark.asyncio
    async def test_delete_not_found(self):
        ctx = _make_ctx()
        ctx.deps.context.resolve_core_memory_ref = MagicMock(return_value=None)
        request = MemoryCrudRequest(
            action=DeleteMemoryRequest(memory_kind="core_memory", target_ref="CM:404")
        )
        result = await _memory_crud(ctx, request=request)
        assert "错误" in result
