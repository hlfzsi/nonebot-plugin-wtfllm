"""llm/tools/tool_group/knowledge_base.py 单元测试"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from nonebot_plugin_wtfllm.llm.tools.tool_group.knowledge_base import (
    add_knowledge as _add_wrapped,
    update_knowledge as _update_wrapped,
    delete_knowledge as _delete_wrapped,
)

import nonebot_plugin_wtfllm.llm.deps as _deps

_add = _add_wrapped.__wrapped__
_update = _update_wrapped.__wrapped__
_delete = _delete_wrapped.__wrapped__

AgentDeps = _deps.AgentDeps
IDs = _deps.IDs


def _make_ctx(group_id=None):
    from nonebot_plugin_wtfllm.memory import MemoryContextBuilder

    mock_ctx_builder = MagicMock(spec=MemoryContextBuilder)
    deps = AgentDeps(
        ids=IDs(user_id="u1", group_id=group_id, agent_id="a1"),
        context=mock_ctx_builder,
        active_tool_groups={"KnowledgeBase"},
    )
    ctx = MagicMock()
    ctx.deps = deps
    return ctx


MODULE = "nonebot_plugin_wtfllm.llm.tools.tool_group.knowledge_base"


class TestAddKnowledge:
    @pytest.mark.asyncio
    @patch(f"{MODULE}.knowledge_base_repo")
    @patch(f"{MODULE}.count_tokens", return_value=15)
    async def test_add_private(self, mock_tokens, mock_repo):
        mock_repo.save_knowledge = AsyncMock()
        ctx = _make_ctx()
        result = await _add(ctx, title="React Hooks", content="React Hooks是...")
        assert "已记录" in result
        assert "React Hooks" in result
        mock_repo.save_knowledge.assert_called_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.knowledge_base_repo")
    @patch(f"{MODULE}.count_tokens", return_value=10)
    async def test_add_group(self, mock_tokens, mock_repo):
        mock_repo.save_knowledge = AsyncMock()
        ctx = _make_ctx(group_id="g1")
        result = await _add(ctx, title="量子纠缠", content="量子纠缠是...", category="物理")
        assert "已记录" in result
        assert "物理" in result

    @pytest.mark.asyncio
    @patch(f"{MODULE}.knowledge_base_repo")
    @patch(f"{MODULE}.count_tokens", return_value=5)
    async def test_add_with_tags(self, mock_tokens, mock_repo):
        mock_repo.save_knowledge = AsyncMock()
        ctx = _make_ctx()
        result = await _add(ctx, title="test", content="c", tags=["a", "b"])
        assert "已记录" in result


class TestUpdateKnowledge:
    @pytest.mark.asyncio
    @patch(f"{MODULE}.knowledge_base_repo")
    @patch(f"{MODULE}.count_tokens", return_value=12)
    async def test_update_found(self, mock_tokens, mock_repo):
        mock_repo.save_knowledge = AsyncMock()
        ctx = _make_ctx()
        mock_entry = MagicMock()
        mock_entry.storage_id = "kb_1"
        ctx.deps.context.resolve_knowledge_ref = MagicMock(return_value=mock_entry)

        result = await _update(ctx, knowledge_ref="KB:1", new_content="更新知识")
        assert "已更新" in result
        assert mock_entry.content == "更新知识"

    @pytest.mark.asyncio
    @patch(f"{MODULE}.knowledge_base_repo")
    @patch(f"{MODULE}.count_tokens", return_value=12)
    async def test_update_with_new_title(self, mock_tokens, mock_repo):
        mock_repo.save_knowledge = AsyncMock()
        ctx = _make_ctx()
        mock_entry = MagicMock()
        ctx.deps.context.resolve_knowledge_ref = MagicMock(return_value=mock_entry)

        await _update(ctx, knowledge_ref="KB:1", new_content="c", new_title="新标题")
        assert mock_entry.title == "新标题"

    @pytest.mark.asyncio
    async def test_update_not_found(self):
        ctx = _make_ctx()
        ctx.deps.context.resolve_knowledge_ref = MagicMock(return_value=None)
        result = await _update(ctx, knowledge_ref="KB:99", new_content="x")
        assert "错误" in result


class TestDeleteKnowledge:
    @pytest.mark.asyncio
    @patch(f"{MODULE}.knowledge_base_repo")
    async def test_delete_found(self, mock_repo):
        mock_repo.delete_knowledge = AsyncMock()
        ctx = _make_ctx()
        mock_entry = MagicMock()
        mock_entry.storage_id = "kb_del"
        ctx.deps.context.resolve_knowledge_ref = MagicMock(return_value=mock_entry)

        result = await _delete(ctx, knowledge_ref="KB:1")
        assert "已删除" in result
        mock_repo.delete_knowledge.assert_called_once_with("kb_del")

    @pytest.mark.asyncio
    async def test_delete_not_found(self):
        ctx = _make_ctx()
        ctx.deps.context.resolve_knowledge_ref = MagicMock(return_value=None)
        result = await _delete(ctx, knowledge_ref="KB:404")
        assert "错误" in result
