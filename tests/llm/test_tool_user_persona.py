"""llm/tools/tool_group/user_persona.py 单元测试"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from nonebot_plugin_wtfllm.llm.tools.tool_group.user_persona import (
    get_user_persona as _get_wrapped,
    update_user_persona as _update_wrapped,
)

import nonebot_plugin_wtfllm.llm.deps as _deps

_get = _get_wrapped.__wrapped__
_update = _update_wrapped.__wrapped__

AgentDeps = _deps.AgentDeps
IDs = _deps.IDs


def _make_ctx(group_id=None):
    from nonebot_plugin_wtfllm.memory import MemoryContextBuilder

    mock_ctx_builder = MagicMock(spec=MemoryContextBuilder)
    deps = AgentDeps(
        ids=IDs(user_id="u1", group_id=group_id, agent_id="a1"),
        context=mock_ctx_builder,
        active_tool_groups={"UserPersona"},
    )
    ctx = MagicMock()
    ctx.deps = deps
    return ctx


MODULE = "nonebot_plugin_wtfllm.llm.tools.tool_group.user_persona"


class TestGetUserPersona:
    @pytest.mark.asyncio
    @patch(f"{MODULE}.user_persona_repo")
    async def test_has_persona(self, mock_repo):
        mock_repo.get_persona_text = AsyncMock(return_value="用户: Alice\n整体印象: 技术宅")
        ctx = _make_ctx(group_id="g1")
        result = await _get(ctx, user_id="u2")
        assert "Alice" in result

    @pytest.mark.asyncio
    @patch(f"{MODULE}.user_persona_repo")
    async def test_no_persona(self, mock_repo):
        mock_repo.get_persona_text = AsyncMock(return_value=None)
        ctx = _make_ctx()
        result = await _get(ctx, user_id="u_ghost")
        assert "暂无" in result


class TestUpdateUserPersona:
    @pytest.mark.asyncio
    @patch(f"{MODULE}.user_persona_repo")
    async def test_update_with_fields(self, mock_repo):
        mock_persona = MagicMock()
        mock_persona.render_to_llm.return_value = "用户: u2\n整体印象: 技术宅"
        mock_repo.update_persona = AsyncMock(return_value=mock_persona)
        ctx = _make_ctx()
        ctx.deps.context.resolve_aliases = MagicMock(return_value="real_u2")

        result = await _update(
            ctx, user_id="u2", impression="技术宅", note="喜欢猫"
        )
        assert "技术宅" in result
        mock_repo.update_persona.assert_called_once()
        call_kwargs = mock_repo.update_persona.call_args
        assert call_kwargs.kwargs["user_id"] == "real_u2"
        assert call_kwargs.kwargs["impression"] == "技术宅"
        assert call_kwargs.kwargs["note"] == "喜欢猫"

    @pytest.mark.asyncio
    @patch(f"{MODULE}.user_persona_repo")
    async def test_update_fallback_when_render_none(self, mock_repo):
        mock_persona = MagicMock()
        mock_persona.render_to_llm.return_value = None
        mock_repo.update_persona = AsyncMock(return_value=mock_persona)
        ctx = _make_ctx()
        ctx.deps.context.resolve_aliases = MagicMock(return_value=None)

        result = await _update(ctx, user_id="u_new")
        assert "暂无" in result

    @pytest.mark.asyncio
    @patch(f"{MODULE}.user_persona_repo")
    async def test_alias_not_resolved(self, mock_repo):
        mock_persona = MagicMock()
        mock_persona.render_to_llm.return_value = "ok"
        mock_repo.update_persona = AsyncMock(return_value=mock_persona)
        ctx = _make_ctx()
        ctx.deps.context.resolve_aliases = MagicMock(return_value=None)

        await _update(ctx, user_id="original_id")
        call_kwargs = mock_repo.update_persona.call_args
        assert call_kwargs.kwargs["user_id"] == "original_id"
