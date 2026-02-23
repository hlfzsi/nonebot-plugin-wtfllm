"""llm/tools/tool_group/chat.py 单元测试"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from nonebot_plugin_wtfllm.llm.tools.tool_group.chat import (
    build_uni_from_metions_and_reply,
    send as _send_wrapped,
    ask as _ask_wrapped,
)

import nonebot_plugin_wtfllm.llm.deps as _deps

AgentDeps = _deps.AgentDeps
IDs = _deps.IDs

_send = _send_wrapped.__wrapped__
_ask = _ask_wrapped.__wrapped__

MODULE = "nonebot_plugin_wtfllm.llm.tools.tool_group.chat"


def _make_ctx(user_id="u1", group_id=None, with_runtime=True):
    from nonebot_plugin_wtfllm.memory import MemoryContextBuilder

    mock_ctx_builder = MagicMock(spec=MemoryContextBuilder)
    deps = AgentDeps(
        ids=IDs(user_id=user_id, group_id=group_id, agent_id="a1"),
        context=mock_ctx_builder,
        active_tool_groups={"Chat"},
    )
    if with_runtime:
        deps.nb_runtime = MagicMock()
    ctx = MagicMock()
    ctx.deps = deps
    return ctx


# ===================== build_uni_from_metions_and_reply =====================


class TestBuildUniFromMentionsAndReply:
    @patch(f"{MODULE}.UniMessage")
    def test_plain_content_only(self, MockUniMsg):
        ctx = _make_ctx()
        mock_msg = MagicMock()
        MockUniMsg.return_value = mock_msg

        result = build_uni_from_metions_and_reply(ctx, "hello")
        mock_msg.text.assert_called_once_with("hello")
        mock_msg.at.assert_not_called()

    @patch(f"{MODULE}.UniMessage")
    def test_with_mentions_resolves_aliases(self, MockUniMsg):
        ctx = _make_ctx()
        ctx.deps.context.resolve_aliases = MagicMock(side_effect=lambda x: f"resolved_{x}")
        mock_msg = MagicMock()
        MockUniMsg.return_value = mock_msg

        build_uni_from_metions_and_reply(ctx, "hi", mentions=["alice"])
        ctx.deps.context.resolve_aliases.assert_called_once_with("alice")
        mock_msg.at.assert_called_once_with(user_id="resolved_alice")

    @patch(f"{MODULE}.UniMessage")
    def test_with_reply_resolves_memory_ref(self, MockUniMsg):
        ctx = _make_ctx()
        mock_ref = MagicMock()
        mock_ref.message_id = "msg_123"
        ctx.deps.context.resolve_memory_ref = MagicMock(return_value=mock_ref)
        mock_msg = MagicMock()
        MockUniMsg.return_value = mock_msg

        build_uni_from_metions_and_reply(ctx, "reply", reply_to="5")
        ctx.deps.context.resolve_memory_ref.assert_called_once_with(5)
        mock_msg.reply.assert_called_once_with(id="msg_123")

    @patch(f"{MODULE}.UniMessage")
    def test_mention_alias_not_found(self, MockUniMsg):
        ctx = _make_ctx()
        ctx.deps.context.resolve_aliases = MagicMock(return_value=None)
        mock_msg = MagicMock()
        MockUniMsg.return_value = mock_msg

        build_uni_from_metions_and_reply(ctx, "hi", mentions=["unknown"])
        mock_msg.at.assert_not_called()


# ===================== send 测试 =====================


class TestSend:
    @pytest.mark.asyncio
    async def test_requires_nb_runtime(self):
        ctx = _make_ctx(with_runtime=False)
        ctx.deps.nb_runtime = None
        with pytest.raises(ValueError, match="NonebotRuntime"):
            await _send(ctx, message="msg")

    @pytest.mark.asyncio
    async def test_requires_user_id(self):
        ctx = _make_ctx(user_id=None)
        with pytest.raises(ValueError, match="User ID"):
            await _send(ctx, message="msg")

    @pytest.mark.asyncio
    @patch(f"{MODULE}.msg_tracker")
    @patch(f"{MODULE}.convert_and_store_item", new_callable=AsyncMock)
    @patch(f"{MODULE}.ensure_msgid_from_receipt", return_value="sent_id")
    @patch(f"{MODULE}.reschedule_deadline")
    @patch(f"{MODULE}.build_uni_from_metions_and_reply")
    async def test_sends_message_and_stores(
        self, mock_build, mock_resched, mock_ensure, mock_convert, mock_tracker
    ):
        ctx = _make_ctx()
        mock_msg = MagicMock()
        mock_msg.send = AsyncMock(return_value=MagicMock())
        mock_build.return_value = mock_msg

        result = await _send(ctx, message="test msg")
        assert "已发送" in result
        mock_msg.send.assert_called_once()
        mock_convert.assert_called_once()
        mock_tracker.track.assert_called_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.msg_tracker")
    @patch(f"{MODULE}.convert_and_store_item", new_callable=AsyncMock)
    @patch(f"{MODULE}.ensure_msgid_from_receipt", return_value="sent_id")
    @patch(f"{MODULE}.reschedule_deadline")
    @patch(f"{MODULE}.build_uni_from_metions_and_reply")
    async def test_reschedules_deadline(
        self, mock_build, mock_resched, mock_ensure, mock_convert, mock_tracker
    ):
        ctx = _make_ctx()
        mock_msg = MagicMock()
        mock_msg.send = AsyncMock(return_value=MagicMock())
        mock_build.return_value = mock_msg

        await _send(ctx, message="hi", added_timeout=120.0)
        mock_resched.assert_called_once()
        # delay = min(4.0, max(0.0, len("hi") * 0.1)) = 0.2
        call_args = mock_resched.call_args
        assert call_args[0][1] == pytest.approx(0.2 + 120.0, abs=0.1)


# ===================== ask 测试 =====================


class TestAsk:
    @pytest.mark.asyncio
    async def test_requires_nb_runtime(self):
        ctx = _make_ctx(with_runtime=False)
        ctx.deps.nb_runtime = None
        with pytest.raises(ValueError, match="NonebotRuntime"):
            await _ask(ctx, question="q?")

    @pytest.mark.asyncio
    async def test_requires_user_id(self):
        ctx = _make_ctx(user_id=None)
        with pytest.raises(ValueError, match="User ID"):
            await _ask(ctx, question="q?")

    @pytest.mark.asyncio
    @patch(f"{MODULE}.MemoryItemStream")
    @patch(f"{MODULE}.extract_memoryitem_from_unimsg")
    @patch(f"{MODULE}.msg_tracker")
    @patch(f"{MODULE}.convert_and_store_item", new_callable=AsyncMock)
    @patch(f"{MODULE}.ensure_msgid_from_receipt", return_value="sent_id")
    @patch(f"{MODULE}.reschedule_deadline")
    @patch(f"{MODULE}.waiter")
    @patch(f"{MODULE}.build_uni_from_metions_and_reply")
    async def test_timeout_returns_no_reply(
        self, mock_build, mock_waiter, mock_resched, mock_ensure,
        mock_convert, mock_tracker, mock_extract, mock_stream,
    ):
        ctx = _make_ctx()
        mock_msg = MagicMock()
        mock_msg.send = AsyncMock(return_value=MagicMock())
        mock_build.return_value = mock_msg

        # waiter 返回的 wait 对象返回 None（超时）
        mock_wait_obj = MagicMock()
        mock_wait_obj.wait = AsyncMock(return_value=None)
        mock_waiter.return_value = MagicMock(return_value=mock_wait_obj)

        result = await _ask(ctx, question="你在吗？")
        assert result.return_value == "用户未回复"

    @pytest.mark.asyncio
    @patch(f"{MODULE}.MemoryItemStream")
    @patch(f"{MODULE}.extract_memoryitem_from_unimsg", return_value=MagicMock())
    @patch(f"{MODULE}.msg_tracker")
    @patch(f"{MODULE}.convert_and_store_item", new_callable=AsyncMock)
    @patch(f"{MODULE}.ensure_msgid_from_receipt", return_value="sent_id")
    @patch(f"{MODULE}.reschedule_deadline")
    @patch(f"{MODULE}.waiter")
    @patch(f"{MODULE}.build_uni_from_metions_and_reply")
    async def test_receives_reply_and_returns_prompt(
        self, mock_build, mock_waiter, mock_resched, mock_ensure,
        mock_convert, mock_tracker, mock_extract, mock_stream,
    ):
        ctx = _make_ctx()
        mock_msg = MagicMock()
        mock_msg.send = AsyncMock(return_value=MagicMock())
        mock_build.return_value = mock_msg

        # waiter 返回 (uni_msg, msg_id) 元组
        reply_msg = MagicMock()
        mock_wait_obj = MagicMock()
        mock_wait_obj.wait = AsyncMock(return_value=(reply_msg, "reply_id"))
        mock_waiter.return_value = MagicMock(return_value=mock_wait_obj)

        # context.copy 返回新 builder
        new_builder = MagicMock()
        new_builder.to_prompt.return_value = "new prompt"
        ctx.deps.context.copy = MagicMock(return_value=new_builder)

        result = await _ask(ctx, question="你好？")
        assert result.return_value == "已收到回复"
        assert result.content == "new prompt"

    @pytest.mark.asyncio
    @patch(f"{MODULE}.MemoryItemStream")
    @patch(f"{MODULE}.extract_memoryitem_from_unimsg")
    @patch(f"{MODULE}.msg_tracker")
    @patch(f"{MODULE}.convert_and_store_item", new_callable=AsyncMock)
    @patch(f"{MODULE}.ensure_msgid_from_receipt", return_value="sent_id")
    @patch(f"{MODULE}.reschedule_deadline")
    @patch(f"{MODULE}.waiter")
    @patch(f"{MODULE}.build_uni_from_metions_and_reply")
    async def test_reschedules_deadline(
        self, mock_build, mock_waiter, mock_resched, mock_ensure,
        mock_convert, mock_tracker, mock_extract, mock_stream,
    ):
        ctx = _make_ctx()
        mock_msg = MagicMock()
        mock_msg.send = AsyncMock(return_value=MagicMock())
        mock_build.return_value = mock_msg

        mock_wait_obj = MagicMock()
        mock_wait_obj.wait = AsyncMock(return_value=None)
        mock_waiter.return_value = MagicMock(return_value=mock_wait_obj)

        await _ask(ctx, question="q?", timeout=45.0, added_timeout=90.0)
        mock_resched.assert_called_once_with(ctx, 45.0 + 90.0)
