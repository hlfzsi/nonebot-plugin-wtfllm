"""llm/response_models.py 单元测试

覆盖 TextResponse, MarkdownResponse, RejectResponse 
模型创建send 行为（使mock NonebotRuntime）
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

# 直接导入子模块避llm.__init__ 的循环导入问题
import nonebot_plugin_wtfllm.llm.response_models as _rm
import nonebot_plugin_wtfllm.llm.deps as _deps

TextResponse = _rm.TextResponse
MarkdownResponse = _rm.MarkdownResponse
RejectResponse = _rm.RejectResponse
CHAT_OUTPUT = _rm.CHAT_OUTPUT

AgentDeps = _deps.AgentDeps
IDs = _deps.IDs
NonebotRuntime = _deps.NonebotRuntime


# ===================== 辅助构建测试数据 =====================


def _make_deps(with_runtime: bool = True, user_id: str = "u1") -> AgentDeps:
    """创建 mock AgentDeps"""
    from nonebot_plugin_wtfllm.memory import MemoryContextBuilder

    mock_ctx = MagicMock(spec=MemoryContextBuilder)
    mock_ctx.ctx = MagicMock()
    mock_ctx.ctx.alias_provider = MagicMock()
    mock_ctx.ctx.alias_provider.resolve_alias = MagicMock(return_value=user_id)
    mock_ctx.resolve_aliases = MagicMock(return_value=user_id)
    mock_ctx.resolve_media_ref = MagicMock(side_effect=ValueError("no media"))

    nb_runtime = None
    if with_runtime:
        nb_runtime = MagicMock(spec=NonebotRuntime)
        nb_runtime.bot = MagicMock()
        nb_runtime.session = MagicMock()
        nb_runtime.target = MagicMock()

    deps = AgentDeps(
        ids=IDs(user_id=user_id, group_id="g1", agent_id="a1"),
        context=mock_ctx,
        nb_runtime=nb_runtime,
    )
    return deps


# ===================== 模型创建测试 =====================


class TestTextResponseModel:
    """TextResponse 模型测试"""

    def test_create_basic(self):
        resp = TextResponse(thought_of_chain="ok", responses=["Hello!"], meme=None, interested_topics=None)
        assert resp.responses == ["Hello!"]
        assert resp.meme is None
        assert resp.mentions == []
        assert resp.interested_topics is None

    def test_create_with_mentions(self):
        resp = TextResponse(
            thought_of_chain="ok",
            responses=["Hi there"],
            mentions=["user_a", "user_b"],
            meme=None,
            interested_topics=None,
        )
        assert len(resp.mentions) == 2

    def test_create_with_meme(self):
        resp = TextResponse(thought_of_chain="ok", responses=["look"], meme="uuid-123", interested_topics=None)
        assert resp.meme == "uuid-123"


class TestMarkdownResponseModel:
    """MarkdownResponse 模型测试"""

    def test_create(self):
        resp = MarkdownResponse(
            thought_of_chain="ok",
            markdown_content="# Title\n\nContent",
            summary="Title content summary",
            interested_topics=None,
        )
        assert "# Title" in resp.markdown_content
        assert resp.summary == "Title content summary"


class TestRejectResponseModel:
    """RejectResponse 模型测试"""

    def test_create_basic(self):
        resp = RejectResponse(thought_of_chain="ok", reason="not relevant", interested_topics=None)
        assert resp.reason == "not relevant"
        assert resp.message_to_user is None

    def test_create_show_user(self):
        resp = RejectResponse(
            thought_of_chain="ok",
            reason="off-topic",
            message_to_user="这不是我能回答的",
            interested_topics=None,
        )
        assert resp.message_to_user == "这不是我能回答的"


# ===================== Output 类型元组测试 =====================


class TestOutputTypes:
    """CHAT_OUTPUT 类型定义测试"""

    def test_chat_output_types(self):
        assert TextResponse in CHAT_OUTPUT
        assert MarkdownResponse in CHAT_OUTPUT
        assert RejectResponse in CHAT_OUTPUT


# ===================== send 行为测试 =====================


class TestTextResponseSend:
    """TextResponse.send 测试"""

    @pytest.mark.asyncio
    async def test_send_requires_runtime(self):
        deps = _make_deps(with_runtime=False)
        resp = TextResponse(thought_of_chain="ok", responses=["test"], meme=None, interested_topics=None)
        with pytest.raises(ValueError, match="NonebotRuntime"):
            await resp.send(deps)

    @pytest.mark.asyncio
    async def test_send_requires_user_id(self):
        deps = _make_deps(user_id=None)
        resp = TextResponse(thought_of_chain="ok", responses=["test"], meme=None, interested_topics=None)
        with pytest.raises(ValueError, match="User ID"):
            await resp.send(deps)

    @pytest.mark.asyncio
    @patch(
        "nonebot_plugin_wtfllm.llm.response_models.store_message_with_context",
        new_callable=AsyncMock,
    )
    async def test_send_basic(self, mock_store):
        deps = _make_deps()

        # Mock UniMessage().send()
        mock_receipt = MagicMock()
        mock_receipt.get_reply = MagicMock(return_value=MagicMock(id="sent_1"))

        with patch(
            "nonebot_plugin_wtfllm.llm.response_models.UniMessage"
        ) as MockUniMsg:
            mock_msg = MagicMock()
            mock_msg.send = AsyncMock(return_value=mock_receipt)
            mock_msg.__iadd__ = MagicMock(return_value=mock_msg)
            MockUniMsg.return_value = mock_msg

            resp = TextResponse(thought_of_chain="ok", responses=["Hello!"], meme=None, interested_topics=None)
            await resp.send(deps)

            mock_store.assert_called_once()


class TestRejectResponseSend:
    """RejectResponse.send 测试"""

    @pytest.mark.asyncio
    async def test_send_silent_reject(self):
        """静默拒绝不发送消息"""
        deps = _make_deps()
        resp = RejectResponse(thought_of_chain="ok", reason="no", interested_topics=None)

        # 不应抛出异常
        with patch(
            "nonebot_plugin_wtfllm.llm.response_models.store_message_with_context",
            new_callable=AsyncMock,
        ) as mock_store:
            await resp.send(deps)
            # 无消息发送时不应调用 store
            mock_store.assert_not_called()

    @pytest.mark.asyncio
    @patch(
        "nonebot_plugin_wtfllm.llm.response_models.store_message_with_context",
        new_callable=AsyncMock,
    )
    async def test_send_visible_reject(self, mock_store):
        deps = _make_deps()

        mock_receipt = MagicMock()
        mock_receipt.get_reply = MagicMock(return_value=MagicMock(id="rej_1"))

        with patch(
            "nonebot_plugin_wtfllm.llm.response_models.UniMessage"
        ) as MockUniMsg:
            mock_msg = MagicMock()
            mock_msg.send = AsyncMock(return_value=mock_receipt)
            mock_msg.__iadd__ = MagicMock(return_value=mock_msg)
            MockUniMsg.return_value = mock_msg

            resp = RejectResponse(
                thought_of_chain="ok",
                reason="off topic",
                message_to_user="不好意思",
                interested_topics=None,
            )
            await resp.send(deps)

            mock_store.assert_called_once()


class TestMarkdownResponseSend:
    """MarkdownResponse.send 测试"""

    @pytest.mark.asyncio
    async def test_send_requires_runtime(self):
        deps = _make_deps(with_runtime=False)
        resp = MarkdownResponse(
            thought_of_chain="ok",
            markdown_content="# Test",
            summary="Test summary",
            interested_topics=None,
        )
        with pytest.raises(ValueError, match="Context information missing"):
            await resp.send(deps)


# ===================== 补充测试：SendableResponse.send 工具链持久化逻辑 =====================

MODULE = "nonebot_plugin_wtfllm.llm.response_models"

ToolCallInfo = _deps.ToolCallInfo


def _make_tool_call_info(**overrides) -> ToolCallInfo:
    """创建一个ToolCallInfo 实例用于测试"""
    defaults = dict(
        run_id="run-1",
        round_index=0,
        tool_name="test_tool",
        kwargs={"arg": "val"},
        timestamp=1700000000,
    )
    defaults.update(overrides)
    return ToolCallInfo(**defaults)


class TestSendableResponseSend:
    """SendableResponse.send() 工具链持久化逻辑测试"""

    @pytest.mark.asyncio
    @patch(f"{MODULE}.store_message_with_context", new_callable=AsyncMock)
    @patch(f"{MODULE}.thought_record_repo")
    @patch(f"{MODULE}.tool_call_record_repo")
    async def test_send_with_tool_chain(
        self, mock_tool_repo, mock_thought_repo, mock_store
    ):
        """tool_chain 非空时调save_batch_from_tool_call_info"""
        deps = _make_deps()
        tc = _make_tool_call_info()
        deps.tool_chain = [tc]

        mock_tool_repo.save_batch_from_tool_call_info = AsyncMock()
        mock_tool_repo.save_empty_record = AsyncMock()
        mock_thought_repo.save_record = AsyncMock()

        with patch(f"{MODULE}.UniMessage") as MockUniMsg:
            mock_msg = MagicMock()
            mock_msg.send = AsyncMock(return_value=MagicMock())
            mock_msg.__iadd__ = MagicMock(return_value=mock_msg)
            MockUniMsg.return_value = mock_msg

            resp = TextResponse(thought_of_chain="ok", responses=["ok"], meme=None, interested_topics=None)
            await resp.send(deps)

        mock_tool_repo.save_batch_from_tool_call_info.assert_called_once_with(
            infos=[tc],
            agent_id="a1",
            group_id="g1",
            user_id="u1",
        )
        mock_tool_repo.save_empty_record.assert_not_called()
        mock_thought_repo.save_record.assert_called_once_with(
            thought_of_chain="ok",
            agent_id="a1",
            group_id="g1",
            user_id="u1",
            run_id="run-1",
        )

    @pytest.mark.asyncio
    @patch(f"{MODULE}.store_message_with_context", new_callable=AsyncMock)
    @patch(f"{MODULE}.thought_record_repo")
    @patch(f"{MODULE}.tool_call_record_repo")
    async def test_send_without_tool_chain(
        self, mock_tool_repo, mock_thought_repo, mock_store
    ):
        """tool_chain 为空时调save_empty_record"""
        deps = _make_deps()
        deps.tool_chain = []

        mock_tool_repo.save_batch_from_tool_call_info = AsyncMock()
        mock_tool_repo.save_empty_record = AsyncMock()
        mock_thought_repo.save_record = AsyncMock()

        with patch(f"{MODULE}.UniMessage") as MockUniMsg:
            mock_msg = MagicMock()
            mock_msg.send = AsyncMock(return_value=MagicMock())
            mock_msg.__iadd__ = MagicMock(return_value=mock_msg)
            MockUniMsg.return_value = mock_msg

            resp = TextResponse(thought_of_chain="ok", responses=["ok"], meme=None, interested_topics=None)
            await resp.send(deps)

        mock_tool_repo.save_empty_record.assert_called_once_with(
            agent_id="a1",
            group_id="g1",
            user_id="u1",
        )
        mock_tool_repo.save_batch_from_tool_call_info.assert_not_called()
        mock_thought_repo.save_record.assert_called_once_with(
            thought_of_chain="ok",
            agent_id="a1",
            group_id="g1",
            user_id="u1",
            run_id=None,
        )

    @pytest.mark.asyncio
    @patch(f"{MODULE}.store_message_with_context", new_callable=AsyncMock)
    @patch(f"{MODULE}.thought_record_repo")
    @patch(f"{MODULE}.tool_call_record_repo")
    async def test_send_tool_chain_error_continues(
        self, mock_tool_repo, mock_thought_repo, mock_store
    ):
        """save_batch 抛出 SQLAlchemyError 时仍继续调用 _perform_send"""
        from sqlalchemy.exc import SQLAlchemyError

        deps = _make_deps()
        deps.tool_chain = [_make_tool_call_info()]

        mock_tool_repo.save_batch_from_tool_call_info = AsyncMock(
            side_effect=SQLAlchemyError("db down")
        )
        mock_thought_repo.save_record = AsyncMock()

        with patch(f"{MODULE}.UniMessage") as MockUniMsg:
            mock_msg = MagicMock()
            mock_msg.send = AsyncMock(return_value=MagicMock())
            mock_msg.__iadd__ = MagicMock(return_value=mock_msg)
            MockUniMsg.return_value = mock_msg

            resp = TextResponse(thought_of_chain="ok", responses=["ok"], meme=None, interested_topics=None)
            # 不应抛出异常，_perform_send 仍应执行
            await resp.send(deps)

        # 验证 _perform_send 仍被执行（store 被调用）
        mock_store.assert_called_once()
        mock_thought_repo.save_record.assert_called_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.store_message_with_context", new_callable=AsyncMock)
    @patch(f"{MODULE}.thought_record_repo")
    @patch(f"{MODULE}.tool_call_record_repo")
    async def test_send_thought_persist_error_continues(
        self, mock_tool_repo, mock_thought_repo, mock_store
    ):
        from sqlalchemy.exc import SQLAlchemyError

        deps = _make_deps()
        mock_tool_repo.save_empty_record = AsyncMock()
        mock_tool_repo.save_batch_from_tool_call_info = AsyncMock()
        mock_thought_repo.save_record = AsyncMock(
            side_effect=SQLAlchemyError("thought repo down")
        )

        with patch(f"{MODULE}.UniMessage") as MockUniMsg:
            mock_msg = MagicMock()
            mock_msg.send = AsyncMock(return_value=MagicMock())
            mock_msg.__iadd__ = MagicMock(return_value=mock_msg)
            MockUniMsg.return_value = mock_msg

            resp = TextResponse(thought_of_chain="ok", responses=["ok"], meme=None, interested_topics=None)
            await resp.send(deps)

        mock_store.assert_called_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.topic_interest_store")
    @patch(f"{MODULE}.store_message_with_context", new_callable=AsyncMock)
    async def test_send_updates_topic_interest_store(
        self, mock_store, mock_topic_interest_store
    ):
        deps = _make_deps()

        with patch(f"{MODULE}.UniMessage") as MockUniMsg:
            mock_msg = MagicMock()
            mock_msg.send = AsyncMock(return_value=MagicMock())
            mock_msg.__iadd__ = MagicMock(return_value=mock_msg)
            MockUniMsg.return_value = mock_msg

            resp = TextResponse(
                thought_of_chain="ok",
                responses=["ok"],
                meme=None,
                interested_topics=["天气", "周末安排"],
            )
            await resp.send(deps)

        mock_topic_interest_store.set_topics.assert_called_once_with(
            agent_id="a1",
            user_id="u1",
            group_id="g1",
            topics=["天气", "周末安排"],
        )

    @pytest.mark.asyncio
    @patch(f"{MODULE}.topic_interest_store")
    @patch(f"{MODULE}.store_message_with_context", new_callable=AsyncMock)
    async def test_send_without_results_still_updates_topic_interest_store(
        self, mock_store, mock_topic_interest_store
    ):
        deps = _make_deps()
        resp = RejectResponse(thought_of_chain="ok", reason="no", interested_topics=["天气"])

        await resp.send(deps)

        mock_store.assert_not_called()
        mock_topic_interest_store.set_topics.assert_called_once_with(
            agent_id="a1",
            user_id="u1",
            group_id="g1",
            topics=["天气"],
        )


# ===================== 补充测试：TextResponse._perform_send =====================


class TestTextResponsePerformSend:
    """TextResponse._perform_send 详细行为测试"""

    @pytest.mark.asyncio
    async def test_no_runtime_raises(self):
        """nb_runtime=None 时抛ValueError"""
        deps = _make_deps(with_runtime=False)
        resp = TextResponse(thought_of_chain="ok", responses=["test"], meme=None, interested_topics=None)
        with pytest.raises(ValueError, match="NonebotRuntime"):
            await resp._perform_send(deps)

    @pytest.mark.asyncio
    async def test_no_user_id_raises(self):
        """user_id=None 时抛ValueError"""
        deps = _make_deps(user_id=None)
        resp = TextResponse(thought_of_chain="ok", responses=["test"], meme=None, interested_topics=None)
        with pytest.raises(ValueError, match="User ID"):
            await resp._perform_send(deps)

    @pytest.mark.asyncio
    @patch(f"{MODULE}.store_message_with_context", new_callable=AsyncMock)
    @patch(f"{MODULE}.ensure_msgid_from_receipt", return_value="msg-001")
    async def test_basic_send(self, mock_ensure, mock_store):
        """基础文本发送：无 meme、无 mentions"""
        deps = _make_deps()

        mock_receipt = MagicMock()
        with patch(f"{MODULE}.UniMessage") as MockUniMsg:
            mock_msg = MagicMock()
            mock_msg.send = AsyncMock(return_value=mock_receipt)
            mock_msg.__iadd__ = MagicMock(return_value=mock_msg)
            MockUniMsg.return_value = mock_msg

            resp = TextResponse(thought_of_chain="ok", responses=["Hello world"], meme=None, interested_topics=None)
            results = await resp._perform_send(deps)

            # 应调text() 而不调用 at()
            mock_msg.text.assert_called_once_with("Hello world")
            mock_msg.at.assert_not_called()
            mock_msg.send.assert_called_once_with(
                target=deps.nb_runtime.target
            )
            assert results == [(mock_msg, mock_receipt, True)]

        mock_ensure.assert_not_called()
        mock_store.assert_not_called()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.store_message_with_context", new_callable=AsyncMock)
    @patch(f"{MODULE}.ensure_msgid_from_receipt", return_value="msg-002")
    async def test_with_mentions(self, mock_ensure, mock_store):
        """mentions 列表非空时调resolve_aliases at()"""
        deps = _make_deps()
        # resolve_aliases 返回解析后的别名
        deps.context.resolve_aliases = MagicMock(
            side_effect=lambda m: f"resolved_{m}"
        )

        mock_receipt = MagicMock()
        with patch(f"{MODULE}.UniMessage") as MockUniMsg:
            mock_msg = MagicMock()
            mock_msg.send = AsyncMock(return_value=mock_receipt)
            mock_msg.__iadd__ = MagicMock(return_value=mock_msg)
            MockUniMsg.return_value = mock_msg

            resp = TextResponse(
                thought_of_chain="ok",
                responses=["Hi!"],
                mentions=["alice", "bob"],
                meme=None,
                interested_topics=None,
            )
            await resp._perform_send(deps)

            # resolve_aliases 应被每个 mention 调用
            assert deps.context.resolve_aliases.call_count == 2
            deps.context.resolve_aliases.assert_any_call("alice")
            deps.context.resolve_aliases.assert_any_call("bob")

            # at() 应被调用两次，参数为解析后的别名
            assert mock_msg.at.call_count == 2
            mock_msg.at.assert_any_call("resolved_alice")
            mock_msg.at.assert_any_call("resolved_bob")

    @pytest.mark.asyncio
    @patch(f"{MODULE}.store_message_with_context", new_callable=AsyncMock)
    @patch(f"{MODULE}.ensure_msgid_from_receipt", return_value="msg-003")
    async def test_with_meme_from_context(
        self, mock_ensure, mock_store
    ):
        """meme 设置后通过 context.resolve_media_ref 获取图片"""
        from nonebot_plugin_wtfllm.memory import ImageSegment

        deps = _make_deps()

        # 构mock ImageSegment
        mock_image_seg = MagicMock(spec=ImageSegment)
        mock_image_seg.available = True
        mock_image_seg.local_path = "/tmp/meme.webp"
        mock_image_seg.url = None
        mock_image_seg.get_bytes_async = AsyncMock(return_value=b"fake-image-bytes")

        deps.context.resolve_media_ref = MagicMock(return_value=mock_image_seg)

        mock_receipt = MagicMock()
        with patch(f"{MODULE}.UniMessage") as MockUniMsg:
            mock_msg = MagicMock()
            mock_msg.send = AsyncMock(return_value=mock_receipt)
            mock_msg.__iadd__ = MagicMock(return_value=mock_msg)
            MockUniMsg.return_value = mock_msg

            resp = TextResponse(
                thought_of_chain="ok",
                responses=["Look at this!"],
                meme="meme-uuid-1",
                interested_topics=None,
            )
            await resp._perform_send(deps)

            # resolve_media_ref 应被调用
            deps.context.resolve_media_ref.assert_called_once_with(
                "meme-uuid-1", ImageSegment
            )

            # 应通过 local_path 路径读取字节并调image(raw=...)
            mock_image_seg.get_bytes_async.assert_called_once()
            mock_msg.image.assert_called_once_with(raw=b"fake-image-bytes")


# ===================== 补充测试：RejectResponse._perform_send =====================


class TestRejectResponsePerformSend:
    """RejectResponse._perform_send 详细行为测试"""

    @pytest.mark.asyncio
    async def test_no_runtime_raises(self):
        """nb_runtime=None 时抛ValueError"""
        deps = _make_deps(with_runtime=False)
        resp = RejectResponse(thought_of_chain="ok", reason="no", interested_topics=None)
        with pytest.raises(ValueError, match="NonebotRuntime"):
            await resp._perform_send(deps)

    @pytest.mark.asyncio
    @patch(f"{MODULE}.store_message_with_context", new_callable=AsyncMock)
    async def test_silent_reject(self, mock_store):
        """message_to_user=None 时不发送消息、不 store"""
        deps = _make_deps()
        resp = RejectResponse(thought_of_chain="ok", reason="not relevant", interested_topics=None)

        await resp._perform_send(deps)

        mock_store.assert_not_called()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.store_message_with_context", new_callable=AsyncMock)
    @patch(f"{MODULE}.ensure_msgid_from_receipt", return_value="rej-msg-001")
    async def test_show_user_reject(
        self, mock_ensure, mock_store
    ):
        """message_to_user 非空时发送消息"""
        deps = _make_deps()

        mock_receipt = MagicMock()
        with patch(f"{MODULE}.UniMessage") as MockUniMsg:
            mock_msg = MagicMock()
            mock_msg.send = AsyncMock(return_value=mock_receipt)
            mock_msg.__iadd__ = MagicMock(return_value=mock_msg)
            MockUniMsg.return_value = mock_msg

            resp = RejectResponse(
                thought_of_chain="ok",
                reason="off topic",
                message_to_user="Sorry, I cannot help with that.",
                interested_topics=None,
            )
            results = await resp._perform_send(deps)

            # 应调text() 并发
            mock_msg.text.assert_called_once_with(
                "Sorry, I cannot help with that."
            )
            mock_msg.send.assert_called_once_with(
                target=deps.nb_runtime.target
            )
            assert results == [(mock_msg, mock_receipt, True)]

        mock_ensure.assert_not_called()
        mock_store.assert_not_called()


# ===================== 补充测试：MarkdownResponse._perform_send =====================


class TestMarkdownResponsePerformSend:
    """MarkdownResponse._perform_send 详细行为测试"""

    @pytest.mark.asyncio
    async def test_no_runtime_raises(self):
        """nb_runtime=None 时抛ValueError"""
        deps = _make_deps(with_runtime=False)
        resp = MarkdownResponse(
            thought_of_chain="ok",
            markdown_content="# Test",
            summary="Test",
            interested_topics=None,
        )
        with pytest.raises(ValueError, match="Context information missing"):
            await resp._perform_send(deps)

    @pytest.mark.asyncio
    @patch(f"{MODULE}.store_message_with_context", new_callable=AsyncMock)
    @patch(f"{MODULE}.ensure_msgid_from_receipt", return_value="md-msg-001")
    @patch(f"{MODULE}.render_markdown_to_image", new_callable=AsyncMock)
    async def test_basic_send(
        self, mock_render, mock_ensure, mock_store
    ):
        """基础 Markdown 渲染并发送图片"""
        deps = _make_deps()
        deps.reply_segments = None

        mock_render.return_value = b"fake-png-bytes"

        mock_receipt = MagicMock()
        with patch(f"{MODULE}.UniMessage") as MockUniMsg, \
             patch(f"{MODULE}.Image") as MockImage:
            mock_msg = MagicMock()
            mock_msg.send = AsyncMock(return_value=mock_receipt)
            mock_msg.__iadd__ = MagicMock(return_value=mock_msg)
            MockUniMsg.return_value = mock_msg

            mock_image_obj = MagicMock()
            MockImage.return_value = mock_image_obj

            resp = MarkdownResponse(
                thought_of_chain="ok",
                markdown_content="# Hello\n\nWorld",
                summary="Hello world summary",
                interested_topics=None,
            )
            results = await resp._perform_send(deps)

            # 应渲markdown
            mock_render.assert_called_once_with("# Hello\n\nWorld")

            # 应创Image 对象
            MockImage.assert_called_once_with(
                raw=b"fake-png-bytes", name="response.webp"
            )

            # 应发送消
            mock_msg.send.assert_called_once_with(
                target=deps.nb_runtime.target
            )
            assert results == [(mock_msg, mock_receipt, False)]

        mock_ensure.assert_not_called()
        mock_store.assert_not_called()


# ===================== 补充测试：TextResponse meme URL 路径 =====================


class TestTextResponseMemeUrl:
    """TextResponse meme 通过 URL 发送的路径"""

    @pytest.mark.asyncio
    @patch(f"{MODULE}.store_message_with_context", new_callable=AsyncMock)
    @patch(f"{MODULE}.ensure_msgid_from_receipt", return_value="msg-url-1")
    async def test_meme_from_url(self, mock_ensure, mock_store):
        """ImageSegment 有 url 无 local_path 时通过 URL 发送"""
        from nonebot_plugin_wtfllm.memory import ImageSegment

        deps = _make_deps()

        mock_image_seg = MagicMock(spec=ImageSegment)
        mock_image_seg.available = True
        mock_image_seg.local_path = None
        mock_image_seg.url = "http://example.com/meme.jpg"

        deps.context.resolve_media_ref = MagicMock(return_value=mock_image_seg)

        mock_receipt = MagicMock()
        with patch(f"{MODULE}.UniMessage") as MockUniMsg:
            mock_msg = MagicMock()
            mock_msg.send = AsyncMock(return_value=mock_receipt)
            mock_msg.__iadd__ = MagicMock(return_value=mock_msg)
            MockUniMsg.return_value = mock_msg

            resp = TextResponse(thought_of_chain="ok", responses=["check this"], meme="IMG:1", interested_topics=None)
            await resp._perform_send(deps)

            mock_msg.image.assert_called_once_with(url="http://example.com/meme.jpg")


class TestTextResponseMemeFromRepo:
    """TextResponse meme meme_repo 回退获取"""

    @pytest.mark.asyncio
    @patch(f"{MODULE}.store_message_with_context", new_callable=AsyncMock)
    @patch(f"{MODULE}.ensure_msgid_from_receipt", return_value="msg-repo-1")
    @patch(f"{MODULE}.meme_repo")
    async def test_meme_from_repo_found(
        self, mock_meme_repo, mock_ensure, mock_store
    ):
        """resolve_media_ref 失败后从 meme_repo 获取成功"""
        deps = _make_deps()

        # resolve_media_ref ValueError -> _meme = None
        deps.context.resolve_media_ref = MagicMock(side_effect=ValueError("no ref"))

        mock_payload = AsyncMock()
        mock_payload.storage_id = "meme-123"
        mock_payload.get_bytes_async = AsyncMock(return_value=b"meme-bytes")
        mock_meme_repo.get_meme_by_id = AsyncMock(return_value=mock_payload)

        mock_receipt = MagicMock()
        with patch(f"{MODULE}.UniMessage") as MockUniMsg:
            mock_msg = MagicMock()
            mock_msg.send = AsyncMock(return_value=mock_receipt)
            mock_msg.__iadd__ = MagicMock(return_value=mock_msg)
            MockUniMsg.return_value = mock_msg

            resp = TextResponse(thought_of_chain="ok", responses=["look"], meme="meme-uuid", interested_topics=None)
            await resp._perform_send(deps)

            mock_meme_repo.get_meme_by_id.assert_called_once_with("meme-uuid")
            mock_msg.image.assert_called_once_with(
                raw=b"meme-bytes", name="meme-123.webp"
            )

    @pytest.mark.asyncio
    @patch(f"{MODULE}.store_message_with_context", new_callable=AsyncMock)
    @patch(f"{MODULE}.ensure_msgid_from_receipt", return_value="msg-repo-2")
    @patch(f"{MODULE}.meme_repo")
    async def test_meme_from_repo_not_found(
        self, mock_meme_repo, mock_ensure, mock_store
    ):
        """meme_repo 返回 None 时追加 '哎呀图丢了'"""
        deps = _make_deps()
        deps.context.resolve_media_ref = MagicMock(side_effect=ValueError("no ref"))
        mock_meme_repo.get_meme_by_id = AsyncMock(return_value=None)

        mock_receipt = MagicMock()
        with patch(f"{MODULE}.UniMessage") as MockUniMsg:
            mock_msg = MagicMock()
            mock_msg.send = AsyncMock(return_value=mock_receipt)
            mock_msg.__iadd__ = MagicMock(return_value=mock_msg)
            MockUniMsg.return_value = mock_msg

            resp = TextResponse(thought_of_chain="ok", responses=["look"], meme="nonexistent", interested_topics=None)
            await resp._perform_send(deps)

            # 应调用 text("\n哎呀图丢了")
            mock_msg.text.assert_any_call("\n哎呀图丢了")

    @pytest.mark.asyncio
    @patch(f"{MODULE}.store_message_with_context", new_callable=AsyncMock)
    @patch(f"{MODULE}.ensure_msgid_from_receipt", return_value="msg-repo-3")
    @patch(f"{MODULE}.meme_repo")
    async def test_meme_from_repo_error(
        self, mock_meme_repo, mock_ensure, mock_store
    ):
        """meme_repo 抛出 OSError 时追加 '哎呀图丢了'"""
        deps = _make_deps()
        deps.context.resolve_media_ref = MagicMock(side_effect=ValueError("no ref"))
        mock_meme_repo.get_meme_by_id = AsyncMock(side_effect=OSError("disk error"))

        mock_receipt = MagicMock()
        with patch(f"{MODULE}.UniMessage") as MockUniMsg:
            mock_msg = MagicMock()
            mock_msg.send = AsyncMock(return_value=mock_receipt)
            mock_msg.__iadd__ = MagicMock(return_value=mock_msg)
            MockUniMsg.return_value = mock_msg

            resp = TextResponse(thought_of_chain="ok", responses=["look"], meme="broken", interested_topics=None)
            await resp._perform_send(deps)

            mock_msg.text.assert_any_call("\n哎呀图丢了")


class TestTextResponseWithReplySegments:
    """TextResponse 对 reply_segments 的路径测试"""

    @pytest.mark.asyncio
    @patch(f"{MODULE}.store_message_with_context", new_callable=AsyncMock)
    @patch(f"{MODULE}.ensure_msgid_from_receipt", return_value="msg-reply-1")
    async def test_reply_segments_appended(
        self, mock_ensure, mock_store
    ):
        deps = _make_deps()
        mock_reply = MagicMock()
        deps.reply_segments = mock_reply

        mock_receipt = MagicMock()
        with patch(f"{MODULE}.UniMessage") as MockUniMsg:
            mock_msg = MagicMock()
            mock_msg.send = AsyncMock(return_value=mock_receipt)
            mock_msg.__iadd__ = MagicMock(return_value=mock_msg)
            MockUniMsg.return_value = mock_msg

            resp = TextResponse(thought_of_chain="ok", responses=["Hello!"], meme=None, interested_topics=None)
            await resp._perform_send(deps)

            # reply_segments 应通过 += 追加到 msg
            mock_msg.__iadd__.assert_any_call(mock_reply)


class TestTextResponseWithExtraSegments:
    """TextResponse 对 extra_segments 的路径测试"""

    @pytest.mark.asyncio
    @patch(f"{MODULE}.store_message_with_context", new_callable=AsyncMock)
    @patch(f"{MODULE}.ensure_msgid_from_receipt", return_value="msg-extra-1")
    async def test_extra_segments_appended(
        self, mock_ensure, mock_store
    ):
        deps = _make_deps()
        mock_extra = MagicMock()

        mock_receipt = MagicMock()
        with patch(f"{MODULE}.UniMessage") as MockUniMsg:
            mock_msg = MagicMock()
            mock_msg.send = AsyncMock(return_value=mock_receipt)
            mock_msg.__iadd__ = MagicMock(return_value=mock_msg)
            MockUniMsg.return_value = mock_msg

            resp = TextResponse(thought_of_chain="ok", responses=["Hi"], meme=None, interested_topics=None)
            await resp._perform_send(deps, extra_segments=mock_extra)

            mock_msg.__iadd__.assert_any_call(mock_extra)


class TestMarkdownResponseWithReplyAndExtra:
    """MarkdownResponse reply_segments + extra_segments"""

    @pytest.mark.asyncio
    @patch(f"{MODULE}.store_message_with_context", new_callable=AsyncMock)
    @patch(f"{MODULE}.ensure_msgid_from_receipt", return_value="md-reply-1")
    @patch(f"{MODULE}.render_markdown_to_image", new_callable=AsyncMock)
    async def test_reply_and_extra_second_message(
        self, mock_render, mock_ensure, mock_store
    ):
        """reply_segments + extra_segments 应产生第二条消息"""
        deps = _make_deps()
        mock_reply = MagicMock()
        deps.reply_segments = mock_reply
        mock_extra = MagicMock()

        mock_render.return_value = b"fake-png"
        mock_receipt = MagicMock()

        with patch(f"{MODULE}.UniMessage") as MockUniMsg, \
             patch(f"{MODULE}.Image") as MockImage:
            mock_msg = MagicMock()
            mock_msg.send = AsyncMock(return_value=mock_receipt)
            mock_msg.__iadd__ = MagicMock(return_value=mock_msg)
            MockUniMsg.return_value = mock_msg
            MockImage.return_value = MagicMock()

            resp = MarkdownResponse(
                thought_of_chain="ok",
                markdown_content="# Test",
                summary="test",
                interested_topics=None,
            )
            await resp._perform_send(deps, extra_segments=mock_extra)

            # UniMessage() 应被调用至少 2 (msg1 + msg2)
            assert MockUniMsg.call_count >= 2
            # send 应被调用 2 (msg1 发图 msg2 reply+extra)
            assert mock_msg.send.call_count == 2


class TestRejectResponseWithExtraSegments:
    """RejectResponse extra_segments"""

    @pytest.mark.asyncio
    @patch(f"{MODULE}.store_message_with_context", new_callable=AsyncMock)
    @patch(f"{MODULE}.ensure_msgid_from_receipt", return_value="rej-extra-1")
    async def test_extra_segments_appended_to_visible_reject(
        self, mock_ensure, mock_store
    ):
        deps = _make_deps()
        mock_extra = MagicMock()

        mock_receipt = MagicMock()
        with patch(f"{MODULE}.UniMessage") as MockUniMsg:
            mock_msg = MagicMock()
            mock_msg.send = AsyncMock(return_value=mock_receipt)
            mock_msg.__iadd__ = MagicMock(return_value=mock_msg)
            MockUniMsg.return_value = mock_msg

            resp = RejectResponse(
                thought_of_chain="ok",
                reason="off topic",
                message_to_user="Sorry",
                interested_topics=None,
            )
            await resp._perform_send(deps, extra_segments=mock_extra)

            # extra_segments 应通过 += 追加
            mock_msg.__iadd__.assert_any_call(mock_extra)
            mock_msg.send.assert_called_once()

