"""stream_processing/extract.py 单元测试

覆盖: 各 _to_*_segment 转换器、extract 函数、convert_and_store_item
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from nonebot_plugin_wtfllm.stream_processing.extract import (
    _to_text_segment,
    _to_emoji_segment,
    _to_unknown_segment,
    _to_image_segment,
    _to_file_segment,
    _to_audio_segment,
    _to_mention_segment,
    _to_video_segment,
    _to_hyper_segment,
    _to_forward_segment,
    extract_memeorymsg_from_unimsg,
    extract_memoryitem_from_unimsg,
    convert_and_store_item,
)
from nonebot_plugin_wtfllm.memory.content.segments import (
    TextSegment,
    EmojiSegment,
    UnknownSegment,
    ImageSegment,
    FileSegment,
    AudioSegment,
    MentionSegment,
    VideoSegment,
    HyperSegment,
    ForwardSegment,
)
from nonebot_plugin_wtfllm.memory.items.base_items import (
    PrivateMemoryItem,
    GroupMemoryItem,
)


def _mock_seg(cls_name, **attrs):
    """创建模拟 Segment 对象"""
    seg = MagicMock()
    seg.__class__ = type(cls_name, (), {})
    type(seg).__name__ = cls_name
    for k, v in attrs.items():
        setattr(seg, k, v)
    return seg


# ===================== _to_text_segment =====================


class TestToTextSegment:
    def test_basic(self):
        seg = _mock_seg("Text", text="hello")
        result = _to_text_segment(seg, 1000)
        assert isinstance(result, TextSegment)
        assert result.content == "hello"
        assert result.created_at == 1000

    def test_none_created_at(self):
        seg = _mock_seg("Text", text="world")
        result = _to_text_segment(seg, None)
        assert isinstance(result, TextSegment)
        assert result.created_at > 0


# ===================== _to_emoji_segment =====================


class TestToEmojiSegment:
    def test_with_name(self):
        seg = _mock_seg("Emoji", name="smile", id="123", url="http://emoji.png")
        result = _to_emoji_segment(seg, 1000)
        assert isinstance(result, EmojiSegment)
        assert result.name == "smile"

    def test_with_id_fallback(self):
        seg = _mock_seg("Emoji", name=None, id="456", url=None)
        result = _to_emoji_segment(seg, 1000)
        assert result.name == "456"


# ===================== _to_unknown_segment =====================


class TestToUnknownSegment:
    def test_basic(self):
        seg = _mock_seg("WeirdType")
        result = _to_unknown_segment(seg, 1000)
        assert isinstance(result, UnknownSegment)
        assert result.original_type == "WeirdType"


# ===================== _to_image_segment =====================


class TestToImageSegment:
    def test_from_url(self):
        seg = _mock_seg("Image", url="http://img.jpg", raw_bytes=None)
        # no desc attr
        delattr(seg, "desc") if hasattr(seg, "desc") else None
        result = _to_image_segment(seg, 1000)
        assert isinstance(result, ImageSegment)
        assert result.url == "http://img.jpg"

    def test_from_raw_bytes(self, tmp_path):
        seg = _mock_seg("Image", url=None, raw_bytes=b"\x89PNG")
        with patch("nonebot_plugin_wtfllm.stream_processing.extract.MEDIA_DIR", tmp_path):
            result = _to_image_segment(seg, 1000)
        assert isinstance(result, ImageSegment)
        assert result.local_path is not None

    def test_neither_raises(self):
        seg = _mock_seg("Image", url=None, raw_bytes=None)
        with pytest.raises(ValueError, match="url or raw_bytes"):
            _to_image_segment(seg, 1000)

    def test_with_desc(self):
        seg = _mock_seg("Image", url="http://img.jpg", raw_bytes=None, desc="a cat")
        result = _to_image_segment(seg, 1000)
        assert result.desc == "a cat"


# ===================== _to_file_segment =====================


class TestToFileSegment:
    def test_from_url(self):
        seg = _mock_seg("File", url="http://file.zip", raw_bytes=None, name="doc.zip")
        result = _to_file_segment(seg, 1000)
        assert isinstance(result, FileSegment)
        assert result.url == "http://file.zip"

    def test_from_raw_bytes(self, tmp_path):
        seg = _mock_seg("File", url=None, raw_bytes=b"data", name="test.bin")
        with patch("nonebot_plugin_wtfllm.stream_processing.extract.MEDIA_DIR", tmp_path):
            result = _to_file_segment(seg, 1000)
        assert isinstance(result, FileSegment)

    def test_neither_raises(self):
        seg = _mock_seg("File", url=None, raw_bytes=None, name="x")
        with pytest.raises(ValueError, match="url or raw_bytes"):
            _to_file_segment(seg, 1000)


# ===================== _to_audio_segment =====================


class TestToAudioSegment:
    def test_from_url(self):
        seg = _mock_seg("Voice", url="http://audio.mp3", raw_bytes=None)
        result = _to_audio_segment(seg, 1000)
        assert isinstance(result, AudioSegment)

    def test_neither_raises(self):
        seg = _mock_seg("Voice", url=None, raw_bytes=None)
        with pytest.raises(ValueError, match="url or raw_bytes"):
            _to_audio_segment(seg, 1000)


# ===================== _to_video_segment =====================


class TestToVideoSegment:
    def test_from_url(self):
        seg = _mock_seg("Video", url="http://video.mp4", raw_bytes=None)
        result = _to_video_segment(seg, 1000)
        assert isinstance(result, VideoSegment)

    def test_neither_raises(self):
        seg = _mock_seg("Video", url=None, raw_bytes=None)
        with pytest.raises(ValueError, match="url or raw_bytes"):
            _to_video_segment(seg, 1000)


# ===================== _to_mention_segment =====================


class TestToMentionSegment:
    def test_at_user(self):
        # AtAll is mocked (MagicMock), so isinstance() won't work normally.
        # Patch AtAll with a real class so isinstance check in _to_mention_segment works.
        real_atall = type("AtAll", (), {})
        seg = _mock_seg("At", target="user_123")
        with patch("nonebot_plugin_wtfllm.stream_processing.extract.AtAll", real_atall):
            result = _to_mention_segment(seg, 1000)
        assert isinstance(result, MentionSegment)
        assert result.user_id == "user_123"

    def test_at_all(self):
        real_atall = type("AtAll", (), {})
        seg = real_atall()
        with patch("nonebot_plugin_wtfllm.stream_processing.extract.AtAll", real_atall):
            result = _to_mention_segment(seg, 1000)
        assert isinstance(result, MentionSegment)
        assert result.at_all is True


# ===================== _to_hyper_segment =====================


class TestToHyperSegment:
    @patch("nonebot_plugin_wtfllm.stream_processing.extract.clean_hyper_content")
    def test_basic(self, mock_clean):
        mock_clean.return_value = "cleaned content"
        seg = _mock_seg("Hyper", raw="<xml>data</xml>", format="xml")
        result = _to_hyper_segment(seg, 1000)
        assert isinstance(result, HyperSegment)
        assert result.content == "cleaned content"


# ===================== _to_forward_segment =====================


class TestToForwardSegment:
    @patch("nonebot_plugin_wtfllm.stream_processing.extract.APP_CONFIG")
    def test_ignore_reference(self, mock_config):
        mock_config.ignore_reference = True
        seg = _mock_seg("Reference", children=[])
        result = _to_forward_segment(seg, 1000)
        assert isinstance(result, ForwardSegment)
        assert len(result.children) == 1
        assert "忽略" in result.children[0].content.get_plain_text()


# ===================== extract_memeorymsg_from_unimsg =====================


class TestExtractMemoryMsg:
    def test_basic_text(self):
        from nonebot_plugin_wtfllm.stream_processing.extract import Reply

        text_seg = _mock_seg("Text", text="hello")
        seg_type = type(text_seg)

        unimsg = MagicMock()
        unimsg.__iter__ = MagicMock(return_value=iter([text_seg]))
        unimsg.has = MagicMock(return_value=False)

        # Patch Reply to a real class so isinstance check works
        real_reply = type("Reply", (), {})
        with patch(
            "nonebot_plugin_wtfllm.stream_processing.extract.UNISEG_TO_MEMORYSEG_MAP",
            {seg_type: _to_text_segment},
        ), patch(
            "nonebot_plugin_wtfllm.stream_processing.extract.Reply", real_reply,
        ):
            result = extract_memeorymsg_from_unimsg(unimsg)
        assert result is not None


# ===================== extract_memoryitem_from_unimsg =====================


class TestExtractMemoryItem:
    def test_missing_sender_raises(self):
        unimsg = MagicMock()
        with pytest.raises(ValueError, match="Sender"):
            extract_memoryitem_from_unimsg(
                unimsg, sender="", group_id=None, user_id=None,
                agent_id="a1", message_id="m1",
            )

    def test_private_item(self):
        text_seg = _mock_seg("Text", text="hello")
        seg_type = type(text_seg)

        unimsg = MagicMock()
        unimsg.__iter__ = MagicMock(return_value=iter([text_seg]))
        unimsg.has = MagicMock(return_value=False)

        real_reply = type("Reply", (), {})
        with patch(
            "nonebot_plugin_wtfllm.stream_processing.extract.UNISEG_TO_MEMORYSEG_MAP",
            {seg_type: _to_text_segment},
        ), patch(
            "nonebot_plugin_wtfllm.stream_processing.extract.Reply", real_reply,
        ):
            result = extract_memoryitem_from_unimsg(
                unimsg, sender="u1", group_id=None, user_id="u1",
                agent_id="a1", message_id="m1",
            )
        assert isinstance(result, PrivateMemoryItem)

    def test_group_item(self):
        text_seg = _mock_seg("Text", text="hello")
        seg_type = type(text_seg)

        unimsg = MagicMock()
        unimsg.__iter__ = MagicMock(return_value=iter([text_seg]))
        unimsg.has = MagicMock(return_value=False)

        real_reply = type("Reply", (), {})
        with patch(
            "nonebot_plugin_wtfllm.stream_processing.extract.UNISEG_TO_MEMORYSEG_MAP",
            {seg_type: _to_text_segment},
        ), patch(
            "nonebot_plugin_wtfllm.stream_processing.extract.Reply", real_reply,
        ):
            result = extract_memoryitem_from_unimsg(
                unimsg, sender="u1", group_id="g1", user_id=None,
                agent_id="a1", message_id="m1",
            )
        assert isinstance(result, GroupMemoryItem)


# ===================== _to_forward_segment (non-ignore path) =====================


class TestToForwardSegmentNonIgnore:
    """ignore_reference=False 时的转发段处理"""

    @patch("nonebot_plugin_wtfllm.stream_processing.extract.APP_CONFIG")
    def test_custom_node_with_string_content(self, mock_config):
        """CustomNode 包含纯字符串内容"""
        import datetime
        mock_config.ignore_reference = False

        CustomNode = type("CustomNode", (), {})
        node = CustomNode()
        node.uid = "user_1"
        node.context = "group_1"
        node.content = "hello forward"
        node.time = datetime.datetime(2024, 1, 1, 12, 0, 0)

        seg = _mock_seg("Reference", children=[node])

        with patch(
            "nonebot_plugin_wtfllm.stream_processing.extract.CustomNode", CustomNode
        ), patch(
            "nonebot_plugin_wtfllm.stream_processing.extract.RefNode",
            type("RefNode", (), {}),
        ):
            result = _to_forward_segment(seg, 1000)

        assert isinstance(result, ForwardSegment)
        assert len(result.children) == 1
        assert result.children[0].sender == "user_1"
        assert "hello forward" in result.children[0].content.get_plain_text()

    @patch("nonebot_plugin_wtfllm.stream_processing.extract.APP_CONFIG")
    def test_custom_node_with_segments(self, mock_config):
        """CustomNode 包含 segment 列表"""
        import datetime
        mock_config.ignore_reference = False

        CustomNode = type("CustomNode", (), {})
        text_seg = _mock_seg("Text", text="nested text")
        seg_type = type(text_seg)

        node = CustomNode()
        node.uid = "user_2"
        node.context = None
        node.content = [text_seg]
        node.time = datetime.datetime(2024, 1, 1, 12, 0, 0)

        seg = _mock_seg("Reference", children=[node])

        with patch(
            "nonebot_plugin_wtfllm.stream_processing.extract.CustomNode", CustomNode
        ), patch(
            "nonebot_plugin_wtfllm.stream_processing.extract.RefNode",
            type("RefNode", (), {}),
        ), patch(
            "nonebot_plugin_wtfllm.stream_processing.extract.UNISEG_TO_MEMORYSEG_MAP",
            {seg_type: _to_text_segment},
        ):
            result = _to_forward_segment(seg, 1000)

        assert isinstance(result, ForwardSegment)
        assert len(result.children) == 1
        assert "nested text" in result.children[0].content.get_plain_text()

    @patch("nonebot_plugin_wtfllm.stream_processing.extract.APP_CONFIG")
    def test_ref_node(self, mock_config):
        """RefNode 子节点"""
        mock_config.ignore_reference = False

        RefNode = type("RefNode", (), {})
        node = RefNode()
        node.context = "group_x"

        seg = _mock_seg("Reference", children=[node])

        with patch(
            "nonebot_plugin_wtfllm.stream_processing.extract.CustomNode",
            type("CustomNode", (), {}),
        ), patch(
            "nonebot_plugin_wtfllm.stream_processing.extract.RefNode", RefNode
        ):
            result = _to_forward_segment(seg, 1000)

        assert isinstance(result, ForwardSegment)
        assert len(result.children) == 1
        assert "未知引用消息" in result.children[0].content.get_plain_text()


# ===================== extract_memoryitem_from_unimsg (Reply) =====================


class TestExtractMemoryItemWithReply:
    """带 Reply 段的提取测试"""

    def test_with_reply_sets_related_message_id(self):
        """UniMessage 含 Reply 段时设置 related_message_id"""
        text_seg = _mock_seg("Text", text="replying")
        seg_type = type(text_seg)

        Reply = type("Reply", (), {})
        reply_seg = Reply()
        reply_seg.id = "replied_msg_id"

        unimsg = MagicMock()
        unimsg.__iter__ = MagicMock(return_value=iter([reply_seg, text_seg]))
        unimsg.has = MagicMock(return_value=True)
        unimsg.get = MagicMock(return_value=[reply_seg])

        with patch(
            "nonebot_plugin_wtfllm.stream_processing.extract.UNISEG_TO_MEMORYSEG_MAP",
            {seg_type: _to_text_segment},
        ), patch(
            "nonebot_plugin_wtfllm.stream_processing.extract.Reply", Reply,
        ):
            result = extract_memoryitem_from_unimsg(
                unimsg, sender="u1", group_id="g1", user_id=None,
                agent_id="a1", message_id="m1",
            )

        assert isinstance(result, GroupMemoryItem)
        assert result.related_message_id == "replied_msg_id"

    def test_no_user_no_group_raises(self):
        """user_id=None 且 group_id=None 时抛出 ValueError"""
        unimsg = MagicMock()
        with pytest.raises(ValueError):
            extract_memoryitem_from_unimsg(
                unimsg, sender="u1", group_id=None, user_id=None,
                agent_id="a1", message_id="m1",
            )


# ===================== _to_video_segment (raw_bytes) =====================


class TestToVideoSegmentRawBytes:
    def test_from_raw_bytes(self, tmp_path):
        seg = _mock_seg("Video", url=None, raw_bytes=b"\x00\x00\x00\x1c")
        with patch("nonebot_plugin_wtfllm.stream_processing.extract.MEDIA_DIR", tmp_path):
            result = _to_video_segment(seg, 1000)
        assert isinstance(result, VideoSegment)
        assert result.local_path is not None


# ===================== _to_audio_segment (raw_bytes) =====================


class TestToAudioSegmentRawBytes:
    def test_from_raw_bytes(self, tmp_path):
        seg = _mock_seg("Voice", url=None, raw_bytes=b"\xff\xfb")
        with patch("nonebot_plugin_wtfllm.stream_processing.extract.MEDIA_DIR", tmp_path):
            result = _to_audio_segment(seg, 1000)
        assert isinstance(result, AudioSegment)
        assert result.local_path is not None


# ===================== convert_and_store_item =====================


class TestConvertAndStoreItem:
    @pytest.mark.asyncio
    @patch("nonebot_plugin_wtfllm.stream_processing.extract.memory_item_repo")
    async def test_basic_flow(self, mock_repo):
        mock_repo.save_memory_item = AsyncMock()

        text_seg = _mock_seg("Text", text="test")
        seg_type = type(text_seg)

        unimsg = MagicMock()
        unimsg.__iter__ = MagicMock(return_value=iter([text_seg]))
        unimsg.has = MagicMock(return_value=False)

        real_reply = type("Reply", (), {})
        with patch(
            "nonebot_plugin_wtfllm.stream_processing.extract.UNISEG_TO_MEMORYSEG_MAP",
            {seg_type: _to_text_segment},
        ), patch(
            "nonebot_plugin_wtfllm.stream_processing.extract.Reply", real_reply,
        ), patch(
            "nonebot_plugin_wtfllm.memory.content.message.get_http_client",
        ):
            result = await convert_and_store_item(
                agent_id="a1",
                uni_msg=unimsg,
                group_id=None,
                user_id="u1",
                sender="u1",
                msg_id="msg_123",
            )
        assert isinstance(result, PrivateMemoryItem)
        mock_repo.save_memory_item.assert_called_once()
