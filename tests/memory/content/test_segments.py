# tests/memory/content/test_segments.py
"""memory/content/segments.py 单元测试"""

import os
import tempfile
from pathlib import Path

import pytest
from unittest.mock import MagicMock

from nonebot_plugin_wtfllm.memory.content.segments import (
    TextSegment,
    MentionSegment,
    ImageSegment,
    VideoSegment,
    FileSegment,
    AudioSegment,
    ForwardSegment,
    Node,
)
from nonebot_plugin_wtfllm.memory.content.message import Message


class TestTextSegment:
    """TextSegment 测试"""

    def test_create_basic(self):
        """测试基本创建"""
        seg = TextSegment(content="Hello World")
        assert seg.type == "text"
        assert seg.content == "Hello World"
        assert seg.created_at > 0

    def test_format_content_short(self):
        """测试短文本 _format_content 返回原始内容"""
        seg = TextSegment(content="Plain text content")
        ctx = MagicMock()

        result = seg._format_content(ctx)
        assert result == "Plain text content"

    def test_format_content_long_condensed(self):
        """测试长文本 _format_content 触发压缩"""
        long_text = "A" * 30 + "B" * 40 + "C" * 30
        seg = TextSegment(content=long_text)
        ctx = MagicMock()
        ctx.condense = True

        result = seg._format_content(ctx)
        assert "[...省略...]" in result
        assert result.startswith("A" * 30)
        assert result.endswith("C" * 30)
        # 原始 content 未被修改
        assert seg.content == long_text

    def test_format_content_long_no_condense(self):
        """ctx.condense=False 时长文本原样返回"""
        long_text = "A" * 30 + "B" * 40 + "C" * 30
        seg = TextSegment(content=long_text)
        ctx = MagicMock()
        ctx.condense = False

        result = seg._format_content(ctx)
        assert result == long_text

    def test_to_llm_context(self):
        """测试 to_llm_context 设置 message_id 并返回内容"""
        seg = TextSegment(content="Test content")
        ctx = MagicMock()

        result = seg.to_llm_context(ctx, "new_msg_id")
        assert result == "Test content"
        assert seg.message_id == "new_msg_id"


class TestMentionSegment:
    """MentionSegment 测试"""

    def test_create_with_user_id(self):
        """测试使用 user_id 创建"""
        seg = MentionSegment(user_id="user_123")
        assert seg.type == "mention"
        assert seg.user_id == "user_123"
        assert seg.at_all is False

    def test_create_with_at_all(self):
        """测试使用 at_all 创建"""
        seg = MentionSegment(at_all=True)
        assert seg.type == "mention"
        assert seg.user_id is None
        assert seg.at_all is True

    def test_create_neither_raises(self):
        """测试既无 user_id 也无 at_all 抛出错误"""
        with pytest.raises(ValueError, match="must have either 'user_id' or 'at_all'"):
            MentionSegment()

    def test_create_both_raises(self):
        """测试同时有 user_id 和 at_all 抛出错误"""
        with pytest.raises(ValueError, match="cannot have both"):
            MentionSegment(user_id="user_123", at_all=True)

    def test_format_content_user_id(self):
        """测试 _format_content 使用 user_id"""
        seg = MentionSegment(user_id="user_456")

        ctx = MagicMock()
        ctx.alias_provider.get_alias.return_value = "User_1"

        result = seg._format_content(ctx)
        assert result == "<@User_1>"

    def test_format_content_user_id_no_alias(self):
        """测试 _format_content 使用 user_id 但无别名"""
        seg = MentionSegment(user_id="user_789")

        ctx = MagicMock()
        ctx.alias_provider.get_alias.return_value = None

        result = seg._format_content(ctx)
        assert result == "<@user_789>"

    def test_format_content_at_all(self):
        """测试 _format_content 使用 at_all"""
        seg = MentionSegment(at_all=True)
        ctx = MagicMock()

        result = seg._format_content(ctx)
        assert result == "<@全体成员>"


class TestImageSegment:
    """ImageSegment 测试"""

    def test_create_with_url(self):
        """测试使用 URL 创建"""
        seg = ImageSegment(url="http://example.com/image.jpg")
        assert seg.type == "image"
        assert seg.url == "http://example.com/image.jpg"
        assert seg.local_path is None

    def test_create_with_local_path(self):
        """测试使用 local_path 创建，validator 会将 str 转为 Path"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"fake image data")
            path = f.name
        try:
            seg = ImageSegment(local_path=path)
            assert seg.type == "image"
            assert isinstance(seg.local_path, Path)
            assert seg.local_path == Path(path)
            assert seg.url is None
        finally:
            os.unlink(path)

    def test_create_neither_raises(self):
        """测试既无 url 也无 local_path 抛出错误"""
        with pytest.raises(ValueError, match="must have either 'url' or 'local_path'"):
            ImageSegment()

    def test_format_content(self):
        """测试 _format_content 返回媒体引用"""
        seg = ImageSegment(url="http://example.com/img.jpg")

        ctx = MagicMock()
        ctx.ref_provider.next_media_ref.return_value = "IMG:1"

        result = seg._format_content(ctx, memory_ref=5)
        assert result == "[IMG:1]"
        ctx.ref_provider.next_media_ref.assert_called_once_with(seg, 5)


class TestVideoSegment:
    """VideoSegment 测试"""

    def test_create_with_url(self):
        """测试使用 URL 创建"""
        seg = VideoSegment(url="http://example.com/video.mp4")
        assert seg.type == "video"
        assert seg.url == "http://example.com/video.mp4"

    def test_create_with_local_path(self):
        """测试使用 local_path 创建，validator 会将 str 转为 Path"""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video data")
            path = f.name
        try:
            seg = VideoSegment(local_path=path)
            assert seg.local_path == Path(path)
        finally:
            os.unlink(path)

    def test_create_neither_raises(self):
        """测试既无 url 也无 local_path 抛出错误"""
        with pytest.raises(ValueError, match="must have either 'url' or 'local_path'"):
            VideoSegment()

    def test_format_content(self):
        """测试 _format_content 返回媒体引用"""
        seg = VideoSegment(url="http://example.com/video.mp4")

        ctx = MagicMock()
        ctx.ref_provider.next_media_ref.return_value = "VIDEO:1"

        result = seg._format_content(ctx)
        assert result == "[VIDEO:1]"


class TestFileSegment:
    """FileSegment 测试"""

    def test_create_with_url(self):
        """测试使用 URL 创建"""
        seg = FileSegment(filename="document.pdf", url="http://example.com/doc.pdf")
        assert seg.type == "file"
        assert seg.filename == "document.pdf"
        assert seg.url == "http://example.com/doc.pdf"

    def test_create_with_local_path(self):
        """测试使用 local_path 创建，validator 会将 str 转为 Path"""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(b"csv_data")
            path = f.name
        try:
            seg = FileSegment(filename="data.csv", local_path=path)
            assert seg.filename == "data.csv"
            assert seg.local_path == Path(path)
        finally:
            os.unlink(path)

    def test_create_neither_raises(self):
        """测试既无 url 也无 local_path 抛出错误"""
        with pytest.raises(ValueError, match="must have either 'url' or 'local_path'"):
            FileSegment(filename="test.txt")

    def test_format_content(self):
        """测试 _format_content 返回带文件名的引用"""
        seg = FileSegment(filename="report.pdf", url="http://example.com/report.pdf")

        ctx = MagicMock()
        ctx.ref_provider.next_media_ref.return_value = "FILE:1"

        result = seg._format_content(ctx)
        assert result == "[FILE:1: report.pdf]"


class TestAudioSegment:
    """AudioSegment 测试"""

    def test_create_with_url(self):
        """测试使用 URL 创建"""
        seg = AudioSegment(url="http://example.com/audio.mp3")
        assert seg.type == "audio"
        assert seg.url == "http://example.com/audio.mp3"

    def test_create_with_local_path(self):
        """测试使用 local_path 创建，validator 会将 str 转为 Path"""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"audio_data")
            path = f.name
        try:
            seg = AudioSegment(local_path=path)
            assert seg.local_path == Path(path)
        finally:
            os.unlink(path)

    def test_create_neither_raises(self):
        """测试既无 url 也无 local_path 抛出错误"""
        with pytest.raises(ValueError, match="must have either 'url' or 'local_path'"):
            AudioSegment()

    def test_format_content(self):
        """测试 _format_content 返回媒体引用"""
        seg = AudioSegment(url="http://example.com/audio.mp3")

        ctx = MagicMock()
        ctx.ref_provider.next_media_ref.return_value = "AUDIO:1"

        result = seg._format_content(ctx)
        assert result == "[AUDIO:1]"


class TestSegmentSerialization:
    """Segment 序列化测试"""

    def test_text_segment_roundtrip(self):
        """测试 TextSegment 序列化往返"""
        seg = TextSegment(content="Test content")
        data = seg.model_dump()
        restored = TextSegment.model_validate(data)

        assert restored.content == seg.content
        assert restored.type == "text"

    def test_mention_segment_roundtrip(self):
        """测试 MentionSegment 序列化往返"""
        seg = MentionSegment(user_id="user_123")
        data = seg.model_dump()
        restored = MentionSegment.model_validate(data)

        assert restored.user_id == seg.user_id
        assert restored.at_all == seg.at_all
        assert restored.type == "mention"

    def test_image_segment_roundtrip(self):
        """测试 ImageSegment 序列化往返"""
        seg = ImageSegment(url="http://example.com/img.jpg")
        data = seg.model_dump()
        restored = ImageSegment.model_validate(data)

        assert restored.url == seg.url
        assert restored.type == "image"

    def test_file_segment_roundtrip(self):
        """测试 FileSegment 序列化往返"""
        seg = FileSegment(filename="doc.pdf", url="http://example.com/doc.pdf")
        data = seg.model_dump()
        restored = FileSegment.model_validate(data)

        assert restored.filename == seg.filename
        assert restored.url == seg.url
        assert restored.type == "file"


class TestForwardSegment:
    """ForwardSegment 测试"""

    @staticmethod
    def _make_nodes(count: int, text_template: str = "msg {}") -> list:
        nodes = []
        for i in range(count):
            msg = Message.create().text(text_template.format(i))
            nodes.append(Node(sender=f"user_{i}", content=msg))
        return nodes

    def test_empty_children(self):
        """测试空子消息"""
        seg = ForwardSegment(children=[])
        seg._message_id = "msg_1"
        ctx = MagicMock()
        result = seg._format_content(ctx)
        assert result == "[合并转发消息, 共0条, 空]"

    def test_few_children_no_condense(self):
        """子消息数量较少（<=7），走原有渲染"""
        nodes = self._make_nodes(3)
        seg = ForwardSegment(children=nodes)
        seg._message_id = "msg_1"
        ctx = MagicMock()
        ctx.alias_provider.get_alias.side_effect = lambda x: x

        result = seg._format_content(ctx)
        assert "合并转发消息, 共3条:" in result
        assert "合并转发结束" in result
        assert "user_0:" in result
        assert "user_1:" in result
        assert "user_2:" in result
        assert "省略" not in result

    def test_many_children_condensed(self):
        """子消息数量 > 7，触发压缩渲染"""
        nodes = self._make_nodes(10)
        seg = ForwardSegment(children=nodes)
        seg._message_id = "msg_1"
        ctx = MagicMock()
        ctx.condense = True
        ctx.alias_provider.get_alias.side_effect = lambda x: x

        result = seg._format_content(ctx)
        assert "合并转发消息, 共10条:" in result
        assert "省略中间5条消息" in result
        assert "合并转发结束" in result
        # 前3条
        assert "user_0:" in result
        assert "user_1:" in result
        assert "user_2:" in result
        # 后2条
        assert "user_8:" in result
        assert "user_9:" in result
        # 中间消息不应出现
        assert "user_4:" not in result

    def test_many_children_no_condense_flag(self):
        """ctx.condense=False 时 >7 条子消息也完整渲染"""
        nodes = self._make_nodes(10)
        seg = ForwardSegment(children=nodes)
        seg._message_id = "msg_1"
        ctx = MagicMock()
        ctx.condense = False
        ctx.alias_provider.get_alias.side_effect = lambda x: x

        result = seg._format_content(ctx)
        assert "合并转发消息, 共10条:" in result
        assert "合并转发结束" in result
        # 所有消息都应存在
        for i in range(10):
            assert f"user_{i}:" in result
        # 不应有省略
        assert "省略" not in result
