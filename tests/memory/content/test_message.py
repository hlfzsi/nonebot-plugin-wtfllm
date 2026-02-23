# tests/memory/content/test_message.py
"""memory/content/message.py 单元测试"""

import pytest
from unittest.mock import MagicMock

from nonebot_plugin_wtfllm.memory.content.message import Message
from nonebot_plugin_wtfllm.memory.content.segments import (
    TextSegment,
    MentionSegment,
    ImageSegment,
    FileSegment,
    AudioSegment,
)


class TestMessageCreate:
    """Message.create 静态方法测试"""

    def test_create_empty(self):
        """测试创建空消息"""
        msg = Message.create()
        assert msg.segments == []

    def test_create_with_none(self):
        """测试使用 None 创建"""
        msg = Message.create(None)
        assert msg.segments == []

    def test_create_with_single_segment(self):
        """测试使用单个片段创建"""
        seg = TextSegment(content="Hello")
        msg = Message.create(seg)

        assert len(msg.segments) == 1
        assert msg.segments[0].content == "Hello"

    def test_create_with_list(self):
        """测试使用列表创建"""
        segs = [
            TextSegment(content="Hello"),
            TextSegment(content="World"),
        ]
        msg = Message.create(segs)

        assert len(msg.segments) == 2
        assert msg.segments[0].content == "Hello"
        assert msg.segments[1].content == "World"


class TestMessageCreatedAt:
    """Message.created_at 属性测试"""

    def test_created_at_empty_returns_zero(self):
        """测试空消息返回 0"""
        msg = Message.create()
        assert msg.created_at == 0

    def test_created_at_single_segment(self):
        """测试单个片段的创建时间"""
        seg = TextSegment(content="Test", created_at=1000)
        msg = Message.create(seg)
        assert msg.created_at == 1000

    def test_created_at_returns_minimum(self):
        """测试多个片段返回最小时间"""
        segs = [
            TextSegment(content="A", created_at=2000),
            TextSegment(content="B", created_at=1000),
            TextSegment(content="C", created_at=3000),
        ]
        msg = Message.create(segs)
        assert msg.created_at == 1000


class TestMessageHas:
    """Message.has 方法测试"""

    def test_has_by_string_type_true(self):
        """测试使用字符串类型检测 - 存在"""
        msg = Message.create(
            [
                TextSegment(content="Hello"),
                ImageSegment(url="http://example.com/img.jpg"),
            ]
        )

        assert msg.has("text") is True
        assert msg.has("image") is True

    def test_has_by_string_type_false(self):
        """测试使用字符串类型检测 - 不存在"""
        msg = Message.create([TextSegment(content="Hello")])

        assert msg.has("image") is False
        assert msg.has("audio") is False

    def test_has_by_class_type_true(self):
        """测试使用类类型检测 - 存在"""
        msg = Message.create(
            [
                TextSegment(content="Hello"),
                MentionSegment(user_id="user_123"),
            ]
        )

        assert msg.has(TextSegment) is True
        assert msg.has(MentionSegment) is True

    def test_has_by_class_type_false(self):
        """测试使用类类型检测 - 不存在"""
        msg = Message.create([TextSegment(content="Hello")])

        assert msg.has(ImageSegment) is False
        assert msg.has(AudioSegment) is False


class TestMessageGet:
    """Message.get 方法测试"""

    def test_get_by_string_type(self):
        """测试使用字符串类型获取"""
        msg = Message.create(
            [
                TextSegment(content="A"),
                ImageSegment(url="http://example.com/1.jpg"),
                TextSegment(content="B"),
                ImageSegment(url="http://example.com/2.jpg"),
            ]
        )

        texts = msg.get("text")
        assert len(texts) == 2
        assert all(seg.type == "text" for seg in texts)

        images = msg.get("image")
        assert len(images) == 2
        assert all(seg.type == "image" for seg in images)

    def test_get_by_class_type(self):
        """测试使用类类型获取"""
        msg = Message.create(
            [
                TextSegment(content="Hello"),
                MentionSegment(user_id="user_1"),
                MentionSegment(user_id="user_2"),
            ]
        )

        mentions = msg.get(MentionSegment)
        assert len(mentions) == 2
        assert all(isinstance(seg, MentionSegment) for seg in mentions)

    def test_get_empty_result(self):
        """测试获取不存在的类型返回空列表"""
        msg = Message.create([TextSegment(content="Hello")])

        images = msg.get("image")
        assert images == []

        audios = msg.get(AudioSegment)
        assert audios == []


class TestMessageGetPlainText:
    """Message.get_plain_text 方法测试"""

    def test_get_plain_text_only_text(self):
        """测试只有文本片段"""
        msg = Message.create(
            [
                TextSegment(content="Hello "),
                TextSegment(content="World"),
            ]
        )

        assert msg.get_plain_text() == "Hello World"

    def test_get_plain_text_mixed(self):
        """测试混合片段"""
        msg = Message.create(
            [
                TextSegment(content="Hello "),
                ImageSegment(url="http://example.com/img.jpg"),
                TextSegment(content="World"),
                MentionSegment(user_id="user_1"),
            ]
        )

        assert msg.get_plain_text() == "Hello World"

    def test_get_plain_text_empty(self):
        """测试空消息"""
        msg = Message.create()
        assert msg.get_plain_text() == ""

    def test_get_plain_text_no_text_segments(self):
        """测试无文本片段"""
        msg = Message.create(
            [
                ImageSegment(url="http://example.com/img.jpg"),
                MentionSegment(user_id="user_1"),
            ]
        )

        assert msg.get_plain_text() == ""


class TestMessageFluentBuilder:
    """Message 流式构建测试"""

    def test_text_method(self):
        """测试 text() 方法"""
        msg = Message.create().text("Hello")

        assert len(msg.segments) == 1
        assert isinstance(msg.segments[0], TextSegment)
        assert msg.segments[0].content == "Hello"

    def test_mention_method(self):
        """测试 mention() 方法"""
        msg = Message.create().mention("user_123")

        assert len(msg.segments) == 1
        assert isinstance(msg.segments[0], MentionSegment)
        assert msg.segments[0].user_id == "user_123"

    def test_image_method_with_url(self):
        """测试 image() 方法使用 URL"""
        msg = Message.create().image(url="http://example.com/img.jpg")

        assert len(msg.segments) == 1
        assert isinstance(msg.segments[0], ImageSegment)
        assert msg.segments[0].url == "http://example.com/img.jpg"

    def test_file_method(self):
        """测试 file() 方法"""
        msg = Message.create().file("doc.pdf", url="http://example.com/doc.pdf")

        assert len(msg.segments) == 1
        assert isinstance(msg.segments[0], FileSegment)
        assert msg.segments[0].filename == "doc.pdf"

    def test_audio_method(self):
        """测试 audio() 方法"""
        msg = Message.create().audio(url="http://example.com/audio.mp3")

        assert len(msg.segments) == 1
        assert isinstance(msg.segments[0], AudioSegment)

    def test_chained_building(self):
        """测试链式构建"""
        msg = (
            Message.create()
            .text("Hello ")
            .mention("user_123")
            .text(" check this ")
            .image(url="http://example.com/img.jpg")
        )

        assert len(msg.segments) == 4
        assert isinstance(msg.segments[0], TextSegment)
        assert isinstance(msg.segments[1], MentionSegment)
        assert isinstance(msg.segments[2], TextSegment)
        assert isinstance(msg.segments[3], ImageSegment)


class TestMessageOperators:
    """Message 运算符测试"""

    def test_add_message(self):
        """测试 + 运算符添加消息"""
        msg1 = Message.create().text("Hello")
        msg2 = Message.create().text("World")

        result = msg1 + msg2
        assert len(result.segments) == 2
        assert result.segments[0].content == "Hello"
        assert result.segments[1].content == "World"

        # 原始消息不应被修改
        assert len(msg1.segments) == 1
        assert len(msg2.segments) == 1

    def test_add_list(self):
        """测试 + 运算符添加列表"""
        msg = Message.create().text("Hello")
        segs = [TextSegment(content="World"), MentionSegment(user_id="user_1")]

        result = msg + segs
        assert len(result.segments) == 3
        assert len(msg.segments) == 1  # 原始不变

    def test_add_single_segment(self):
        """测试 + 运算符添加单个片段"""
        msg = Message.create().text("Hello")
        seg = TextSegment(content="World")

        result = msg + seg
        assert len(result.segments) == 2

    def test_iadd_message(self):
        """测试 += 运算符添加消息"""
        msg1 = Message.create().text("Hello")
        msg2 = Message.create().text("World")

        msg1 += msg2
        assert len(msg1.segments) == 2
        assert msg1.segments[0].content == "Hello"
        assert msg1.segments[1].content == "World"

    def test_iadd_list(self):
        """测试 += 运算符添加列表"""
        msg = Message.create().text("Hello")
        segs = [TextSegment(content="World")]

        msg += segs
        assert len(msg.segments) == 2

    def test_iadd_single_segment(self):
        """测试 += 运算符添加单个片段"""
        msg = Message.create().text("Hello")
        seg = TextSegment(content="World")

        msg += seg
        assert len(msg.segments) == 2


class TestMessageIter:
    """Message.iter 方法测试"""

    def test_iter_empty(self):
        """测试空消息迭代"""
        msg = Message.create()
        result = list(msg.iter())
        assert result == []

    def test_iter_multiple_segments(self):
        """测试多片段迭代"""
        msg = Message.create().text("A").mention("user_1").text("B")
        result = list(msg.iter())

        assert len(result) == 3
        assert isinstance(result[0], TextSegment)
        assert isinstance(result[1], MentionSegment)
        assert isinstance(result[2], TextSegment)


class TestMessageToLLMContext:
    """Message.to_llm_context 方法测试"""

    def test_to_llm_context_single_segment(self):
        """测试单个片段的 LLM 上下文转换"""
        msg = Message.create().text("Hello World")

        ctx = MagicMock()
        result = msg.to_llm_context(ctx, "msg_123", memory_ref=1)

        assert result == "Hello World"

    def test_to_llm_context_multiple_segments(self):
        """测试多个片段的 LLM 上下文转换"""
        msg = Message.create(
            [
                TextSegment(content="Hello"),
                TextSegment(content="World"),
            ]
        )

        ctx = MagicMock()
        result = msg.to_llm_context(ctx, "msg_456")

        assert result == "Hello World"

    def test_to_llm_context_with_image(self):
        """测试包含图片的 LLM 上下文转换"""
        msg = Message.create(
            [
                TextSegment(content="Check this:"),
                ImageSegment(url="http://example.com/img.jpg"),
            ]
        )

        ctx = MagicMock()
        ctx.ref_provider.next_media_ref.return_value = "IMG:1"

        result = msg.to_llm_context(ctx, "msg_789", memory_ref=1)

        assert result == "Check this: [IMG:1]"


class TestMessageAppend:
    """Message.append 方法测试"""

    def test_append_returns_self(self):
        """测试 append 返回 self"""
        msg = Message.create()
        seg = TextSegment(content="Test")

        result = msg.append(seg)
        assert result is msg

    def test_append_adds_segment(self):
        """测试 append 添加片段"""
        msg = Message.create()
        seg = TextSegment(content="Test")

        msg.append(seg)
        assert len(msg.segments) == 1
        assert msg.segments[0] is seg
