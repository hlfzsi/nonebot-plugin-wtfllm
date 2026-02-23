# tests/memory/test_segments.py
"""memory/content/segments.py 单元测试 - 全面覆盖各 Segment 类型"""

import os
from pathlib import Path

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from nonebot_plugin_wtfllm.memory.content.segments import (
    TextSegment,
    EmojiSegment,
    UnknownSegment,
    MentionSegment,
    ImageSegment,
    VideoSegment,
    FileSegment,
    AudioSegment,
    HyperSegment,
    ForwardSegment,
    Node,
)
from nonebot_plugin_wtfllm.memory.content.message import Message


# ---------------------------------------------------------------------------
# Helper: build a mock LLMContext
# ---------------------------------------------------------------------------


def _make_ctx(
    condense: bool = False,
    alias_return: str | None = "alias_name",
    ref_return: str = "IMG:1",
):
    """Create a mock LLMContext with configurable behaviour."""
    ctx = MagicMock()
    ctx.condense = condense
    ctx.alias_provider.get_alias.return_value = alias_return
    ctx.ref_provider.next_media_ref.return_value = ref_return
    ctx.ref_provider.get_item_by_ref.return_value = None
    return ctx


# ===========================================================================
# TextSegment
# ===========================================================================


class TestTextSegment:
    """TextSegment 的创建、unique_key、_format_content、__eq__、__hash__ 测试"""

    def test_creation(self):
        seg = TextSegment(content="hello")
        assert seg.type == "text"
        assert seg.content == "hello"
        assert seg.created_at > 0

    def test_unique_key_contains_content_and_type(self):
        seg = TextSegment(content="abc", created_at=100)
        key = seg.unique_key
        assert "self.type:text" in key
        assert "content:abc" in key
        assert "created_at:100" in key

    def test_format_content_no_condense(self):
        """ctx.condense=False: content is returned as-is regardless of length."""
        long_text = "X" * 200
        seg = TextSegment(content=long_text)
        ctx = _make_ctx(condense=False)

        result = seg._format_content(ctx)
        assert result == long_text

    def test_format_content_condense_short_text(self):
        """ctx.condense=True but text <= max_chars: no truncation."""
        short_text = "short"
        seg = TextSegment(content=short_text)
        ctx = _make_ctx(condense=True)

        result = seg._format_content(ctx)
        assert result == short_text

    def test_format_content_condense_long_text(self):
        """ctx.condense=True and text > max_chars (60): triggers truncation."""
        # Build text longer than memory_item_max_chars (60)
        long_text = "A" * 30 + "B" * 40 + "C" * 30
        seg = TextSegment(content=long_text)
        ctx = _make_ctx(condense=True)

        result = seg._format_content(ctx)
        assert "[...省略...]" in result
        assert result.startswith("A" * 30)
        assert result.endswith("C" * 30)
        # original content is not mutated
        assert seg.content == long_text

    def test_eq_same_content_same_time(self):
        a = TextSegment(content="hi", created_at=1)
        b = TextSegment(content="hi", created_at=1)
        assert a == b

    def test_eq_different_content(self):
        a = TextSegment(content="hi", created_at=1)
        b = TextSegment(content="bye", created_at=1)
        assert a != b

    def test_eq_different_type(self):
        a = TextSegment(content="hi", created_at=1)
        b = EmojiSegment(name="hi", created_at=1)
        assert a != b

    def test_eq_with_non_segment(self):
        a = TextSegment(content="hi", created_at=1)
        assert a != "not a segment"

    def test_hash_consistent_with_eq(self):
        a = TextSegment(content="hi", created_at=1)
        b = TextSegment(content="hi", created_at=1)
        assert hash(a) == hash(b)

    def test_hash_differs_for_different_content(self):
        a = TextSegment(content="hi", created_at=1)
        b = TextSegment(content="bye", created_at=1)
        assert hash(a) != hash(b)

    def test_to_llm_context_sets_message_id(self):
        seg = TextSegment(content="test")
        ctx = _make_ctx()
        seg.to_llm_context(ctx, "msg_42")
        assert seg.message_id == "msg_42"

    def test_message_id_raises_when_not_set(self):
        seg = TextSegment(content="test")
        with pytest.raises(ValueError, match="Message ID has not been set"):
            _ = seg.message_id


# ===========================================================================
# EmojiSegment
# ===========================================================================


class TestEmojiSegment:
    """EmojiSegment 的创建和 _format_content 测试"""

    def test_creation(self):
        seg = EmojiSegment(name="smile")
        assert seg.type == "emoji"
        assert seg.name == "smile"
        assert seg.url is None

    def test_creation_with_url(self):
        seg = EmojiSegment(name="grin", url="http://example.com/emoji.png")
        assert seg.name == "grin"
        assert seg.url == "http://example.com/emoji.png"

    def test_format_content(self):
        seg = EmojiSegment(name="laugh")
        ctx = _make_ctx()
        result = seg._format_content(ctx)
        assert result == "[表情: laugh]"

    def test_unique_key(self):
        seg = EmojiSegment(name="cry", url="http://x.com/cry.png", created_at=50)
        key = seg.unique_key
        assert "self.type:emoji" in key
        assert "name:cry" in key
        assert "url:http://x.com/cry.png" in key
        assert "created_at:50" in key


# ===========================================================================
# UnknownSegment
# ===========================================================================


class TestUnknownSegment:
    """UnknownSegment 的创建和 _format_content 测试"""

    def test_creation(self):
        seg = UnknownSegment(original_type="poke")
        assert seg.type == "unknown"
        assert seg.original_type == "poke"

    def test_creation_with_none(self):
        seg = UnknownSegment(original_type=None)
        assert seg.original_type is None

    def test_format_content(self):
        seg = UnknownSegment(original_type="poke")
        ctx = _make_ctx()
        result = seg._format_content(ctx)
        assert result == "[未知格式消息]"

    def test_unique_key(self):
        seg = UnknownSegment(original_type="share", created_at=77)
        key = seg.unique_key
        assert "self.type:unknown" in key
        assert "original_type:share" in key
        assert "created_at:77" in key


# ===========================================================================
# MentionSegment
# ===========================================================================


class TestMentionSegment:
    """MentionSegment 的创建、验证、_format_content 测试"""

    def test_create_with_user_id(self):
        seg = MentionSegment(user_id="user_123")
        assert seg.type == "mention"
        assert seg.user_id == "user_123"
        assert seg.at_all is False

    def test_create_with_at_all(self):
        seg = MentionSegment(at_all=True)
        assert seg.user_id is None
        assert seg.at_all is True

    def test_validation_neither_raises(self):
        with pytest.raises(ValueError, match="must have either 'user_id' or 'at_all'"):
            MentionSegment()

    def test_validation_both_raises(self):
        with pytest.raises(ValueError, match="cannot have both"):
            MentionSegment(user_id="user_1", at_all=True)

    def test_format_content_user_id_with_alias(self):
        seg = MentionSegment(user_id="user_456")
        ctx = _make_ctx(alias_return="AliasName")
        result = seg._format_content(ctx)
        assert result == "<@AliasName>"
        ctx.alias_provider.get_alias.assert_called_once_with("user_456")

    def test_format_content_user_id_no_alias(self):
        seg = MentionSegment(user_id="user_789")
        ctx = _make_ctx(alias_return=None)
        result = seg._format_content(ctx)
        assert result == "<@user_789>"

    def test_format_content_at_all(self):
        seg = MentionSegment(at_all=True)
        ctx = _make_ctx()
        result = seg._format_content(ctx)
        assert result == "<@全体成员>"

    def test_unique_key(self):
        seg = MentionSegment(user_id="uid", created_at=10)
        key = seg.unique_key
        assert "self.type:mention" in key
        assert "user_id:uid" in key
        assert "at_all:False" in key
        assert "created_at:10" in key


# ===========================================================================
# MediaBaseSegment (tested via ImageSegment)
# ===========================================================================


class TestMediaBaseSegmentViaImage:
    """MediaBaseSegment 行为通过 ImageSegment 测试"""

    # --- model_post_init validation ---

    def test_must_have_url_or_local_path_or_expired(self):
        """Neither url, local_path, nor expired=True raises ValueError."""
        with pytest.raises(
            ValueError, match="must have either 'url' or 'local_path' or be expired"
        ):
            ImageSegment()

    def test_create_with_url(self):
        seg = ImageSegment(url="http://example.com/img.jpg")
        assert seg.type == "image"
        assert seg.url == "http://example.com/img.jpg"
        assert seg.local_path is None
        assert seg.expired is False

    def test_create_with_local_path(self, tmp_path):
        """local_path must be absolute; tmp_path provides an absolute path."""
        img = tmp_path / "photo.png"
        img.write_bytes(b"fake png data")
        seg = ImageSegment(local_path=img)
        assert seg.local_path == img

    def test_create_expired(self):
        """expired=True is a valid state even without url/local_path."""
        seg = ImageSegment(expired=True)
        assert seg.expired is True
        assert seg.url is None
        assert seg.local_path is None

    # --- available property ---

    def test_available_expired(self):
        seg = ImageSegment(expired=True)
        assert seg.available is False

    def test_available_local_path_exists(self, tmp_path):
        img = tmp_path / "photo.png"
        img.write_bytes(b"data")
        seg = ImageSegment(local_path=img)
        assert seg.available is True

    def test_available_local_path_missing(self, tmp_path):
        img = tmp_path / "missing.png"
        # Create with url so post_init passes, then set local_path to missing file
        seg = ImageSegment(url="http://example.com/img.jpg")
        seg.local_path = img  # file does not exist
        seg.url = None
        assert seg.available is False

    def test_available_url_only(self):
        seg = ImageSegment(url="http://example.com/img.jpg")
        assert seg.available is True

    def test_available_nothing(self):
        seg = ImageSegment(expired=True)
        seg.expired = False
        # Now it has no url, no local_path, not expired
        assert seg.available is False

    # --- unbound_local ---

    def test_unbound_local_deletes_file(self, tmp_path):
        img = tmp_path / "to_delete.png"
        img.write_bytes(b"data")
        seg = ImageSegment(local_path=img)
        assert img.exists()

        seg.unbound_local()
        assert not img.exists()
        assert seg.local_path is None
        assert seg.expired is True

    def test_unbound_local_no_file(self):
        seg = ImageSegment(url="http://example.com/img.jpg")
        seg.unbound_local()
        assert seg.local_path is None
        assert seg.expired is True

    def test_unbound_local_expired_false(self, tmp_path):
        img = tmp_path / "keep.png"
        img.write_bytes(b"data")
        seg = ImageSegment(local_path=img)
        seg.unbound_local(expired=False)
        assert seg.expired is False
        assert seg.local_path is None

    # --- get_bytes_async ---

    @pytest.mark.asyncio
    async def test_get_bytes_from_cache(self):
        seg = ImageSegment(url="http://example.com/img.jpg")
        seg._bytes = b"cached_data"
        result = await seg.get_bytes_async()
        assert result == b"cached_data"

    @pytest.mark.asyncio
    async def test_get_bytes_from_local_file(self, tmp_path):
        img = tmp_path / "local.png"
        img.write_bytes(b"local_file_data")
        seg = ImageSegment(local_path=img)
        result = await seg.get_bytes_async()
        assert result == b"local_file_data"
        # Subsequent call returns from cache
        assert seg._bytes == b"local_file_data"

    @pytest.mark.asyncio
    async def test_get_bytes_url_no_download_raises(self):
        seg = ImageSegment(url="http://example.com/img.jpg")
        with pytest.raises(ValueError, match="Cannot get bytes from URL without downloading"):
            await seg.get_bytes_async(download=False)

    @pytest.mark.asyncio
    async def test_get_bytes_nothing_raises(self):
        seg = ImageSegment(expired=True)
        with pytest.raises(ValueError, match="Cannot get bytes without local_path or url"):
            await seg.get_bytes_async()

    # --- _format_content ---

    def test_format_content_returns_ref(self):
        seg = ImageSegment(url="http://example.com/img.jpg")
        ctx = _make_ctx(ref_return="IMG:1")
        result = seg._format_content(ctx, memory_ref=5)
        assert result == "[IMG:1]"
        ctx.ref_provider.next_media_ref.assert_called_once_with(seg, 5)

    def test_format_content_with_desc(self):
        seg = ImageSegment(url="http://example.com/img.jpg", desc="a photo")
        ctx = _make_ctx(ref_return="IMG:2")
        result = seg._format_content(ctx)
        assert result == "[IMG:2 - a photo]"

    # --- unique_key ---

    def test_unique_key(self):
        seg = ImageSegment(url="http://x.com/a.jpg", desc="desc1", created_at=99)
        key = seg.unique_key
        assert "self.type:image" in key
        assert "url:http://x.com/a.jpg" in key
        assert "desc:desc1" in key
        assert "created_at:99" in key


# ===========================================================================
# FileSegment
# ===========================================================================


class TestFileSegment:
    """FileSegment 的创建和自定义 _format_content 测试"""

    def test_creation(self):
        seg = FileSegment(filename="report.pdf", url="http://x.com/report.pdf")
        assert seg.type == "file"
        assert seg.filename == "report.pdf"

    def test_format_content_includes_filename(self):
        seg = FileSegment(filename="report.pdf", url="http://x.com/report.pdf")
        ctx = _make_ctx(ref_return="FILE:1")
        result = seg._format_content(ctx)
        assert result == "[FILE:1: report.pdf]"

    def test_format_content_with_desc(self):
        seg = FileSegment(
            filename="data.csv", url="http://x.com/data.csv", desc="monthly data"
        )
        ctx = _make_ctx(ref_return="FILE:2")
        result = seg._format_content(ctx)
        assert result == "[FILE:2: data.csv - monthly data]"

    def test_unique_key_includes_filename(self):
        seg = FileSegment(filename="doc.txt", url="http://x.com/doc.txt", created_at=10)
        key = seg.unique_key
        assert "filename:doc.txt" in key
        assert "self.type:file" in key


# ===========================================================================
# VideoSegment / AudioSegment (brief coverage)
# ===========================================================================


class TestVideoSegment:
    def test_creation_with_duration(self):
        seg = VideoSegment(url="http://x.com/v.mp4", duration=120)
        assert seg.type == "video"
        assert seg.duration == 120

    def test_unique_key_includes_duration(self):
        seg = VideoSegment(url="http://x.com/v.mp4", duration=60, created_at=1)
        key = seg.unique_key
        assert "duration:60" in key


class TestAudioSegment:
    def test_creation_with_duration(self):
        seg = AudioSegment(url="http://x.com/a.mp3", duration=30)
        assert seg.type == "audio"
        assert seg.duration == 30

    def test_unique_key_includes_duration(self):
        seg = AudioSegment(url="http://x.com/a.mp3", duration=15, created_at=1)
        key = seg.unique_key
        assert "duration:15" in key


# ===========================================================================
# HyperSegment
# ===========================================================================


class TestHyperSegment:
    """HyperSegment 的创建和 _format_content 测试 (json / xml)"""

    def test_creation_json(self):
        seg = HyperSegment(format="json", content='{"key":"value"}')
        assert seg.type == "hyper"
        assert seg.format == "json"

    def test_creation_xml(self):
        seg = HyperSegment(format="xml", content="<root><child/></root>")
        assert seg.format == "xml"

    def test_format_content_json_valid(self):
        seg = HyperSegment(format="json", content='{"a":1,"b":2}')
        ctx = _make_ctx()
        result = seg._format_content(ctx)
        assert result.startswith("[Rich Message: JSON]")
        assert "```json" in result
        # orjson pretty-prints, so we should see indentation
        assert '"a"' in result
        assert '"b"' in result

    def test_format_content_json_invalid(self):
        """Invalid JSON is rendered as-is."""
        seg = HyperSegment(format="json", content="not valid json {{{")
        ctx = _make_ctx()
        result = seg._format_content(ctx)
        assert "[Rich Message: JSON]" in result
        assert "not valid json {{{" in result

    def test_format_content_xml_valid(self):
        seg = HyperSegment(format="xml", content="<root><child/></root>")
        ctx = _make_ctx()
        result = seg._format_content(ctx)
        assert "[Rich Message: XML]" in result
        assert "```xml" in result
        assert "<root>" in result

    def test_format_content_xml_invalid_raises_type_error(self):
        """Completely unparseable XML (lxml recover returns None) raises TypeError.

        This documents a known gap in the except clause -- only XMLSyntaxError
        and UnicodeDecodeError are caught, so the TypeError from
        etree.tostring(None, ...) propagates.
        """
        seg = HyperSegment(format="xml", content="<<<totally broken>>>")
        ctx = _make_ctx()
        with pytest.raises(TypeError):
            seg._format_content(ctx)

    def test_format_content_xml_recoverable(self):
        """Partially malformed XML that lxml can recover renders successfully."""
        seg = HyperSegment(format="xml", content="<root><unclosed>")
        ctx = _make_ctx()
        result = seg._format_content(ctx)
        assert "[Rich Message: XML]" in result
        assert "```xml" in result

    def test_unique_key(self):
        seg = HyperSegment(format="json", content='{"x":1}', created_at=5)
        key = seg.unique_key
        assert "self.type:hyper" in key
        assert "format:json" in key
        assert "created_at:5" in key


# ===========================================================================
# ForwardSegment
# ===========================================================================


class TestForwardSegment:
    """ForwardSegment 的 _format_content 和 unique_key 测试"""

    @staticmethod
    def _make_nodes(count: int, text_template: str = "msg {}") -> list:
        nodes = []
        for i in range(count):
            msg = Message.create().text(text_template.format(i))
            nodes.append(Node(sender=f"user_{i}", content=msg))
        return nodes

    def test_empty_children(self):
        seg = ForwardSegment(children=[])
        seg._message_id = "msg_1"
        ctx = _make_ctx()
        result = seg._format_content(ctx)
        assert result == "[合并转发消息, 共0条, 空]"

    def test_non_empty_children_no_condense(self):
        """With condense=False and a small number of children, all are rendered."""
        nodes = self._make_nodes(3)
        seg = ForwardSegment(children=nodes)
        seg._message_id = "msg_1"

        ctx = _make_ctx(condense=False)
        ctx.alias_provider.get_alias.side_effect = lambda x: x

        result = seg._format_content(ctx)
        assert "合并转发消息, 共3条:" in result
        assert "合并转发结束" in result
        for i in range(3):
            assert f"user_{i}:" in result
        assert "省略" not in result

    def test_many_children_condensed(self):
        """With condense=True and > 7 children, middle messages are omitted."""
        nodes = self._make_nodes(10)
        seg = ForwardSegment(children=nodes)
        seg._message_id = "msg_1"

        ctx = _make_ctx(condense=True)
        ctx.alias_provider.get_alias.side_effect = lambda x: x

        result = seg._format_content(ctx)
        assert "合并转发消息, 共10条:" in result
        assert "省略中间5条消息" in result
        assert "合并转发结束" in result
        # Head 3 nodes present
        assert "user_0:" in result
        assert "user_1:" in result
        assert "user_2:" in result
        # Tail 2 nodes present
        assert "user_8:" in result
        assert "user_9:" in result
        # Middle nodes absent
        assert "user_4:" not in result

    def test_many_children_no_condense(self):
        """With condense=False, even > 7 children are all rendered."""
        nodes = self._make_nodes(10)
        seg = ForwardSegment(children=nodes)
        seg._message_id = "msg_1"

        ctx = _make_ctx(condense=False)
        ctx.alias_provider.get_alias.side_effect = lambda x: x

        result = seg._format_content(ctx)
        assert "合并转发消息, 共10条:" in result
        assert "合并转发结束" in result
        for i in range(10):
            assert f"user_{i}:" in result
        assert "省略" not in result

    def test_unique_key_empty(self):
        seg = ForwardSegment(children=[], created_at=1)
        key = seg.unique_key
        assert "self.type:forward" in key
        assert "children:[]" in key
        assert "created_at:1" in key

    def test_unique_key_with_children(self):
        nodes = self._make_nodes(2)
        seg = ForwardSegment(children=nodes, created_at=99)
        key = seg.unique_key
        assert "self.type:forward" in key
        assert "user_0:" in key
        assert "user_1:" in key
        assert "created_at:99" in key

    def test_node_created_at(self):
        """Node.created_at delegates to its content message."""
        msg = Message.create([TextSegment(content="x", created_at=42)])
        node = Node(sender="s", content=msg)
        assert node.created_at == 42

    def test_node_group_id_default(self):
        msg = Message.create().text("hi")
        node = Node(sender="s", content=msg)
        assert node.group_id is None

    def test_node_with_group_id(self):
        msg = Message.create().text("hi")
        node = Node(sender="s", group_id="g_1", content=msg)
        assert node.group_id == "g_1"
