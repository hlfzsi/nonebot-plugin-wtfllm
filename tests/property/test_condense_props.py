"""condense 模块的属性测试"""

import pytest
from hypothesis import given
from hypothesis import strategies as st


# ===== condense_text =====


@st.composite
def short_text_and_max_chars(draw):
    """生成 len(content) <= max_chars 的组合"""
    max_chars = draw(st.integers(min_value=2, max_value=200))
    content = draw(st.text(max_size=max_chars))
    return content, max_chars


@st.composite
def long_text_and_max_chars(draw):
    """生成 len(content) > max_chars 的组合"""
    max_chars = draw(st.integers(min_value=2, max_value=200))
    content = draw(st.text(min_size=max_chars + 1, max_size=max_chars + 300))
    return content, max_chars


@pytest.mark.property
class TestCondenseTextProperties:

    @given(data=short_text_and_max_chars())
    def test_short_text_unchanged(self, data):
        """当 len(content) <= max_chars 时, 输出不变"""
        from nonebot_plugin_wtfllm.memory.content.condense import condense_text

        content, max_chars = data
        text, was_condensed = condense_text(content, max_chars)
        assert text == content
        assert was_condensed is False

    @given(data=long_text_and_max_chars())
    def test_long_text_condensed(self, data):
        """当 len(content) > max_chars 时, was_condensed 为 True 且包含省略标记"""
        from nonebot_plugin_wtfllm.memory.content.condense import condense_text

        content, max_chars = data
        text, was_condensed = condense_text(content, max_chars)
        assert was_condensed is True
        assert "[...省略...]" in text

    @given(data=long_text_and_max_chars())
    def test_condensed_has_correct_head_tail(self, data):
        """压缩后文本的头尾匹配原文"""
        from nonebot_plugin_wtfllm.memory.content.condense import condense_text

        content, max_chars = data
        text, _ = condense_text(content, max_chars)
        half = max_chars // 2
        assert text.startswith(content[:half])
        assert text.endswith(content[-half:])

    @given(
        content=st.text(max_size=500),
        max_chars=st.integers(min_value=2, max_value=200),
    )
    def test_idempotent_flag(self, content, max_chars):
        """was_condensed 标志严格等于 len(content) > max_chars"""
        from nonebot_plugin_wtfllm.memory.content.condense import condense_text

        _, was_condensed = condense_text(content, max_chars)
        assert was_condensed == (len(content) > max_chars)


# ===== condense_forward =====


@pytest.mark.property
class TestCondenseForwardProperties:

    @given(count=st.integers(min_value=0, max_value=7))
    def test_small_count_returns_none(self, count):
        """当 count <= KEEP_HEAD + KEEP_TAIL + 2 (=7) 时, 返回 None"""
        from nonebot_plugin_wtfllm.memory.content.condense import condense_forward
        from nonebot_plugin_wtfllm.memory.content.segments import Node
        from nonebot_plugin_wtfllm.memory.content.message import Message
        from unittest.mock import MagicMock

        ctx = MagicMock()
        nodes = []
        for i in range(count):
            msg = Message.create().text(f"msg {i}")
            nodes.append(Node(sender=f"user_{i}", content=msg))

        result = condense_forward(nodes, ctx, "msg_id", None, 60)
        assert result is None

    @given(count=st.integers(min_value=8, max_value=50))
    def test_large_count_returns_string(self, count):
        """当 count > 7 时, 返回包含正确总数的字符串"""
        from nonebot_plugin_wtfllm.memory.content.condense import condense_forward
        from nonebot_plugin_wtfllm.memory.content.segments import Node
        from nonebot_plugin_wtfllm.memory.content.message import Message
        from unittest.mock import MagicMock

        ctx = MagicMock()
        ctx.alias_provider.get_alias.side_effect = lambda x: x
        ctx.condense = False

        nodes = []
        for i in range(count):
            msg = Message.create().text(f"msg {i}")
            nodes.append(Node(sender=f"user_{i}", content=msg))

        result = condense_forward(nodes, ctx, "msg_id", None, 60)
        assert result is not None
        assert f"共{count}条" in result

        expected_skipped = count - 3 - 2  # KEEP_HEAD=3, KEEP_TAIL=2
        assert f"省略中间{expected_skipped}条消息" in result
