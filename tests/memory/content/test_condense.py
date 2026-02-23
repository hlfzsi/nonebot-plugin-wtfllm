# tests/memory/content/test_condense.py
"""memory/content/condense.py 单元测试"""

from unittest.mock import MagicMock

from nonebot_plugin_wtfllm.memory.content.condense import (
    condense_text,
    condense_forward,
)
from nonebot_plugin_wtfllm.memory.content.segments import Node
from nonebot_plugin_wtfllm.memory.content.message import Message


class TestCondenseText:
    """condense_text 测试"""

    def test_short_text_unchanged(self):
        """短文本不压缩"""
        text, condensed = condense_text("Hello", 60)
        assert text == "Hello"
        assert condensed is False

    def test_exact_limit_unchanged(self):
        """恰好等于限制不压缩"""
        content = "a" * 60
        text, condensed = condense_text(content, 60)
        assert text == content
        assert condensed is False

    def test_over_limit_condensed(self):
        """超过限制时取头尾各半"""
        content = "a" * 30 + "b" * 31  # 61 字符
        text, condensed = condense_text(content, 60)
        assert condensed is True
        assert "\n[...省略...]\n" in text

    def test_head_tail_content(self):
        """验证头尾内容正确"""
        # 创建 100 字符的文本，max_chars=60，head=30, tail=30
        head_part = "H" * 30
        middle_part = "M" * 40
        tail_part = "T" * 30
        content = head_part + middle_part + tail_part  # 100 chars

        text, condensed = condense_text(content, 60)
        assert condensed is True
        # head 取前30字符
        assert text.startswith(head_part)
        # tail 取后30字符
        assert text.endswith(tail_part)
        # 中间有省略标记
        assert "[...省略...]" in text

    def test_empty_text(self):
        """空文本不压缩"""
        text, condensed = condense_text("", 60)
        assert text == ""
        assert condensed is False

    def test_custom_max_chars(self):
        """自定义 max_chars"""
        content = "abcdefghij"  # 10 chars
        text, condensed = condense_text(content, 6)
        assert condensed is True
        # half = 3, head = "abc", tail = "hij"
        assert text.startswith("abc")
        assert text.endswith("hij")

    def test_default_max_chars(self):
        """默认 max_chars 为 60"""
        content = "x" * 61
        text, condensed = condense_text(content)
        assert condensed is True


class TestCondenseForward:
    """condense_forward 测试"""

    @staticmethod
    def _make_nodes(count: int, text_template: str = "msg {}") -> list:
        """创建指定数量的 Node 列表"""
        nodes = []
        for i in range(count):
            msg = Message.create().text(text_template.format(i))
            nodes.append(Node(sender=f"user_{i}", content=msg))
        return nodes

    def test_few_children_returns_none(self):
        """子消息数量 <= KEEP_HEAD + KEEP_TAIL + 2 (即 <= 7) 返回 None"""
        ctx = MagicMock()
        nodes = self._make_nodes(7)
        result = condense_forward(nodes, ctx, "msg_id", None, 60)
        assert result is None

    def test_exactly_threshold_returns_none(self):
        """恰好 7 条返回 None"""
        ctx = MagicMock()
        nodes = self._make_nodes(7)
        result = condense_forward(nodes, ctx, "msg_id", None, 60)
        assert result is None

    def test_above_threshold_condensed(self):
        """超过阈值（8条）触发压缩"""
        ctx = MagicMock()
        ctx.alias_provider.get_alias.side_effect = lambda x: f"Alias_{x}"
        nodes = self._make_nodes(8)

        result = condense_forward(nodes, ctx, "msg_id", None, 60)
        assert result is not None
        assert "合并转发消息, 共8条:" in result
        assert "省略中间" in result
        assert "合并转发结束" in result

    def test_keeps_head_3_tail_2(self):
        """保留前3条和后2条"""
        ctx = MagicMock()
        ctx.alias_provider.get_alias.side_effect = lambda x: x

        nodes = self._make_nodes(10, "message_{}")

        result = condense_forward(nodes, ctx, "msg_id", None, 200)
        assert result is not None

        # 前3条: user_0 ~ user_2
        assert "user_0:" in result
        assert "user_1:" in result
        assert "user_2:" in result
        # 后2条: user_8, user_9
        assert "user_8:" in result
        assert "user_9:" in result
        # 中间被省略: user_3 ~ user_7
        assert "user_3:" not in result
        assert "user_5:" not in result
        # 省略提示
        assert "省略中间5条消息" in result

    def test_child_text_condensed(self):
        """子消息文本也应用 condense_text"""
        ctx = MagicMock()
        ctx.alias_provider.get_alias.side_effect = lambda x: x

        nodes = []
        for i in range(10):
            # 创建超长文本的子消息
            msg = Message.create().text("x" * 100)
            nodes.append(Node(sender=f"user_{i}", content=msg))

        result = condense_forward(nodes, ctx, "msg_id", None, 20)
        assert result is not None
        # 子消息文本应该被压缩，包含省略标记
        assert "[...省略...]" in result

    def test_large_forward_message(self):
        """大量子消息的转发"""
        ctx = MagicMock()
        ctx.alias_provider.get_alias.side_effect = lambda x: x

        nodes = self._make_nodes(50)
        result = condense_forward(nodes, ctx, "msg_id", None, 60)
        assert result is not None
        assert "共50条" in result
        assert "省略中间45条消息" in result
