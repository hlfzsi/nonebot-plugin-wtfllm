# tests/memory/items/test_storages.py
"""memory/items/storages.py 单元测试"""

import time
import pytest
from unittest.mock import MagicMock

from nonebot_plugin_wtfllm.memory.items.storages import MemoryItemStream
from nonebot_plugin_wtfllm.memory.items.base_items import PrivateMemoryItem
from nonebot_plugin_wtfllm.memory.content.message import Message


class TestMemoryItemStreamCreate:
    """MemoryItemStream.create 测试"""

    def test_create_empty(self):
        """测试创建空流"""
        stream = MemoryItemStream.create()
        assert len(stream) == 0

    def test_create_with_none(self):
        """测试使用 None 创建"""
        stream = MemoryItemStream.create(None)
        assert len(stream) == 0

    def test_create_with_single_item(self):
        """测试使用单个项创建"""
        item = PrivateMemoryItem(
            message_id="msg_001",
            sender="user_123",
            content=Message.create().text("Test"),
            agent_id="agent_1",
            user_id="user_123",
        )

        stream = MemoryItemStream.create(items=item)
        assert len(stream) == 1

    def test_create_with_list(self):
        """测试使用列表创建"""
        items = [
            PrivateMemoryItem(
                message_id=f"msg_{i}",
                sender="user_123",
                content=Message.create().text(f"Test {i}"),
                agent_id="agent_1",
                user_id="user_123",
            )
            for i in range(3)
        ]

        stream = MemoryItemStream.create(items=items)
        assert len(stream) == 3


class TestMemoryItemStreamProperties:
    """MemoryItemStream 属性测试"""

    def test_started_at_empty(self):
        """测试空流的 started_at"""
        stream = MemoryItemStream.create()
        assert stream.started_at == 0

    def test_started_at_returns_min(self):
        """测试 started_at 返回最小值"""
        items = [
            PrivateMemoryItem(
                message_id="msg_1",
                sender="user",
                content=Message.create(),
                created_at=3000,
                agent_id="agent",
                user_id="user",
            ),
            PrivateMemoryItem(
                message_id="msg_2",
                sender="user",
                content=Message.create(),
                created_at=1000,  # 最小
                agent_id="agent",
                user_id="user",
            ),
            PrivateMemoryItem(
                message_id="msg_3",
                sender="user",
                content=Message.create(),
                created_at=2000,
                agent_id="agent",
                user_id="user",
            ),
        ]

        stream = MemoryItemStream.create(items=items)
        assert stream.started_at == 1000

    def test_ended_at_empty(self):
        """测试空流的 ended_at"""
        stream = MemoryItemStream.create()
        assert stream.ended_at == 0

    def test_ended_at_returns_max(self):
        """测试 ended_at 返回最大值"""
        items = [
            PrivateMemoryItem(
                message_id="msg_1",
                sender="user",
                content=Message.create(),
                created_at=1000,
                agent_id="agent",
                user_id="user",
            ),
            PrivateMemoryItem(
                message_id="msg_2",
                sender="user",
                content=Message.create(),
                created_at=3000,  # 最大
                agent_id="agent",
                user_id="user",
            ),
        ]

        stream = MemoryItemStream.create(items=items)
        assert stream.ended_at == 3000

    def test_source_id_format(self):
        """测试 source_id 格式"""
        item = PrivateMemoryItem(
            message_id="msg_001",
            sender="user",
            content=Message.create(),
            agent_id="agent",
            user_id="user",
        )
        stream = MemoryItemStream.create(items=item)

        assert stream.source_id.startswith("stream-")

    def test_priority_is_zero(self):
        """测试优先级为 0"""
        stream = MemoryItemStream.create()
        assert stream.priority == pytest.approx(0)


class TestMemoryItemStreamOperations:
    """MemoryItemStream 操作测试"""

    def test_append(self):
        """测试 append 方法"""
        stream = MemoryItemStream.create()
        item = PrivateMemoryItem(
            message_id="msg_001",
            sender="user",
            content=Message.create(),
            agent_id="agent",
            user_id="user",
        )

        result = stream.append(item)

        assert len(stream) == 1
        assert result is stream  # 返回 self

    def test_add_stream(self):
        """测试 + 运算符添加流"""
        item1 = PrivateMemoryItem(
            message_id="msg_1",
            sender="user",
            content=Message.create(),
            agent_id="agent",
            user_id="user",
        )
        item2 = PrivateMemoryItem(
            message_id="msg_2",
            sender="user",
            content=Message.create(),
            agent_id="agent",
            user_id="user",
        )

        stream1 = MemoryItemStream.create(items=item1)
        stream2 = MemoryItemStream.create(items=item2)

        result = stream1 + stream2

        assert len(result) == 2
        assert len(stream1) == 1  # 原始不变

    def test_add_list(self):
        """测试 + 运算符添加列表"""
        stream = MemoryItemStream.create()
        items = [
            PrivateMemoryItem(
                message_id=f"msg_{i}",
                sender="user",
                content=Message.create(),
                agent_id="agent",
                user_id="user",
            )
            for i in range(2)
        ]

        result = stream + items
        assert len(result) == 2

    def test_add_single_item(self):
        """测试 + 运算符添加单个项"""
        stream = MemoryItemStream.create()
        item = PrivateMemoryItem(
            message_id="msg_1",
            sender="user",
            content=Message.create(),
            agent_id="agent",
            user_id="user",
        )

        result = stream + item
        assert len(result) == 1

    def test_iadd_stream(self):
        """测试 += 运算符添加流"""
        item1 = PrivateMemoryItem(
            message_id="msg_1",
            sender="user",
            content=Message.create(),
            agent_id="agent",
            user_id="user",
        )
        item2 = PrivateMemoryItem(
            message_id="msg_2",
            sender="user",
            content=Message.create(),
            agent_id="agent",
            user_id="user",
        )

        stream1 = MemoryItemStream.create(items=item1)
        stream2 = MemoryItemStream.create(items=item2)

        stream1 += stream2

        assert len(stream1) == 2

    def test_iadd_list(self):
        """测试 += 运算符添加列表"""
        stream = MemoryItemStream.create()
        items = [
            PrivateMemoryItem(
                message_id=f"msg_{i}",
                sender="user",
                content=Message.create(),
                agent_id="agent",
                user_id="user",
            )
            for i in range(2)
        ]

        stream += items
        assert len(stream) == 2

    def test_iadd_single_item(self):
        """测试 += 运算符添加单个项"""
        stream = MemoryItemStream.create()
        item = PrivateMemoryItem(
            message_id="msg_1",
            sender="user",
            content=Message.create(),
            agent_id="agent",
            user_id="user",
        )

        stream += item
        assert len(stream) == 1


class TestMemoryItemStreamToLLMContext:
    """MemoryItemStream.to_llm_context 测试"""

    def test_to_llm_context_empty(self):
        """测试空流的 LLM 上下文"""
        stream = MemoryItemStream.create()
        ctx = MagicMock()

        result = stream.to_llm_context(ctx)
        assert result == ""

    def test_to_llm_context_sorts_by_time(self):
        """测试按时间排序"""
        # 创建乱序的项
        items = [
            PrivateMemoryItem(
                message_id="msg_3",
                sender="user",
                content=Message.create().text("Third"),
                created_at=3000,
                agent_id="agent",
                user_id="user",
            ),
            PrivateMemoryItem(
                message_id="msg_1",
                sender="user",
                content=Message.create().text("First"),
                created_at=1000,
                agent_id="agent",
                user_id="user",
            ),
            PrivateMemoryItem(
                message_id="msg_2",
                sender="user",
                content=Message.create().text("Second"),
                created_at=2000,
                agent_id="agent",
                user_id="user",
            ),
        ]

        stream = MemoryItemStream.create(items=items)

        ctx = MagicMock()
        ctx.alias_provider.get_alias.return_value = "User_1"
        ctx.ref_provider.next_memory_ref.side_effect = [1, 2, 3]

        result = stream.to_llm_context(ctx)

        # 验证排序：First 应该在 Second 之前，Second 在 Third 之前
        first_pos = result.find("First")
        second_pos = result.find("Second")
        third_pos = result.find("Third")

        assert first_pos < second_pos < third_pos

    def test_to_llm_context_no_max_token(self):
        """没有 _max_token 时不熔断"""
        items = [
            PrivateMemoryItem(
                message_id=f"msg_{i}",
                sender="user",
                content=Message.create().text("x" * 500),
                created_at=1000 + i,
                agent_id="agent",
                user_id="user",
            )
            for i in range(5)
        ]
        stream = MemoryItemStream.create(items=items)
        ctx = MagicMock()
        ctx.alias_provider.get_alias.return_value = "User_1"
        ctx.ref_provider.next_memory_ref.side_effect = list(range(1, 6))

        result = stream.to_llm_context(ctx)
        # 所有5条消息都应存在
        assert result.count("User_1:") == 5

    def test_to_llm_context_max_token_fuse(self):
        """_max_token 熔断：丢弃最旧的消息直到 token 数符合限制"""
        items = [
            PrivateMemoryItem(
                message_id=f"msg_{i}",
                sender="user",
                content=Message.create().text(f"Content_{i} " + "pad " * 50),
                created_at=1000 + i,
                agent_id="agent",
                user_id="user",
            )
            for i in range(5)
        ]
        # 设置很小的 max_token，迫使一些消息被丢弃
        stream = MemoryItemStream.create(items=items, max_token=100)
        ctx = MagicMock()
        ctx.alias_provider.get_alias.return_value = "User_1"
        ctx.ref_provider.next_memory_ref.side_effect = list(range(1, 6))

        result = stream.to_llm_context(ctx)

        # 最旧的消息应该被丢弃，最新的消息应该保留
        # Content_4 是最新的，应该存在
        assert "Content_4" in result
        # Content_0 是最旧的，很可能被丢弃
        # （具体取决于 token 数，但至少部分消息应被移除）
        assert result.count("User_1:") < 5

    def test_to_llm_context_max_token_preserves_prefix_suffix(self):
        """_max_token 熔断时保留 prefix 和 suffix"""
        items = [
            PrivateMemoryItem(
                message_id=f"msg_{i}",
                sender="user",
                content=Message.create().text("pad " * 100),
                created_at=1000 + i,
                agent_id="agent",
                user_id="user",
            )
            for i in range(5)
        ]
        stream = MemoryItemStream.create(items=items, max_token=50)
        stream.prefix = "[PREFIX]"
        stream.suffix = "[SUFFIX]"
        ctx = MagicMock()
        ctx.alias_provider.get_alias.return_value = "User_1"
        ctx.ref_provider.next_memory_ref.side_effect = list(range(1, 6))

        result = stream.to_llm_context(ctx)

        # prefix 和 suffix 必须保留
        assert result.startswith("[PREFIX]")
        assert result.endswith("[SUFFIX]")

    def test_to_llm_context_max_token_large_enough(self):
        """_max_token 足够大时不丢弃任何消息"""
        items = [
            PrivateMemoryItem(
                message_id=f"msg_{i}",
                sender="user",
                content=Message.create().text(f"Short_{i}"),
                created_at=1000 + i,
                agent_id="agent",
                user_id="user",
            )
            for i in range(3)
        ]
        stream = MemoryItemStream.create(items=items, max_token=10000)
        ctx = MagicMock()
        ctx.alias_provider.get_alias.return_value = "User_1"
        ctx.ref_provider.next_memory_ref.side_effect = list(range(1, 4))

        result = stream.to_llm_context(ctx)

        # 所有消息都保留
        assert "Short_0" in result
        assert "Short_1" in result
        assert "Short_2" in result
