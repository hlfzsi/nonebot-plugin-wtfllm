# tests/memory/items/test_base_items.py
"""memory/items/base_items.py 单元测试"""

import pytest
from unittest.mock import MagicMock

from nonebot_plugin_wtfllm.memory.items.base_items import (
    PrivateMemoryItem,
    GroupMemoryItem,
)
from nonebot_plugin_wtfllm.memory.content.message import Message
from nonebot_plugin_wtfllm.memory.content.segments import MentionSegment


class TestPrivateMemoryItem:
    """PrivateMemoryItem 测试"""

    def test_create_basic(self):
        """测试基本创建"""
        content = Message.create().text("Hello")
        item = PrivateMemoryItem(
            message_id="msg_001",
            sender="user_123",
            content=content,
            created_at=1000000,
            agent_id="agent_1",
            user_id="user_123",
        )

        assert item.message_id == "msg_001"
        assert item.sender == "user_123"
        assert item.created_at == 1000000
        assert item.agent_id == "agent_1"
        assert item.user_id == "user_123"
        assert item.memory_type == "private"

    def test_is_from_agent_true(self):
        """测试消息来自 Agent"""
        item = PrivateMemoryItem(
            message_id="msg_001",
            sender="agent_1",  # sender == agent_id
            content=Message.create(),
            agent_id="agent_1",
            user_id="user_123",
        )
        assert item.is_from_agent is True

    def test_is_from_agent_false(self):
        """测试消息不是来自 Agent"""
        item = PrivateMemoryItem(
            message_id="msg_001",
            sender="user_123",  # sender != agent_id
            content=Message.create(),
            agent_id="agent_1",
            user_id="user_123",
        )
        assert item.is_from_agent is False

    def test_get_plain_text(self):
        """测试获取纯文本"""
        content = Message.create().text("Hello ").mention("user_456").text("World")
        item = PrivateMemoryItem(
            message_id="msg_001",
            sender="user_123",
            content=content,
            agent_id="agent_1",
            user_id="user_123",
        )
        assert item.get_plain_text() == "Hello World"

    def test_to_llm_context_basic(self):
        """测试 to_llm_context 基本格式"""
        content = Message.create().text("Test message")
        item = PrivateMemoryItem(
            message_id="msg_001",
            sender="user_123",
            content=content,
            agent_id="agent_1",
            user_id="user_123",
        )

        ctx = MagicMock()
        ctx.alias_provider.get_alias.return_value = "User_1"
        ctx.ref_provider.next_memory_ref.return_value = 1

        result = item.to_llm_context(ctx)
        assert "[1]" in result
        assert "User_1" in result
        assert "Test message" in result

    def test_to_llm_context_with_reply(self):
        """测试 to_llm_context 带回复"""
        content = Message.create().text("Reply content")
        item = PrivateMemoryItem(
            message_id="msg_002",
            related_message_id="msg_001",
            sender="user_123",
            content=content,
            agent_id="agent_1",
            user_id="user_123",
        )

        ctx = MagicMock()
        ctx.alias_provider.get_alias.return_value = "User_1"
        ctx.ref_provider.next_memory_ref.return_value = 2
        ctx.ref_provider.get_ref_by_item_id.return_value = 1

        result = item.to_llm_context(ctx)
        assert "in reply to [1]" in result

    def test_register_entities(self):
        """测试实体注册"""
        content = Message.create().text("Hi ").mention("user_mentioned")
        item = PrivateMemoryItem(
            message_id="msg_001",
            sender="user_123",
            content=content,
            agent_id="agent_1",
            user_id="user_123",
        )

        ctx = MagicMock()
        item.register_entities(ctx)

        # 应注册 agent
        ctx.alias_provider.register_agent.assert_called_once_with("agent_1")
        # 应注册 sender（因为不是 agent）
        ctx.alias_provider.register_user.assert_called()


class TestGroupMemoryItem:
    """GroupMemoryItem 测试"""

    def test_create_basic(self):
        """测试基本创建"""
        content = Message.create().text("Group message")
        item = GroupMemoryItem(
            message_id="msg_001",
            sender="user_123",
            content=content,
            created_at=1000000,
            agent_id="agent_1",
            group_id="group_001",
        )

        assert item.message_id == "msg_001"
        assert item.sender == "user_123"
        assert item.agent_id == "agent_1"
        assert item.group_id == "group_001"
        assert item.memory_type == "group"

    def test_to_llm_context_includes_group(self):
        """测试 to_llm_context 包含群组信息"""
        content = Message.create().text("Group message")
        item = GroupMemoryItem(
            message_id="msg_001",
            sender="user_123",
            content=content,
            agent_id="agent_1",
            group_id="group_001",
        )

        ctx = MagicMock()
        ctx.alias_provider.get_alias.side_effect = lambda x: {
            "user_123": "User_1",
            "group_001": "Group_1",
        }.get(x)
        ctx.ref_provider.next_memory_ref.return_value = 1

        result = item.to_llm_context(ctx)
        assert "[1]" in result
        assert "User_1" in result

    def test_to_llm_context_with_reply(self):
        """测试 to_llm_context 带回复"""
        content = Message.create().text("Reply in group")
        item = GroupMemoryItem(
            message_id="msg_002",
            related_message_id="msg_001",
            sender="user_123",
            content=content,
            agent_id="agent_1",
            group_id="group_001",
        )

        ctx = MagicMock()
        ctx.alias_provider.get_alias.side_effect = lambda x: {
            "user_123": "User_1",
            "group_001": "Group_1",
        }.get(x)
        ctx.ref_provider.next_memory_ref.return_value = 2
        ctx.ref_provider.get_ref_by_item_id.return_value = 1

        result = item.to_llm_context(ctx)
        assert "in reply to [1]" in result

    def test_register_entities_includes_group(self):
        """测试实体注册包含群组"""
        content = Message.create().text("Hi")
        item = GroupMemoryItem(
            message_id="msg_001",
            sender="user_123",
            content=content,
            agent_id="agent_1",
            group_id="group_001",
        )

        ctx = MagicMock()
        item.register_entities(ctx)

        # 应注册 agent
        ctx.alias_provider.register_agent.assert_called_once_with("agent_1")
        # 应注册 group
        ctx.alias_provider.register_group.assert_called_once_with("group_001")


class TestMemoryItemRelatedMessage:
    """MemoryItem related_message_id 测试"""

    def test_related_message_id_none(self):
        """测试无关联消息"""
        item = PrivateMemoryItem(
            message_id="msg_001",
            sender="user_123",
            content=Message.create(),
            agent_id="agent_1",
            user_id="user_123",
        )
        assert item.related_message_id is None

    def test_related_message_id_set(self):
        """测试有关联消息"""
        item = PrivateMemoryItem(
            message_id="msg_002",
            related_message_id="msg_001",
            sender="user_123",
            content=Message.create(),
            agent_id="agent_1",
            user_id="user_123",
        )
        assert item.related_message_id == "msg_001"


class TestMemoryItemContent:
    """MemoryItem content 相关测试"""

    def test_content_with_text(self):
        """测试文本内容"""
        content = Message.create().text("Just text")
        item = PrivateMemoryItem(
            message_id="msg_001",
            sender="user_123",
            content=content,
            agent_id="agent_1",
            user_id="user_123",
        )
        assert item.content.has("text")
        assert item.get_plain_text() == "Just text"

    def test_content_with_mixed(self):
        """测试混合内容"""
        content = (
            Message.create()
            .text("Hello ")
            .mention("user_456")
            .image(url="http://example.com/img.jpg")
        )
        item = PrivateMemoryItem(
            message_id="msg_001",
            sender="user_123",
            content=content,
            agent_id="agent_1",
            user_id="user_123",
        )

        assert item.content.has("text")
        assert item.content.has("mention")
        assert item.content.has("image")


class TestMemoryItemSerialization:
    """MemoryItem 序列化测试"""

    def test_private_item_model_dump(self):
        """测试 PrivateMemoryItem 序列化"""
        content = Message.create().text("Test")
        item = PrivateMemoryItem(
            message_id="msg_001",
            sender="user_123",
            content=content,
            created_at=1000000,
            agent_id="agent_1",
            user_id="user_123",
        )

        data = item.model_dump()
        assert data["message_id"] == "msg_001"
        assert data["memory_type"] == "private"
        assert data["user_id"] == "user_123"

    def test_group_item_model_dump(self):
        """测试 GroupMemoryItem 序列化"""
        content = Message.create().text("Test")
        item = GroupMemoryItem(
            message_id="msg_001",
            sender="user_123",
            content=content,
            created_at=1000000,
            agent_id="agent_1",
            group_id="group_001",
        )

        data = item.model_dump()
        assert data["message_id"] == "msg_001"
        assert data["memory_type"] == "group"
        assert data["group_id"] == "group_001"
