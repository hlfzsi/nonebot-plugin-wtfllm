"""集成测试的共享 fixtures — 使用真实对象而非 mock"""

import time

import pytest

from nonebot_plugin_wtfllm.memory.context import LLMContext
from nonebot_plugin_wtfllm.memory.content.message import Message
from nonebot_plugin_wtfllm.memory.items.base_items import GroupMemoryItem, PrivateMemoryItem
from nonebot_plugin_wtfllm.memory.items.storages import MemoryItemStream


@pytest.fixture
def llm_context():
    """创建真实的 LLMContext（condense=True）"""
    return LLMContext.create(condense=True)


@pytest.fixture
def llm_context_no_condense():
    """创建真实的 LLMContext（condense=False）"""
    return LLMContext.create(condense=False)


@pytest.fixture
def sample_group_items():
    """创建一批真实的 GroupMemoryItem 实例"""
    base_ts = int(time.time()) - 600  # 10 分钟前
    items = []
    for i in range(5):
        msg = Message.create().text(f"Group message {i}")
        if i == 2:
            msg = msg.mention("user_1")
        if i == 3:
            msg = msg.image(url=f"http://example.com/img_{i}.jpg")
        items.append(
            GroupMemoryItem(
                message_id=f"msg_grp_{i}",
                sender=f"user_{i % 4}",
                content=msg,
                created_at=base_ts + i * 60,
                agent_id="agent_bot",
                group_id="group_main",
            )
        )
    return items


@pytest.fixture
def sample_private_items():
    """创建一批真实的 PrivateMemoryItem 实例"""
    base_ts = int(time.time()) - 600
    items = []
    for i in range(3):
        msg = Message.create().text(f"Private message {i}")
        items.append(
            PrivateMemoryItem(
                message_id=f"msg_priv_{i}",
                sender=f"user_{i % 2}",
                content=msg,
                created_at=base_ts + i * 60,
                agent_id="agent_bot",
                user_id="user_0",
            )
        )
    return items


@pytest.fixture
def memory_stream(sample_group_items):
    """从 sample_group_items 创建 MemoryItemStream"""
    return MemoryItemStream.create(
        items=sample_group_items,
        prefix="--- Recent Messages ---",
        suffix="--- End ---",
    )
