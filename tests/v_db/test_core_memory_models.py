"""v_db/models/core_memory.py 单元测试

覆盖 CoreMemoryPayload 的:
- 模型创建和字段验证
- point_id / get_text_for_embedding
- from_core_memory / to_core_memory 转换
"""

import pytest

from nonebot_plugin_wtfllm.v_db.models.core_memory import CoreMemoryPayload
from nonebot_plugin_wtfllm.memory.items.core_memory import CoreMemory


def _make_core_memory(**kwargs) -> CoreMemory:
    defaults = dict(
        storage_id="cm-test-123",
        content="test content",
        agent_id="a1",
        created_at=1000,
        updated_at=1000,
        source="agent",
        token_count=5,
        related_entities=["u1"],
    )
    defaults.update(kwargs)
    return CoreMemory(**defaults)


def _make_payload(**kwargs) -> CoreMemoryPayload:
    defaults = dict(
        storage_id="cm-test-123",
        content="test content",
        agent_id="a1",
        created_at=1000,
        updated_at=1000,
        source="agent",
        token_count=5,
        related_entities=["u1"],
    )
    defaults.update(kwargs)
    return CoreMemoryPayload(**defaults)


class TestCoreMemoryPayloadCreate:
    """CoreMemoryPayload 创建测试"""

    def test_create_basic(self):
        p = _make_payload()
        assert p.storage_id == "cm-test-123"
        assert p.content == "test content"
        assert p.agent_id == "a1"
        assert p.source == "agent"
        assert p.token_count == 5
        assert p.related_entities == ["u1"]

    def test_create_with_group(self):
        p = _make_payload(group_id="g1")
        assert p.group_id == "g1"
        assert p.user_id is None

    def test_create_with_user(self):
        p = _make_payload(user_id="u1")
        assert p.user_id == "u1"
        assert p.group_id is None


class TestCoreMemoryPayloadProperties:
    """CoreMemoryPayload 属性测试"""

    def test_point_id(self):
        p = _make_payload()
        assert p.point_id == "cm-test-123"

    def test_get_text_for_embedding(self):
        p = _make_payload(content="semantic text here")
        assert p.get_text_for_embedding() == "semantic text here"

    def test_collection_name(self):
        assert CoreMemoryPayload.collection_name == "wtfllm_core_memory"

    def test_point_id_field(self):
        assert CoreMemoryPayload.point_id_field == "storage_id"


class TestCoreMemoryPayloadConversion:
    """CoreMemoryPayload 转换测试"""

    def test_from_core_memory(self):
        memory = _make_core_memory()
        payload = CoreMemoryPayload.from_core_memory(memory)

        assert payload.storage_id == memory.storage_id
        assert payload.content == memory.content
        assert payload.agent_id == memory.agent_id
        assert payload.created_at == memory.created_at
        assert payload.updated_at == memory.updated_at
        assert payload.source == memory.source
        assert payload.token_count == memory.token_count
        assert payload.related_entities == memory.related_entities

    def test_from_core_memory_with_group(self):
        memory = _make_core_memory(group_id="g1")
        payload = CoreMemoryPayload.from_core_memory(memory)
        assert payload.group_id == "g1"

    def test_to_core_memory(self):
        payload = _make_payload()
        memory = payload.to_core_memory()

        assert isinstance(memory, CoreMemory)
        assert memory.storage_id == payload.storage_id
        assert memory.content == payload.content
        assert memory.agent_id == payload.agent_id
        assert memory.token_count == payload.token_count
        assert memory.related_entities == payload.related_entities

    def test_roundtrip_conversion(self):
        """测试 CoreMemory -> Payload -> CoreMemory 往返转换"""
        original = _make_core_memory(
            group_id="g1",
            related_entities=["u1", "u2"],
        )
        payload = CoreMemoryPayload.from_core_memory(original)
        restored = payload.to_core_memory()

        assert restored.storage_id == original.storage_id
        assert restored.content == original.content
        assert restored.group_id == original.group_id
        assert restored.related_entities == original.related_entities


class TestCoreMemoryPayloadIndexes:
    """CoreMemoryPayload 索引配置测试"""

    def test_indexes_defined(self):
        indexes = CoreMemoryPayload.indexes
        assert "agent_id" in indexes
        assert "group_id" in indexes
        assert "user_id" in indexes
        assert "created_at" in indexes
        assert "updated_at" in indexes
        assert "source" in indexes
        assert "related_entities" in indexes
