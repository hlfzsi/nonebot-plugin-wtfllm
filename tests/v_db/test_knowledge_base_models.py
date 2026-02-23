"""v_db/models/knowledge_base.py 单元测试

覆盖 KnowledgeBasePayload 的:
- 模型创建和字段验证
- point_id / get_text_for_embedding
- from_knowledge_entry / to_knowledge_entry 转换
- 索引配置
"""

import pytest

from nonebot_plugin_wtfllm.v_db.models.knowledge_base import KnowledgeBasePayload
from nonebot_plugin_wtfllm.memory.items.knowledge_base import KnowledgeEntry


def _make_knowledge_entry(**kwargs) -> KnowledgeEntry:
    defaults = dict(
        storage_id="kb-test-123",
        content="React Hooks 是 React 16.8 引入的特性",
        title="React Hooks",
        category="技术",
        agent_id="a1",
        created_at=1000,
        updated_at=1000,
        source_session_type="group",
        source_session_id="g1",
        tags=["前端", "React"],
        token_count=15,
    )
    defaults.update(kwargs)
    return KnowledgeEntry(**defaults)


def _make_payload(**kwargs) -> KnowledgeBasePayload:
    defaults = dict(
        storage_id="kb-test-123",
        content="React Hooks 是 React 16.8 引入的特性",
        title="React Hooks",
        category="技术",
        agent_id="a1",
        created_at=1000,
        updated_at=1000,
        source_session_type="group",
        source_session_id="g1",
        tags=["前端", "React"],
        token_count=15,
    )
    defaults.update(kwargs)
    return KnowledgeBasePayload(**defaults)


class TestKnowledgeBasePayloadCreate:
    """KnowledgeBasePayload 创建测试"""

    def test_create_basic(self):
        p = _make_payload()
        assert p.storage_id == "kb-test-123"
        assert p.content == "React Hooks 是 React 16.8 引入的特性"
        assert p.title == "React Hooks"
        assert p.category == "技术"
        assert p.agent_id == "a1"
        assert p.tags == ["前端", "React"]
        assert p.token_count == 15

    def test_create_with_defaults(self):
        p = KnowledgeBasePayload(
            storage_id="kb-1",
            content="test",
            title="test title",
            agent_id="a1",
            created_at=1000,
            updated_at=1000,
        )
        assert p.category == "general"
        assert p.source_session_type == "agent"
        assert p.source_session_id is None
        assert p.tags == []
        assert p.token_count == 0

    def test_create_with_none_session_id(self):
        p = _make_payload(source_session_id=None)
        assert p.source_session_id is None


class TestKnowledgeBasePayloadProperties:
    """KnowledgeBasePayload 属性测试"""

    def test_point_id(self):
        p = _make_payload()
        assert p.point_id == "kb-test-123"

    def test_get_text_for_embedding(self):
        p = _make_payload(title="量子纠缠", content="两个粒子的量子态相互关联")
        assert p.get_text_for_embedding() == "量子纠缠: 两个粒子的量子态相互关联"

    def test_collection_name(self):
        assert KnowledgeBasePayload.collection_name == "wtfllm_knowledge_base"

    def test_point_id_field(self):
        assert KnowledgeBasePayload.point_id_field == "storage_id"


class TestKnowledgeBasePayloadConversion:
    """KnowledgeBasePayload 转换测试"""

    def test_from_knowledge_entry(self):
        entry = _make_knowledge_entry()
        payload = KnowledgeBasePayload.from_knowledge_entry(entry)

        assert payload.storage_id == entry.storage_id
        assert payload.content == entry.content
        assert payload.title == entry.title
        assert payload.category == entry.category
        assert payload.agent_id == entry.agent_id
        assert payload.created_at == entry.created_at
        assert payload.updated_at == entry.updated_at
        assert payload.tags == entry.tags
        assert payload.token_count == entry.token_count

    def test_from_knowledge_entry_with_defaults(self):
        entry = KnowledgeEntry(
            storage_id="kb-min",
            content="minimal",
            title="min",
            agent_id="a1",
            created_at=100,
            updated_at=100,
        )
        payload = KnowledgeBasePayload.from_knowledge_entry(entry)
        assert payload.category == "general"
        assert payload.tags == []

    def test_to_knowledge_entry(self):
        payload = _make_payload()
        entry = payload.to_knowledge_entry()

        assert isinstance(entry, KnowledgeEntry)
        assert entry.storage_id == payload.storage_id
        assert entry.content == payload.content
        assert entry.title == payload.title
        assert entry.category == payload.category
        assert entry.agent_id == payload.agent_id
        assert entry.token_count == payload.token_count
        assert entry.tags == payload.tags

    def test_roundtrip_conversion(self):
        """测试 KnowledgeEntry -> Payload -> KnowledgeEntry 往返转换"""
        original = _make_knowledge_entry(
            tags=["AI", "ML", "深度学习"],
            source_session_type="private",
            source_session_id="u1",
        )
        payload = KnowledgeBasePayload.from_knowledge_entry(original)
        restored = payload.to_knowledge_entry()

        assert restored.storage_id == original.storage_id
        assert restored.content == original.content
        assert restored.title == original.title
        assert restored.category == original.category
        assert restored.tags == original.tags
        assert restored.source_session_type == original.source_session_type
        assert restored.source_session_id == original.source_session_id


class TestKnowledgeBasePayloadIndexes:
    """KnowledgeBasePayload 索引配置测试"""

    def test_indexes_defined(self):
        indexes = KnowledgeBasePayload.indexes
        assert "agent_id" in indexes
        assert "category" in indexes
        assert "tags" in indexes
        assert "created_at" in indexes
        assert "updated_at" in indexes
