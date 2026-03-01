"""memory/items/knowledge_base.py 单元测试

覆盖 KnowledgeEntry 和 KnowledgeBlock 的:
- 模型创建和默认值
- 属性 (source_id, priority, sort_key)
- register_all_alias (空操作)
- to_llm_context 上下文输出
- KnowledgeBlock 空/非空场景
- hash / eq
"""

import pytest
from unittest.mock import MagicMock

from nonebot_plugin_wtfllm.memory.items.knowledge_base import (
    KnowledgeEntry,
    KnowledgeBlock,
)


def _make_ctx():
    """创建 mock LLMContext"""
    ctx = MagicMock()
    ctx.alias_provider = MagicMock()
    ctx.ref_provider = MagicMock()
    return ctx


def _make_entry(**kwargs) -> KnowledgeEntry:
    defaults = dict(
        storage_id="kb-test-1",
        content="React Hooks 是 React 16.8 引入的特性",
        title="React Hooks",
        category="技术",
        agent_id="a1",
        created_at=1000,
        updated_at=1000,
        tags=["前端", "React"],
        token_count=15,
    )
    defaults.update(kwargs)
    return KnowledgeEntry(**defaults)


class TestKnowledgeEntryCreate:
    """KnowledgeEntry 创建测试"""

    def test_create_basic(self):
        e = _make_entry()
        assert e.content == "React Hooks 是 React 16.8 引入的特性"
        assert e.title == "React Hooks"
        assert e.category == "技术"
        assert e.agent_id == "a1"
        assert e.tags == ["前端", "React"]
        assert e.token_count == 15

    def test_create_with_defaults(self):
        e = KnowledgeEntry(
            content="test",
            title="test",
            agent_id="a1",
        )
        assert e.storage_id  # UUID 自动生成
        assert e.category == "general"
        assert e.source_session_type == "agent"
        assert e.source_session_id is None
        assert e.tags == []
        assert e.token_count == 0
        assert e.created_at > 0
        assert e.updated_at > 0

    def test_create_auto_uuid(self):
        e1 = KnowledgeEntry(content="a", title="a", agent_id="a1")
        e2 = KnowledgeEntry(content="b", title="b", agent_id="a1")
        assert e1.storage_id != e2.storage_id

    def test_factory_create_with_priority(self):
        e = KnowledgeEntry.create(
            content="工厂创建内容",
            title="工厂标题",
            agent_id="a1",
            priority=0.2,
        )
        assert e.content == "工厂创建内容"
        assert e.title == "工厂标题"
        assert e.priority == pytest.approx(3.2)

    def test_factory_create_invalid_priority(self):
        with pytest.raises(ValueError, match="priority must be between 0 and 1"):
            KnowledgeEntry.create(content="x", title="t", agent_id="a1", priority=1)


class TestKnowledgeEntryProperties:
    """KnowledgeEntry 属性测试"""

    def test_source_id_format(self):
        e = _make_entry()
        assert e.source_id == "knowledge-kb-test-1"

    def test_priority(self):
        e = _make_entry()
        assert e.priority == pytest.approx(3)

    def test_sort_key(self):
        e = _make_entry(updated_at=12345)
        assert e.sort_key == (12345, "kb-test-1")

    def test_hash(self):
        e = _make_entry()
        assert hash(e) == hash("knowledge-kb-test-1")

    def test_hash_stability(self):
        """同一 source_id 的 hash 一致"""
        e1 = _make_entry(storage_id="same-id")
        e2 = _make_entry(storage_id="same-id", content="different content")
        assert hash(e1) == hash(e2)

    def test_hash_different_entries(self):
        e1 = _make_entry(storage_id="id-1")
        e2 = _make_entry(storage_id="id-2")
        assert hash(e1) != hash(e2)


class TestKnowledgeEntryRegisterAlias:
    """KnowledgeEntry.register_all_alias 测试"""

    def test_register_all_alias_is_noop(self):
        ctx = _make_ctx()
        e = _make_entry()
        e.register_all_alias(ctx)
        # 不应调用任何 alias 方法
        ctx.alias_provider.register_user.assert_not_called()
        ctx.alias_provider.register_group.assert_not_called()
        ctx.alias_provider.register_agent.assert_not_called()


class TestKnowledgeEntryToLLMContext:
    """KnowledgeEntry.to_llm_context 测试"""

    def test_format_with_tags(self):
        ctx = _make_ctx()
        ctx.ref_provider.next_knowledge_ref.return_value = "KB:1"

        e = _make_entry(
            title="React Hooks",
            content="用于在函数组件中管理状态",
            tags=["前端", "React"],
        )
        result = e.to_llm_context(ctx)

        assert result == "[KB:1] 【React Hooks】 [前端, React] 用于在函数组件中管理状态"

    def test_format_without_tags(self):
        ctx = _make_ctx()
        ctx.ref_provider.next_knowledge_ref.return_value = "KB:2"

        e = _make_entry(
            title="量子纠缠",
            content="两个粒子的量子态相互关联",
            tags=[],
        )
        result = e.to_llm_context(ctx)

        assert result == "[KB:2] 【量子纠缠】 两个粒子的量子态相互关联"

    def test_format_calls_ref_provider(self):
        ctx = _make_ctx()
        ctx.ref_provider.next_knowledge_ref.return_value = "KB:5"

        e = _make_entry()
        e.to_llm_context(ctx)

        ctx.ref_provider.next_knowledge_ref.assert_called_once_with(e)


# ===================== KnowledgeBlock 测试 =====================


class TestKnowledgeBlockCreate:
    """KnowledgeBlock 创建测试"""

    def test_create_empty(self):
        block = KnowledgeBlock()
        assert block.entries == []
        assert block.prefix is None
        assert block.suffix is None

    def test_create_with_entries(self):
        entries = [
            _make_entry(storage_id="kb-1"),
            _make_entry(storage_id="kb-2"),
        ]
        block = KnowledgeBlock(
            entries=entries,
            prefix="[开始]",
            suffix="[结束]",
        )
        assert len(block.entries) == 2
        assert block.prefix == "[开始]"
        assert block.suffix == "[结束]"

    def test_factory_create_with_priority(self):
        block = KnowledgeBlock.create(entries=[_make_entry()], priority=0.3)
        assert block.priority == pytest.approx(3.3)

    def test_factory_create_invalid_priority(self):
        with pytest.raises(ValueError, match="priority must be between 0 and 1"):
            KnowledgeBlock.create(entries=[], priority=1)


class TestKnowledgeBlockProperties:
    """KnowledgeBlock 属性测试"""

    def test_source_id(self):
        block = KnowledgeBlock()
        assert block.source_id.startswith("knowledge-block-")

    def test_priority(self):
        block = KnowledgeBlock()
        assert block.priority == pytest.approx(3)

    def test_sort_key_empty(self):
        block = KnowledgeBlock()
        key = block.sort_key
        assert key[0] == 0

    def test_sort_key_with_entries(self):
        entries = [
            _make_entry(storage_id="kb-1", updated_at=100),
            _make_entry(storage_id="kb-2", updated_at=300),
            _make_entry(storage_id="kb-3", updated_at=200),
        ]
        block = KnowledgeBlock(entries=entries)
        assert block.sort_key[0] == 300

    def test_hash_unique_per_instance(self):
        block1 = KnowledgeBlock()
        block2 = KnowledgeBlock()
        assert hash(block1) != hash(block2)


class TestKnowledgeBlockToLLMContext:
    """KnowledgeBlock.to_llm_context 测试"""

    def test_empty_block(self):
        ctx = _make_ctx()
        block = KnowledgeBlock()
        result = block.to_llm_context(ctx)
        assert result == ""

    def test_with_prefix_suffix(self):
        ctx = _make_ctx()
        ctx.ref_provider.next_knowledge_ref.side_effect = ["KB:1", "KB:2"]

        entries = [
            _make_entry(storage_id="kb-1", title="First", content="content1", tags=[], updated_at=100),
            _make_entry(storage_id="kb-2", title="Second", content="content2", tags=[], updated_at=200),
        ]
        block = KnowledgeBlock(
            entries=entries,
            prefix="[开始]",
            suffix="[结束]",
        )
        result = block.to_llm_context(ctx)

        assert result.startswith("[开始]")
        assert result.endswith("[结束]")
        # 验证排序：First 在 Second 之前 (按 updated_at 升序)
        assert result.index("First") < result.index("Second")

    def test_sorts_by_updated_at(self):
        ctx = _make_ctx()
        ctx.ref_provider.next_knowledge_ref.side_effect = ["KB:1", "KB:2"]

        entries = [
            _make_entry(storage_id="kb-1", title="Newer", content="newer content", tags=[], updated_at=200),
            _make_entry(storage_id="kb-2", title="Older", content="older content", tags=[], updated_at=100),
        ]
        block = KnowledgeBlock(entries=entries)
        result = block.to_llm_context(ctx)

        # older 应排在 newer 之前（按 updated_at 升序）
        assert result.index("Older") < result.index("Newer")


class TestKnowledgeBlockRegisterAlias:
    """KnowledgeBlock.register_all_alias 测试"""

    def test_register_all_alias_is_noop(self):
        ctx = _make_ctx()
        entries = [
            _make_entry(storage_id="kb-1"),
            _make_entry(storage_id="kb-2"),
        ]
        block = KnowledgeBlock(entries=entries)
        block.register_all_alias(ctx)
        # 不应调用任何 alias 方法
        ctx.alias_provider.register_user.assert_not_called()
        ctx.alias_provider.register_group.assert_not_called()
