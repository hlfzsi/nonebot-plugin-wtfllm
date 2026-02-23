# tests/memory/test_director.py
"""memory/director.py 单元测试"""

import pytest
from unittest.mock import MagicMock, patch, call

from nonebot_plugin_wtfllm.memory.director import MemoryContextBuilder
from nonebot_plugin_wtfllm.memory.context import LLMContext


def _make_source(priority=0, sort_key=(0.0, ""), role=None, llm_context_output=""):
    """创建一个 mock MemorySource 对象"""
    source = MagicMock()
    source.priority = priority
    source.sort_key = sort_key
    source.role = role
    source.to_llm_context = MagicMock(return_value=llm_context_output)
    source.register_all_alias = MagicMock()
    return source


# ===== __init__ 测试 =====


class TestMemoryContextBuilderInit:
    """MemoryContextBuilder.__init__ 测试"""

    def test_default_values(self):
        """测试默认值初始化"""
        builder = MemoryContextBuilder()

        assert builder.suffix_prompt is None
        assert builder.prefix_prompt is None
        assert builder.ctx is not None
        assert isinstance(builder.ctx, LLMContext)
        assert builder._sources == []
        assert builder.agent_ids is None
        assert builder._dirty is True

    def test_with_suffix_prompt(self):
        """测试带 suffix_prompt 初始化"""
        builder = MemoryContextBuilder(suffix_prompt="End of context")
        assert builder.suffix_prompt == "End of context"

    def test_with_prefix_prompt(self):
        """测试带 prefix_prompt 初始化"""
        builder = MemoryContextBuilder(prefix_prompt="Start of context")
        assert builder.prefix_prompt == "Start of context"

    def test_with_both_prompts(self):
        """测试同时设置 prefix 和 suffix"""
        builder = MemoryContextBuilder(
            prefix_prompt="PREFIX", suffix_prompt="SUFFIX"
        )
        assert builder.prefix_prompt == "PREFIX"
        assert builder.suffix_prompt == "SUFFIX"

    def test_with_ctx(self):
        """测试传入自定义 LLMContext"""
        ctx = LLMContext.create()
        builder = MemoryContextBuilder(ctx=ctx)
        assert builder.ctx is ctx

    def test_with_sources(self):
        """测试传入 sources 列表"""
        s1 = _make_source()
        s2 = _make_source()
        builder = MemoryContextBuilder(sources=[s1, s2])
        assert builder._sources == [s1, s2]

    def test_with_agent_id_string(self):
        """测试 agent_id 为字符串时转为列表"""
        builder = MemoryContextBuilder(agent_id="agent_001")
        assert builder.agent_ids == ["agent_001"]

    def test_with_agent_id_list(self):
        """测试 agent_id 为列表时直接使用"""
        builder = MemoryContextBuilder(agent_id=["agent_001", "agent_002"])
        assert builder.agent_ids == ["agent_001", "agent_002"]

    def test_with_agent_id_none(self):
        """测试 agent_id 为 None"""
        builder = MemoryContextBuilder(agent_id=None)
        assert builder.agent_ids is None

    def test_agent_id_registers_agents(self):
        """测试 agent_id 触发 alias_provider.register_agent 调用"""
        ctx = LLMContext.create()
        builder = MemoryContextBuilder(
            ctx=ctx, agent_id=["agent_a", "agent_b"]
        )
        # 验证别名已被注册
        assert ctx.alias_provider.get_alias("agent_a") is not None
        assert ctx.alias_provider.get_alias("agent_b") is not None

    def test_with_custom_ref(self):
        """测试 custom_ref 传入自定义别名映射"""
        custom_ref = {"entity_1": "Alice", "entity_2": "Bob"}
        builder = MemoryContextBuilder(custom_ref=custom_ref)

        assert builder.ctx.alias_provider.get_alias("entity_1") == "Alice"
        assert builder.ctx.alias_provider.get_alias("entity_2") == "Bob"

    def test_custom_ref_none_does_not_call_update(self):
        """测试 custom_ref 为 None 时不调用 update_aliases"""
        ctx = LLMContext.create()
        builder = MemoryContextBuilder(ctx=ctx, custom_ref=None)
        # 无异常即可，alias_map 应为空
        assert ctx.alias_provider.alias_map == {}


# ===== is_dirty 属性测试 =====


class TestIsDirty:
    """MemoryContextBuilder.is_dirty 属性测试"""

    def test_initial_dirty_state(self):
        """测试初始化后 is_dirty 为 True"""
        builder = MemoryContextBuilder()
        assert builder.is_dirty is True

    def test_dirty_reflects_internal_flag(self):
        """测试 is_dirty 反映 _dirty 内部标志"""
        builder = MemoryContextBuilder()
        builder._dirty = False
        assert builder.is_dirty is False

        builder._dirty = True
        assert builder.is_dirty is True


# ===== agent_refs 属性测试 =====


class TestAgentRefs:
    """MemoryContextBuilder.agent_refs 属性测试"""

    def test_agent_refs_without_agent_ids(self):
        """测试无 agent_ids 时返回 None"""
        builder = MemoryContextBuilder()
        assert builder.agent_refs is None

    def test_agent_refs_with_registered_agents(self):
        """测试有 agent_ids 时返回别名列表"""
        builder = MemoryContextBuilder(agent_id=["agent_1", "agent_2"])
        refs = builder.agent_refs

        assert refs is not None
        assert len(refs) == 2
        # 别名应为 Agent_1 和 Agent_2
        assert "Agent_1" in refs
        assert "Agent_2" in refs

    def test_agent_refs_filters_none(self):
        """测试 agent_refs 过滤掉 None 值"""
        builder = MemoryContextBuilder(agent_id=["agent_1"])
        # 手动清除别名以模拟 get_alias 返回 None 的场景
        builder.ctx.alias_provider._entity_to_alias.clear()
        builder.ctx.alias_provider._alias_to_entity.clear()

        refs = builder.agent_refs
        assert refs == []


# ===== add / extend / remove / index 测试 =====


class TestSourceOperations:
    """MemoryContextBuilder 源操作测试"""

    def test_add_returns_self(self):
        """测试 add 返回 self（支持链式调用）"""
        builder = MemoryContextBuilder()
        source = _make_source()
        result = builder.add(source)
        assert result is builder

    def test_add_appends_source(self):
        """测试 add 将源添加到列表"""
        builder = MemoryContextBuilder()
        s1 = _make_source()
        s2 = _make_source()
        builder.add(s1)
        builder.add(s2)
        assert builder._sources == [s1, s2]

    def test_add_marks_dirty(self):
        """测试 add 标记 _dirty 为 True"""
        builder = MemoryContextBuilder()
        builder._dirty = False
        source = _make_source()
        builder.add(source)
        assert builder._dirty is True

    def test_extend_returns_self(self):
        """测试 extend 返回 self"""
        builder = MemoryContextBuilder()
        result = builder.extend([_make_source()])
        assert result is builder

    def test_extend_adds_multiple_sources(self):
        """测试 extend 添加多个源"""
        builder = MemoryContextBuilder()
        s1 = _make_source()
        s2 = _make_source()
        s3 = _make_source()
        builder.extend([s1, s2, s3])
        assert builder._sources == [s1, s2, s3]

    def test_index_returns_correct_position(self):
        """测试 index 返回正确索引"""
        builder = MemoryContextBuilder()
        s1 = _make_source()
        s2 = _make_source()
        s3 = _make_source()
        builder.extend([s1, s2, s3])

        assert builder.index(s1) == 0
        assert builder.index(s2) == 1
        assert builder.index(s3) == 2

    def test_index_raises_value_error_for_missing(self):
        """测试 index 对不存在的源抛出 ValueError"""
        builder = MemoryContextBuilder()
        source = _make_source()

        with pytest.raises(ValueError):
            builder.index(source)

    def test_remove_returns_self(self):
        """测试 remove 返回 self"""
        builder = MemoryContextBuilder()
        source = _make_source()
        builder.add(source)
        result = builder.remove(source)
        assert result is builder

    def test_remove_removes_source(self):
        """测试 remove 从列表中移除源"""
        builder = MemoryContextBuilder()
        s1 = _make_source()
        s2 = _make_source()
        builder.extend([s1, s2])
        builder.remove(s1)
        assert builder._sources == [s2]

    def test_remove_raises_value_error_for_missing(self):
        """测试 remove 对不存在的源抛出 ValueError"""
        builder = MemoryContextBuilder()
        source = _make_source()

        with pytest.raises(ValueError):
            builder.remove(source)


# ===== to_prompt 测试 =====


class TestToPrompt:
    """MemoryContextBuilder.to_prompt 测试"""

    def test_empty_sources(self):
        """测试无源时生成空字符串"""
        builder = MemoryContextBuilder()
        # to_prompt 需要先清理脏状态
        assert builder.to_prompt() == ""

    def test_single_source(self):
        """测试单个源的输出"""
        source = _make_source(llm_context_output="Memory line 1")
        builder = MemoryContextBuilder(sources=[source])

        result = builder.to_prompt()
        assert result == "Memory line 1"

    def test_multiple_sources_joined_by_separator(self):
        """测试多个源以分隔符连接"""
        s1 = _make_source(priority=0, sort_key=(0.0, "a"), llm_context_output="Line A")
        s2 = _make_source(priority=0, sort_key=(0.0, "b"), llm_context_output="Line B")
        builder = MemoryContextBuilder(sources=[s1, s2])

        result = builder.to_prompt()
        assert "Line A" in result
        assert "Line B" in result

    def test_custom_separator(self):
        """测试自定义分隔符"""
        s1 = _make_source(priority=0, sort_key=(0.0, "a"), llm_context_output="A")
        s2 = _make_source(priority=0, sort_key=(0.0, "b"), llm_context_output="B")
        builder = MemoryContextBuilder(sources=[s1, s2])

        result = builder.to_prompt(sep=" | ")
        assert result == "A | B"

    def test_sorting_by_priority_descending(self):
        """测试按优先级降序排序"""
        low = _make_source(priority=1, sort_key=(0.0, ""), llm_context_output="LOW")
        high = _make_source(priority=10, sort_key=(0.0, ""), llm_context_output="HIGH")
        builder = MemoryContextBuilder(sources=[low, high])

        result = builder.to_prompt()
        # HIGH 应该在 LOW 之前（priority 越高排越前）
        assert result.index("HIGH") < result.index("LOW")

    def test_sorting_by_sort_key_when_same_priority(self):
        """测试同优先级下按 sort_key 排序"""
        s_b = _make_source(priority=5, sort_key=(2.0, "x"), llm_context_output="B-ITEM")
        s_a = _make_source(priority=5, sort_key=(1.0, "x"), llm_context_output="A-ITEM")
        builder = MemoryContextBuilder(sources=[s_b, s_a])

        result = builder.to_prompt()
        # sort_key[0] 较小的应该排在前面
        assert result.index("A-ITEM") < result.index("B-ITEM")

    def test_sorting_by_sort_key_second_element(self):
        """测试同优先级同 sort_key[0] 下按 sort_key[1] 排序"""
        s2 = _make_source(priority=5, sort_key=(1.0, "z"), llm_context_output="SECOND")
        s1 = _make_source(priority=5, sort_key=(1.0, "a"), llm_context_output="FIRST")
        builder = MemoryContextBuilder(sources=[s2, s1])

        result = builder.to_prompt()
        assert result.index("FIRST") < result.index("SECOND")

    def test_filters_empty_lines(self):
        """测试过滤空行"""
        s1 = _make_source(priority=0, sort_key=(0.0, "a"), llm_context_output="Content")
        s_empty = _make_source(priority=0, sort_key=(0.0, "b"), llm_context_output="")
        s_whitespace = _make_source(priority=0, sort_key=(0.0, "c"), llm_context_output="   ")
        builder = MemoryContextBuilder(sources=[s1, s_empty, s_whitespace])

        result = builder.to_prompt()
        assert result == "Content"

    def test_with_prefix_prompt(self):
        """测试带 prefix_prompt 的输出"""
        source = _make_source(llm_context_output="Body")
        builder = MemoryContextBuilder(
            prefix_prompt="=== START ===", sources=[source]
        )

        result = builder.to_prompt()
        lines = result.split("\n")
        assert lines[0] == "=== START ==="
        assert "Body" in result

    def test_with_suffix_prompt(self):
        """测试带 suffix_prompt 的输出"""
        source = _make_source(llm_context_output="Body")
        builder = MemoryContextBuilder(
            suffix_prompt="=== END ===", sources=[source]
        )

        result = builder.to_prompt()
        lines = result.split("\n")
        assert lines[-1] == "=== END ==="
        assert "Body" in result

    def test_with_both_prefix_and_suffix(self):
        """测试同时有 prefix 和 suffix"""
        source = _make_source(llm_context_output="Middle")
        builder = MemoryContextBuilder(
            prefix_prompt="HEADER", suffix_prompt="FOOTER", sources=[source]
        )

        result = builder.to_prompt()
        lines = result.split("\n")
        assert lines[0] == "HEADER"
        assert lines[-1] == "FOOTER"
        assert "Middle" in result

    def test_to_prompt_calls_ensure_clean_when_dirty(self):
        """测试 to_prompt 在脏状态时调用 _ensure_clean"""
        source = _make_source(llm_context_output="Data")
        builder = MemoryContextBuilder(sources=[source])
        assert builder._dirty is True

        builder.to_prompt()

        # _ensure_clean 应调用 register_all_alias
        source.register_all_alias.assert_called_once_with(builder.ctx)
        assert builder._dirty is False

    def test_to_prompt_skips_ensure_clean_when_not_dirty(self):
        """测试 to_prompt 在净状态时不调用 _ensure_clean"""
        source = _make_source(llm_context_output="Data")
        builder = MemoryContextBuilder(sources=[source])

        # 第一次调用：脏状态，会清理
        builder.to_prompt()
        source.register_all_alias.reset_mock()

        # 第二次调用：净状态，不应再调用
        builder.to_prompt()
        source.register_all_alias.assert_not_called()

    def test_prefix_suffix_only_no_sources(self):
        """测试只有 prefix 和 suffix 无源时的输出"""
        builder = MemoryContextBuilder(
            prefix_prompt="HEADER", suffix_prompt="FOOTER"
        )
        result = builder.to_prompt()
        assert result == "HEADER\nFOOTER"


# ===== _ensure_clean 测试 =====


class TestEnsureClean:
    """MemoryContextBuilder._ensure_clean 测试"""

    def test_calls_register_all_alias_on_each_source(self):
        """测试 _ensure_clean 对每个源调用 register_all_alias"""
        s1 = _make_source()
        s2 = _make_source()
        s3 = _make_source()
        builder = MemoryContextBuilder(sources=[s1, s2, s3])

        builder._ensure_clean()

        s1.register_all_alias.assert_called_once_with(builder.ctx)
        s2.register_all_alias.assert_called_once_with(builder.ctx)
        s3.register_all_alias.assert_called_once_with(builder.ctx)

    def test_marks_clean_after_call(self):
        """测试 _ensure_clean 后 _dirty 为 False"""
        builder = MemoryContextBuilder()
        assert builder._dirty is True

        builder._ensure_clean()
        assert builder._dirty is False


# ===== resolve_* 委托方法测试 =====


class TestResolveDelegation:
    """MemoryContextBuilder resolve_* 委托方法测试"""

    def test_resolve_aliases(self):
        """测试 resolve_aliases 委托给 alias_provider.resolve_alias"""
        builder = MemoryContextBuilder()
        builder.ctx.alias_provider.register_user("user_123")

        result = builder.resolve_aliases("User_1")
        assert result == "user_123"

    def test_resolve_aliases_not_found(self):
        """测试 resolve_aliases 不存在时返回 None"""
        builder = MemoryContextBuilder()
        assert builder.resolve_aliases("nonexistent") is None

    def test_resolve_memory_ref(self):
        """测试 resolve_memory_ref 委托给 ref_provider.get_item_by_ref"""
        builder = MemoryContextBuilder()
        mock_item = MagicMock()
        mock_item.message_id = "msg_1"
        builder.ctx.ref_provider.next_memory_ref(mock_item)

        result = builder.resolve_memory_ref(1)
        assert result is mock_item

    def test_resolve_memory_ref_not_found(self):
        """测试 resolve_memory_ref 不存在时返回 None"""
        builder = MemoryContextBuilder()
        assert builder.resolve_memory_ref(999) is None

    def test_resolve_core_memory_ref(self):
        """测试 resolve_core_memory_ref 委托给 ref_provider.get_core_memory_by_ref"""
        builder = MemoryContextBuilder()
        mock_cm = MagicMock()
        mock_cm.storage_id = "cm_1"
        builder.ctx.ref_provider.next_core_memory_ref(mock_cm)

        result = builder.resolve_core_memory_ref("CM:1")
        assert result is mock_cm

    def test_resolve_core_memory_ref_not_found(self):
        """测试 resolve_core_memory_ref 不存在时返回 None"""
        builder = MemoryContextBuilder()
        assert builder.resolve_core_memory_ref("CM:999") is None

    def test_resolve_knowledge_ref(self):
        """测试 resolve_knowledge_ref 委托给 ref_provider.get_knowledge_by_ref"""
        builder = MemoryContextBuilder()
        mock_kb = MagicMock()
        mock_kb.storage_id = "kb_1"
        builder.ctx.ref_provider.next_knowledge_ref(mock_kb)

        result = builder.resolve_knowledge_ref("KB:1")
        assert result is mock_kb

    def test_resolve_knowledge_ref_not_found(self):
        """测试 resolve_knowledge_ref 不存在时返回 None"""
        builder = MemoryContextBuilder()
        assert builder.resolve_knowledge_ref("KB:999") is None

    def test_resolve_media_ref(self):
        """测试 resolve_media_ref 委托给 ref_provider.get_media_typed"""
        from nonebot_plugin_wtfllm.memory.content.segments import ImageSegment

        builder = MemoryContextBuilder()
        img = ImageSegment(url="http://example.com/1.jpg")
        builder.ctx.ref_provider.next_media_ref(img)

        result = builder.resolve_media_ref("IMG:1", ImageSegment)
        assert result is img

    def test_resolve_media_ref_wrong_type(self):
        """测试 resolve_media_ref 类型不匹配返回 None"""
        from nonebot_plugin_wtfllm.memory.content.segments import (
            ImageSegment,
            FileSegment,
        )

        builder = MemoryContextBuilder()
        img = ImageSegment(url="http://example.com/1.jpg")
        builder.ctx.ref_provider.next_media_ref(img)

        result = builder.resolve_media_ref("IMG:1", FileSegment)
        assert result is None

    def test_resolve_media_by_memory_ref(self):
        """测试 resolve_media_by_memory_ref 委托给 ref_provider"""
        from nonebot_plugin_wtfllm.memory.content.segments import ImageSegment

        builder = MemoryContextBuilder()
        img = ImageSegment(url="http://example.com/1.jpg")
        builder.ctx.ref_provider.next_media_ref(img, memory_ref=1)

        result = builder.resolve_media_by_memory_ref(1, ImageSegment)
        assert len(result) == 1
        assert result[0] is img

    def test_resolve_media_by_memory_ref_empty(self):
        """测试 resolve_media_by_memory_ref 无结果返回空列表"""
        from nonebot_plugin_wtfllm.memory.content.segments import ImageSegment

        builder = MemoryContextBuilder()
        result = builder.resolve_media_by_memory_ref(999, ImageSegment)
        assert result == []


# ===== get_source_by_role 测试 =====


class TestGetSourceByRole:
    """MemoryContextBuilder.get_source_by_role 测试"""

    def test_found(self):
        """测试找到匹配 role 的源"""
        s1 = _make_source(role="chat_history")
        s2 = _make_source(role="core_memory")
        builder = MemoryContextBuilder(sources=[s1, s2])

        result = builder.get_source_by_role("core_memory")
        assert result is s2

    def test_not_found(self):
        """测试未找到匹配 role 的源"""
        s1 = _make_source(role="chat_history")
        builder = MemoryContextBuilder(sources=[s1])

        result = builder.get_source_by_role("nonexistent_role")
        assert result is None

    def test_returns_first_match(self):
        """测试返回第一个匹配的源"""
        s1 = _make_source(role="dup_role")
        s2 = _make_source(role="dup_role")
        builder = MemoryContextBuilder(sources=[s1, s2])

        result = builder.get_source_by_role("dup_role")
        assert result is s1

    def test_empty_sources(self):
        """测试源列表为空时返回 None"""
        builder = MemoryContextBuilder()
        assert builder.get_source_by_role("any") is None


# ===== copy 测试 =====


class TestCopy:
    """MemoryContextBuilder.copy 测试"""

    def test_copy_share_context_true(self):
        """测试共享上下文的副本"""
        s1 = _make_source()
        s2 = _make_source()
        builder = MemoryContextBuilder(
            sources=[s1, s2], agent_id=["agent_1"]
        )

        copied = builder.copy(share_context=True)

        # 共享同一个 ctx
        assert copied.ctx is builder.ctx
        # 源列表是副本（不同对象但相同内容）
        assert copied._sources == builder._sources
        assert copied._sources is not builder._sources
        # agent_ids 共享引用
        assert copied.agent_ids is builder.agent_ids

    def test_copy_share_context_false(self):
        """测试不共享上下文的副本"""
        builder = MemoryContextBuilder(
            agent_id=["agent_1"],
            custom_ref={"entity_x": "Alice"},
        )

        copied = builder.copy(share_context=False)

        # 不共享 ctx（新的实例）
        assert copied.ctx is not builder.ctx
        # agent_ids 是副本
        assert copied.agent_ids == builder.agent_ids
        assert copied.agent_ids is not builder.agent_ids

    def test_copy_empty_true(self):
        """测试创建空副本"""
        s1 = _make_source()
        builder = MemoryContextBuilder(sources=[s1])

        copied = builder.copy(empty=True)

        assert copied._sources == []
        assert len(builder._sources) == 1  # 原始不受影响

    def test_copy_empty_false(self):
        """测试非空副本保留源"""
        s1 = _make_source()
        s2 = _make_source()
        builder = MemoryContextBuilder(sources=[s1, s2])

        copied = builder.copy(empty=False)

        assert copied._sources == [s1, s2]

    def test_copy_without_agent_ids(self):
        """测试无 agent_ids 时副本的 agent_ids 为 None"""
        builder = MemoryContextBuilder()
        copied = builder.copy(share_context=False)
        assert copied.agent_ids is None

    def test_copy_source_list_independence(self):
        """测试副本的源列表与原始独立"""
        s1 = _make_source()
        builder = MemoryContextBuilder(sources=[s1])

        copied = builder.copy(share_context=True)
        new_source = _make_source()
        copied.add(new_source)

        assert new_source not in builder._sources
        assert new_source in copied._sources


# ===== __add__ 测试 =====


class TestDunderAdd:
    """MemoryContextBuilder.__add__ 测试"""

    def test_add_creates_new_builder(self):
        """测试 + 运算符创建新的 builder"""
        builder = MemoryContextBuilder()
        source = _make_source()

        new_builder = builder + source

        assert new_builder is not builder
        assert source in new_builder
        assert source not in builder

    def test_add_does_not_share_context(self):
        """测试 + 运算符不共享上下文"""
        builder = MemoryContextBuilder()
        source = _make_source()

        new_builder = builder + source
        assert new_builder.ctx is not builder.ctx


# ===== __iadd__ 测试 =====


class TestDunderIadd:
    """MemoryContextBuilder.__iadd__ 测试"""

    def test_iadd_adds_to_same_builder(self):
        """测试 += 运算符在同一 builder 上添加源"""
        builder = MemoryContextBuilder()
        source = _make_source()

        original_id = id(builder)
        builder += source

        # __iadd__ 返回 self.add(other) 即 self
        assert id(builder) == original_id
        assert source in builder


# ===== __len__ 测试 =====


class TestDunderLen:
    """MemoryContextBuilder.__len__ 测试"""

    def test_len_empty(self):
        """测试空 builder 长度为 0"""
        builder = MemoryContextBuilder()
        assert len(builder) == 0

    def test_len_with_sources(self):
        """测试有源的 builder 长度"""
        builder = MemoryContextBuilder(
            sources=[_make_source(), _make_source(), _make_source()]
        )
        assert len(builder) == 3


# ===== __contains__ 测试 =====


class TestDunderContains:
    """MemoryContextBuilder.__contains__ 测试"""

    def test_contains_true(self):
        """测试源在 builder 中"""
        source = _make_source()
        builder = MemoryContextBuilder(sources=[source])
        assert source in builder

    def test_contains_false(self):
        """测试源不在 builder 中"""
        source = _make_source()
        builder = MemoryContextBuilder()
        assert source not in builder


# ===== __getitem__ 测试 =====


class TestDunderGetitem:
    """MemoryContextBuilder.__getitem__ 测试"""

    def test_getitem_int(self):
        """测试整数索引返回单个源"""
        s1 = _make_source()
        s2 = _make_source()
        s3 = _make_source()
        builder = MemoryContextBuilder(sources=[s1, s2, s3])

        assert builder[0] is s1
        assert builder[1] is s2
        assert builder[2] is s3

    def test_getitem_negative_int(self):
        """测试负数索引"""
        s1 = _make_source()
        s2 = _make_source()
        builder = MemoryContextBuilder(sources=[s1, s2])

        assert builder[-1] is s2
        assert builder[-2] is s1

    def test_getitem_out_of_range(self):
        """测试超出范围的索引抛出 IndexError"""
        builder = MemoryContextBuilder(sources=[_make_source()])

        with pytest.raises(IndexError):
            _ = builder[5]

    def test_getitem_slice(self):
        """测试切片返回新的 MemoryContextBuilder"""
        s1 = _make_source()
        s2 = _make_source()
        s3 = _make_source()
        builder = MemoryContextBuilder(sources=[s1, s2, s3])

        sliced = builder[1:]

        assert isinstance(sliced, MemoryContextBuilder)
        assert sliced is not builder
        assert len(sliced) == 2
        assert sliced[0] is s2
        assert sliced[1] is s3

    def test_getitem_slice_shares_context(self):
        """测试切片共享上下文"""
        builder = MemoryContextBuilder()
        builder.add(_make_source())
        builder.add(_make_source())

        sliced = builder[0:1]
        assert sliced.ctx is builder.ctx

    def test_getitem_slice_marks_dirty(self):
        """测试切片后标记为脏"""
        builder = MemoryContextBuilder(sources=[_make_source(), _make_source()])
        builder._dirty = False

        sliced = builder[0:1]
        assert sliced._dirty is True

    def test_getitem_empty_slice(self):
        """测试空切片返回空 builder"""
        builder = MemoryContextBuilder(sources=[_make_source()])
        sliced = builder[5:10]

        assert isinstance(sliced, MemoryContextBuilder)
        assert len(sliced) == 0


# ===== __bool__ 测试 =====


class TestDunderBool:
    """MemoryContextBuilder.__bool__ 测试"""

    def test_bool_empty(self):
        """测试空 builder 为 False"""
        builder = MemoryContextBuilder()
        assert bool(builder) is False

    def test_bool_non_empty(self):
        """测试非空 builder 为 True"""
        builder = MemoryContextBuilder(sources=[_make_source()])
        assert bool(builder) is True


# ===== __iter__ 测试 =====


class TestDunderIter:
    """MemoryContextBuilder.__iter__ 测试"""

    def test_iter_empty(self):
        """测试迭代空 builder"""
        builder = MemoryContextBuilder()
        assert list(builder) == []

    def test_iter_returns_all_sources(self):
        """测试迭代返回所有源"""
        s1 = _make_source()
        s2 = _make_source()
        s3 = _make_source()
        builder = MemoryContextBuilder(sources=[s1, s2, s3])

        result = list(builder)
        assert result == [s1, s2, s3]

    def test_iter_preserves_order(self):
        """测试迭代保持插入顺序"""
        sources = [_make_source() for _ in range(5)]
        builder = MemoryContextBuilder(sources=sources)

        for i, source in enumerate(builder):
            assert source is sources[i]
