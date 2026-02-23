"""memory/items/core_memory.py 单元测试

覆盖 CoreMemory 和 CoreMemoryBlock 的:
- 模型创建和默认值
- 属性 (source_id, priority, sort_key)
- normalize_placeholders 占位符归一化
- _render_content 渲染
- to_llm_context 上下文输出
- register_all_alias 别名注册
- CoreMemoryBlock 空/非空场景
"""

import pytest
from unittest.mock import MagicMock

from nonebot_plugin_wtfllm.memory.items.core_memory import CoreMemory, CoreMemoryBlock


def _make_ctx():
    """创建 mock LLMContext"""
    ctx = MagicMock()
    ctx.alias_provider = MagicMock()
    ctx.ref_provider = MagicMock()
    return ctx


class TestCoreMemoryCreate:
    """CoreMemory 创建测试"""

    def test_create_basic(self):
        m = CoreMemory(
            content="User likes Python",
            agent_id="a1",
        )
        assert m.content == "User likes Python"
        assert m.agent_id == "a1"
        assert m.storage_id  # UUID 自动生成
        assert m.source == "agent"
        assert m.token_count == 0
        assert m.related_entities == []
        assert m.group_id is None
        assert m.user_id is None

    def test_create_with_group(self):
        m = CoreMemory(
            content="Group discussion topic",
            agent_id="a1",
            group_id="g1",
        )
        assert m.group_id == "g1"
        assert m.user_id is None

    def test_create_with_user(self):
        m = CoreMemory(
            content="Private memory",
            agent_id="a1",
            user_id="u1",
        )
        assert m.user_id == "u1"
        assert m.group_id is None

    def test_create_compression_source(self):
        m = CoreMemory(
            content="Compressed content",
            agent_id="a1",
            source="compression",
            token_count=50,
        )
        assert m.source == "compression"
        assert m.token_count == 50


class TestCoreMemoryProperties:
    """CoreMemory 属性测试"""

    def test_source_id_format(self):
        m = CoreMemory(content="test", agent_id="a1")
        assert m.source_id.startswith("core-memory-")
        assert m.storage_id in m.source_id

    def test_priority(self):
        m = CoreMemory(content="test", agent_id="a1")
        assert m.priority == 2

    def test_sort_key(self):
        m = CoreMemory(
            content="test",
            agent_id="a1",
            updated_at=12345,
        )
        assert m.sort_key == (12345, m.storage_id)

    def test_hash(self):
        m = CoreMemory(content="test", agent_id="a1")
        assert hash(m) == hash(m.source_id)


class TestCoreMemoryNormalizePlaceholders:
    """CoreMemory.normalize_placeholders 测试"""

    def test_replaces_alias_with_entity_id(self):
        ctx = _make_ctx()
        ctx.alias_provider.resolve_alias.return_value = "real_user_123"

        m = CoreMemory(
            content="{{User_1}} likes coding",
            agent_id="a1",
        )
        m.normalize_placeholders(ctx)

        assert "{{real_user_123}}" in m.content
        assert "real_user_123" in m.related_entities
        ctx.alias_provider.resolve_alias.assert_called_with("User_1")

    def test_no_match_keeps_original(self):
        ctx = _make_ctx()
        ctx.alias_provider.resolve_alias.return_value = None

        m = CoreMemory(
            content="{{User_1}} likes coding",
            agent_id="a1",
        )
        m.normalize_placeholders(ctx)

        assert "{{User_1}}" in m.content
        assert m.related_entities == []

    def test_multiple_placeholders(self):
        ctx = _make_ctx()

        def resolve(alias):
            mapping = {"User_1": "u1", "User_2": "u2"}
            return mapping.get(alias)

        ctx.alias_provider.resolve_alias.side_effect = resolve

        m = CoreMemory(
            content="{{User_1}} and {{User_2}} are friends",
            agent_id="a1",
        )
        m.normalize_placeholders(ctx)

        assert "{{u1}}" in m.content
        assert "{{u2}}" in m.content
        assert "u1" in m.related_entities
        assert "u2" in m.related_entities

    def test_no_duplicates_in_related_entities(self):
        ctx = _make_ctx()
        ctx.alias_provider.resolve_alias.return_value = "u1"

        m = CoreMemory(
            content="{{User_1}} said {{User_1}} likes coding",
            agent_id="a1",
        )
        m.normalize_placeholders(ctx)

        assert m.related_entities.count("u1") == 1


class TestCoreMemoryRenderContent:
    """CoreMemory._render_content 测试"""

    def test_replaces_entity_id_with_alias(self):
        ctx = _make_ctx()
        ctx.alias_provider.get_alias.return_value = "User_1"

        m = CoreMemory(
            content="{{real_user_123}} likes coding",
            agent_id="a1",
        )
        result = m._render_content(ctx)

        assert "{{User_1}}" in result

    def test_no_alias_keeps_original(self):
        ctx = _make_ctx()
        ctx.alias_provider.get_alias.return_value = None

        m = CoreMemory(
            content="{{unknown_id}} test",
            agent_id="a1",
        )
        result = m._render_content(ctx)

        assert "{{unknown_id}}" in result


class TestCoreMemoryToLLMContext:
    """CoreMemory.to_llm_context 测试"""

    def test_format_with_ref_no_session(self):
        """无 group_id/user_id 时不包含会话标签"""
        ctx = _make_ctx()
        ctx.ref_provider.next_core_memory_ref.return_value = "CM:1"
        ctx.alias_provider.get_alias.return_value = "User_1"

        m = CoreMemory(
            content="{{u1}} likes coding",
            agent_id="a1",
        )
        result = m.to_llm_context(ctx)

        assert result.startswith("[CM:1]")
        assert "{{User_1}}" in result
        # 无会话标签
        assert "(Group" not in result
        assert "私聊" not in result

    def test_format_with_group_session_tag(self):
        """有 group_id 时包含群组别名标签"""
        ctx = _make_ctx()
        ctx.ref_provider.next_core_memory_ref.return_value = "CM:2"

        def get_alias(entity_id):
            if entity_id == "g1":
                return "Group_1"
            return None

        ctx.alias_provider.get_alias.side_effect = get_alias

        m = CoreMemory(
            content="some memory",
            agent_id="a1",
            group_id="g1",
        )
        result = m.to_llm_context(ctx)

        assert "[CM:2]" in result
        assert "(Group_1)" in result
        assert "some memory" in result

    def test_format_with_user_session_tag(self):
        """有 user_id 时包含用户私聊别名标签"""
        ctx = _make_ctx()
        ctx.ref_provider.next_core_memory_ref.return_value = "CM:3"

        def get_alias(entity_id):
            if entity_id == "u1":
                return "User_1"
            return None

        ctx.alias_provider.get_alias.side_effect = get_alias

        m = CoreMemory(
            content="private memory",
            agent_id="a1",
            user_id="u1",
        )
        result = m.to_llm_context(ctx)

        assert "[CM:3]" in result
        assert "(User_1 私聊)" in result
        assert "private memory" in result

    def test_format_with_group_no_alias_fallback(self):
        """group_id 无别名时使用原始 ID"""
        ctx = _make_ctx()
        ctx.ref_provider.next_core_memory_ref.return_value = "CM:4"
        ctx.alias_provider.get_alias.return_value = None

        m = CoreMemory(
            content="memory",
            agent_id="a1",
            group_id="raw_group_id_123",
        )
        result = m.to_llm_context(ctx)

        assert "(raw_group_id_123)" in result

    def test_format_with_user_no_alias_fallback(self):
        """user_id 无别名时使用原始 ID"""
        ctx = _make_ctx()
        ctx.ref_provider.next_core_memory_ref.return_value = "CM:5"
        ctx.alias_provider.get_alias.return_value = None

        m = CoreMemory(
            content="memory",
            agent_id="a1",
            user_id="raw_user_id_456",
        )
        result = m.to_llm_context(ctx)

        assert "(raw_user_id_456 私聊)" in result


class TestCoreMemoryRegisterAlias:
    """CoreMemory.register_all_alias 测试"""

    def test_registers_group(self):
        ctx = _make_ctx()
        m = CoreMemory(
            content="test",
            agent_id="a1",
            group_id="g1",
        )
        m.register_all_alias(ctx)
        ctx.alias_provider.register_group.assert_called_once_with("g1")

    def test_registers_user_id(self):
        ctx = _make_ctx()
        m = CoreMemory(
            content="test",
            agent_id="a1",
            user_id="u1",
        )
        m.register_all_alias(ctx)
        ctx.alias_provider.register_user.assert_any_call("u1")

    def test_registers_related_entities(self):
        ctx = _make_ctx()
        m = CoreMemory(
            content="test",
            agent_id="a1",
            related_entities=["u1", "u2"],
        )
        m.register_all_alias(ctx)
        assert ctx.alias_provider.register_user.call_count == 2

    def test_no_group_no_register(self):
        ctx = _make_ctx()
        m = CoreMemory(
            content="test",
            agent_id="a1",
        )
        m.register_all_alias(ctx)
        ctx.alias_provider.register_group.assert_not_called()

    def test_fallback_scans_content_for_unregistered_entities(self):
        """related_entities 为空但 content 中有 {{entity_id}} 时，兜底扫描注册"""
        ctx = _make_ctx()
        m = CoreMemory(
            content="{{user123}} likes Python",
            agent_id="a1",
            related_entities=[],
        )
        m.register_all_alias(ctx)
        ctx.alias_provider.register_user.assert_any_call("user123")

    def test_fallback_no_duplicate_when_already_in_related_entities(self):
        """entity 已在 related_entities 中时，不重复注册"""
        ctx = _make_ctx()
        m = CoreMemory(
            content="{{user123}} likes Python",
            agent_id="a1",
            related_entities=["user123"],
        )
        m.register_all_alias(ctx)
        # register_user 应只被调用一次（来自 related_entities 循环）
        ctx.alias_provider.register_user.assert_called_once_with("user123")

    def test_fallback_skips_group_id_and_user_id(self):
        """group_id 和 user_id 已由专门逻辑注册，兜底扫描应跳过"""
        ctx = _make_ctx()
        m = CoreMemory(
            content="{{g1}} and {{u1}} discussion",
            agent_id="a1",
            group_id="g1",
            user_id="u1",
        )
        m.register_all_alias(ctx)
        ctx.alias_provider.register_group.assert_called_once_with("g1")
        # register_user 仅对 user_id 调用一次，不因 content 中的 {{u1}} 和 {{g1}} 重复
        ctx.alias_provider.register_user.assert_called_once_with("u1")


# ===================== CoreMemoryBlock 测试 =====================


class TestCoreMemoryBlockCreate:
    """CoreMemoryBlock 创建测试"""

    def test_create_empty(self):
        block = CoreMemoryBlock()
        assert block.memories == []
        assert block.prefix is None
        assert block.suffix is None

    def test_create_with_memories(self):
        memories = [
            CoreMemory(content="m1", agent_id="a1"),
            CoreMemory(content="m2", agent_id="a1"),
        ]
        block = CoreMemoryBlock(
            memories=memories,
            prefix="[开始]",
            suffix="[结束]",
        )
        assert len(block.memories) == 2
        assert block.prefix == "[开始]"
        assert block.suffix == "[结束]"


class TestCoreMemoryBlockProperties:
    """CoreMemoryBlock 属性测试"""

    def test_source_id(self):
        block = CoreMemoryBlock()
        assert block.source_id.startswith("core-memory-block-")

    def test_priority(self):
        block = CoreMemoryBlock()
        assert block.priority == 2

    def test_sort_key_empty(self):
        block = CoreMemoryBlock()
        key = block.sort_key
        assert key[0] == 0

    def test_sort_key_with_memories(self):
        memories = [
            CoreMemory(content="m1", agent_id="a1", updated_at=100),
            CoreMemory(content="m2", agent_id="a1", updated_at=300),
            CoreMemory(content="m3", agent_id="a1", updated_at=200),
        ]
        block = CoreMemoryBlock(memories=memories)
        assert block.sort_key[0] == 300


class TestCoreMemoryBlockToLLMContext:
    """CoreMemoryBlock.to_llm_context 测试"""

    def test_empty_block(self):
        ctx = _make_ctx()
        block = CoreMemoryBlock()
        result = block.to_llm_context(ctx)
        assert result == ""

    def test_with_prefix_suffix(self):
        ctx = _make_ctx()
        ctx.ref_provider.next_core_memory_ref.side_effect = ["CM:1", "CM:2"]
        ctx.alias_provider.get_alias.return_value = None

        memories = [
            CoreMemory(content="first", agent_id="a1", updated_at=100),
            CoreMemory(content="second", agent_id="a1", updated_at=200),
        ]
        block = CoreMemoryBlock(
            memories=memories,
            prefix="[开始]",
            suffix="[结束]",
        )
        result = block.to_llm_context(ctx)

        assert result.startswith("[开始]")
        assert result.endswith("[结束]")
        # 验证排序：first 在 second 之前
        assert result.index("first") < result.index("second")

    def test_sorts_by_updated_at(self):
        ctx = _make_ctx()
        ctx.ref_provider.next_core_memory_ref.side_effect = ["CM:1", "CM:2"]
        ctx.alias_provider.get_alias.return_value = None

        memories = [
            CoreMemory(content="newer", agent_id="a1", updated_at=200),
            CoreMemory(content="older", agent_id="a1", updated_at=100),
        ]
        block = CoreMemoryBlock(memories=memories)
        result = block.to_llm_context(ctx)

        # older 应排在 newer 之前（按 updated_at 升序）
        assert result.index("older") < result.index("newer")


class TestCoreMemoryBlockRegisterAlias:
    """CoreMemoryBlock.register_all_alias 测试"""

    def test_registers_all_memories(self):
        ctx = _make_ctx()
        memories = [
            CoreMemory(content="m1", agent_id="a1", group_id="g1"),
            CoreMemory(content="m2", agent_id="a1", related_entities=["u1"]),
        ]
        block = CoreMemoryBlock(memories=memories)
        block.register_all_alias(ctx)

        ctx.alias_provider.register_group.assert_called_once_with("g1")
        ctx.alias_provider.register_user.assert_called_once_with("u1")


# ===================== 反渲染 → 渲染 全流程测试 =====================


class TestCoreMemoryRoundTrip:
    """normalize_placeholders (反渲染) → _render_content (渲染) 全流程测试

    使用真实 AliasProvider 而非 mock，验证完整的双向转换。
    """

    def _make_real_ctx(self):
        from nonebot_plugin_wtfllm.memory.context import LLMContext
        return LLMContext.create()

    def test_roundtrip_single_user(self):
        """{{User_1}} → normalize → {{real_id}} → render → {{User_1}}"""
        ctx = self._make_real_ctx()
        ctx.alias_provider.register_user("ID123")  # User_1

        m = CoreMemory(
            content="{{User_1}} 是一名大三CS学生",
            agent_id="a1",
        )
        # 反渲染：{{User_1}} → {{ID123}}
        m.normalize_placeholders(ctx)
        assert m.content == "{{ID123}} 是一名大三CS学生"
        assert "ID123" in m.related_entities

        # 渲染：{{ID123}} → {{User_1}}
        rendered = m._render_content(ctx)
        assert rendered == "{{User_1}} 是一名大三CS学生"

    def test_roundtrip_multiple_entities(self):
        """多个实体的双向转换"""
        ctx = self._make_real_ctx()
        ctx.alias_provider.register_user("alice_id")   # User_1
        ctx.alias_provider.register_user("bob_id")     # User_2
        ctx.alias_provider.register_group("group_id")  # Group_1

        m = CoreMemory(
            content="{{User_1}} and {{User_2}} are friends in {{Group_1}}",
            agent_id="a1",
        )
        m.normalize_placeholders(ctx)
        assert "{{alice_id}}" in m.content
        assert "{{bob_id}}" in m.content
        assert "{{group_id}}" in m.content

        rendered = m._render_content(ctx)
        assert "{{User_1}}" in rendered
        assert "{{User_2}}" in rendered
        assert "{{Group_1}}" in rendered

    def test_roundtrip_preserves_unresolvable(self):
        """无法解析的占位符在两个方向上都保持原样"""
        ctx = self._make_real_ctx()
        ctx.alias_provider.register_user("known_id")  # User_1

        m = CoreMemory(
            content="{{User_1}} talks to {{User_99}}",
            agent_id="a1",
        )
        m.normalize_placeholders(ctx)
        # User_1 被解析，User_99 保持原样
        assert "{{known_id}}" in m.content
        assert "{{User_99}}" in m.content

    def test_roundtrip_with_new_context(self):
        """在新 context 中注册同一实体后，渲染应生成新的别名"""
        ctx1 = self._make_real_ctx()
        ctx1.alias_provider.register_user("ID123")  # User_1

        m = CoreMemory(
            content="{{User_1}} likes Python",
            agent_id="a1",
        )
        m.normalize_placeholders(ctx1)
        assert m.content == "{{ID123}} likes Python"

        # 新 context，ID123 仍被注册但可能有不同的别名编号
        ctx2 = self._make_real_ctx()
        ctx2.alias_provider.register_user("other_user")  # User_1
        ctx2.alias_provider.register_user("ID123")       # User_2

        rendered = m._render_content(ctx2)
        assert "{{User_2}}" in rendered

    def test_roundtrip_after_register_all_alias(self):
        """通过 register_all_alias 注册后渲染，验证完整流程"""
        ctx1 = self._make_real_ctx()
        ctx1.alias_provider.register_user("user_abc")  # User_1

        m = CoreMemory(
            content="{{User_1}} 喜欢编程",
            agent_id="a1",
        )
        m.normalize_placeholders(ctx1)
        assert m.content == "{{user_abc}} 喜欢编程"
        assert "user_abc" in m.related_entities

        # 模拟新会话：用新 context，通过 register_all_alias 自动注册
        ctx2 = self._make_real_ctx()
        m.register_all_alias(ctx2)

        rendered = m._render_content(ctx2)
        assert "{{User_1}}" in rendered

    def test_roundtrip_compressed_memory_with_fallback(self):
        """压缩来源的记忆（related_entities 可能为空）通过兜底扫描仍能正确渲染"""
        ctx = self._make_real_ctx()

        m = CoreMemory(
            content="{{user_abc}} 喜欢编程",
            agent_id="a1",
            source="compression",
            related_entities=[],  # 模拟压缩后 related_entities 为空的情况
        )
        # register_all_alias 应通过兜底扫描注册 user_abc
        m.register_all_alias(ctx)

        rendered = m._render_content(ctx)
        assert "{{User_1}}" in rendered


class TestCoreMemoryRoundTripWithCustomRef:
    """使用自定义映射字典 (custom_ref) 的反渲染 → 渲染全流程测试

    模拟实际场景：custom_ref 来自 cached_aliases，格式为 {entity_id: 用户昵称}，
    如 {"测试2288383": "小明", "group123": "测试群"}。
    """

    def _make_real_ctx(self):
        from nonebot_plugin_wtfllm.memory.context import LLMContext
        return LLMContext.create()

    def test_custom_ref_normalize_and_render(self):
        """自定义映射字典场景下的完整反渲染→渲染流程

        custom_ref {"测试2288383": "小明"} 设入后，
        LLM 写 {{小明}} → normalize 解析为 {{测试2288383}} → 再次渲染回 {{小明}}
        """
        ctx = self._make_real_ctx()
        ctx.alias_provider.update_aliases({"测试2288383": "小明"})

        m = CoreMemory(
            content="{{小明}} 是一名大三CS学生",
            agent_id="a1",
        )
        m.normalize_placeholders(ctx)
        # {{小明}} 通过 resolve_alias("小明") 解析为 "测试2288383"
        assert m.content == "{{测试2288383}} 是一名大三CS学生"
        assert "测试2288383" in m.related_entities

        # 渲染回来
        rendered = m._render_content(ctx)
        assert "{{小明}}" in rendered

    def test_custom_ref_with_register_user_then_normalize(self):
        """先 register_user 再设 custom_ref 时的行为

        register_user 会生成 User_1 格式别名，
        之后 update_aliases 会用自定义别名覆盖。
        LLM 写 {{小明}} 能正确反渲染。
        """
        ctx = self._make_real_ctx()
        ctx.alias_provider.register_user("测试2288383")  # -> User_1
        # update_aliases 覆盖为自定义昵称
        ctx.alias_provider.update_aliases({"测试2288383": "小明"})

        # 此时 resolve_alias("User_1") 应返回 None（被覆盖了）
        assert ctx.alias_provider.resolve_alias("User_1") is None
        # resolve_alias("小明") 应返回 entity_id
        assert ctx.alias_provider.resolve_alias("小明") == "测试2288383"
        # get_alias 应返回新别名
        assert ctx.alias_provider.get_alias("测试2288383") == "小明"

        # LLM 写 {{小明}} — 能通过 resolve_alias 正确解析
        m = CoreMemory(
            content="{{小明}} 是一名大三CS学生",
            agent_id="a1",
        )
        m.normalize_placeholders(ctx)
        assert m.content == "{{测试2288383}} 是一名大三CS学生"
        assert "测试2288383" in m.related_entities

        # LLM 写 {{User_1}} — User_1 已被覆盖，resolve 返回 None，保持原样
        m2 = CoreMemory(
            content="{{User_1}} 是一名大三CS学生",
            agent_id="a1",
        )
        m2.normalize_placeholders(ctx)
        assert m2.content == "{{User_1}} 是一名大三CS学生"
        assert m2.related_entities == []

    def test_custom_ref_render_uses_custom_alias(self):
        """渲染时使用自定义别名"""
        ctx = self._make_real_ctx()
        ctx.alias_provider.update_aliases({"测试2288383": "小明"})

        # 假设 content 中已经存储了 entity_id 格式
        m = CoreMemory(
            content="{{测试2288383}} 是一名大三CS学生",
            agent_id="a1",
            related_entities=["测试2288383"],
        )
        rendered = m._render_content(ctx)
        # 渲染出 {{小明}} 而不是 {{User_1}}
        assert "{{小明}}" in rendered

    def test_custom_ref_roundtrip_via_director(self):
        """模拟 MemoryContextBuilder 使用 custom_ref 的完整流程"""
        from nonebot_plugin_wtfllm.memory.director import MemoryContextBuilder

        custom_ref = {
            "测试2288383": "小明",
            "group456": "测试群",
        }
        builder = MemoryContextBuilder(
            agent_id="agent1",
            custom_ref=custom_ref,
        )
        ctx = builder.ctx

        # custom_ref 和 register_agent 都已应用
        assert ctx.alias_provider.get_alias("测试2288383") == "小明"
        assert ctx.alias_provider.get_alias("group456") == "测试群"
        assert ctx.alias_provider.get_alias("agent1") == "Agent_1"

        # LLM 使用自定义别名写核心记忆
        m = CoreMemory(
            content="{{Agent_1}} 记住了 {{小明}} 喜欢编程",
            agent_id="agent1",
        )
        m.normalize_placeholders(ctx)
        # Agent_1 能被 resolve 为 "agent1"
        assert "{{agent1}}" in m.content
        # {{小明}} 通过 resolve_alias 解析为 "测试2288383"
        assert "{{测试2288383}}" in m.content
        assert "测试2288383" in m.related_entities

        # 渲染回来
        rendered = m._render_content(ctx)
        assert "{{Agent_1}}" in rendered
        assert "{{小明}}" in rendered

    def test_custom_ref_does_not_break_standard_alias_flow(self):
        """自定义映射不影响标准 User_X 流程（当两者不冲突时）"""
        ctx = self._make_real_ctx()
        # 先注册标准用户，再添加自定义映射（不同 entity）
        ctx.alias_provider.register_user("standard_user")   # -> User_1
        ctx.alias_provider.update_aliases({"custom_user": "自定义用户"})

        # 两者共存
        assert ctx.alias_provider.get_alias("standard_user") == "User_1"
        assert ctx.alias_provider.get_alias("custom_user") == "自定义用户"
        assert ctx.alias_provider.resolve_alias("User_1") == "standard_user"
        assert ctx.alias_provider.resolve_alias("自定义用户") == "custom_user"

        # normalize 能正确处理两种格式
        m = CoreMemory(
            content="{{User_1}} 和 {{自定义用户}} 聊天",
            agent_id="a1",
        )
        m.normalize_placeholders(ctx)
        assert "{{standard_user}}" in m.content
        # {{自定义用户}} 现在也能被 resolve_alias 解析为 "custom_user"
        assert "{{custom_user}}" in m.content

        # 渲染回来
        rendered = m._render_content(ctx)
        assert "{{User_1}}" in rendered
        assert "{{自定义用户}}" in rendered
