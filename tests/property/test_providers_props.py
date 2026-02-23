"""Providers 的属性测试"""

import pytest
from hypothesis import given, assume
from hypothesis import strategies as st
from unittest.mock import MagicMock


# ===== RefProvider =====


@pytest.mark.property
class TestRefProviderMonotonicity:

    @given(count=st.integers(min_value=1, max_value=50))
    def test_memory_refs_are_monotonically_increasing(self, count):
        from nonebot_plugin_wtfllm.memory.providers import RefProvider

        provider = RefProvider()
        refs = []
        for i in range(count):
            item = MagicMock()
            item.message_id = f"msg_{i}"
            refs.append(provider.next_memory_ref(item))

        assert refs == list(range(1, count + 1))

    @given(msg_id=st.text(min_size=1, max_size=50))
    def test_idempotent_same_item(self, msg_id):
        from nonebot_plugin_wtfllm.memory.providers import RefProvider

        provider = RefProvider()
        item = MagicMock()
        item.message_id = msg_id

        ref1 = provider.next_memory_ref(item)
        ref2 = provider.next_memory_ref(item)
        ref3 = provider.next_memory_ref(item)
        assert ref1 == ref2 == ref3

    @given(count=st.integers(min_value=1, max_value=20))
    def test_total_memories_matches_count(self, count):
        from nonebot_plugin_wtfllm.memory.providers import RefProvider

        provider = RefProvider()
        for i in range(count):
            item = MagicMock()
            item.message_id = f"msg_{i}"
            provider.next_memory_ref(item)

        assert provider.total_memories == count

    @given(count=st.integers(min_value=1, max_value=20))
    def test_get_item_by_ref_roundtrip(self, count):
        from nonebot_plugin_wtfllm.memory.providers import RefProvider

        provider = RefProvider()
        items = []
        for i in range(count):
            item = MagicMock()
            item.message_id = f"msg_{i}"
            ref = provider.next_memory_ref(item)
            items.append((ref, item))

        for ref, item in items:
            assert provider.get_item_by_ref(ref) is item


# ===== AliasProvider =====


@pytest.mark.property
class TestAliasProviderBidirectionalConsistency:

    @given(
        entity_ids=st.lists(
            st.from_regex(r"[a-z]{3,10}", fullmatch=True),
            min_size=1,
            max_size=20,
            unique=True,
        )
    )
    def test_alias_map_and_reverse_map_are_inverses(self, entity_ids):
        from nonebot_plugin_wtfllm.memory.providers import AliasProvider

        provider = AliasProvider()
        for eid in entity_ids:
            provider.register_user(eid)

        alias_map = provider.alias_map
        reverse_map = provider.reverse_map

        # 正向: entity -> alias, 反向: alias -> entity
        for eid, alias in alias_map.items():
            assert reverse_map[alias] == eid

        # 反向: alias -> entity, 正向: entity -> alias
        for alias, eid in reverse_map.items():
            assert alias_map[eid] == alias

        # 大小一致
        assert len(alias_map) == len(reverse_map) == len(entity_ids)

    @given(
        entity_ids=st.lists(
            st.from_regex(r"[a-z]{3,10}", fullmatch=True),
            min_size=1,
            max_size=20,
            unique=True,
        )
    )
    def test_auto_generated_format(self, entity_ids):
        import re
        from nonebot_plugin_wtfllm.memory.providers import AliasProvider

        provider = AliasProvider()
        for eid in entity_ids:
            alias = provider.register_user(eid)
            assert re.match(r"User_\d+", alias)


@pytest.mark.property
class TestAliasProviderSetAliasOverride:

    @given(
        entity_id=st.from_regex(r"[a-z]{5,10}", fullmatch=True),
        alias1=st.from_regex(r"Name_[a-z]{3,8}", fullmatch=True),
        alias2=st.from_regex(r"Name_[a-z]{3,8}", fullmatch=True),
    )
    def test_override_clears_old_reverse(self, entity_id, alias1, alias2):
        from nonebot_plugin_wtfllm.memory.providers import AliasProvider

        assume(alias1 != alias2)

        provider = AliasProvider()
        provider.set_alias(entity_id, alias1)
        assert provider.resolve_alias(alias1) == entity_id

        provider.set_alias(entity_id, alias2)
        assert provider.resolve_alias(alias2) == entity_id
        assert provider.resolve_alias(alias1) is None  # 旧别名已清除

    @given(
        entity_id=st.from_regex(r"[a-z]{5,10}", fullmatch=True),
        alias=st.from_regex(r"Alias_[a-z]{3,8}", fullmatch=True),
    )
    def test_get_or_create_then_set_alias_overrides(self, entity_id, alias):
        from nonebot_plugin_wtfllm.memory.providers import AliasProvider

        provider = AliasProvider()
        auto_alias = provider.register_user(entity_id)
        assert provider.get_alias(entity_id) == auto_alias

        provider.set_alias(entity_id, alias)
        assert provider.get_alias(entity_id) == alias
        assert provider.resolve_alias(auto_alias) is None
        assert provider.resolve_alias(alias) == entity_id
