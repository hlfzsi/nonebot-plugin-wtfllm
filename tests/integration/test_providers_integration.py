"""集成测试: RefProvider + AliasProvider 联合使用"""

import pytest

from nonebot_plugin_wtfllm.memory.providers import RefProvider, AliasProvider
from nonebot_plugin_wtfllm.memory.content.segments import ImageSegment, AudioSegment
from unittest.mock import MagicMock


@pytest.mark.integration
class TestProvidersCrossResolution:

    def test_ref_and_alias_together(self):
        ref = RefProvider()
        alias = AliasProvider()

        # 注册用户
        alias.register_user("user_abc")
        alias.register_user("user_def")

        # 注册记忆项
        item1 = MagicMock()
        item1.message_id = "msg_1"
        item2 = MagicMock()
        item2.message_id = "msg_2"

        r1 = ref.next_memory_ref(item1)
        r2 = ref.next_memory_ref(item2)

        # 注册媒体
        img = ImageSegment(url="http://example.com/img.jpg")
        media_ref = ref.next_media_ref(img, memory_ref=r1)

        # 交叉解析
        assert alias.get_alias("user_abc") == "User_1"
        assert alias.resolve_alias("User_1") == "user_abc"
        assert ref.get_item_by_ref(r1) is item1
        assert ref.get_item_by_ref(r2) is item2
        assert ref.get_media_by_ref(media_ref) is img
        assert ref.get_media_by_memory_ref(r1) == [img]
        assert ref.get_media_by_memory_ref(r2) == []

    def test_multiple_media_on_same_memory(self):
        ref = RefProvider()

        item = MagicMock()
        item.message_id = "msg_1"
        r = ref.next_memory_ref(item)

        img = ImageSegment(url="http://example.com/img.jpg")
        audio = AudioSegment(url="http://example.com/audio.mp3")

        ref.next_media_ref(img, memory_ref=r)
        ref.next_media_ref(audio, memory_ref=r)

        media_list = ref.get_media_by_memory_ref(r)
        assert len(media_list) == 2
        assert img in media_list
        assert audio in media_list

        # 按类型获取
        images = ref.get_media_by_memory_ref_typed(r, ImageSegment)
        assert len(images) == 1
        assert images[0] is img

    def test_media_dedup_by_unique_key(self):
        ref = RefProvider()

        img = ImageSegment(url="http://example.com/img.jpg", created_at=1000000000)
        ref1 = ref.next_media_ref(img)
        ref2 = ref.next_media_ref(img)  # 同一个对象
        assert ref1 == ref2

        # 创建一个具有相同数据的新对象
        img_copy = ImageSegment(url="http://example.com/img.jpg", created_at=1000000000)
        ref3 = ref.next_media_ref(img_copy)
        assert ref3 == ref1  # 同 unique_key 返回同一引用

    def test_reset_clears_everything(self):
        ref = RefProvider()
        item = MagicMock()
        item.message_id = "msg_1"
        ref.next_memory_ref(item)

        assert ref.total_memories == 1
        ref.reset()
        assert ref.total_memories == 0
        assert ref.get_item_by_ref(1) is None

    def test_core_memory_refs(self):
        ref = RefProvider()

        cm = MagicMock()
        cm.storage_id = "cm_001"
        r1 = ref.next_core_memory_ref(cm)
        assert r1 == "CM:1"

        # 幂等
        r2 = ref.next_core_memory_ref(cm)
        assert r2 == "CM:1"

        # 新的 core memory
        cm2 = MagicMock()
        cm2.storage_id = "cm_002"
        r3 = ref.next_core_memory_ref(cm2)
        assert r3 == "CM:2"

        # 反查
        assert ref.get_core_memory_by_ref("CM:1") is cm
        assert ref.get_core_memory_by_ref("CM:2") is cm2
        assert ref.get_core_memory_ref_by_id("cm_001") == "CM:1"

    def test_knowledge_refs(self):
        ref = RefProvider()

        kb = MagicMock()
        kb.storage_id = "kb_001"
        r = ref.next_knowledge_ref(kb)
        assert r == "KB:1"
        assert ref.get_knowledge_by_ref("KB:1") is kb
        assert ref.get_knowledge_ref_by_id("kb_001") == "KB:1"

    def test_alias_provider_mixed_types(self):
        alias = AliasProvider()

        u1 = alias.register_user("user_1")
        g1 = alias.register_group("group_1")
        a1 = alias.register_agent("agent_1")

        assert u1 == "User_1"
        assert g1 == "Group_1"
        assert a1 == "Agent_1"

        # 不同类型的计数器独立
        u2 = alias.register_user("user_2")
        assert u2 == "User_2"

        g2 = alias.register_group("group_2")
        assert g2 == "Group_2"

        # 反查
        assert alias.resolve_alias("User_1") == "user_1"
        assert alias.resolve_alias("Group_1") == "group_1"
        assert alias.resolve_alias("Agent_1") == "agent_1"
