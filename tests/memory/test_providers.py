# tests/memory/test_providers.py
"""memory/providers.py 单元测试"""

import pytest
from unittest.mock import MagicMock

from nonebot_plugin_wtfllm.memory.providers import RefProvider, AliasProvider
from nonebot_plugin_wtfllm.memory.content.segments import (
    ImageSegment,
    FileSegment,
    AudioSegment,
    VideoSegment,
)


class TestRefProviderMemoryRef:
    """RefProvider 记忆项引用测试"""

    def test_next_memory_ref_sequential(self):
        """测试记忆项引用按顺序分配"""
        provider = RefProvider()

        item1 = MagicMock()
        item1.message_id = "msg_1"
        item2 = MagicMock()
        item2.message_id = "msg_2"
        item3 = MagicMock()
        item3.message_id = "msg_3"

        assert provider.next_memory_ref(item1) == 1
        assert provider.next_memory_ref(item2) == 2
        assert provider.next_memory_ref(item3) == 3

    def test_get_ref_by_item_id_found(self):
        """测试通过 item_id 获取引用号 - 找到"""
        provider = RefProvider()

        item = MagicMock()
        item.message_id = "msg_123"
        provider.next_memory_ref(item)

        assert provider.get_ref_by_item_id("msg_123") == 1

    def test_get_ref_by_item_id_not_found(self):
        """测试通过 item_id 获取引用号 - 未找到"""
        provider = RefProvider()
        assert provider.get_ref_by_item_id("nonexistent") is None

    def test_get_item_by_ref_found(self):
        """测试通过引用号获取记忆项 - 找到"""
        provider = RefProvider()

        item = MagicMock()
        item.message_id = "msg_456"
        provider.next_memory_ref(item)

        result = provider.get_item_by_ref(1)
        assert result is item

    def test_get_item_by_ref_not_found(self):
        """测试通过引用号获取记忆项 - 未找到"""
        provider = RefProvider()
        assert provider.get_item_by_ref(999) is None


class TestRefProviderMediaRef:
    """RefProvider 媒体引用测试"""

    def test_next_media_ref_image(self):
        """测试图片引用分配"""
        provider = RefProvider()

        img1 = ImageSegment(url="http://example.com/1.jpg")
        img2 = ImageSegment(url="http://example.com/2.jpg")

        assert provider.next_media_ref(img1) == "IMG:1"
        assert provider.next_media_ref(img2) == "IMG:2"

    def test_next_media_ref_file(self):
        """测试文件引用分配"""
        provider = RefProvider()

        file1 = FileSegment(filename="doc.pdf", url="http://example.com/doc.pdf")
        file2 = FileSegment(filename="data.csv", url="http://example.com/data.csv")

        assert provider.next_media_ref(file1) == "FILE:1"
        assert provider.next_media_ref(file2) == "FILE:2"

    def test_next_media_ref_audio(self):
        """测试音频引用分配"""
        provider = RefProvider()

        audio = AudioSegment(url="http://example.com/audio.mp3")
        assert provider.next_media_ref(audio) == "AUDIO:1"

    def test_next_media_ref_video(self):
        """测试视频引用分配"""
        provider = RefProvider()

        video = VideoSegment(url="http://example.com/video.mp4")
        assert provider.next_media_ref(video) == "VIDEO:1"

    def test_media_ref_independent_counters(self):
        """测试不同媒体类型使用独立计数器"""
        provider = RefProvider()

        img = ImageSegment(url="http://example.com/img.jpg")
        file = FileSegment(filename="file.txt", url="http://example.com/file.txt")
        audio = AudioSegment(url="http://example.com/audio.mp3")

        # 每种类型独立计数
        assert provider.next_media_ref(img) == "IMG:1"
        assert provider.next_media_ref(file) == "FILE:1"
        assert provider.next_media_ref(audio) == "AUDIO:1"

        # 同类型继续递增
        img2 = ImageSegment(url="http://example.com/img2.jpg")
        assert provider.next_media_ref(img2) == "IMG:2"

    def test_next_media_ref_with_memory_ref(self):
        """测试媒体引用关联到记忆项"""
        provider = RefProvider()

        img1 = ImageSegment(url="http://example.com/1.jpg")
        img2 = ImageSegment(url="http://example.com/2.jpg")
        img3 = ImageSegment(url="http://example.com/3.jpg")

        provider.next_media_ref(img1, memory_ref=1)
        provider.next_media_ref(img2, memory_ref=1)
        provider.next_media_ref(img3, memory_ref=2)

        # 检查记忆项索引
        media_ref_1 = provider.get_media_by_memory_ref(1)
        assert len(media_ref_1) == 2
        assert img1 in media_ref_1
        assert img2 in media_ref_1

        media_ref_2 = provider.get_media_by_memory_ref(2)
        assert len(media_ref_2) == 1
        assert img3 in media_ref_2

    def test_get_media_by_ref_found(self):
        """测试通过引用获取媒体 - 找到"""
        provider = RefProvider()

        img = ImageSegment(url="http://example.com/img.jpg")
        provider.next_media_ref(img)

        result = provider.get_media_by_ref("IMG:1")
        assert result is img

    def test_get_media_by_ref_not_found(self):
        """测试通过引用获取媒体 - 未找到"""
        provider = RefProvider()
        result = provider.get_media_by_ref("IMG:999")
        assert result is None

    def test_get_media_by_ref_invalid_format(self):
        """测试无效的引用格式"""
        provider = RefProvider()

        with pytest.raises(ValueError, match="Invalid media ref format"):
            provider.get_media_by_ref("invalid")

    def test_get_media_typed_correct_type(self):
        """测试类型匹配的媒体获取"""
        provider = RefProvider()

        img = ImageSegment(url="http://example.com/img.jpg")
        provider.next_media_ref(img)

        result = provider.get_media_typed("IMG:1", ImageSegment)
        assert result is img

    def test_get_media_typed_wrong_type(self):
        """测试类型不匹配的媒体获取"""
        provider = RefProvider()

        img = ImageSegment(url="http://example.com/img.jpg")
        provider.next_media_ref(img)

        result = provider.get_media_typed("IMG:1", FileSegment)
        assert result is None

    def test_get_media_by_memory_ref_empty(self):
        """测试获取无媒体的记忆项返回空列表"""
        provider = RefProvider()
        result = provider.get_media_by_memory_ref(999)
        assert result == []

    def test_get_media_by_memory_ref_typed(self):
        """测试按类型获取记忆项的媒体"""
        provider = RefProvider()

        img = ImageSegment(url="http://example.com/img.jpg")
        file = FileSegment(filename="doc.pdf", url="http://example.com/doc.pdf")

        provider.next_media_ref(img, memory_ref=1)
        provider.next_media_ref(file, memory_ref=1)

        images = provider.get_media_by_memory_ref_typed(1, ImageSegment)
        assert len(images) == 1
        assert images[0] is img

        files = provider.get_media_by_memory_ref_typed(1, FileSegment)
        assert len(files) == 1
        assert files[0] is file


class TestRefProviderStatistics:
    """RefProvider 统计属性测试"""

    def test_total_memories(self):
        """测试记忆项总数统计"""
        provider = RefProvider()
        assert provider.total_memories == 0

        for i in range(5):
            item = MagicMock()
            item.message_id = f"msg_{i}"
            provider.next_memory_ref(item)

        assert provider.total_memories == 5

    def test_total_images(self):
        """测试图片总数统计"""
        provider = RefProvider()
        assert provider.total_images == 0

        for i in range(4):
            img = ImageSegment(url=f"http://example.com/{i}.jpg")
            provider.next_media_ref(img)

        assert provider.total_images == 4


class TestRefProviderKnowledgeRef:
    """RefProvider 知识库引用测试"""

    def test_next_knowledge_ref_sequential(self):
        """测试知识库引用按顺序分配"""
        provider = RefProvider()

        entry1 = MagicMock()
        entry1.storage_id = "kb_1"
        entry2 = MagicMock()
        entry2.storage_id = "kb_2"
        entry3 = MagicMock()
        entry3.storage_id = "kb_3"

        assert provider.next_knowledge_ref(entry1) == "KB:1"
        assert provider.next_knowledge_ref(entry2) == "KB:2"
        assert provider.next_knowledge_ref(entry3) == "KB:3"

    def test_next_knowledge_ref_idempotent(self):
        """测试同一条目多次调用返回相同引用"""
        provider = RefProvider()

        entry = MagicMock()
        entry.storage_id = "kb_1"

        ref1 = provider.next_knowledge_ref(entry)
        ref2 = provider.next_knowledge_ref(entry)
        ref3 = provider.next_knowledge_ref(entry)

        assert ref1 == ref2 == ref3 == "KB:1"

    def test_get_knowledge_ref_by_id_found(self):
        """测试通过 storage_id 获取引用号 - 找到"""
        provider = RefProvider()

        entry = MagicMock()
        entry.storage_id = "kb_123"
        provider.next_knowledge_ref(entry)

        assert provider.get_knowledge_ref_by_id("kb_123") == "KB:1"

    def test_get_knowledge_ref_by_id_not_found(self):
        """测试通过 storage_id 获取引用号 - 未找到"""
        provider = RefProvider()
        assert provider.get_knowledge_ref_by_id("nonexistent") is None

    def test_get_knowledge_by_ref_found(self):
        """测试通过引用号获取知识条目 - 找到"""
        provider = RefProvider()

        entry = MagicMock()
        entry.storage_id = "kb_456"
        provider.next_knowledge_ref(entry)

        result = provider.get_knowledge_by_ref("KB:1")
        assert result is entry

    def test_get_knowledge_by_ref_not_found(self):
        """测试通过引用号获取知识条目 - 未找到"""
        provider = RefProvider()
        assert provider.get_knowledge_by_ref("KB:999") is None

    def test_knowledge_ref_independent_from_core_memory(self):
        """测试知识库和核心记忆使用独立计数器"""
        provider = RefProvider()

        cm = MagicMock()
        cm.storage_id = "cm_1"

        kb = MagicMock()
        kb.storage_id = "kb_1"

        assert provider.next_core_memory_ref(cm) == "CM:1"
        assert provider.next_knowledge_ref(kb) == "KB:1"

        cm2 = MagicMock()
        cm2.storage_id = "cm_2"
        kb2 = MagicMock()
        kb2.storage_id = "kb_2"

        assert provider.next_core_memory_ref(cm2) == "CM:2"
        assert provider.next_knowledge_ref(kb2) == "KB:2"


class TestRefProviderIdempotency:
    """RefProvider 幂等性测试"""

    def test_next_memory_ref_idempotent(self):
        """测试 next_memory_ref 对同一对象多次调用返回相同结果"""
        provider = RefProvider()

        item = MagicMock()
        item.message_id = "msg_1"

        ref1 = provider.next_memory_ref(item)
        ref2 = provider.next_memory_ref(item)
        ref3 = provider.next_memory_ref(item)

        assert ref1 == ref2 == ref3 == 1
        assert provider.total_memories == 1

    def test_next_media_ref_idempotent(self):
        """测试 next_media_ref 对同一对象多次调用返回相同结果"""
        provider = RefProvider()

        img = ImageSegment(url="http://example.com/test.jpg")

        ref1 = provider.next_media_ref(img)
        ref2 = provider.next_media_ref(img)
        ref3 = provider.next_media_ref(img)

        assert ref1 == ref2 == ref3 == "IMG:1"
        assert provider.total_images == 1

    def test_idempotency_mixed_operations(self):
        """测试混合操作中的幂等性"""
        provider = RefProvider()

        item1 = MagicMock()
        item1.message_id = "msg_1"
        item2 = MagicMock()
        item2.message_id = "msg_2"

        # 第一次调用
        assert provider.next_memory_ref(item1) == 1
        assert provider.next_memory_ref(item2) == 2

        # 重复调用不影响计数器
        assert provider.next_memory_ref(item1) == 1
        assert provider.next_memory_ref(item2) == 2

        # 新对象继续递增
        item3 = MagicMock()
        item3.message_id = "msg_3"
        assert provider.next_memory_ref(item3) == 3

        assert provider.total_memories == 3


class TestAliasProviderGetOrCreateAlias:
    """AliasProvider.get_or_create_alias 测试"""

    def test_create_new_alias(self):
        """测试创建新别名"""
        provider = AliasProvider()

        alias = provider.get_or_create_alias("user_12345", "User")
        assert alias == "User_1"

    def test_get_existing_alias(self):
        """测试获取已存在的别名"""
        provider = AliasProvider()

        provider.get_or_create_alias("user_12345", "User")
        alias = provider.get_or_create_alias("user_12345", "User")
        assert alias == "User_1"

    def test_different_entities_different_aliases(self):
        """测试不同实体获得不同别名"""
        provider = AliasProvider()

        alias1 = provider.get_or_create_alias("user_1", "User")
        alias2 = provider.get_or_create_alias("user_2", "User")

        assert alias1 == "User_1"
        assert alias2 == "User_2"

    def test_different_types_independent_counters(self):
        """测试不同类型使用独立计数器"""
        provider = AliasProvider()

        user_alias = provider.get_or_create_alias("user_1", "User")
        group_alias = provider.get_or_create_alias("group_1", "Group")
        agent_alias = provider.get_or_create_alias("agent_1", "Agent")

        assert user_alias == "User_1"
        assert group_alias == "Group_1"
        assert agent_alias == "Agent_1"


class TestAliasProviderSetAlias:
    """AliasProvider.set_alias 测试"""

    def test_set_alias_new_entity(self):
        """测试为新实体设置别名"""
        provider = AliasProvider()

        provider.set_alias("user_123", "CustomName")

        assert provider.get_alias("user_123") == "CustomName"
        assert provider.resolve_alias("CustomName") == "user_123"

    def test_set_alias_override_existing(self):
        """测试覆盖已存在的别名"""
        provider = AliasProvider()

        provider.set_alias("user_123", "OldName")
        provider.set_alias("user_123", "NewName")

        assert provider.get_alias("user_123") == "NewName"
        assert provider.resolve_alias("NewName") == "user_123"
        # 旧别名应该被清除
        assert provider.resolve_alias("OldName") is None


class TestAliasProviderGetAndResolve:
    """AliasProvider.get_alias 和 resolve_alias 测试"""

    def test_get_alias_found(self):
        """测试获取存在的别名"""
        provider = AliasProvider()
        provider.get_or_create_alias("user_123", "User")

        assert provider.get_alias("user_123") == "User_1"

    def test_get_alias_not_found(self):
        """测试获取不存在的别名"""
        provider = AliasProvider()
        assert provider.get_alias("nonexistent") is None

    def test_resolve_alias_found(self):
        """测试解析存在的别名"""
        provider = AliasProvider()
        provider.get_or_create_alias("user_123", "User")

        assert provider.resolve_alias("User_1") == "user_123"

    def test_resolve_alias_not_found(self):
        """测试解析不存在的别名"""
        provider = AliasProvider()
        assert provider.resolve_alias("User_999") is None


class TestAliasProviderConvenienceMethods:
    """AliasProvider 便捷方法测试"""

    def test_register_user(self):
        """测试注册用户"""
        provider = AliasProvider()

        alias = provider.register_user("user_abc")
        assert alias == "User_1"

    def test_register_group(self):
        """测试注册群组"""
        provider = AliasProvider()

        alias = provider.register_group("group_xyz")
        assert alias == "Group_1"

    def test_register_agent(self):
        """测试注册智能体"""
        provider = AliasProvider()

        alias = provider.register_agent("agent_001")
        assert alias == "Agent_1"

    def test_convenience_methods_sequential(self):
        """测试便捷方法的序号递增"""
        provider = AliasProvider()

        assert provider.register_user("u1") == "User_1"
        assert provider.register_user("u2") == "User_2"
        assert provider.register_group("g1") == "Group_1"
        assert provider.register_agent("a1") == "Agent_1"
        assert provider.register_agent("a2") == "Agent_2"


class TestAliasProviderMaps:
    """AliasProvider 映射属性测试"""

    def test_alias_map_returns_copy(self):
        """测试 alias_map 返回副本"""
        provider = AliasProvider()
        provider.register_user("user_1")
        provider.register_group("group_1")

        map_copy = provider.alias_map

        # 验证内容
        assert map_copy == {"user_1": "User_1", "group_1": "Group_1"}

        # 修改副本不影响原始
        map_copy["user_1"] = "Modified"
        assert provider.get_alias("user_1") == "User_1"

    def test_reverse_map_returns_copy(self):
        """测试 reverse_map 返回副本"""
        provider = AliasProvider()
        provider.register_user("user_1")
        provider.register_group("group_1")

        reverse = provider.reverse_map

        # 验证内容
        assert reverse == {"User_1": "user_1", "Group_1": "group_1"}

        # 修改副本不影响原始
        reverse["User_1"] = "modified"
        assert provider.resolve_alias("User_1") == "user_1"
