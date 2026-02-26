# tests/db/test_models.py
"""db/models/memory_item.py 单元测试"""

import pytest

from nonebot_plugin_wtfllm.db.models.memory_item import MemoryItemTable
from nonebot_plugin_wtfllm.memory.items.base_items import (
    PrivateMemoryItem,
    GroupMemoryItem,
)
from nonebot_plugin_wtfllm.memory.content.message import Message
from nonebot_plugin_wtfllm.memory.content.segments import TextSegment


class TestMemoryItemTableFromMemoryItem:
    """MemoryItemTable.from_MemoryItem 测试"""

    def test_from_private_memory_item(self):
        """测试从 PrivateMemoryItem 创建"""
        content = Message.create().text("Hello World")
        item = PrivateMemoryItem(
            message_id="msg_001",
            related_message_id="msg_000",
            sender="user_123",
            content=content,
            created_at=1000000,
            agent_id="agent_1",
            user_id="user_123",
        )

        table = MemoryItemTable.from_MemoryItem(item)

        assert table.message_id == "msg_001"
        assert table.related_message_id == "msg_000"
        assert table.sender == "user_123"
        assert table.created_at == 1000000
        assert table.agent_id == "agent_1"
        assert table.memory_type == "private"
        assert table.user_id == "user_123"
        assert table.group_id is None
        # content 应该是 JSON 字符串
        assert "Hello World" in table.content

    def test_from_group_memory_item(self):
        """测试从 GroupMemoryItem 创建"""
        content = Message.create().text("Group message")
        item = GroupMemoryItem(
            message_id="msg_002",
            related_message_id=None,
            sender="user_456",
            content=content,
            created_at=2000000,
            agent_id="agent_2",
            group_id="group_001",
        )

        table = MemoryItemTable.from_MemoryItem(item)

        assert table.message_id == "msg_002"
        assert table.related_message_id is None
        assert table.sender == "user_456"
        assert table.created_at == 2000000
        assert table.agent_id == "agent_2"
        assert table.memory_type == "group"
        assert table.group_id == "group_001"
        assert table.user_id is None

    def test_from_unsupported_type_raises(self):
        """测试不支持的类型抛出异常"""

        class UnsupportedItem:
            pass

        with pytest.raises(TypeError, match="Unsupported MemoryItem type"):
            MemoryItemTable.from_MemoryItem(UnsupportedItem())


class TestMemoryItemTableToMemoryItem:
    """MemoryItemTable.to_MemoryItem 测试"""

    def test_to_private_memory_item(self):
        """测试转换为 PrivateMemoryItem"""
        content_json = '{"segments":[{"type":"text","content":"Test message","created_at":1000,"message_id":null}]}'
        table = MemoryItemTable(
            message_id="msg_001",
            related_message_id="msg_000",
            sender="user_123",
            content=content_json,
            created_at=1000000,
            agent_id="agent_1",
            memory_type="private",
            user_id="user_123",
            group_id=None,
        )

        item = table.to_MemoryItem()

        assert isinstance(item, PrivateMemoryItem)
        assert item.message_id == "msg_001"
        assert item.related_message_id == "msg_000"
        assert item.sender == "user_123"
        assert item.created_at == 1000000
        assert item.agent_id == "agent_1"
        assert item.user_id == "user_123"

    def test_to_group_memory_item(self):
        """测试转换为 GroupMemoryItem"""
        content_json = '{"segments":[{"type":"text","content":"Group msg","created_at":1000,"message_id":null}]}'
        table = MemoryItemTable(
            message_id="msg_002",
            related_message_id=None,
            sender="user_456",
            content=content_json,
            created_at=2000000,
            agent_id="agent_2",
            memory_type="group",
            user_id=None,
            group_id="group_001",
        )

        item = table.to_MemoryItem()

        assert isinstance(item, GroupMemoryItem)
        assert item.message_id == "msg_002"
        assert item.related_message_id is None
        assert item.sender == "user_456"
        assert item.created_at == 2000000
        assert item.agent_id == "agent_2"
        assert item.group_id == "group_001"


class TestMemoryItemTableRoundtrip:
    """MemoryItemTable 往返序列化测试"""

    def test_roundtrip_private_item(self):
        """测试 PrivateMemoryItem 往返一致性"""
        content = Message.create().text("Test roundtrip")
        original = PrivateMemoryItem(
            message_id="msg_rt_001",
            related_message_id="msg_rt_000",
            sender="user_rt",
            content=content,
            created_at=3000000,
            agent_id="agent_rt",
            user_id="user_rt",
        )

        # 转换为 Table 再转回
        table = MemoryItemTable.from_MemoryItem(original)
        restored = table.to_MemoryItem()

        assert isinstance(restored, PrivateMemoryItem)
        assert restored.message_id == original.message_id
        assert restored.related_message_id == original.related_message_id
        assert restored.sender == original.sender
        assert restored.created_at == original.created_at
        assert restored.agent_id == original.agent_id
        assert restored.user_id == original.user_id
        assert restored.get_plain_text() == original.get_plain_text()

    def test_roundtrip_group_item(self):
        """测试 GroupMemoryItem 往返一致性"""
        content = Message.create().text("Group roundtrip test")
        original = GroupMemoryItem(
            message_id="msg_rt_002",
            related_message_id=None,
            sender="user_grp",
            content=content,
            created_at=4000000,
            agent_id="agent_grp",
            group_id="group_rt",
        )

        table = MemoryItemTable.from_MemoryItem(original)
        restored = table.to_MemoryItem()

        assert isinstance(restored, GroupMemoryItem)
        assert restored.message_id == original.message_id
        assert restored.sender == original.sender
        assert restored.agent_id == original.agent_id
        assert restored.group_id == original.group_id
        assert restored.get_plain_text() == original.get_plain_text()

    def test_roundtrip_complex_content(self):
        """测试复杂内容的往返"""
        content = (
            Message.create()
            .text("Hello ")
            .mention("user_mentioned")
            .text(" check this ")
            .image(url="http://example.com/img.jpg")
        )

        original = PrivateMemoryItem(
            message_id="msg_complex",
            sender="user_complex",
            content=content,
            created_at=5000000,
            agent_id="agent_complex",
            user_id="user_complex",
        )

        table = MemoryItemTable.from_MemoryItem(original)
        restored = table.to_MemoryItem()

        # 验证内容结构
        assert len(restored.content.segments) == 4
        assert restored.content.has("text")
        assert restored.content.has("mention")
        assert restored.content.has("image")


class TestMemoryItemTableFields:
    """MemoryItemTable 字段测试"""

    def test_content_is_json_string(self):
        """测试 content 是 JSON 字符串"""
        content = Message.create().text("Test")
        item = PrivateMemoryItem(
            message_id="msg_json",
            sender="user_json",
            content=content,
            created_at=1000,
            agent_id="agent_json",
            user_id="user_json",
        )

        table = MemoryItemTable.from_MemoryItem(item)

        assert isinstance(table.content, str)
        # 应该能解析为 JSON
        import json
        parsed = json.loads(table.content)
        assert "segments" in parsed

    def test_message_id_as_primary_key(self):
        """测试 message_id 作为主键"""
        content = Message.create().text("Test")
        item = PrivateMemoryItem(
            message_id="unique_msg_id",
            sender="user",
            content=content,
            created_at=1000,
            agent_id="agent",
            user_id="user",
        )

        table = MemoryItemTable.from_MemoryItem(item)
        assert table.message_id == "unique_msg_id"


# ===================== UserPersona 模型测试 =====================


class TestUserPersonaModel:
    """UserPersona 模型字段与方法测试"""

    def test_create_default(self):
        """测试使用默认值创建"""
        from nonebot_plugin_wtfllm.db.models.user_persona import UserPersona

        persona = UserPersona(user_id="u1")
        assert persona.user_id == "u1"
        assert persona.interaction_style is None
        assert persona.structured_preferences == {}
        assert persona.impression is None
        assert persona.note is None
        assert persona.other is None
        assert persona.version == 1

    def test_render_to_llm_empty(self):
        """测试空 persona 渲染返回 None"""
        from nonebot_plugin_wtfllm.db.models.user_persona import UserPersona

        persona = UserPersona(user_id="u1", version=0)
        result = persona.render_to_llm("用户A")
        assert result is None

    def test_render_to_llm_full(self):
        """测试完整 persona 渲染"""
        from nonebot_plugin_wtfllm.db.models.user_persona import UserPersona

        persona = UserPersona(
            user_id="u1",
            interaction_style="幽默风趣",
            structured_preferences={"语言": "Python", "框架": "FastAPI"},
            impression="技术宅",
            note="喜欢猫",
            other="夜猫子",
            version=3,
        )
        result = persona.render_to_llm("Alice")
        assert result is not None
        assert "用户: Alice" in result
        assert "交互风格: 幽默风趣" in result
        assert "结构化偏好:" in result
        assert "Python" in result
        assert "整体印象: 技术宅" in result
        assert "备注: 喜欢猫" in result
        assert "其他信息: 夜猫子" in result

    def test_render_to_llm_partial(self):
        """测试只有部分字段的渲染"""
        from nonebot_plugin_wtfllm.db.models.user_persona import UserPersona

        persona = UserPersona(
            user_id="u2",
            impression="善良的人",
            version=1,
        )
        result = persona.render_to_llm("Bob")
        assert result is not None
        assert "用户: Bob" in result
        assert "整体印象: 善良的人" in result
        assert "交互风格" not in result

    def test_updated_at_auto_set(self):
        """测试 updated_at 自动设置"""
        import time
        from nonebot_plugin_wtfllm.db.models.user_persona import UserPersona

        before = int(time.time())
        persona = UserPersona(user_id="u3")
        after = int(time.time())
        assert persona.updated_at is not None
        assert before <= persona.updated_at <= after


# ===================== ScheduledJob 模型字段测试 =====================


class TestScheduledJobModel:
    """ScheduledJob 模型字段测试"""

    def test_status_enum_values(self):
        """测试状态枚举定义"""
        from nonebot_plugin_wtfllm.db.models.scheduled_job import (
            ScheduledJobStatus,
        )

        assert ScheduledJobStatus.PENDING == "pending"
        assert ScheduledJobStatus.COMPLETED == "completed"
        assert ScheduledJobStatus.FAILED == "failed"
        assert ScheduledJobStatus.MISSED == "missed"
        assert ScheduledJobStatus.CANCELED == "canceled"

    def test_create_basic_scheduled_job(self):
        """测试直接构建 ScheduledJob"""
        from nonebot_plugin_wtfllm.db.models.scheduled_job import (
            ScheduledJob,
            ScheduledJobStatus,
        )

        job = ScheduledJob(
            job_id="job_test",
            task_name="send_static_message",
            task_params={"user_id": "user_1", "messages": [{"type": "text", "content": "hello"}]},
            trigger_config={"type": "date", "run_timestamp": 999999},
            user_id="user_1",
            group_id="group_1",
            agent_id="agent_1",
            status=ScheduledJobStatus.PENDING,
            created_at=100000,
            description="Test job",
        )

        assert job.job_id == "job_test"
        assert job.task_name == "send_static_message"
        assert job.status == "pending"
        assert job.executed_at is None
        assert job.error_message is None
        assert job.id is None  # auto-increment
        assert job.description == "Test job"
        assert job.trigger_config["type"] == "date"
        assert job.task_params["user_id"] == "user_1"

    def test_private_scheduled_job(self):
        """测试私聊定时任务（group_id 为 None）"""
        from nonebot_plugin_wtfllm.db.models.scheduled_job import (
            ScheduledJob,
        )

        job = ScheduledJob(
            job_id="job_private",
            task_name="send_static_message",
            task_params={},
            trigger_config={"type": "date", "run_timestamp": 100},
            user_id="user_1",
            group_id=None,
            agent_id="agent_1",
            created_at=50,
        )
        assert job.group_id is None

    def test_created_at_default(self):
        """测试 created_at 默认值"""
        import time
        from nonebot_plugin_wtfllm.db.models.scheduled_job import ScheduledJob

        before = int(time.time())
        job = ScheduledJob(
            job_id="job_default",
            task_name="test",
            task_params={},
            trigger_config={"type": "date", "run_timestamp": 100},
        )
        after = int(time.time())
        assert before <= job.created_at <= after
