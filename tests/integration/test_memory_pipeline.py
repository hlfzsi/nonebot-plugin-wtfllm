"""集成测试: MemoryContextBuilder 端到端管道"""

import pytest

from nonebot_plugin_wtfllm.memory.director import MemoryContextBuilder
from nonebot_plugin_wtfllm.memory.items.note import Note
from nonebot_plugin_wtfllm.memory.items.storages import MemoryItemStream
from nonebot_plugin_wtfllm.memory.content.segments import ImageSegment


@pytest.mark.integration
class TestMemoryContextBuilderPipeline:

    def test_to_prompt_with_single_stream(self, memory_stream):
        builder = MemoryContextBuilder(agent_id="agent_bot")
        builder.add(memory_stream)

        prompt = builder.to_prompt()

        # 应包含前后缀
        assert "Recent Messages" in prompt
        assert "End" in prompt

        # 应包含自动生成的用户别名
        assert "User_" in prompt

        # Agent 别名已注册但不会出现在 prompt 中（没有 agent 发送的消息）
        assert builder.resolve_aliases("Agent_1") == "agent_bot"

        # 应包含引用号
        assert "[1]" in prompt
        assert "[2]" in prompt

    def test_to_prompt_with_multiple_streams(
        self, sample_group_items, sample_private_items
    ):
        stream1 = MemoryItemStream.create(
            items=sample_group_items[:2], prefix="[Group]"
        )
        stream2 = MemoryItemStream.create(
            items=sample_private_items[:2], prefix="[Private]"
        )

        builder = MemoryContextBuilder(agent_id="agent_bot")
        builder.add(stream1)
        builder.add(stream2)

        prompt = builder.to_prompt()
        assert "[Group]" in prompt
        assert "[Private]" in prompt

    def test_resolve_aliases_after_render(self, memory_stream):
        builder = MemoryContextBuilder(agent_id="agent_bot")
        builder.add(memory_stream)
        builder.to_prompt()  # 触发别名注册

        # 应能将别名解析回实体 ID
        entity_id = builder.resolve_aliases("User_1")
        assert entity_id is not None

    def test_resolve_memory_ref_after_render(self, memory_stream):
        builder = MemoryContextBuilder(agent_id="agent_bot")
        builder.add(memory_stream)
        builder.to_prompt()

        item = builder.resolve_memory_ref(1)
        assert item is not None
        assert item.message_id.startswith("msg_grp_")

    def test_resolve_media_ref_after_render(self, sample_group_items):
        # sample_group_items[3] 有一张图片
        stream = MemoryItemStream.create(items=sample_group_items)
        builder = MemoryContextBuilder(agent_id="agent_bot")
        builder.add(stream)
        builder.to_prompt()

        img = builder.resolve_media_ref("IMG:1", ImageSegment)
        assert img is not None
        assert isinstance(img, ImageSegment)

    def test_custom_ref_aliases(self, memory_stream):
        builder = MemoryContextBuilder(
            agent_id="agent_bot",
            custom_ref={"user_0": "Alice", "user_1": "Bob"},
        )
        builder.add(memory_stream)
        prompt = builder.to_prompt()

        # 自定义别名应出现在输出中
        assert "Alice" in prompt or "Bob" in prompt

    def test_resolve_note_ref_after_render(self):
        note = Note.create(
            content="提醒：这一轮会话结束前给出总结",
            agent_id="agent_bot",
            user_id="user_1",
            expires_at=4102444800,
        )
        builder = MemoryContextBuilder(agent_id="agent_bot")
        builder.add(note)
        prompt = builder.to_prompt()

        assert "NT:" in prompt
        resolved = builder.resolve_note_ref("NT:1")
        assert resolved is not None
        assert resolved.content.startswith("提醒")

    def test_prefix_and_suffix_prompt(self, memory_stream):
        builder = MemoryContextBuilder(
            agent_id="agent_bot",
            prefix_prompt="=== SYSTEM PREFIX ===",
            suffix_prompt="=== SYSTEM SUFFIX ===",
        )
        builder.add(memory_stream)
        prompt = builder.to_prompt()

        assert prompt.startswith("=== SYSTEM PREFIX ===")
        assert prompt.endswith("=== SYSTEM SUFFIX ===")

    def test_dirty_state_after_add(self, memory_stream):
        builder = MemoryContextBuilder(agent_id="agent_bot")
        builder.add(memory_stream)
        assert builder.is_dirty is True

        builder.to_prompt()
        assert builder.is_dirty is False

        # 再次 add 应标记为 dirty
        stream2 = MemoryItemStream.create(items=[])
        builder.add(stream2)
        assert builder.is_dirty is True

    def test_len_and_contains(self, memory_stream):
        builder = MemoryContextBuilder(agent_id="agent_bot")
        assert len(builder) == 0
        assert not builder

        builder.add(memory_stream)
        assert len(builder) == 1
        assert memory_stream in builder
        assert builder
