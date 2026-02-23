"""集成测试: Message -> LLMContext 渲染"""

import pytest

from nonebot_plugin_wtfllm.memory.context import LLMContext
from nonebot_plugin_wtfllm.memory.content.message import Message
from nonebot_plugin_wtfllm.memory.items.base_items import GroupMemoryItem


@pytest.mark.integration
class TestMessageToLLMContextIntegration:

    def test_text_only_rendering(self, llm_context):
        msg = Message.create().text("Hello").text(" World")
        result = msg.to_llm_context(llm_context, "msg_1")
        assert "Hello" in result
        assert "World" in result

    def test_mixed_content_rendering(self, llm_context):
        llm_context.alias_provider.register_user("user_42")

        msg = (
            Message.create()
            .text("Check this ")
            .mention("user_42")
            .image(url="http://example.com/pic.jpg")
        )

        result = msg.to_llm_context(llm_context, "msg_1", memory_ref=1)

        assert "Check this" in result
        assert "<@User_1>" in result
        assert "IMG:1" in result

    def test_long_text_condensed(self, llm_context):
        long_text = "A" * 200
        msg = Message.create().text(long_text)
        result = msg.to_llm_context(llm_context, "msg_1")
        assert "[...省略...]" in result

    def test_long_text_no_condense(self, llm_context_no_condense):
        long_text = "A" * 200
        msg = Message.create().text(long_text)
        result = msg.to_llm_context(llm_context_no_condense, "msg_1")
        assert result == long_text

    def test_multiple_images_get_sequential_refs(self, llm_context):
        msg = (
            Message.create()
            .image(url="http://example.com/a.jpg")
            .image(url="http://example.com/b.jpg")
        )
        result = msg.to_llm_context(llm_context, "msg_1", memory_ref=1)
        assert "IMG:1" in result
        assert "IMG:2" in result

    def test_media_with_description(self, llm_context):
        from nonebot_plugin_wtfllm.memory.content.segments import ImageSegment

        img = ImageSegment(url="http://example.com/pic.jpg", desc="sunset photo")
        msg = Message.create([img])
        result = msg.to_llm_context(llm_context, "msg_1", memory_ref=1)
        assert "IMG:1" in result
        assert "sunset photo" in result


@pytest.mark.integration
class TestMemoryItemToLLMContextIntegration:

    def test_group_item_rendering(self, llm_context):
        msg = Message.create().text("Hello group!")
        item = GroupMemoryItem(
            message_id="msg_1",
            sender="user_alice",
            content=msg,
            created_at=1700000000,
            agent_id="agent_bot",
            group_id="group_main",
        )

        item.register_entities(llm_context)
        result = item.to_llm_context(llm_context)

        assert "[1]" in result
        assert "User_" in result or "Agent_" in result
        assert "Hello group!" in result

    def test_group_item_with_reply(self, llm_context):
        # 先创建被引用的消息
        original = GroupMemoryItem(
            message_id="msg_original",
            sender="user_alice",
            content=Message.create().text("Original message"),
            created_at=1700000000,
            agent_id="agent_bot",
            group_id="group_main",
        )
        original.register_entities(llm_context)
        original.to_llm_context(llm_context)  # 注册 ref

        # 创建回复消息
        reply = GroupMemoryItem(
            message_id="msg_reply",
            related_message_id="msg_original",
            sender="user_bob",
            content=Message.create().text("Reply to you"),
            created_at=1700000060,
            agent_id="agent_bot",
            group_id="group_main",
        )
        reply.register_entities(llm_context)
        result = reply.to_llm_context(llm_context)

        assert "in reply to" in result
        assert "[1]" in result  # 被引用的消息 ref

    def test_agent_message_rendering(self, llm_context):
        msg = Message.create().text("I am the bot")
        item = GroupMemoryItem(
            message_id="msg_bot",
            sender="agent_bot",
            content=msg,
            created_at=1700000000,
            agent_id="agent_bot",
            group_id="group_main",
        )

        item.register_entities(llm_context)
        result = item.to_llm_context(llm_context)

        assert "Agent_" in result
        assert "I am the bot" in result
        assert item.is_from_agent is True
