"""history_processor 测试

验证 inject_new_messages 能在 Agent 多轮迭代间注入新到达的消息。
使用 pydantic-ai 的 FunctionModel 模拟 LLM 行为。
"""

import pytest
from unittest.mock import MagicMock

from pydantic_ai import Agent, models
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel, AgentInfo

from pydantic import BaseModel

import nonebot_plugin_wtfllm.llm.deps as _deps
import nonebot_plugin_wtfllm.llm.history_processor as _hp

from nonebot_plugin_wtfllm.memory import (
    GroupMemoryItem,
    MemoryContextBuilder,
    Message,
    TextSegment,
)

AgentDeps = _deps.AgentDeps
IDs = _deps.IDs
inject_new_messages = _hp.inject_new_messages


class SimpleOutput(BaseModel):
    reason: str


models.ALLOW_MODEL_REQUESTS = False


def _make_memory_item(
    sender: str = "user1", text: str = "hello", msg_id: str = "m1"
) -> GroupMemoryItem:
    """创建一个真实的 GroupMemoryItem 用于测试"""
    return GroupMemoryItem(
        message_id=msg_id,
        sender=sender,
        group_id="g1",
        agent_id="a1",
        content=Message.create([TextSegment(content=text)]),
    )


def _make_deps(message_queue=None) -> AgentDeps:
    """创建带有 message_queue 的 mock AgentDeps"""
    mock_ctx = MagicMock(spec=MemoryContextBuilder)
    mock_ctx.ctx = MagicMock()
    mock_ctx.ctx.alias_provider = MagicMock()

    def mock_copy(share_context=True, empty=False):
        new_mock = MagicMock(spec=MemoryContextBuilder)
        new_mock.ctx = mock_ctx.ctx if share_context else MagicMock()
        new_mock._sources = []
        new_mock.to_prompt = MagicMock(return_value="[新消息内容]")
        return new_mock

    mock_ctx.copy = mock_copy

    return AgentDeps(
        ids=IDs(user_id="u1", group_id="g1", agent_id="a1"),
        context=mock_ctx,
        active_tool_groups={"Core"},
        message_queue=message_queue,
    )


class TestInjectNewMessagesBasic:
    """inject_new_messages 基础行为测试"""

    @pytest.mark.asyncio
    async def test_no_queue_returns_unchanged(self):
        """message_queue 为 None 时原样返回 messages"""
        deps = _make_deps(message_queue=None)
        ctx = MagicMock()
        ctx.deps = deps

        original_messages = [MagicMock(spec=ModelRequest)]
        result = await inject_new_messages(ctx, original_messages)
        assert result is original_messages
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_empty_queue_returns_unchanged(self):
        """空队列时原样返回 messages"""
        queue = []
        deps = _make_deps(message_queue=queue)
        ctx = MagicMock()
        ctx.deps = deps

        original_messages = [MagicMock(spec=ModelRequest)]
        result = await inject_new_messages(ctx, original_messages)
        assert result is original_messages
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_non_empty_queue_injects_user_prompt(self):
        """队列有消息时在最后一条 ModelRequest 的 parts 末尾追加 UserPromptPart"""
        queue = [_make_memory_item()]
        deps = _make_deps(message_queue=queue)
        ctx = MagicMock()
        ctx.deps = deps

        last_request = ModelRequest(parts=[UserPromptPart(content="original")])
        result = await inject_new_messages(ctx, [last_request])

        # 不新增 ModelRequest，仍然是 1 条
        assert len(result) == 1
        # 原有的 UserPromptPart 仍在前面
        assert isinstance(result[0].parts[0], UserPromptPart)
        assert result[0].parts[0].content == "original"
        # UserPromptPart 被追加到 parts 末尾
        assert isinstance(result[0].parts[1], UserPromptPart)
        assert "[新消息内容]" in result[0].parts[1].content

    @pytest.mark.asyncio
    async def test_queue_cleared_after_injection(self):
        """注入后队列被清空"""
        queue = [_make_memory_item(msg_id="m1"), _make_memory_item(msg_id="m2")]
        deps = _make_deps(message_queue=queue)
        ctx = MagicMock()
        ctx.deps = deps

        last_request = ModelRequest(parts=[UserPromptPart(content="original")])
        await inject_new_messages(ctx, [last_request])
        assert len(queue) == 0

    @pytest.mark.asyncio
    async def test_second_call_after_drain_returns_unchanged(self):
        """队列被清空后，下次调用不再修改 messages"""
        queue = [_make_memory_item()]
        deps = _make_deps(message_queue=queue)
        ctx = MagicMock()
        ctx.deps = deps

        last_request = ModelRequest(parts=[UserPromptPart(content="original")])
        messages = [last_request]
        await inject_new_messages(ctx, messages)
        # 第一次注入：追加了 UserPromptPart，但消息条数不变
        assert len(messages) == 1
        parts_count_after_first = len(messages[0].parts)
        assert (
            parts_count_after_first == 2
        )  # UserPromptPart(original) + UserPromptPart(注入)

        result = await inject_new_messages(ctx, messages)
        # 第二次调用：队列已空，不再修改
        assert len(result) == 1
        assert len(result[0].parts) == parts_count_after_first


class TestInjectNewMessagesWithAgent:
    """使用 FunctionModel Agent 测试 history_processor 注入行为"""

    @pytest.mark.asyncio
    async def test_processor_called_between_tool_rounds(self):
        """验证 history_processor 在工具调用轮次间被调用，新消息被注入"""
        queue = []
        turn = 0
        saw_injected = False

        def fake_llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal turn, saw_injected
            turn += 1
            if turn == 1:
                queue.append(_make_memory_item(text="new msg during processing"))
                return ModelResponse(parts=[ToolCallPart("my_tool", args={"x": "1"})])
            else:
                for msg in messages:
                    if isinstance(msg, ModelRequest):
                        for part in msg.parts:
                            if isinstance(part, UserPromptPart):
                                content = part.content
                                if "[新消息内容]" in content:
                                    saw_injected = True
                return ModelResponse(parts=[TextPart(content='{"reason":"done"}')])

        agent = Agent(
            FunctionModel(fake_llm),
            output_type=SimpleOutput,
            deps_type=AgentDeps,
            history_processors=[inject_new_messages],
        )

        @agent.tool_plain
        def my_tool(x: str) -> str:
            return f"result_{x}"

        result = await agent.run("test prompt", deps=_make_deps(message_queue=queue))
        assert isinstance(result.output, SimpleOutput)
        assert turn == 2
        assert saw_injected, "注入的新消息应出现在第二轮 LLM 的消息历史中"

    @pytest.mark.asyncio
    async def test_no_injection_without_new_messages(self):
        """无新消息时 history_processor 不修改历史"""
        queue = []
        message_counts = []

        def fake_llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            message_counts.append(len(messages))
            return ModelResponse(parts=[TextPart(content='{"reason":"done"}')])

        agent = Agent(
            FunctionModel(fake_llm),
            output_type=SimpleOutput,
            deps_type=AgentDeps,
            history_processors=[inject_new_messages],
        )

        result = await agent.run("test", deps=_make_deps(message_queue=queue))
        assert isinstance(result.output, SimpleOutput)
        assert len(message_counts) == 1

    @pytest.mark.asyncio
    async def test_multiple_messages_injected_at_once(self):
        """多条新消息同时入队后被一次性注入"""
        queue = []
        turn = 0
        injected_count = 0

        def fake_llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal turn, injected_count
            turn += 1
            if turn == 1:
                queue.extend(
                    [
                        _make_memory_item(msg_id=f"new_{i}", text=f"msg {i}")
                        for i in range(3)
                    ]
                )
                return ModelResponse(parts=[ToolCallPart("my_tool", args={"x": "1"})])
            else:
                for msg in messages:
                    if isinstance(msg, ModelRequest):
                        for part in msg.parts:
                            if (
                                isinstance(part, UserPromptPart)
                                and "[新消息内容]" in part.content
                            ):
                                injected_count += 1
                return ModelResponse(parts=[TextPart(content='{"reason":"done"}')])

        agent = Agent(
            FunctionModel(fake_llm),
            output_type=SimpleOutput,
            deps_type=AgentDeps,
            history_processors=[inject_new_messages],
        )

        @agent.tool_plain
        def my_tool(x: str) -> str:
            return f"result_{x}"

        await agent.run("test", deps=_make_deps(message_queue=queue))
        assert turn == 2
        assert injected_count == 1
        assert len(queue) == 0
