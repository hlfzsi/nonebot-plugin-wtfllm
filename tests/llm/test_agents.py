"""llm/agents.py 集成测试

使用 pydantic-ai 的 FunctionModel 和 TestModel 对 Agent 进行端到端测试，
不发起真实 LLM 请求。
"""

import json

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from pydantic import BaseModel
from pydantic_ai import Agent, models
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.models.test import TestModel

# 直接导入子模块避免 llm.__init__ 的循环导入
import nonebot_plugin_wtfllm.llm.deps as _deps
import nonebot_plugin_wtfllm.llm.response_models as _rm

AgentDeps = _deps.AgentDeps
IDs = _deps.IDs

TextResponse = _rm.TextResponse
MarkdownResponse = _rm.MarkdownResponse
RejectResponse = _rm.RejectResponse
CHAT_OUTPUT = _rm.CHAT_OUTPUT

# 禁止真实请求
models.ALLOW_MODEL_REQUESTS = False


# 简单的输出模型，用于工具调用测试
class SimpleOutput(BaseModel):
    reason: str


# ===================== Fixtures =====================


def _make_deps() -> AgentDeps:
    """创建 mock AgentDeps"""
    from nonebot_plugin_wtfllm.memory import MemoryContextBuilder

    mock_ctx = MagicMock(spec=MemoryContextBuilder)
    mock_ctx.ctx = MagicMock()
    mock_ctx.ctx.alias_provider = MagicMock()
    mock_ctx.ctx.alias_provider.resolve_alias = MagicMock(return_value="u1")
    mock_ctx.resolve_aliases = MagicMock(return_value="u1")
    mock_ctx.resolve_media_ref = MagicMock(side_effect=ValueError("no media"))

    return AgentDeps(
        ids=IDs(user_id="u1", group_id="g1", agent_id="a1"),
        context=mock_ctx,
        active_tool_groups={"Core"},
    )


# ===================== Chat Agent 测试 (使用 FunctionModel) =====================


class TestChatAgentFake:
    """使用 FunctionModel fake 回复测试 Chat Agent"""

    @pytest.mark.asyncio
    async def test_chat_agent_returns_text_response(self):
        """测试 chat agent 返回 TextResponse"""

        def chat_response(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            # union output 需要使用 ToolCallPart，工具名格式为 final_result_{ClassName}
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="final_result_TextResponse",
                        args=json.dumps(
                            {
                                "responses": ["你好！很高兴认识你"],
                                "mentions": [],
                                "meme": None,
                                "interested_topics": None,
                            }
                        ),
                    )
                ]
            )

        agent = Agent(
            FunctionModel(chat_response),
            output_type=CHAT_OUTPUT,
            deps_type=AgentDeps,
        )

        result = await agent.run("你好", deps=_make_deps())
        assert isinstance(result.output, TextResponse)
        assert any("你好" in text for text in result.output.responses)

    @pytest.mark.asyncio
    async def test_chat_agent_returns_reject_response(self):
        """测试 chat agent 返回 RejectResponse"""

        def reject_response(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="final_result_RejectResponse",
                        args=json.dumps(
                            {
                                "reason": "不想回答",
                                "message_to_user": None,
                                "interested_topics": None,
                            }
                        ),
                    )
                ]
            )

        agent = Agent(
            FunctionModel(reject_response),
            output_type=CHAT_OUTPUT,
            deps_type=AgentDeps,
        )

        result = await agent.run("无聊的问题", deps=_make_deps())
        assert isinstance(result.output, RejectResponse)
        assert result.output.reason == "不想回答"

    @pytest.mark.asyncio
    async def test_chat_agent_returns_markdown_response(self):
        """测试 chat agent 返回 MarkdownResponse"""

        def md_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="final_result_MarkdownResponse",
                        args=json.dumps(
                            {
                                "markdown_content": "# Hello\n\nWorld",
                                "summary": "Hello World 摘要",
                                "interested_topics": None,
                            }
                        ),
                    )
                ]
            )

        agent = Agent(
            FunctionModel(md_response),
            output_type=CHAT_OUTPUT,
            deps_type=AgentDeps,
        )

        result = await agent.run("写点什么", deps=_make_deps())
        assert isinstance(result.output, MarkdownResponse)
        assert "Hello" in result.output.markdown_content


# ===================== Agent 带工具调用测试 =====================


class TestAgentWithTools:
    """Agent 工具调用测试（使用 FunctionModel）"""

    @pytest.mark.asyncio
    async def test_agent_with_simple_tool(self):
        """测试 agent 调用工具后返回结果"""
        call_count = 0

        def func_with_tool(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # 第一轮：调用工具
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="get_info",
                            args={"query": "test"},
                        )
                    ]
                )
            else:
                # 第二轮：根据工具返回生成最终回复
                return ModelResponse(parts=[TextPart(content='{"reason":"done"}')])

        agent = Agent(
            FunctionModel(func_with_tool),
            output_type=SimpleOutput,
            deps_type=AgentDeps,
        )

        @agent.tool_plain
        def get_info(query: str) -> str:
            """获取信息"""
            return f"info about {query}"

        result = await agent.run("需要信息", deps=_make_deps())
        assert isinstance(result.output, SimpleOutput)
        assert call_count == 2  # 调用了两轮

    @pytest.mark.asyncio
    async def test_agent_multi_turn_tool(self):
        """测试 agent 多轮工具调用"""
        turn = 0

        def multi_tool_func(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            nonlocal turn
            turn += 1
            if turn == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart("tool_a", args={"x": "1"}),
                        ToolCallPart("tool_b", args={"y": "2"}),
                    ]
                )
            else:
                return ModelResponse(
                    parts=[TextPart(content='{"reason":"all tools done"}')]
                )

        agent = Agent(
            FunctionModel(multi_tool_func),
            output_type=SimpleOutput,
            deps_type=AgentDeps,
        )

        @agent.tool_plain
        def tool_a(x: str) -> str:
            return f"result_a_{x}"

        @agent.tool_plain
        def tool_b(y: str) -> str:
            return f"result_b_{y}"

        result = await agent.run("多工具", deps=_make_deps())
        assert isinstance(result.output, SimpleOutput)
        assert turn == 2


# ===================== Agent.override 测试 =====================
# 注意：由于 agents.py 在模块级别创建 OpenAIProvider（需要真实配置），
# 直接导入 CHAT_AGENT 在测试环境中不可用。
# override 功能已通过 TestModel / FunctionModel 覆盖。


class TestAgentOverride:
    """测试 agent.override 模式替换模型（不导入真实 Agent）"""

    @pytest.mark.asyncio
    async def test_override_with_test_model(self):
        """Agent.override 可以替换模型为 TestModel"""
        dummy_agent = Agent(
            TestModel(),
            output_type=SimpleOutput,
            deps_type=AgentDeps,
        )
        with dummy_agent.override(model=TestModel()):
            result = await dummy_agent.run("test prompt", deps=_make_deps())
            assert isinstance(result.output, SimpleOutput)

    @pytest.mark.asyncio
    async def test_override_with_function_model(self):
        """Agent.override 可以替换模型为 FunctionModel"""
        dummy_agent = Agent(
            TestModel(),
            output_type=CHAT_OUTPUT,
            deps_type=AgentDeps,
        )

        def fake_chat(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="final_result_TextResponse",
                        args=json.dumps(
                            {
                                "responses": ["fake response"],
                                "mentions": [],
                                "meme": None,
                                "interested_topics": None,
                            }
                        ),
                    )
                ]
            )

        with dummy_agent.override(model=FunctionModel(fake_chat)):
            result = await dummy_agent.run("hello", deps=_make_deps())
            assert isinstance(result.output, TextResponse)
            assert result.output.responses == ["fake response"]


# ===================== 消息追踪测试 =====================


class TestAgentMessages:
    """测试 agent 运行后消息记录"""

    @pytest.mark.asyncio
    async def test_messages_are_recorded(self):
        """运行后可以访问所有消息"""

        def simple_resp(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"reason":"recorded"}')])

        agent = Agent(
            FunctionModel(simple_resp),
            output_type=SimpleOutput,
            deps_type=AgentDeps,
        )

        result = await agent.run("test", deps=_make_deps())
        all_msgs = result.all_messages()
        assert len(all_msgs) >= 2  # 至少有 request 和 response
