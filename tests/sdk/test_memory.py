"""SDK 记忆链组装测试。

验证：
- build_chat_retrieval_chain 返回包含正确 task 类型的 RetrievalChain
- build_context_from_sources 正确构建 MemoryContextBuilder
- 自定义参数覆盖默认值
"""

import pytest
from unittest.mock import MagicMock

from nonebot_plugin_wtfllm.services.func.memory_retrieval.chain import RetrievalChain
from nonebot_plugin_wtfllm.services.func.memory_retrieval.main_chat import MainChatTask
from nonebot_plugin_wtfllm.services.func.memory_retrieval.core_memory import (
    CoreMemoryTask,
    CrossSessionMemoryTask,
)
from nonebot_plugin_wtfllm.services.func.memory_retrieval.tool_history import (
    ToolCallHistoryTask,
)
from nonebot_plugin_wtfllm.services.func.memory_retrieval.knowledge import (
    KnowledgeSearchTask,
)
from nonebot_plugin_wtfllm.services.func.memory_retrieval.topic_context import (
    TopicContextTask,
)

from nonebot_plugin_wtfllm.memory.director import MemoryContextBuilder

# 直接导入 SDK 子模块避免 sdk/__init__.py 的连锁导入
import nonebot_plugin_wtfllm.sdk.memory as sdk_memory


# ===================== build_chat_retrieval_chain 测试 =====================


class TestBuildChatRetrievalChain:

    def test_default_chain_has_expected_tasks(self):
        """默认链包含 6 类检索任务"""
        chain = sdk_memory.build_chat_retrieval_chain(
            agent_id="agent_1",
            group_id="group_1",
            query="你好",
        )
        assert isinstance(chain, RetrievalChain)
        task_types = {type(t) for t in chain}
        assert MainChatTask in task_types
        assert CoreMemoryTask in task_types
        assert CrossSessionMemoryTask in task_types
        assert ToolCallHistoryTask in task_types
        assert KnowledgeSearchTask in task_types
        assert TopicContextTask in task_types

    def test_private_chat_chain(self):
        """私聊场景：user_id 而非 group_id"""
        chain = sdk_memory.build_chat_retrieval_chain(
            agent_id="agent_1",
            user_id="user_1",
            query="hi",
        )
        assert isinstance(chain, RetrievalChain)
        assert len(chain) == 6

    def test_custom_params_override_defaults(self):
        """自定义参数覆盖 APP_CONFIG 的默认值"""
        chain = sdk_memory.build_chat_retrieval_chain(
            agent_id="agent_1",
            group_id="group_1",
            query="test",
            short_memory_limit=20,
            tool_history_limit=5,
            knowledge_limit=10,
            knowledge_max_tokens=2000,
            topic_max_messages=15,
        )
        # 验证 chain 接受了参数并且长度正确
        assert len(chain) == 6

        # 验证 MainChatTask 的 limit
        main_chat_tasks = [t for t in chain if isinstance(t, MainChatTask)]
        assert len(main_chat_tasks) == 1
        assert main_chat_tasks[0].limit == 20

        # 验证 ToolCallHistoryTask 的 limit
        tool_tasks = [t for t in chain if isinstance(t, ToolCallHistoryTask)]
        assert len(tool_tasks) == 1
        assert tool_tasks[0].limit == 5

        # 验证 KnowledgeSearchTask 的参数
        kb_tasks = [t for t in chain if isinstance(t, KnowledgeSearchTask)]
        assert len(kb_tasks) == 1
        assert kb_tasks[0].limit == 10
        assert kb_tasks[0].max_tokens == 2000

        # 验证 TopicContextTask 参数
        topic_tasks = [t for t in chain if isinstance(t, TopicContextTask)]
        assert len(topic_tasks) == 1
        assert topic_tasks[0].max_topic_messages == 15

    def test_chain_is_composable(self):
        """返回的 chain 仍可继续追加 task"""
        chain = sdk_memory.build_chat_retrieval_chain(
            agent_id="agent_1",
            group_id="group_1",
            query="test",
        )
        original_len = len(chain)
        chain.entity_memory(entity_ids=["entity_1"])
        assert len(chain) == original_len + 1


# ===================== build_context_from_sources 测试 =====================


class TestBuildContextFromSources:

    def test_builds_builder_from_sources(self):
        """从 MemorySource 集合构建 builder"""
        mock_source_1 = MagicMock()
        mock_source_2 = MagicMock()
        sources = {mock_source_1, mock_source_2}

        builder = sdk_memory.build_context_from_sources(
            sources,
            agent_id="agent_1",
            group_id="group_1",
            prefix_prompt="Current Scene: TestGroup",
        )
        assert isinstance(builder, MemoryContextBuilder)

    def test_builder_with_custom_ref(self):
        """custom_ref 被正确传入 builder"""
        refs = {"user_123": "Alice", "agent_1": "Bot"}
        builder = sdk_memory.build_context_from_sources(
            [],
            agent_id="agent_1",
            custom_ref=refs,
        )
        # 验证 alias_provider 中注册了自定义别名
        alias = builder.ctx.alias_provider.get_alias("user_123")
        assert alias == "Alice"

    def test_empty_sources_returns_valid_builder(self):
        """空 sources 返回合法 builder"""
        builder = sdk_memory.build_context_from_sources(
            [],
            agent_id="agent_1",
        )
        assert isinstance(builder, MemoryContextBuilder)

    def test_suffix_prompt_is_set(self):
        """suffix_prompt 被正确设置"""
        builder = sdk_memory.build_context_from_sources(
            [],
            agent_id="agent_1",
            suffix_prompt="<extra>info</extra>",
        )
        assert builder.suffix_prompt == "<extra>info</extra>"
