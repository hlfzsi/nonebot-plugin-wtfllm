"""
用法::

    from nonebot_plugin_wtfllm.sdk import (
        # 工具组
        ToolGroupMeta,
        register_tool_groups,
        list_registered_groups,
        get_tool_group,
        # 记忆链
        MemoryContextBuilder,
        RetrievalChain,
        LLMContext,
        build_chat_retrieval_chain,
        build_context_from_sources,
        # Agent
        AgentDeps,
        IDs,
        run_chat_agent,
    )
"""

__all__ = [
    # tools
    "ToolGroupMeta",
    "register_tool_groups",
    "list_registered_groups",
    "get_tool_group",
    # memory
    "MemoryContextBuilder",
    "RetrievalChain",
    "LLMContext",
    "build_chat_retrieval_chain",
    "build_context_from_sources",
    # agent
    "AgentDeps",
    "IDs",
    "run_chat_agent",
]

from .tools import (
    ToolGroupMeta,
    register_tool_groups,
    list_registered_groups,
    get_tool_group,
)
from .memory import (
    MemoryContextBuilder,
    RetrievalChain,
    LLMContext,
    build_chat_retrieval_chain,
    build_context_from_sources,
)
from .agent import (
    AgentDeps,
    IDs,
    run_chat_agent,
)
