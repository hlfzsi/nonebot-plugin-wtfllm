__all__ = [
    "MemoryContextBuilder",
    "RetrievalChain",
    "LLMContext",
    "build_chat_retrieval_chain",
    "build_context_from_sources",
]

from typing import Dict, Iterable

from ..memory.director import MemoryContextBuilder
from ..memory.context import LLMContext
from ..memory.items import MemorySource
from ..services.func.memory_retrieval import RetrievalChain
from ..utils import APP_CONFIG


def build_chat_retrieval_chain(
    agent_id: str,
    *,
    group_id: str | None = None,
    user_id: str | None = None,
    query: str = "",
    short_memory_limit: int | None = None,
    tool_history_limit: int | None = None,
    knowledge_limit: int | None = None,
    knowledge_max_tokens: int | None = None,
    topic_max_messages: int | None = None,
) -> RetrievalChain:
    """按默认配置组装标准聊天场景的检索链。

    Returns:
        配置好的 RetrievalChain，调用方需自行 ``await chain.resolve()``。
    """
    chain = RetrievalChain(
        agent_id=agent_id,
        group_id=group_id,
        user_id=user_id,
        query=query,
    )
    chain.main_chat(
        limit=short_memory_limit or APP_CONFIG.short_memory_max_count,
    ).note().core_memory().cross_session_memory().tool_history(
        limit=tool_history_limit or APP_CONFIG.tool_call_record_max_count,
    ).knowledge(
        limit=knowledge_limit or APP_CONFIG.knowledge_base_max_results,
        max_tokens=knowledge_max_tokens or APP_CONFIG.knowledge_base_max_tokens,
    ).topic_context(
        max_topic_messages=topic_max_messages or APP_CONFIG.topic_max_context_messages,
    )
    return chain


def build_context_from_sources(
    sources: Iterable[MemorySource],
    *,
    agent_id: str,
    group_id: str | None = None,
    user_id: str | None = None,
    prefix_prompt: str = "",
    suffix_prompt: str = "",
    custom_ref: Dict[str, str] | None = None,
) -> MemoryContextBuilder:
    """从已检索的 MemorySource 集合构建 MemoryContextBuilder。

    用法::

        chain = build_chat_retrieval_chain(agent_id, group_id=gid, query=q)
        sources = await chain.resolve()
        builder = build_context_from_sources(sources, agent_id=agent_id, group_id=gid)
        prompt = builder.to_prompt()
    """
    builder = MemoryContextBuilder(
        prefix_prompt=prefix_prompt or None,
        suffix_prompt=suffix_prompt or None,
        agent_id=agent_id,
        custom_ref=custom_ref,
    )
    builder.extend(sources)
    return builder
