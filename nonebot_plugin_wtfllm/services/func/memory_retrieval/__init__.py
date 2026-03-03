"""记忆检索链模块

提供链式、可组合的记忆检索能力。每种记忆检索逻辑封装为独立的
``RetrievalTask``，通过 ``RetrievalChain`` 链式组装后并发执行。

``RetrievalChain`` 提供快捷方法，调用方无需额外导入任何 Task::

    sources = await (
        RetrievalChain()
        .main_chat(agent_id=aid, group_id=gid, limit=50)
        .core_memory(agent_id=aid, group_id=gid)
        .cross_session_memory(agent_id=aid, query=text, exclude_group_id=gid)
        .knowledge(agent_id=aid, query=text)
        .recent_react(recent_react=react, alias_provider=ap)
        .tool_history(agent_id=aid, group_id=gid)
        .resolve()
    )
"""

__all__ = [
    "RetrievalTask",
    "RetrievalChain",
    "MainChatTask",
    "CoreMemoryTask",
    "CrossSessionMemoryTask",
    "KnowledgeSearchTask",
    "RecentReactTask",
    "ToolCallHistoryTask",
    "TopicContextTask",
    "TopicArchiveTask",
]

from ._base import RetrievalTask
from .chain import RetrievalChain
from .main_chat import MainChatTask
from .core_memory import CoreMemoryTask, CrossSessionMemoryTask
from .knowledge import KnowledgeSearchTask
from .recent_react import RecentReactTask
from .tool_history import ToolCallHistoryTask
from .topic_context import TopicContextTask
from .topic_archive import TopicArchiveTask
