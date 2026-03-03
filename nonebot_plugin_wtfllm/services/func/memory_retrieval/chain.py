import asyncio
import copy
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Self

from ._base import RetrievalTask
from ....memory.items import MemorySource

from .main_chat import MainChatTask
from .core_memory import CoreMemoryTask
from .core_memory import CrossSessionMemoryTask
from .knowledge import KnowledgeSearchTask
from .recent_react import RecentReactTask
from .tool_history import ToolCallHistoryTask
from .topic_context import TopicContextTask
from .topic_archive import TopicArchiveTask


if TYPE_CHECKING:
    from ....memory.providers import AliasProvider

_UNSET: Any = object()


class RetrievalChain:
    """记忆检索链

    调用 ``.resolve()`` 执行全部任务并返回合并的 ``MemorySource`` 集合。

    构造时可传入默认参数，快捷方法会自动使用这些默认值，
    显式传参时覆盖默认值::

        sources = await (
            RetrievalChain(agent_id=aid, group_id=gid, query=text)
            .main_chat(limit=50)
            .core_memory()
            .cross_session_memory()
            .knowledge()
            .recent_react(recent_react=react, alias_provider=ap)
            .tool_history()
            .resolve()
        )

    也支持合并运算::

        chain_a = RetrievalChain(agent_id=aid).main_chat(...)
        chain_b = RetrievalChain(agent_id=aid).knowledge(...)
        merged = chain_a + chain_b          # 返回新链（继承左链默认值）
        chain_a += chain_b                  # 原地合并
    """

    def __init__(
        self,
        tasks: List[RetrievalTask] | None = None,
        *,
        agent_id: str | None = None,
        group_id: str | None = None,
        user_id: str | None = None,
        query: str | None = None,
    ) -> None:
        self._tasks: list[RetrievalTask] = list(tasks) if tasks else []
        self.agent_id = agent_id
        self.group_id = group_id
        self.user_id = user_id
        self.query = query

    def _d(self, value: Any, default: Any) -> Any:
        """解析参数：显式传值优先，否则回退到链默认值"""
        return default if value is _UNSET else value

    def _add(self, task: RetrievalTask) -> Self:
        """链入一个检索任务"""
        self._tasks.append(task)
        return self

    def _extend(self, tasks: Iterable[RetrievalTask]) -> Self:
        """批量链入多个检索任务"""
        self._tasks.extend(tasks)
        return self

    def copy(self) -> "RetrievalChain":
        """深拷贝当前链，返回一条独立的新链"""
        new = RetrievalChain(
            tasks=copy.deepcopy(self._tasks),
            agent_id=self.agent_id,
            group_id=self.group_id,
            user_id=self.user_id,
            query=self.query,
        )
        return new

    async def resolve(self) -> set[MemorySource]:
        """并发执行所有任务，返回合并的 MemorySource 集合"""
        if not self._tasks:
            return set()

        results: set[MemorySource] = set()
        async with asyncio.TaskGroup() as tg:
            futures = [tg.create_task(t.execute()) for t in self._tasks]

        for f in futures:
            results |= f.result()

        return results

    def __add__(self, other: "RetrievalChain") -> "RetrievalChain":
        """合并两条链，返回一条新链（继承左链默认值）"""
        if not isinstance(other, RetrievalChain):
            return NotImplemented
        return RetrievalChain(
            tasks=self._tasks + other._tasks,
            agent_id=self.agent_id,
            group_id=self.group_id,
            user_id=self.user_id,
            query=self.query,
        )

    def __iadd__(self, other: "RetrievalChain") -> Self:
        """原地合并另一条链的任务"""
        if not isinstance(other, RetrievalChain):
            return NotImplemented
        self._extend(other._tasks)
        return self

    def __len__(self) -> int:
        return len(self._tasks)

    def __bool__(self) -> bool:
        return bool(self._tasks)

    def __iter__(self):
        return iter(self._tasks)

    def __contains__(self, task_type: type) -> bool:
        """检查链中是否包含指定类型的任务"""
        return any(isinstance(t, task_type) for t in self._tasks)

    def __repr__(self) -> str:
        task_names = [type(t).__name__ for t in self._tasks]
        return f"RetrievalChain({task_names})"

    def main_chat(
        self,
        *,
        agent_id: str = _UNSET,
        group_id: str | None = _UNSET,
        user_id: str | None = _UNSET,
        limit: int = 50,
    ) -> Self:
        """添加主会话聊天记录检索任务"""
        return self._add(
            MainChatTask(
                agent_id=self._d(agent_id, self.agent_id),
                group_id=self._d(group_id, self.group_id),
                user_id=self._d(user_id, self.user_id),
                limit=limit,
            )
        )

    def core_memory(
        self,
        *,
        agent_id: str = _UNSET,
        group_id: str | None = _UNSET,
        user_id: str | None = _UNSET,
        prefix: str = "<core_memory>",
        suffix: str = "</core_memory>",
    ) -> Self:
        """添加当前会话核心记忆检索任务"""
        return self._add(
            CoreMemoryTask(
                agent_id=self._d(agent_id, self.agent_id),
                group_id=self._d(group_id, self.group_id),
                user_id=self._d(user_id, self.user_id),
                prefix=prefix,
                suffix=suffix,
            )
        )

    def cross_session_memory(
        self,
        *,
        agent_id: str = _UNSET,
        query: str = _UNSET,
        exclude_group_id: str | None = _UNSET,
        exclude_user_id: str | None = _UNSET,
        limit: int = 5,
        prefix: str = "<cross_session_memory>",
        suffix: str = "</cross_session_memory>",
    ) -> Self:
        """添加跨会话核心记忆语义搜索任务

        ``exclude_group_id`` / ``exclude_user_id`` 未显式传入时，
        自动回退到链默认的 ``group_id`` / ``user_id``。
        """
        return self._add(
            CrossSessionMemoryTask(
                agent_id=self._d(agent_id, self.agent_id),
                query=self._d(query, self.query),
                exclude_group_id=self._d(exclude_group_id, self.group_id),
                exclude_user_id=self._d(exclude_user_id, self.user_id),
                limit=limit,
                prefix=prefix,
                suffix=suffix,
            )
        )

    def knowledge(
        self,
        *,
        agent_id: str = _UNSET,
        query: str = _UNSET,
        limit: int = 5,
        max_tokens: int = 4000,
    ) -> Self:
        """添加知识库语义搜索任务"""
        return self._add(
            KnowledgeSearchTask(
                agent_id=self._d(agent_id, self.agent_id),
                query=self._d(query, self.query),
                limit=limit,
                max_tokens=max_tokens,
            )
        )

    def recent_react(
        self,
        *,
        recent_react: Dict[str, List[str]],
        alias_provider: "AliasProvider",
        max_token_per_stream: int = 5000,
    ) -> Self:
        """添加跨会话最近交互记忆检索任务"""
        return self._add(
            RecentReactTask(
                recent_react=recent_react,
                alias_provider=alias_provider,
                max_token_per_stream=max_token_per_stream,
            )
        )

    def tool_history(
        self,
        *,
        agent_id: str = _UNSET,
        group_id: str | None = _UNSET,
        user_id: str | None = _UNSET,
        limit: int = 10,
    ) -> Self:
        """添加工具调用历史检索任务"""
        return self._add(
            ToolCallHistoryTask(
                agent_id=self._d(agent_id, self.agent_id),
                group_id=self._d(group_id, self.group_id),
                user_id=self._d(user_id, self.user_id),
                limit=limit,
            )
        )

    def topic_context(
        self,
        *,
        agent_id: str = _UNSET,
        group_id: str | None = _UNSET,
        user_id: str | None = _UNSET,
        query: str = _UNSET,
        max_topic_messages: int = 10,
    ) -> Self:
        """添加话题上下文检索任务"""
        return self._add(
            TopicContextTask(
                agent_id=self._d(agent_id, self.agent_id),
                group_id=self._d(group_id, self.group_id),
                user_id=self._d(user_id, self.user_id),
                query=self._d(query, self.query),
                max_topic_messages=max_topic_messages,
            )
        )

    def topic_archive(
        self,
        *,
        agent_id: str = _UNSET,
        group_id: str | None = _UNSET,
        user_id: str | None = _UNSET,
        query: str = _UNSET,
        limit: int = 3,
    ) -> Self:
        """添加话题归档长期记忆检索任务"""
        return self._add(
            TopicArchiveTask(
                agent_id=self._d(agent_id, self.agent_id),
                group_id=self._d(group_id, self.group_id),
                user_id=self._d(user_id, self.user_id),
                query=self._d(query, self.query),
                limit=limit,
            )
        )
