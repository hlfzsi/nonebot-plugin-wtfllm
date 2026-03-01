import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

from ._base import RetrievalTask
from ....memory.items import MemorySource, MemoryItemStream, MemoryItemUnion
from ....db import memory_item_repo

if TYPE_CHECKING:
    from ....memory.providers import AliasProvider


@dataclass
class RecentReactTask(RetrievalTask):
    """跨会话最近交互记忆检索

    根据 msg_tracker 提供的最近交互消息 ID，
    检索其他会话的记忆片段，为每个会话生成独立的 MemoryItemStream。
    """

    recent_react: Dict[str, List[str]]
    alias_provider: "AliasProvider"
    max_token_per_stream: int = 5000

    async def execute(self) -> set[MemorySource]:
        if not self.recent_react:
            return set()

        tasks: Dict[str, asyncio.Task[List[MemoryItemUnion]]] = {}
        async with asyncio.TaskGroup() as tg:
            for gid, mid_list in self.recent_react.items():
                if mid_list:
                    tasks[gid] = tg.create_task(
                        memory_item_repo.get_many_by_message_ids(mid_list)
                    )

        sources: set[MemorySource] = set()
        for gid, task in tasks.items():
            items = task.result()
            scene_name = self.alias_provider.get_alias(gid) if gid else "私聊"
            stream = MemoryItemStream.create(
                items=items,
                prefix=f'<memory scene="{scene_name}">',
                suffix="</memory>",
                max_token=self.max_token_per_stream,
                priority=0.3,
            )
            sources.add(stream)

        return sources
