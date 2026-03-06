from dataclasses import dataclass

from ._base import RetrievalTask
from ....memory.items import MemorySource, MemoryItemStream
from ....db import memory_item_repo


@dataclass
class MainChatTask(RetrievalTask):
    """主会话聊天记录检索

    群聊模式按 group_id 检索，私聊模式按 user_id 检索。
    """

    agent_id: str
    group_id: str | None = None
    user_id: str | None = None
    limit: int = 50

    async def execute(self) -> set[MemorySource]:
        if self.group_id:
            items = await memory_item_repo.get_by_group(
                group_id=self.group_id,
                agent_id=self.agent_id,
                limit=self.limit,
            )
        elif self.user_id:
            items = await memory_item_repo.get_in_private_by_user(
                user_id=self.user_id,
                agent_id=self.agent_id,
                limit=self.limit,
            )
        else:
            return set()

        stream = MemoryItemStream.create(items=items, role="main_chat", priority=0.1)
        return {stream}
