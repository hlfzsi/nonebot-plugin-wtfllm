import time
from dataclasses import dataclass

from ._base import RetrievalTask
from ....memory.items import MemorySource, MemoryItemStream
from ....db import memory_item_repo
from ....topic import topic_manager


@dataclass
class TopicContextTask(RetrievalTask):
    """检索当前话题中已滑出滑动窗口的旧消息。"""

    agent_id: str
    group_id: str | None = None
    user_id: str | None = None
    query: str = ""
    max_topic_messages: int = 10
    window_seconds: float = 7200

    async def execute(self) -> set[MemorySource]:
        if not self.query:
            return set()

        cutoff = time.time() - self.window_seconds

        label, topic_msg_ids = topic_manager.query_topic(
            agent_id=self.agent_id,
            group_id=self.group_id,
            user_id=self.user_id,
            query=self.query,
            max_count=self.max_topic_messages,
            before_timestamp=cutoff,
        )

        if not topic_msg_ids:
            return set()

        items = await memory_item_repo.get_many_by_message_ids(topic_msg_ids)
        if not items:
            return set()

        stream = MemoryItemStream.create(
            items=items,
            prefix="<current_session_topic_context>",
            suffix="</current_session_topic_context>",
            role="topic_context",
            priority=0.2,
        )
        return {stream}
