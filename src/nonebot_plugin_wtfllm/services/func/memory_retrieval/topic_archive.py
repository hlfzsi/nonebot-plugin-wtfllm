"""话题归档长期记忆检索任务"""

from dataclasses import dataclass

from ._base import RetrievalTask
from ....memory.items import MemorySource, MemoryItemStream
from ....db import memory_item_repo
from ....v_db.repositories.topic_archive import TopicArchiveRepository

_topic_archive_repo = TopicArchiveRepository()


@dataclass
class TopicArchiveTask(RetrievalTask):
    """检索与当前查询语义相近的归档话题记忆。"""

    agent_id: str
    group_id: str | None = None
    user_id: str | None = None
    query: str = ""
    limit: int = 3
    prefix: str = "<archived_topic_memory>"
    suffix: str = "</archived_topic_memory>"

    async def execute(self) -> set[MemorySource]:
        if not self.query:
            return set()

        results = await _topic_archive_repo.search_by_session(
            agent_id=self.agent_id,
            query=self.query,
            group_id=self.group_id,
            user_id=self.user_id,
            limit=self.limit,
        )
        if not results:
            return set()

        all_msg_ids = [
            mid for r in results for mid in r.item.representative_message_ids
        ]
        if not all_msg_ids:
            return set()

        items = await memory_item_repo.get_many_by_message_ids(all_msg_ids)
        if not items:
            return set()

        stream = MemoryItemStream.create(
            items=items,
            prefix=self.prefix,
            suffix=self.suffix,
            role="topic_archive",
            priority=0.25,
        )
        return {stream}
