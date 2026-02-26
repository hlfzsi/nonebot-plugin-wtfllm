from dataclasses import dataclass

from ._base import RetrievalTask
from ....memory.items import MemorySource
from ....memory.items.knowledge_base import KnowledgeBlock
from ....v_db import knowledge_base_repo


@dataclass
class KnowledgeSearchTask(RetrievalTask):
    """知识库语义搜索"""

    agent_id: str
    query: str
    limit: int = 5
    max_tokens: int = 4000

    async def execute(self) -> set[MemorySource]:
        results = await knowledge_base_repo.search_relevant(
            agent_id=self.agent_id,
            query=self.query,
            limit=self.limit,
        )
        if not results:
            return set()

        entries = []
        total_tokens = 0
        for r in results:
            if total_tokens + r.item.token_count <= self.max_tokens:
                entries.append(r.item)
                total_tokens += r.item.token_count
            else:
                break

        if not entries:
            return set()

        block = KnowledgeBlock(
            entries=entries,
            prefix="<knowledge_base>",
            suffix="</knowledge_base>",
        )
        return {block}
