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
    max_tokens: int | None = 4000
    prefix: str = "<knowledge_base>"
    suffix: str = "</knowledge_base>"

    async def execute(self) -> set[MemorySource]:
        results = await knowledge_base_repo.search_relevant(
            agent_id=self.agent_id,
            query=self.query,
            limit=self.limit,
        )
        if not results:
            return set()

        if self.max_tokens is not None:
            entries = []
            total_tokens = 0
            for r in results:
                if total_tokens + r.item.token_count <= self.max_tokens:
                    entries.append(r.item)
                    total_tokens += r.item.token_count
                else:
                    break
        else:
            entries = [r.item for r in results]

        if not entries:
            return set()

        block = KnowledgeBlock(
            entries=entries,
            prefix=self.prefix,
            suffix=self.suffix,
        )
        return {block}
