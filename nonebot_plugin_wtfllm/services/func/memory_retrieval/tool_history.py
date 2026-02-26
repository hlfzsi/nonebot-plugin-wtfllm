from dataclasses import dataclass

from ._base import RetrievalTask
from ....memory.items import MemorySource
from ....memory.items.tool_call_summary import ToolCallSummaryBlock
from ....db import tool_call_record_repo


@dataclass
class ToolCallHistoryTask(RetrievalTask):
    """工具调用历史检索"""

    agent_id: str
    group_id: str | None = None
    user_id: str | None = None
    limit: int = 10

    async def execute(self) -> set[MemorySource]:
        records = await tool_call_record_repo.get_recent(
            agent_id=self.agent_id,
            group_id=self.group_id,
            user_id=self.user_id,
            limit=self.limit,
        )
        if not records:
            return set()

        block = ToolCallSummaryBlock(
            tool_names=[r.tool_name for r in records],
            prefix="<recent_tools>",
            suffix="</recent_tools>",
        )
        return {block}
