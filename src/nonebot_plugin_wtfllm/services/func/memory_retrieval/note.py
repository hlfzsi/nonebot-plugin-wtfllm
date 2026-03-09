from dataclasses import dataclass
import time

from ._base import RetrievalTask
from ....db import note_memory_repo
from ....memory.items import MemorySource
from ....memory.items.note import NoteBlock


@dataclass
class NoteTask(RetrievalTask):
    """当前会话短期备忘检索。"""

    agent_id: str
    group_id: str | None = None
    user_id: str | None = None
    prefix: str = "<note_memory>"
    suffix: str = "</note_memory>"

    async def execute(self) -> set[MemorySource]:
        await note_memory_repo.delete_expired_by_session(
            agent_id=self.agent_id,
            group_id=self.group_id,
            user_id=self.user_id,
        )
        notes = await note_memory_repo.get_by_session(
            agent_id=self.agent_id,
            group_id=self.group_id,
            user_id=self.user_id,
            include_expired=False,
        )
        notes = [note for note in notes if note.expires_at > int(time.time())]
        if not notes:
            return set()

        block = NoteBlock(
            notes=notes,
            prefix=self.prefix,
            suffix=self.suffix,
        )
        return {block}