"""数据库模块 - 统一数据访问层

提供数据库引擎、模型、Repository 和生命周期管理的统一入口。
"""

__all__ = [
    # 引擎和会话
    "ENGINE",
    "WRITE_LOCK",
    "SESSION_MAKER",
    # 模型
    "MemoryItemTable",
    "NoteMemoryTable",
    "UserPersona",
    "ScheduledJob",
    "ToolCallRecordTable",
    "ThoughtRecordTable",
    # 生命周期
    "init_db",
    "shutdown_db",
    # Repository 类
    "BaseRepository",
    "MemoryItemRepository",
    "NoteMemoryRepository",
    "UserPersonaRepository",
    "ScheduledJobRepository",
    "ToolCallRecordRepository",
    "ThoughtRecordRepository",
    # Repository 单例
    "memory_item_repo",
    "note_memory_repo",
    "user_persona_repo",
    "scheduled_job_repo",
    "tool_call_record_repo",
    "thought_record_repo",
]

from .engine import ENGINE, WRITE_LOCK, SESSION_MAKER
from .models import (
    MemoryItemTable,
    NoteMemoryTable,
    UserPersona,
    ScheduledJob,
    ToolCallRecordTable,
    ThoughtRecordTable,
)
from .lifecycle import init_db, shutdown_db
from .repositories import (
    BaseRepository,
    MemoryItemRepository,
    NoteMemoryRepository,
    UserPersonaRepository,
    ScheduledJobRepository,
    ToolCallRecordRepository,
    ThoughtRecordRepository,
)

memory_item_repo = MemoryItemRepository()
note_memory_repo = NoteMemoryRepository()
user_persona_repo = UserPersonaRepository()
scheduled_job_repo = ScheduledJobRepository()
tool_call_record_repo = ToolCallRecordRepository()
thought_record_repo = ThoughtRecordRepository()
