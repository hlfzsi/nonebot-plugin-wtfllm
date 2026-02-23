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
    "UserPersona",
    "ScheduledMessage",
    "ToolCallRecordTable",
    # 生命周期
    "init_db",
    "shutdown_db",
    # Repository 类
    "BaseRepository",
    "MemoryItemRepository",
    "UserPersonaRepository",
    "ScheduledMessageRepository",
    "ToolCallRecordRepository",
    # Repository 单例
    "memory_item_repo",
    "user_persona_repo",
    "scheduled_message_repo",
    "tool_call_record_repo",
]

from .engine import ENGINE, WRITE_LOCK, SESSION_MAKER
from .models import MemoryItemTable, UserPersona, ScheduledMessage, ToolCallRecordTable
from .lifecycle import init_db, shutdown_db
from .repositories import (
    BaseRepository,
    MemoryItemRepository,
    UserPersonaRepository,
    ScheduledMessageRepository,
    ToolCallRecordRepository,
)

memory_item_repo = MemoryItemRepository()
user_persona_repo = UserPersonaRepository()
scheduled_message_repo = ScheduledMessageRepository()
tool_call_record_repo = ToolCallRecordRepository()
