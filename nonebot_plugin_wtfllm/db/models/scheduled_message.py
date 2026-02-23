import time
from enum import StrEnum
from typing import Any, Dict, List, Optional

from nonebot_plugin_uninfo import Uninfo
from nonebot_plugin_alconna import UniMessage, Target
from sqlmodel import SQLModel, Field, Column, JSON

from ...memory import MemoryContextBuilder
from ...utils import get_agent_id_from_bot, SCHEDULED_MESSAGE_CACHE_DIR


class ScheduledMessageStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    MISSED = "missed"
    CANCELED = "canceled"


class ScheduledFunctionType(StrEnum):
    STATIC_MESSAGE = "static_message"
    DYNAMIC_MESSAGE = "dynamic_message"


class ScheduledMessage(SQLModel, table=True):
    """定时消息记录表"""

    id: Optional[int] = Field(default=None, primary_key=True, description="自增ID")
    job_id: str = Field(..., index=True, description="APScheduler 作业ID")

    target_data: Dict[str, Any] = Field(
        ..., sa_column=Column(JSON), description="消息目标, Target"
    )

    user_id: str = Field(..., index=True, description="目标用户ID")
    group_id: Optional[str] = Field(..., index=True, description="目标群组ID")
    agent_id: str = Field(..., index=True, description="Bot ID")

    messages: List[Dict[str, Any]] = Field(
        ...,
        sa_column=Column(JSON),
        description="消息序列, UniMessage",
    )
    trigger_time: int = Field(..., description="触发时间, 时间戳格式")
    status: ScheduledMessageStatus = Field(
        default=ScheduledMessageStatus.PENDING, description="状态"
    )
    created_at: int = Field(..., description="创建时间, 时间戳格式")
    executed_at: Optional[int] = Field(default=None, description="执行时间, 时间戳格式")
    error_message: Optional[str] = Field(default=None, description="错误信息")

    func_type: ScheduledFunctionType = Field(..., description="函数类型")

    @classmethod
    def create(
        cls,
        job_id: str,
        target: Target,
        session: Uninfo,
        unimsg: UniMessage,
        trigger_time: int,
        func_type: ScheduledFunctionType = ScheduledFunctionType.STATIC_MESSAGE,
        created_at: int = int(time.time()),
    ) -> "ScheduledMessage":
        """
        Args:
            job_id (str): apscheduler 作业ID
            target (Target): 消息目标
            session (Uninfo): 会话信息
            unimsg (UniMessage): 消息内容
            trigger_time (int): 触发时间，时间戳格式
            created_at (int, optional): 创建时间，时间戳格式。
        """
        target_data = target.dump()
        messages = unimsg.dump(media_save_dir=SCHEDULED_MESSAGE_CACHE_DIR)
        return cls(
            job_id=job_id,
            target_data=target_data,
            user_id=session.user.id,
            group_id=session.group.id if session.group else None,
            agent_id=get_agent_id_from_bot(session),
            messages=messages,
            trigger_time=trigger_time,
            created_at=created_at,
            func_type=func_type,
        )

    def to_text(self, ctx: MemoryContextBuilder | None = None) -> str:
        """生成定时消息的文本描述"""
        _user = ctx.ctx.alias_provider.get_alias(self.user_id) if ctx else self.user_id
        _group = (
            ctx.ctx.alias_provider.get_alias(self.group_id)
            if ctx and self.group_id
            else self.group_id
        )

        target_desc = f"用户 {_user}"
        if _group:
            target_desc += f" 在群 {_group}"
        trigger_time_str = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(self.trigger_time)
        )
        return f"定时消息 (job_id={self.job_id}) 触发时间: {trigger_time_str} 状态: {self.status} {target_desc} 消息: {self.messages if self.func_type == ScheduledFunctionType.STATIC_MESSAGE else '<动态消息>'}"
