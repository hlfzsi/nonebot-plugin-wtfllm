__all__ = ["ScheduledJob", "ScheduledJobStatus"]

import time
from enum import StrEnum
from typing import Any, Dict, Optional

from sqlmodel import SQLModel, Field, Column, JSON


class ScheduledJobStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    MISSED = "missed"
    CANCELED = "canceled"


class ScheduledJob(SQLModel, table=True):
    """通用调度任务记录

    通过 task_name + task_params 实现与具体业务逻辑的解耦：
    - task_name 映射到注册表中的 handler 函数
    - task_params 存储 handler 所需的 BaseModel 参数 (JSON)
    - trigger_config 存储 APScheduler 触发器配置 (JSON)
    """

    id: Optional[int] = Field(default=None, primary_key=True, description="自增ID")
    job_id: str = Field(..., index=True, unique=True, description="唯一作业标识")

    task_name: str = Field(..., index=True, description="注册表中的任务类型名称")
    task_params: Dict[str, Any] = Field(
        ...,
        sa_column=Column(JSON),
        description="handler 参数, BaseModel.model_dump()",
    )

    trigger_config: Dict[str, Any] = Field(
        ...,
        sa_column=Column(JSON),
        description="APScheduler 触发器配置, TriggerConfig.model_dump()",
    )

    user_id: Optional[str] = Field(default=None, index=True, description="所属用户ID")
    group_id: Optional[str] = Field(default=None, index=True, description="关联群组ID")
    agent_id: Optional[str] = Field(
        default=None, index=True, description="Bot/Agent ID"
    )

    status: ScheduledJobStatus = Field(
        default=ScheduledJobStatus.PENDING, description="任务状态"
    )
    created_at: int = Field(
        default_factory=lambda: int(time.time()), description="创建时间戳"
    )
    executed_at: Optional[int] = Field(default=None, description="执行时间戳")
    error_message: Optional[str] = Field(default=None, description="失败原因")
    description: Optional[str] = Field(default=None, description="人类可读的任务描述")
