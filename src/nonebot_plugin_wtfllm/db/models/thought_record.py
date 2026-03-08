import time
from typing import Optional

from sqlmodel import SQLModel, Field


class ThoughtRecordTable(SQLModel, table=True):
    """模型显式思考记录表"""

    id: Optional[int] = Field(default=None, primary_key=True, description="自增主键")
    run_id: Optional[str] = Field(default=None, index=True, description="可选运行标识")
    agent_id: str = Field(..., index=True, description="所属智能体ID")
    group_id: Optional[str] = Field(
        default=None, index=True, description="群组ID"
    )
    user_id: Optional[str] = Field(
        default=None, index=True, description="用户ID"
    )
    thought_of_chain: str = Field(..., description="模型输出的显式思考文本")
    timestamp: int = Field(
        default_factory=lambda: int(time.time()), index=True, description="记录时间戳 (秒)"
    )