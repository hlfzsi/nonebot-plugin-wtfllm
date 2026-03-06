from typing import Dict, Optional, Self, TYPE_CHECKING

from sqlmodel import SQLModel, Field, Column, JSON

if TYPE_CHECKING:
    from ...llm.deps import ToolCallInfo


class ToolCallRecordTable(SQLModel, table=True):
    """工具调用记录表"""

    id: Optional[int] = Field(default=None, primary_key=True, description="自增主键")
    run_id: str = Field(..., index=True, description="Agent.run() 唯一标识（分组键）")
    agent_id: str = Field(..., index=True, description="所属智能体ID")
    group_id: Optional[str] = Field(
        default=None, index=True, description="群组ID (群聊时)"
    )
    user_id: Optional[str] = Field(
        default=None, index=True, description="用户ID (私聊时)"
    )
    run_step: int = Field(..., description="agent_ctx.run_step (从 0 开始)")
    tool_name: str = Field(..., description="工具函数名称")
    kwargs: Dict[str, str] = Field(
        default_factory=dict,
        sa_column=Column(JSON),
        description="完整的关键字参数字典",
    )
    timestamp: int = Field(..., index=True, description="工具调用时间戳 (秒)")

    @classmethod
    def from_tool_call_info(
        cls,
        info: "ToolCallInfo",
        agent_id: str,
        group_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Self:
        """从 ToolCallInfo 创建 ToolCallRecordTable 实例

        Args:
            info: ToolCallInfo 实例
            agent_id: 智能体ID
            group_id: 群组ID (群聊时)
            user_id: 用户ID (私聊时)
        """
        return cls(
            run_id=info.run_id,
            agent_id=agent_id,
            group_id=group_id,
            user_id=user_id,
            run_step=info.round_index,
            tool_name=info.tool_name,
            kwargs=info.kwargs,
            timestamp=info.timestamp,
        )
