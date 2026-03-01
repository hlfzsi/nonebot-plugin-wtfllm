import asyncio
from typing import Dict, List, Optional, Set, Annotated, Any

from nonebot.adapters import Bot
from nonebot_plugin_uninfo import Uninfo
from nonebot_plugin_alconna import Target, UniMessage
from pydantic import BaseModel, Field, field_validator, ConfigDict, PlainValidator
from pydantic_ai import RunContext

from ..memory import MemoryContextBuilder, MemoryItemUnion


class ToolCallInfo(BaseModel):
    run_id: str = Field(..., description="当前对话的唯一标识")
    round_index: int = Field(..., description="工具调用所在的对话轮次，从 0 开始计数")
    tool_name: str = Field(..., description="被调用的工具函数名称")
    kwargs: Dict[str, str] = Field(
        ..., description="被调用的工具函数的关键字参数字符串字典"
    )
    timestamp: int = Field(..., description="工具调用的时间戳，单位为秒")


class IDs(BaseModel):
    user_id: Optional[str] = Field(..., description="当前用户的唯一标识")
    group_id: Optional[str] = Field(
        None, description="当前群组的唯一标识，私聊时为 None"
    )
    agent_id: str = Field(..., description="当前助手的唯一标识")


class NonebotRuntime(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    bot: Bot = Field(..., description="Nonebot 适配器 Bot 实例")
    session: Uninfo = Field(..., description="Nonebot 插件 Uninfo 会话实例")
    target: Target = Field(..., description="当前消息的目标对象")


class AgentDeps(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    active_tool_groups: Set[str] = Field(
        default_factory=set, description="当前对话中激活的工具组名称集合"
    )

    ids: IDs = Field(..., description="当前对话的相关唯一标识信息")

    context: MemoryContextBuilder = Field(..., description="当前对话的上下文信息")

    nb_runtime: Optional[NonebotRuntime] = Field(
        default=None, description="Nonebot 运行时相关信息"
    )

    reply_segments: UniMessage = Field(
        default_factory=UniMessage, description="回复消息的内容"
    )

    cm: asyncio.Timeout | None = Field(default=None, description="当前对话的超时控制器")

    message_queue: Annotated[
        List[MemoryItemUnion] | None, PlainValidator(lambda v: v)  # 保持引用
    ] = Field(default=None, description="Agent 执行期间的新消息队列引用")

    tool_chain: List[ToolCallInfo] = Field(
        default_factory=list, description="当前对话中已调用的工具链列表"
    )

    tool_point_budget: int = Field(
        default=0, description="工具点数预算总量，0 表示不启用"
    )

    tool_points_used: int = Field(default=0, description="当前对话中已消耗的工具点数")

    caches: Dict[Any, Any] = Field(
        default_factory=dict, description="当前对话的临时缓存"
    )

    @property
    def tool_budget_enabled(self) -> bool:
        return self.tool_point_budget > 0

    @property
    def tool_points_remaining(self) -> int:
        if not self.tool_budget_enabled:
            return -1
        return max(0, self.tool_point_budget - self.tool_points_used)

    @property
    def tool_budget_exhausted(self) -> bool:
        return self.tool_budget_enabled and self.tool_points_remaining <= 0

    @property
    def tool_budget_ratio(self) -> float:
        if not self.tool_budget_enabled:
            return 1.0
        return self.tool_points_remaining / self.tool_point_budget

    @field_validator("context", mode="before")
    @classmethod
    def validate_context(cls, v):
        if isinstance(v, MemoryContextBuilder):
            return v
        raise ValueError("context must be a MemoryContextBuilder instance")

    @field_validator("nb_runtime", mode="before")
    @classmethod
    def validate_nb_runtime(cls, v):
        if v is None or isinstance(v, NonebotRuntime):
            return v
        raise ValueError("nb_runtime must be a NonebotRuntime instance or None")


Context = RunContext[AgentDeps]
