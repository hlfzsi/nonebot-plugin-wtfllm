import uuid
from time import time
from inspect import iscoroutinefunction
from typing import (
    Any,
    List,
    Callable,
    Awaitable,
    Optional,
    ClassVar,
    Dict,
    TypeVar,
    ParamSpec,
    overload,
)

from makefun import wraps
from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from .utils import build_budget_suffix, append_budget_suffix
from ...deps import Context, AgentDeps, ToolCallInfo
from ....utils import logger

P = ParamSpec("P")
R = TypeVar("R")


def _extract_and_track(
    func_name: str, args: tuple, kwargs: dict, cost: int = 0
) -> AgentDeps:
    agent_deps = None
    agent_ctx = None

    if kwargs and (ctx := list(kwargs.values())[0]):
        if isinstance(ctx, RunContext) and isinstance(ctx.deps, AgentDeps):
            agent_deps = ctx.deps
            agent_ctx = ctx

    if agent_deps is not None and agent_ctx is not None:
        if agent_ctx.run_id is None:
            logger.warning(
                "Agent run_id is None, generating a new run_id for tracking. This may affect the consistency of run_id across tool calls."
            )
            agent_ctx.run_id = str(uuid.uuid4())
        agent_deps.tool_chain.append(
            ToolCallInfo(
                run_id=agent_ctx.run_id,
                round_index=agent_ctx.run_step,
                tool_name=func_name,
                kwargs={k: repr(v) for k, v in kwargs.items() if v is not agent_ctx},
                timestamp=int(time()),
            )
        )

        if agent_deps.tool_budget_enabled:
            agent_deps.tool_points_used += cost

    else:
        raise RuntimeError(
            "Failed to track tool call: AgentDeps not found in the first argument's context."
        )
    return agent_deps


@overload
def tool_call_hook(
    func: Callable[P, Awaitable[R]], cost: int = 0
) -> Callable[P, Awaitable[R]]: ...


@overload
def tool_call_hook(func: Callable[P, R], cost: int = 0) -> Callable[P, R]: ...


def tool_call_hook(func: Callable[P, R], cost: int = 0):
    if not iscoroutinefunction(func):

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger.debug(f"Tool Calling: {func.__name__}")

            agent_deps = _extract_and_track(func.__name__, args, kwargs, cost=cost)

            result = func(*args, **kwargs)
            logger.debug(f"Tool Call Complete: {func.__name__}")

            suffix = build_budget_suffix(agent_deps, cost)
            return append_budget_suffix(result, suffix)

        return sync_wrapper
    else:

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger.debug(f"Tool Calling: {func.__name__}")

            agent_deps = _extract_and_track(func.__name__, args, kwargs, cost=cost)
            result = await func(*args, **kwargs)

            logger.debug(f"Tool Call Complete: {func.__name__}")

            suffix = build_budget_suffix(agent_deps, cost)
            return append_budget_suffix(result, suffix)

        return async_wrapper


class ToolGroupMeta(BaseModel):
    mapping: ClassVar[Dict[str, "ToolGroupMeta"]] = {}

    name: str
    description: str
    tools: List[Callable] = Field(default_factory=list)
    tool_costs: Dict[str, int] = Field(
        default_factory=dict,
        description="工具点数消耗映射 {tool_name: cost}，由 tool() 装饰器自动填充",
    )
    default_tool_cost: int = Field(default=0, description="该组工具的默认点数消耗")
    prem: Optional[Callable[[Context], Awaitable[bool]]] = None
    show: Optional[Callable[[Context], Awaitable[bool]]] = None

    def model_post_init(self, __context: Any) -> None:
        if self.name in ToolGroupMeta.mapping:
            raise ValueError(f"ToolGroupMeta with name '{self.name}' already exists.")
        ToolGroupMeta.mapping[self.name] = self

    def resolve_tool_cost(self, tool_name: str) -> int:
        """解析工具点数消耗，优先级：tool_costs > default_tool_cost > 全局默认"""
        if tool_name in self.tool_costs:
            return self.tool_costs[tool_name]
        if self.default_tool_cost >= 0:
            return self.default_tool_cost
        return 0

    async def get_info(self, context: Context) -> str | None:
        if not await self.should_show(context):
            return None
        lines = []
        lines.append(f"【{self.name}】：{self.description}")
        if not self.tools:
            lines.append("  (该组下暂无可用工具)")
            return "\n".join(lines)

        budget_enabled = context.deps.tool_budget_enabled

        for tool_func in self.tools:
            func_name = tool_func.__name__
            brief = "暂无描述"
            if tool_func.__doc__:
                first_line = tool_func.__doc__.strip().splitlines()[0]
                if first_line:
                    brief = first_line
            if budget_enabled:
                cost = self.resolve_tool_cost(func_name)
                lines.append(f"  - {func_name} ({cost}pt): {brief}")
            else:
                lines.append(f"  - {func_name}: {brief}")

        return "\n".join(lines)

    @overload
    def tool(self, func: Callable) -> Callable: ...

    @overload
    def tool(self, *, cost: int) -> Callable[[Callable], Callable]: ...

    def tool(self, func: Callable | None = None, *, cost: int | None = None):
        """
        将函数注册为该组tools中的一个工具

        支持两种用法:
            @group.tool
            def my_tool(...): ...

            @group.tool(cost=3)
            def my_expensive_tool(...): ...

        Args:
            func: 工具函数
            cost: 该工具的点数消耗，默认为 None 表示使用组默认值
        """

        def decorator(fn: Callable) -> Callable:
            resolved_cost = cost if cost is not None else self.default_tool_cost
            wrapped_func = tool_call_hook(fn, cost=resolved_cost)
            if cost is not None:
                self.tool_costs[fn.__name__] = cost
            if wrapped_func.__name__ not in [t.__name__ for t in self.tools]:
                self.tools.append(wrapped_func)
            return wrapped_func

        if func is not None:
            return decorator(func)
        return decorator

    async def should_show(self, context: Context) -> bool:
        """
        判断在当前上下文中是否应该向LLM展示该工具组
        """
        if self.show is None:
            return True
        return await self.show(context)

    async def check_prem(self, context: Context) -> bool:
        if self.prem is None:
            return True
        return await self.prem(context)

    def __hash__(self) -> int:
        return hash(self.name + self.description)

    def __repr__(self) -> str:
        return f"ToolGroupMeta(name={self.name}, description={self.description}, tools={[tool.__name__ for tool in self.tools]})"
