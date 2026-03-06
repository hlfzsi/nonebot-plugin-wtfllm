__all__ = [
    "AgentDeps",
    "IDs",
    "run_chat_agent",
]

from typing import Any

from pydantic_ai.agent import AgentRunResult

from ..llm.agents import CHAT_AGENT
from ..llm.deps import AgentDeps, IDs


async def run_chat_agent(
    prompt: str,
    deps: AgentDeps,
    **kwargs: Any,
) -> AgentRunResult:
    """调用 CHAT_AGENT 并返回结果。

    Args:
        prompt: 传递给 Agent 的用户提示（通常是 ``builder.to_prompt()``）。
        deps: 预构建的 AgentDeps 实例。
        **kwargs: 透传给 ``Agent.run()`` 的额外参数（如 ``message_history``）。

    Returns:
        AgentRunResult — 与直接调用 ``CHAT_AGENT.run()`` 返回类型一致。
    """
    return await CHAT_AGENT.run(user_prompt=prompt, deps=deps, **kwargs)
