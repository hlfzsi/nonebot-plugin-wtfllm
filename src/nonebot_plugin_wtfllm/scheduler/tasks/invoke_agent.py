import asyncio
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field
from nonebot_plugin_alconna import Target, UniMessage
from nonebot_plugin_uninfo import Session

from ..registry import scheduled_task


class InvokeAgentParams(BaseModel):
    """invoke_agent 任务的参数模型"""

    agent_id: str = Field(..., description="要调用的 Agent ID")
    user_id: str = Field(..., description="创建任务的用户ID")
    group_id: Optional[str] = Field(default=None, description="关联群组ID")
    prompt: Optional[str] = Field(
        default=None, description="调用原因/提示，供 Agent 内部使用"
    )
    target_data: Dict[str, Any] = Field(..., description="Target.dump() 输出")
    session_data: Dict[str, Any] = Field(..., description="Uninfo.dump() 输出")


@scheduled_task("invoke_agent", InvokeAgentParams)
async def handle_invoke_agent(params: InvokeAgentParams) -> None:
    from ...db import user_persona_repo
    from ...llm import CHAT_AGENT
    from ...llm.deps import AgentDeps, IDs, NonebotRuntime
    from ...memory import MemoryContextBuilder
    from ...msg_tracker import msg_tracker
    from ...services.func import RetrievalChain, get_alias_from_cache
    from ...services.func.easy_ban import is_banned
    from ...utils import APP_CONFIG, logger, get_bot_by_agent_id, extract_session_info

    if await is_banned(params.user_id, params.group_id):
        raise RuntimeError(
            f"User {params.user_id} is banned in group {params.group_id}"
        )
    bot = get_bot_by_agent_id(params.agent_id)
    target = Target.load(params.target_data)
    session = Session.load(params.session_data)
    session_info = extract_session_info(session)
    cached_aliases = get_alias_from_cache(bot, session)
    cached_aliases[params.agent_id] = APP_CONFIG.bot_name
    persona = await user_persona_repo.get_persona_text(
        user_id=cached_aliases[params.user_id], real_user_id=params.user_id
    )

    suffix = (
        f'<user_persona user="{cached_aliases[params.user_id]}">\n{persona}\n</user_persona>\n'
        if persona
        else ""
    )
    if params.prompt:
        suffix += f"<called_reason>{params.prompt}</called_reason>"

    builder = MemoryContextBuilder(
        prefix_prompt=(
            f"Current Scene: {cached_aliases.get(session_info['group_id'], 'Private Chat') if session_info['group_id'] else 'Private Chat'}"
        ),
        agent_id=params.agent_id,
        suffix_prompt=suffix,
    )

    recent_react = msg_tracker.get(
        user_id=session_info["user_id"],
        agent_id=params.agent_id,
    )
    recent_react.pop(session_info["group_id"] if session_info["group_id"] else "", None)
    for gid in recent_react.keys():
        if gid:
            builder.ctx.alias_provider.register_group(gid)

    is_group = params.group_id is not None
    if is_group:
        chain = RetrievalChain(
            agent_id=params.agent_id,
            group_id=params.group_id,
        )
    else:
        chain = RetrievalChain(
            agent_id=params.agent_id,
            user_id=params.user_id,
        )
    chain.main_chat(limit=APP_CONFIG.short_memory_max_count).recent_react(
        recent_react=recent_react, alias_provider=builder.ctx.alias_provider
    ).core_memory()
    sources = await chain.resolve()
    builder.extend(sources)

    # 构建依赖注入字典
    deps = AgentDeps(
        ids=IDs(
            user_id=session_info["user_id"],
            group_id=session_info["group_id"],
            agent_id=params.agent_id,
        ),
        context=builder,
        active_tool_groups={"Core", "CoreMemory"},
        nb_runtime=NonebotRuntime(bot=bot, session=session, target=target),
        tool_point_budget=APP_CONFIG.tool_point_budget,
    )
    final_prompt = builder.to_prompt()

    _task_start_at = asyncio.get_running_loop().time()
    try:
        async with asyncio.timeout(APP_CONFIG.agent_base_timeout_seconds) as cm:
            deps.cm = cm
            response = await CHAT_AGENT.run(user_prompt=final_prompt, deps=deps)
    except asyncio.TimeoutError:
        logger.warning("CHATAGENT timeout, cancelling task")
        return
    finally:
        _task_end_at = asyncio.get_running_loop().time()
        consumed = _task_end_at - _task_start_at
        logger.debug(
            f"CHATAGENT finished after running for {consumed:.2f} seconds because of scheduled task"
        )
    if is_group:
        extra_msg = UniMessage().at(params.user_id)
    else:
        extra_msg = None
    await response.output.send(context=deps, extra_segments=extra_msg)

    logger.debug(response.usage())
