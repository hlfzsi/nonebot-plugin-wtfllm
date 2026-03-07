import asyncio

from nonebot import on_message
from nonebot.adapters import Bot
from nonebot.params import EventToMe
from nonebot_plugin_alconna import OriginalUniMsg, MsgTarget, MsgId
from nonebot_plugin_uninfo import Uninfo

from .func import (
    handle_control,
    get_alias_from_cache,
    like_command,
    get_conv_key,
    RetrievalChain,
)
from .func.easy_ban import is_banned
from .func.message_queue import get_queue
from ..llm.agents import CHAT_AGENT
from ..llm.deps import AgentDeps, IDs, NonebotRuntime
from ..memory import MemoryContextBuilder
from ..utils import APP_CONFIG, logger, get_agent_id_from_bot, extract_session_info
from ..db import user_persona_repo
from ..abilities import attention_router
from ..msg_tracker import msg_tracker
from ..proactive import should_proactively_respond, topic_interest_store

matcher = on_message(block=True, priority=99)


@matcher.handle()
async def handle(
    bot: Bot,
    uni_msg: OriginalUniMsg,
    target: MsgTarget,
    session: Uninfo,
    message_id: MsgId,
    is_to_me: bool = EventToMe(),
):
    session_info = extract_session_info(session)
    usable_user_id = session_info["user_id"]
    usable_group_id = session_info["group_id"]

    if await is_banned(usable_user_id, usable_group_id):
        agent_id = get_agent_id_from_bot(session)
        attention_router.remove_poi(
            user_id=usable_user_id,
            group_id=usable_group_id,
            agent_id=agent_id,
        )
        topic_interest_store.clear_topics(
            user_id=usable_user_id,
            group_id=usable_group_id,
            agent_id=agent_id,
        )
        return
    if like_command(uni_msg):
        return

    with handle_control(bot, session) as can_handle:
        if not can_handle:
            return

        cached_aliases = get_alias_from_cache(bot, session)

        agent_id = get_agent_id_from_bot(session)
        cached_aliases[agent_id] = APP_CONFIG.bot_name
        query = uni_msg.extract_plain_text()
        poi = attention_router.get_and_consume_poi(
            user_id=usable_user_id,
            group_id=usable_group_id,
            agent_id=agent_id,
        )
        should_reply_proactively = await should_proactively_respond(
            agent_id=agent_id,
            user_id=usable_user_id,
            group_id=usable_group_id,
            plain_text=query,
        )
        if not is_to_me and not poi and not should_reply_proactively:
            return

        persona = await user_persona_repo.get_persona_text(
            user_id=cached_aliases[usable_user_id], real_user_id=usable_user_id
        )
        suffix = (
            f'<user_persona user="{cached_aliases[usable_user_id]}">\n{persona}\n</user_persona>\n'
            if persona
            else ""
        )
        if poi:
            suffix += f'<poi reason="{poi.reason}">最后一条消息来自你主动追踪的用户。剩余追踪轮数{poi.turns}</poi>'

        builder = MemoryContextBuilder(
            prefix_prompt=(
                f"Current Scene: {cached_aliases.get(usable_group_id, 'Private Chat') if usable_group_id else 'Private Chat'}"
            ),
            suffix_prompt=suffix,
            agent_id=agent_id,
            custom_ref=cached_aliases,
        )

        msg_tracker.track(
            agent_id=agent_id,
            user_id=usable_user_id,
            group_id=usable_group_id,
            msg_id=message_id,
        )

        recent_react = msg_tracker.get(
            user_id=usable_user_id,
            agent_id=agent_id,
        )
        recent_react.pop(usable_group_id if usable_group_id else "", None)
        for gid in recent_react.keys():
            if gid:
                builder.ctx.alias_provider.register_group(gid)

        if usable_group_id:
            chain = RetrievalChain(
                agent_id=agent_id,
                group_id=usable_group_id,
                query=query,
            )
        else:
            chain = RetrievalChain(
                agent_id=agent_id,
                user_id=usable_user_id,
                query=query,
            )

        chain.main_chat(
            limit=APP_CONFIG.short_memory_max_count,
        ).core_memory().cross_session_memory().knowledge(
            limit=APP_CONFIG.knowledge_base_max_results,
            max_tokens=APP_CONFIG.knowledge_base_max_tokens,
        ).recent_react(
            recent_react=recent_react,
            alias_provider=builder.ctx.alias_provider,
        ).topic_context(
            max_topic_messages=APP_CONFIG.topic_max_context_messages,
        )

        final_sources = await chain.resolve()
        builder.extend(final_sources)

        # 构建依赖注入字典
        conv_key = get_conv_key(bot, session)
        queue = get_queue(conv_key)
        deps = AgentDeps(
            ids=IDs(
                user_id=usable_user_id,
                group_id=usable_group_id,
                agent_id=agent_id,
            ),
            context=builder,
            active_tool_groups={"Core", "CoreMemory"},
            nb_runtime=NonebotRuntime(bot=bot, session=session, target=target),
            message_queue=queue,
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
            logger.debug(f"CHATAGENT finished after running for {consumed:.2f} seconds")
        await response.output.send(context=deps)

        logger.debug(response.usage())
