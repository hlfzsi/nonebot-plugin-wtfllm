import time
import asyncio
from typing import Dict, List

from nonebot import on_message
from nonebot.adapters import Bot
from nonebot.params import EventToMe
from nonebot_plugin_alconna import OriginalUniMsg, MsgTarget, MsgId
from nonebot_plugin_uninfo import Uninfo

from .func import handle_control, get_alias_from_cache, like_command, get_conv_key
from .func.easy_ban import is_banned
from .func.message_queue import get_queue
from ..llm.agents import CHAT_AGENT
from ..llm.deps import AgentDeps, IDs, NonebotRuntime
from ..memory import (
    MemoryContextBuilder,
    MemoryItemStream,
    MemoryItemUnion,
)
from ..memory.items import MemorySource
from ..memory.items.core_memory import CoreMemoryBlock
from ..memory.items.knowledge_base import KnowledgeBlock
from ..memory.items.tool_call_summary import ToolCallSummaryBlock
from ..utils import APP_CONFIG, logger, get_agent_id_from_bot
from ..db import memory_item_repo, user_persona_repo, tool_call_record_repo
from ..v_db import core_memory_repo, knowledge_base_repo
from ..abilities import attention_router
from ..msg_tracker import msg_tracker

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
    if await is_banned(session.user.id, session.group.id if session.group else None):
        agent_id = get_agent_id_from_bot(session)
        attention_router.remove_poi(
            user_id=session.user.id,
            group_id=session.group.id if session.group else None,
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
        now_time = int(time.time())
        cached_aliases[agent_id] = APP_CONFIG.bot_name
        poi = attention_router.get_and_consume_poi(
            user_id=session.user.id,
            group_id=session.group.id if session.group else None,
            agent_id=agent_id,
        )
        if not is_to_me and not poi:
            return

        persona = await user_persona_repo.get_persona_text(
            user_id=cached_aliases[session.user.id], real_user_id=session.user.id
        )
        suffix = (
            f'<user_persona user="{cached_aliases[session.user.id]}">\n{persona}\n</user_persona>\n'
            if persona
            else ""
        )
        if poi:
            suffix += (
                f'<poi reason="{poi.reason}">最后一条消息来自你主动追踪的用户。</poi>'
            )

        builder = MemoryContextBuilder(
            prefix_prompt=(
                f"Current Scene: {cached_aliases.get(session.group.id, 'Private Chat') if session.group else 'Private Chat'}"
            ),
            suffix_prompt=suffix,
            agent_id=agent_id,
            custom_ref=cached_aliases,
        )

        msg_tracker.track(
            agent_id=agent_id,
            user_id=session.user.id,
            group_id=session.group.id if session.group else None,
            msg_id=message_id,
        )

        final_sources: set[MemorySource] = set()

        recent_react = msg_tracker.get(
            user_id=session.user.id,
            agent_id=agent_id,
        )
        recent_react.pop(session.group.id if session.group else "", None)
        for gid in recent_react.keys():
            if gid:
                builder.ctx.alias_provider.register_group(gid)
        recent_react_tasks: Dict[str, asyncio.Task[List[MemoryItemUnion]]] = {}

        if session.group:
            # 群聊处理
            async with asyncio.TaskGroup() as tg:
                current_group_items_task = tg.create_task(
                    memory_item_repo.get_by_group_after(
                        group_id=session.group.id,
                        agent_id=agent_id,
                        timestamp=now_time
                        - (APP_CONFIG.short_memory_time_minutes * 60),
                        limit=APP_CONFIG.short_memory_max_count,
                    )
                )
                current_core_memories_task = tg.create_task(
                    core_memory_repo.get_by_session(
                        agent_id=agent_id,
                        group_id=session.group.id,
                    )
                )
                cross_session_memories_task = tg.create_task(
                    core_memory_repo.search_cross_session(
                        agent_id=agent_id,
                        query=uni_msg.extract_plain_text(),
                        exclude_group_id=session.group.id,
                        limit=5,
                    )
                )
                knowledge_search_task = tg.create_task(
                    knowledge_base_repo.search_relevant(
                        agent_id=agent_id,
                        query=uni_msg.extract_plain_text(),
                        limit=APP_CONFIG.knowledge_base_max_results,
                    )
                )
                for gid, mid_list in recent_react.items():
                    if mid_list:
                        recent_react_tasks[gid] = tg.create_task(
                            memory_item_repo.get_many_by_message_ids(mid_list)
                        )

                tool_call_history_task = tg.create_task(
                    tool_call_record_repo.get_recent(
                        agent_id=agent_id,
                        group_id=session.group.id,
                        limit=APP_CONFIG.tool_call_record_max_count,
                    )
                )

            current_group_recent_items = current_group_items_task.result()
            stream = MemoryItemStream.create(
                items=current_group_recent_items, role="main_chat"
            )

            current_core_memories = current_core_memories_task.result()
            if current_core_memories:
                core_memory_block = CoreMemoryBlock(
                    memories=current_core_memories,
                    prefix="<core_memory>",
                    suffix="</core_memory>",
                )
                final_sources.add(core_memory_block)

            cross_session_results = cross_session_memories_task.result()
            if cross_session_results:
                cross_memories = [r.item for r in cross_session_results]
                cross_memory_block = CoreMemoryBlock(
                    memories=cross_memories,
                    prefix="<cross_session_memory>",
                    suffix="</cross_session_memory>",
                )
                final_sources.add(cross_memory_block)

            knowledge_results = knowledge_search_task.result()
            if knowledge_results:
                knowledge_entries = []
                total_tokens = 0
                for r in knowledge_results:
                    if (
                        total_tokens + r.item.token_count
                        <= APP_CONFIG.knowledge_base_max_tokens
                    ):
                        knowledge_entries.append(r.item)
                        total_tokens += r.item.token_count
                    else:
                        break
                if knowledge_entries:
                    knowledge_block = KnowledgeBlock(
                        entries=knowledge_entries,
                        prefix="<knowledge_base>",
                        suffix="</knowledge_base>",
                    )
                    final_sources.add(knowledge_block)

            for gid, task in recent_react_tasks.items():
                _items = task.result()
                _stream = MemoryItemStream.create(
                    items=_items,
                    prefix=f'<memory scene="{builder.ctx.alias_provider.get_alias(gid) if gid else "私聊"}">',
                    suffix="</memory>",
                    max_token=5000,
                )
                final_sources.add(_stream)

            final_sources.add(stream)

            tool_call_records = tool_call_history_task.result()
            if tool_call_records:
                tool_summary_block = ToolCallSummaryBlock(
                    tool_names=[r.tool_name for r in tool_call_records],
                    prefix="<recent_tools>",
                    suffix="</recent_tools>",
                )
                final_sources.add(tool_summary_block)

        else:
            # 私聊处理
            async with asyncio.TaskGroup() as tg:
                items_task = tg.create_task(
                    memory_item_repo.get_in_private_by_user(
                        user_id=session.user.id,
                        agent_id=agent_id,
                        limit=APP_CONFIG.short_memory_max_count,
                    )
                )
                current_core_memories_task = tg.create_task(
                    core_memory_repo.get_by_session(
                        agent_id=agent_id,
                        user_id=session.user.id,
                    )
                )
                cross_session_memories_task = tg.create_task(
                    core_memory_repo.search_cross_session(
                        agent_id=agent_id,
                        query=uni_msg.extract_plain_text(),
                        exclude_user_id=session.user.id,
                        limit=5,
                    )
                )
                knowledge_search_task = tg.create_task(
                    knowledge_base_repo.search_relevant(
                        agent_id=agent_id,
                        query=uni_msg.extract_plain_text(),
                        limit=APP_CONFIG.knowledge_base_max_results,
                    )
                )
                for gid, mid_list in recent_react.items():
                    if mid_list:
                        recent_react_tasks[gid] = tg.create_task(
                            memory_item_repo.get_many_by_message_ids(mid_list)
                        )
                tool_call_history_task = tg.create_task(
                    tool_call_record_repo.get_recent(
                        agent_id=agent_id,
                        user_id=session.user.id,
                        limit=APP_CONFIG.tool_call_record_max_count,
                    )
                )

            items = items_task.result()
            stream = MemoryItemStream.create(items=items, role="main_chat")

            current_core_memories = current_core_memories_task.result()
            if current_core_memories:
                core_memory_block = CoreMemoryBlock(
                    memories=current_core_memories,
                    prefix="<core_memory>",
                    suffix="</core_memory>",
                )
                final_sources.add(core_memory_block)

            cross_session_results = cross_session_memories_task.result()
            if cross_session_results:
                cross_memories = [r.item for r in cross_session_results]
                cross_memory_block = CoreMemoryBlock(
                    memories=cross_memories,
                    prefix="<cross_session_memory>",
                    suffix="</cross_session_memory>",
                )
                final_sources.add(cross_memory_block)

            knowledge_results = knowledge_search_task.result()
            if knowledge_results:
                knowledge_entries = []
                total_tokens = 0
                for r in knowledge_results:
                    if (
                        total_tokens + r.item.token_count
                        <= APP_CONFIG.knowledge_base_max_tokens
                    ):
                        knowledge_entries.append(r.item)
                        total_tokens += r.item.token_count
                    else:
                        break
                if knowledge_entries:
                    knowledge_block = KnowledgeBlock(
                        entries=knowledge_entries,
                        prefix="<knowledge_base>",
                        suffix="</knowledge_base>",
                    )
                    final_sources.add(knowledge_block)

            for gid, task in recent_react_tasks.items():
                _items = task.result()
                _stream = MemoryItemStream.create(
                    items=_items,
                    prefix=f'<memory scene="{builder.ctx.alias_provider.get_alias(gid) if gid else "私聊"}">',
                    suffix="</memory>",
                    max_token=5000,
                )
                final_sources.add(_stream)

            final_sources.add(stream)

            tool_call_records = tool_call_history_task.result()
            if tool_call_records:
                tool_summary_block = ToolCallSummaryBlock(
                    tool_names=[r.tool_name for r in tool_call_records],
                    prefix="<recent_tools>",
                    suffix="</recent_tools>",
                )
                final_sources.add(tool_summary_block)

        builder.extend(final_sources)

        # 构建依赖注入字典
        conv_key = get_conv_key(bot, session)
        queue = get_queue(conv_key)
        deps = AgentDeps(
            ids=IDs(
                user_id=session.user.id,
                group_id=session.group.id if session.group else None,
                agent_id=agent_id,
            ),
            context=builder,
            active_tool_groups={"Core", "CoreMemory", "KnowledgeBase"},
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
