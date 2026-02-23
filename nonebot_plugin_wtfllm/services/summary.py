from datetime import datetime

from arclet.alconna import Alconna, Args, Option
from nonebot_plugin_alconna import on_alconna, Match, Query
from nonebot_plugin_uninfo import Uninfo

from ..memory._types import ID_PATTERN
from ..utils import APP_CONFIG
from ..v_db import core_memory_repo


def _format_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _clean_entity_placeholders(text: str) -> str:
    return ID_PATTERN.sub(lambda m: m.group(1), text)


async def _perm(session: Uninfo) -> bool:
    if session.user.id in APP_CONFIG.admin_users:
        return True
    else:
        return False


summary_alc = Alconna(
    "summary",
    Args["count?", int, 10]["query?", str],
    Option("-g|--group", Args["group_id", str], help_text="指定群组ID"),
)

summary_cmd = on_alconna(
    summary_alc, aliases={"摘要", "记忆"}, use_cmd_start=True, block=True, rule=_perm
)


@summary_cmd.handle()
async def handle_summary(
    count: Match[int],
    query: Match[str],
    session: Uninfo,
    opt_group_id: Query[str] = Query("group.group_id"),
):
    _count = count.result if count.available else 10
    _query = query.result if query.available else None
    _group = None

    if opt_group_id.available:
        _group = opt_group_id.result
    elif session.group:
        _group = str(session.group.id)

    agent_id = session.self_id.removeprefix("llonebot:").removeprefix("napcat:")

    response_lines = []

    if _query and _group:
        search_results = await core_memory_repo.search_cross_session(
            agent_id=agent_id,
            query=_query,
            limit=_count,
        )
        if not search_results:
            await summary_cmd.finish("未找到相关核心记忆")
        for i, result in enumerate(search_results, start=1):
            m = result.item
            content = _clean_entity_placeholders(m.content)
            response_lines.append(
                f"=== 核心记忆 #{i} [相关度: {result.score:.2f}] ===\n"
                f"创建于: {_format_ts(m.created_at)}\n"
                f"更新于: {_format_ts(m.updated_at)}\n"
                f"来源: {m.source}\n"
                f"内容:\n{content}"
            )
    elif _group:
        memories = await core_memory_repo.get_by_session(
            agent_id=agent_id,
            group_id=_group,
        )
        if not memories:
            await summary_cmd.finish("当前会话暂无核心记忆")
        for i, m in enumerate(memories[-_count:], start=1):
            content = _clean_entity_placeholders(m.content)
            response_lines.append(
                f"=== 核心记忆 #{i} ===\n"
                f"创建于: {_format_ts(m.created_at)}\n"
                f"更新于: {_format_ts(m.updated_at)}\n"
                f"来源: {m.source} | tokens: {m.token_count}\n"
                f"内容:\n{content}"
            )
        total_tokens = sum(m.token_count for m in memories)
        response_lines.append(
            f"\n--- 共 {len(memories)} 条核心记忆，"
            f"总 tokens: {total_tokens}/{APP_CONFIG.core_memory_max_tokens} ---"
        )
    else:
        await summary_cmd.finish("请提供群组ID或在群组中使用该命令")

    await summary_cmd.finish("\n\n".join(response_lines))
