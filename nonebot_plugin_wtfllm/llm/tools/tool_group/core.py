import time
import asyncio
import datetime
from typing import Any, Dict, List, Set

import orjson
from pydantic_ai import ImageUrl, BinaryImage, ToolReturn

from .base import ToolGroupMeta
from .utils import reschedule_deadline
from ...deps import Context
from ....utils import APP_CONFIG, logger
from ....memory import ImageSegment
from ....memory.items.storages import MemoryItemStream
from ....db import memory_item_repo, tool_call_record_repo
from ....services.func.memory_retrieval import RetrievalChain
from ....abilities import (
    attention_router,
    PoiInfo,
    identification,
    get_image_desc as _get_image_desc,
)

core_group = ToolGroupMeta(
    name="Core",
    description="基础工具组，包含核心功能的工具",
)


@core_group.tool(cost=1)
def reinforce_persona_anchor(ctx: Context) -> str:
    """
    获取当前人设配置
    仅在需要确认自身角色设定或人设同步时调用。
    """
    logger.debug("Fetching current persona anchor")
    return APP_CONFIG.llm_role_setting


@core_group.tool(cost=0)
async def activate_tool_group(ctx: Context, group_name: str | List[str]) -> str:
    """
    激活指定名称的工具组

    Args:
        group_name (str|List[str]): 工具组名称
    """
    if isinstance(group_name, str):
        group_name = [group_name]

    returns = []

    for _name in group_name:
        group = ToolGroupMeta.mapping.get(_name)
        if not group:
            returns.append(f"工具组 '{_name}' 不存在。")
            continue
        if await group.check_prem(ctx):
            ctx.deps.active_tool_groups.add(_name)
            returns.append(f"工具组 '{_name}' 已激活。")
            logger.debug(
                f"Activated tool group '{_name}' for agent {ctx.deps.ids.agent_id}"
            )
            continue
        else:
            returns.append(f"工具组 '{_name}' 激活失败，权限不足。")
            continue
    return "\n".join(returns)


@core_group.tool(cost=0)
def mark_point_of_interest(
    ctx: Context,
    user_id: str,
    reason: str,
    turns: int = 1,
    timeout_seconds: int = 60 * 30,
) -> str:
    """
    主动追踪用户：使后续对话无需被提及即可直接接收并响应其消息。

    适用场景：
    1. 连续任务：需多轮交互，不希望用户重复 @ 你。
    2. 深度参与：判断后续对话高度相关，需保持“在线”跟进（如：吵架、逻辑辩论）。
    3. 其他适合的场景

    效果：
    该用户下次发言将直接触发你的响应路由。此状态为临时，用于完成短期目标。
    可以重复调用以覆盖当前POI状态

    Args:
        user_id (str): 目标用户 ID。
        reason (str): 追踪原因。
        turns (int): 追踪轮数，默认1轮。超过该轮数后将自动取消追踪。
        timeout_seconds: int = 60 * 30, 追踪持续时间，单位秒，默认30分钟。超过该时间后将自动取消追踪。
    """
    real_user_id = ctx.deps.context.ctx.alias_provider.resolve_alias(user_id) or user_id
    poi = PoiInfo(
        user_id=real_user_id,
        group_id=ctx.deps.ids.group_id,
        agent_id=ctx.deps.ids.agent_id,
        reason=reason,
        turns=turns,
        expires_at=time.time() + timeout_seconds,
    )
    attention_router.mark_poi(poi)
    logger.debug(
        f"Marked point of interest for user {real_user_id} with reason: {reason}"
    )
    return f"已为用户 '{user_id}' 标记关注点，原因：{reason}"


@core_group.tool(cost=0)
async def update_self_identify(ctx: Context, new_identify: Dict[str, Any]) -> str:
    """
    更新你的核心身份与人设特质。当你想根据交互经验进行自我进化、调整性格、
    记录长期记忆、保持行为一致或修正行为准则时，请积极调用此工具。
    警告: 这是全局性更新, 若针对个人和单独某一用户, 请使用其他工具
    支持深度增量合并，将字段设为 null 可将其从你的人设中删除。

    Args:
        new_identify: 增量更新的字典数据。
    """
    await identification.update(new_identify)
    return f"自我认知已进化。最新的自我认知: {await identification.get_all_json()}"


@core_group.tool(cost=1)
async def get_image_description(ctx: Context, media_refs: List[str]) -> str | None:
    """
    获取图片描述信息, 可能不精准
    

    Args:
        media_refs (List[str]): 多媒体文件的引用序号, 如 ["IMG:1", "IMG:2"]

    Returns:
        str | None: 图片描述文本, 如果无法获取则返回None
    """
    logger.debug(f"Describing images for references: {media_refs}")
    reschedule_deadline(ctx, 25)

    ref_to_seg: Dict[str, ImageSegment] = {}
    valid_refs: List[str] = []
    message_ids_to_fetch: Set[str] = set()
    result_dict: Dict[str, str] = {}

    refs_needing_uri: List[str] = []
    segs_needing_uri: List[ImageSegment] = []

    for ref in media_refs:
        if "IMG:" not in ref:
            result_dict[ref] = "错误：无效的引用格式。"
            continue

        seg = ctx.deps.context.resolve_media_ref(ref, ImageSegment)
        if seg is None:
            result_dict[ref] = "错误：未找到对应的图片资源。"
            continue

        if seg.desc:
            result_dict[ref] = seg.desc
            continue

        if not seg.available:
            result_dict[ref] = "图片已过期"
            continue

        if seg.local_path:
            refs_needing_uri.append(ref)
            segs_needing_uri.append(seg)
            ref_to_seg[ref] = seg
            message_ids_to_fetch.add(seg.message_id)
        elif seg.url:
            ref_to_seg[ref] = seg
            valid_refs.append(ref)
            message_ids_to_fetch.add(seg.message_id)
        else:
            result_dict[ref] = "错误：图片数据无效。"

    image_sources: List[str] = []

    if refs_needing_uri:
        uri_tasks = [seg.get_data_uri_async() for seg in segs_needing_uri]
        uris = await asyncio.gather(*uri_tasks)
        for ref, uri in zip(refs_needing_uri, uris):
            valid_refs.append(ref)
            image_sources.append(uri)

    for ref in list(ref_to_seg.keys()):
        if ref not in valid_refs and ref in ref_to_seg:
            seg = ref_to_seg[ref]
            if seg.url:
                valid_refs.append(ref)
                image_sources.append(seg.url)

    if not image_sources:
        return orjson.dumps(result_dict).decode("utf-8")

    if sum(len(s) for s in image_sources) >= 2000000:
        return "错误：图片数据过大，无法处理。"

    try:
        async with asyncio.TaskGroup() as tg:
            descs_task = tg.create_task(_get_image_desc(image_sources))
            items_task = tg.create_task(
                memory_item_repo.get_many_by_message_ids(list(message_ids_to_fetch))
            )
    except (RuntimeError, ValueError, OSError) as e:
        logger.error(f"Error while describing images: {e}")
        for ref in valid_refs:
            result_dict[ref] = f"错误：获取图片描述时发生异常。{e}"
        return orjson.dumps(result_dict).decode("utf-8")

    descs = descs_task.result()
    items = items_task.result()

    if descs is None:
        for ref in valid_refs:
            result_dict[ref] = "错误：视觉模型调用失败。"
        return orjson.dumps(result_dict).decode("utf-8")

    memory_cache = {item.message_id: item for item in items if item.message_id}
    dirty_items = set()

    for ref, full_desc in zip(valid_refs, descs):
        desc = full_desc.to_string()
        result_dict[ref] = desc
        seg = ref_to_seg[ref]

        raw_memory_item = memory_cache.get(seg.message_id)
        if not raw_memory_item:
            logger.warning(
                f"Message {seg.message_id} not found in DB, skipping persistence."
            )
            continue

        updated = raw_memory_item.content.deep_find_and_update(seg, {"desc": desc})

        if updated:
            dirty_items.add(raw_memory_item)

    if dirty_items:
        await asyncio.gather(
            *(memory_item_repo.save_memory_item(item) for item in dirty_items)
        )

    return orjson.dumps(result_dict).decode("utf-8")


@core_group.tool(cost=2)
async def get_image_content(ctx: Context, media_refs: List[str]) -> ToolReturn:
    """
    通过多媒体文件的引用序号获取多媒体资源 仅支持图片, 返回顺序与输入顺序一致
    需要直接获取图片内容进行分析时调用。

        Args:
            media_refs (List[str]): 多媒体文件的引用序号, 如 ["IMG:1", "IMG:2"]
    """
    if not APP_CONFIG.llm_support_vision:
        return ToolReturn(
            return_value="错误：当前模型不支持视觉能力，无法获取图片内容。",
            content=[],
        )
    logger.debug(f"Fetching media references: {media_refs}")

    reschedule_deadline(ctx, 25)

    resolved: List[tuple[str, ImageSegment]] = []  # (ref, seg) — local_path
    url_resolved: List[tuple[str, ImageSegment]] = []  # (ref, seg) — url only
    summary: List[str] = []
    content_parts: List[Any] = []

    for ref in media_refs:
        if "IMG:" not in ref:
            summary.append(f"引用 '{ref}' 格式不正确。")
            continue

        seg = ctx.deps.context.resolve_media_ref(ref, ImageSegment)
        if seg is None:
            summary.append(f"引用 '{ref}' 未找到对应的图片资源。")
            continue

        if not seg.available:
            summary.append(f"引用 '{ref}' 的图片资源已过期。")
            continue

        if seg.local_path:
            resolved.append((ref, seg))
        elif seg.url:
            url_resolved.append((ref, seg))
        else:
            summary.append(f"引用 '{ref}' 未找到对应的图片资源或格式不支持。")

    if resolved:

        async def _load_local(ref: str, seg: ImageSegment):
            image_data = await seg.get_bytes_async()
            img_format = await seg.get_mime_type_async()
            return ref, BinaryImage(data=image_data, media_type=img_format)

        local_tasks = [_load_local(ref, seg) for ref, seg in resolved]
        local_results = await asyncio.gather(*local_tasks, return_exceptions=True)

        for result in local_results:
            if isinstance(result, BaseException):
                summary.append(f"加载图片时发生异常: {result}")
            else:
                ref, binary_img = result
                content_parts.append(binary_img)
                summary.append(f"已加载图片: {ref}")

    for ref, seg in url_resolved:
        content_parts.append(ImageUrl(url=seg.url))  # pyright: ignore[reportArgumentType]
        summary.append(f"已成功加载图片: {ref}")

    return ToolReturn(return_value="\n".join(summary), content=content_parts)


@core_group.tool(cost=2)
async def get_full_message_detail(ctx: Context, message_ref: int) -> str:
    """
    获取消息的完整细节信息，包括文本内容。
    当记忆中的消息被省略或压缩展示时，调用此工具查看完整文本。
    不鼓励调用，仅在确实需要查看完整消息内容时使用。

    Args:
        message_ref: 消息引用ID, 如 1
    """
    item = ctx.deps.context.resolve_memory_ref(message_ref)
    if not item:
        return f"未找到消息ID {message_ref} 的相关信息。"

    items = await memory_item_repo.get_chain_by_message_ids([item.message_id])
    stream = MemoryItemStream.create(
        items=items, prefix="<message_detail>", suffix="</message_detail>"
    )

    llm_ctx = ctx.deps.context.ctx.copy(share_providers=True)
    llm_ctx.set_condense(False)
    new_builder = ctx.deps.context.copy(share_context=llm_ctx, empty=True)
    new_builder.add(stream)
    return new_builder.to_prompt()


@core_group.tool(cost=1)
async def query_memory(
    ctx: Context, query: str, user_id: str | None = None, limit: int = 5
):
    """通过语义搜索查询相关的记忆

    Args:
        query: 查询内容
        user_id: 可选的用户ID，用于过滤记忆归属
        limit: 返回的记忆数量限制
    """
    logger.debug(
        f"Querying memory with query: {query}, user_id: {user_id}, limit: {limit}"
    )
    real_user_id = ctx.deps.context.resolve_aliases(user_id) if user_id else None

    chain = RetrievalChain(
        agent_id=ctx.deps.ids.agent_id,
        group_id=ctx.deps.ids.group_id,
        user_id=ctx.deps.ids.user_id,
        query=query,
    )

    # 核心记忆分支
    if real_user_id and user_id:
        chain.entity_memory(
            entity_ids=[real_user_id],
            limit=limit,
            prefix="<related_memory>",
            suffix="</related_memory>",
        )
    elif real_user_id is None and user_id:
        pass  # user_id 无法解析，跳过核心记忆
    else:
        chain.cross_session_memory(
            limit=limit,
            prefix="<related_memory>",
            suffix="</related_memory>",
        )

    chain.knowledge(
        limit=limit,
        max_tokens=None,
        prefix="<related_knowledge>",
        suffix="</related_knowledge>",
    )
    chain.topic_archive(
        limit=1,
        prefix="<related_topic_archive>",
        suffix="</related_topic_archive>",
    )

    sources = await chain.resolve()

    if not sources:
        if real_user_id is None and user_id:
            return f"未找到与用户ID {user_id} 相关的记忆。"
        return "未找到与查询内容相关的记忆或知识。"

    new_builder = ctx.deps.context.copy(share_context=True, empty=True)
    new_builder.extend(sources)

    logger.debug(f"Found {len(sources)} memory sources for query: {query}")
    return new_builder.to_prompt()


@core_group.tool(cost=1)
async def query_tool_call_history(
    ctx: Context,
    limit: int = 1,
) -> str:
    """查询当前会话最近的工具调用详细记录

    当你需要回顾之前使用过的工具及其参数时调用。

    Args:
        limit: 返回记录数量，默认1
    """
    records = await tool_call_record_repo.get_recent(
        agent_id=ctx.deps.ids.agent_id,
        group_id=ctx.deps.ids.group_id,
        user_id=ctx.deps.ids.user_id if not ctx.deps.ids.group_id else None,
        limit=limit,
    )
    if not records:
        return "当前会话暂无工具调用记录。"
    lines = []
    for r in records:
        ts_str = datetime.datetime.fromtimestamp(r.timestamp).strftime("%m-%d %H:%M")
        line = f"[{ts_str}] step={r.run_step} {r.tool_name}"
        if r.kwargs:
            line += f"({orjson.dumps(r.kwargs, option=orjson.OPT_SORT_KEYS).decode('utf-8')})"
        lines.append(line)
    result = "\n".join(lines)
    return result


@core_group.tool(cost=2)
async def fetch_older_messages(
    ctx: Context,
    count: int = 15,
) -> str:
    """
    加载更早的聊天消息。当前对话上下文中仅包含最近的少量消息，
    当你觉得上下文不够、需要了解更早的对话内容时，调用此工具。

    Args:
        count (int): 要加载的消息条数，默认15条。建议不超过30条。
    """
    count = min(max(1, count), 30)

    source = ctx.deps.context.get_source_by_role("main_chat")
    if not isinstance(source, MemoryItemStream) or not source.items:
        return "当前没有可用的聊天上下文，无法回溯。"

    main_stream = source
    earliest_ts = main_stream.started_at

    ids = ctx.deps.ids
    if ids.group_id:
        older_items = await memory_item_repo.get_by_group_before(
            group_id=ids.group_id,
            agent_id=ids.agent_id,
            timestamp=earliest_ts - 1,
            limit=count,
        )
    elif ids.user_id:
        older_items = await memory_item_repo.get_in_private_by_user_before(
            user_id=ids.user_id,
            agent_id=ids.agent_id,
            timestamp=earliest_ts,
            limit=count,
        )
    else:
        return "当前没有可用的聊天上下文，无法回溯。"

    if not older_items:
        return "已到达最早的历史记录，没有更多消息了。"

    main_stream.items = older_items + main_stream.items

    builder = ctx.deps.context.copy(share_context=True, empty=True)
    temp_stream = MemoryItemStream.create(
        items=older_items,
        prefix="<older_messages>",
        suffix="</older_messages>",
    )
    builder.add(temp_stream)
    rendered = builder.to_prompt()

    return f"已加载 {len(older_items)} 条更早的消息：\n{rendered}"
