from typing import List

from pydantic_ai import ToolReturn, BinaryImage

from .base import ToolGroupMeta
from ...deps import Context
from ....memory import ImageSegment
from ....v_db import meme_repo, MemePayload
from ....db import memory_item_repo
from ....utils import logger, APP_CONFIG


memes_tool_group = ToolGroupMeta(
    name="Memes",
    description="用于存储和发送表情包的工具箱, 表情包发送必须通过本工具箱完成",
)


@memes_tool_group.tool(cost=0)
async def save_meme(
    ctx: Context, img_ref: str, description: str, tags: List[str]
) -> str:
    """保存表情包到数据库

    Args:
        img_ref: 表情包图片的引用，如 IMG:1 等
        description: 图片中包含什么文字或什么内容
        tags: 与图片主题相关的标签列表
    """
    logger.debug(
        f"Saving meme with img_ref: {img_ref}, description: {description}, tags: {tags}"
    )
    _key = f"meme_save_{img_ref}"
    if _key in ctx.deps.caches:
        return "该图片引用已被保存，请勿重复保存。"
    ctx.deps.caches[_key] = True
    image_seg = ctx.deps.context.resolve_media_ref(img_ref, ImageSegment)
    if image_seg is None:
        return "无法找到指定的图片引用，请确认后重试。"

    raw_message_id = image_seg.message_id
    raw_item = await memory_item_repo.get_by_message_id(raw_message_id)
    assert raw_item is not None, "无法找到对应的记忆项"

    forward_node = raw_item.content.deep_find_node(image_seg)
    uploader_id = forward_node.sender if forward_node else raw_item.sender

    if not image_seg.available:
        return "图片资源已过期"

    if image_seg.local_path:
        meme = await MemePayload.from_path(
            path=str(image_seg.local_path),
            docs=description,
            raw_message_id=raw_message_id,
            tags=tags,
            uploader_id=uploader_id,
        )

    elif image_seg.url:
        meme = await MemePayload.from_url(
            url=image_seg.url,
            docs=description,
            raw_message_id=raw_message_id,
            tags=tags,
            uploader_id=uploader_id,
        )

    else:
        raise ValueError("图片段必须包含 URL 或 Base64 数据")

    await meme_repo.save_meme(meme)
    return "Meme 已保存"


@memes_tool_group.tool(cost=0)
async def search_memes(
    ctx: Context, query: str, tags: List[str] | None = None, limit: int = 5
) -> ToolReturn:
    """搜索表情包并获得图片UUID

    Args:
        query: 图片中应当包含什么文字或者什么内容
        tags: 与图片主题相关的标签列表
        limit: 返回的结果数量上限
    """
    logger.debug(f"Searching memes with query: {query}, tags: {tags}, limit: {limit}")
    string_parts = []
    content_parts = []
    if tags is None or len(tags) == 0:
        memes = await meme_repo.search_by_text(query, limit)
    else:
        memes = await meme_repo.search_by_text_with_tags(query, tags, limit)
    memes = [res.item for res in memes]
    for meme in memes:
        string_parts.append(f"Meme UUID: {meme.storage_id} , 描述: {meme.docs}")
        if APP_CONFIG.llm_support_vision:
            _bytes = await meme.get_bytes_async()
            content_parts.append(BinaryImage(data=_bytes, media_type="image/webp"))  # pyright: ignore[reportCallIssue]
    return ToolReturn(return_value="\n".join(string_parts), content=content_parts)


@memes_tool_group.tool(cost=0)
async def list_memes(
    ctx: Context, limit: int = 10, uploader_id: str | None = None
) -> ToolReturn:
    """列出最近上传的表情包并获取图片UUID

    Args:
        limit: 返回的结果数量上限
        uploader_id: 如果指定了上传者ID, 则只列出该用户上传的表情包
    """
    logger.debug(f"Listing memes with limit: {limit}")
    real_uploader_id = (
        ctx.deps.context.resolve_aliases(uploader_id) if uploader_id else None
    )
    string_parts = []
    content_parts = []
    memes = await meme_repo.get_recent(limit=limit, uploader_id=real_uploader_id)
    for meme in memes:
        string_parts.append(f"Meme UUID: {meme.storage_id}, 描述: {meme.docs}")
        if APP_CONFIG.llm_support_vision:
            _bytes = await meme.get_bytes_async()
            content_parts.append(BinaryImage(data=_bytes, media_type="image/webp"))  # pyright: ignore[reportCallIssue]
    return ToolReturn(return_value="\n".join(string_parts), content=content_parts)
