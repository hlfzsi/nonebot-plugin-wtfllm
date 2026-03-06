import asyncio
from urllib.parse import urlparse

from nonebot_plugin_alconna.uniseg import Image as NBImage

from .base import ToolGroupMeta
from .utils import reschedule_deadline
from ...deps import Context
from ....media import convert_to_webp
from ....memory import ImageSegment
from ....abilities import (
    ENABLE_IMAGE_GENERATION,
    text_to_image as _text_to_image,
    modify_image_with_text as _modify_image_with_text,
    combine_images as _combine_images,
)
from ....utils import logger


async def _perm(ctx: Context) -> bool:
    return ENABLE_IMAGE_GENERATION


image_generation_tool_group = ToolGroupMeta(
    name="ImageGeneration",
    description="图片生成工具组, 包含文生图, 图生图, 多图生图。",
    prem=_perm,
)


def _is_valid_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except (ValueError, TypeError):
        return False



@image_generation_tool_group.tool(cost=4)
async def text_to_image(ctx: Context, prompt: str) -> bool:
    """
    基于文本生成图片
    生成的图片将被自动追加到回复中

    Args:
        prompt (str): 优化后的文本提示词, 应当使用英文描述

    Returns:
        bool : 成功生成返回True, 失败返回False
    """
    logger.debug(f"Generating image for prompt: {prompt}")

    reschedule_deadline(ctx, 30)

    url_or_base64 = await _text_to_image(prompt)
    if url_or_base64 is None:
        logger.warning("Image generation failed, received None.")
        return False
    elif _is_valid_url(url_or_base64):
        _image = NBImage(url=url_or_base64, name="generated_image.webp")
        _image.desc = prompt  # pyright: ignore[reportAttributeAccessIssue]
        ctx.deps.reply_segments += _image
        logger.debug(f"Generated image URL: {url_or_base64}")
    else:
        webp_bytes = await asyncio.to_thread(
            convert_to_webp, url_or_base64, quality=85
        )
        _image = NBImage(raw=webp_bytes, name="generated_image.webp")
        _image.desc = prompt  # pyright: ignore[reportAttributeAccessIssue]
        ctx.deps.reply_segments += _image
        logger.debug("Generated image from base64 data.")
    return True


@image_generation_tool_group.tool(cost=4)
async def modify_image_with_text(ctx: Context, image_ref: str, prompt: str) -> bool:
    """
    基于文本修改图片
    生成的图片将被自动追加到回复中

    Args:
        image_ref (str): 图片代号,如IMG:1
        prompt (str): 优化后的文本提示词, 应当使用英文描述

    Returns:
        bool : 成功生成返回True, 失败返回False
    """
    logger.debug(f"Modifying image {image_ref} with prompt: {prompt}")

    reschedule_deadline(ctx, 30)

    try:
        image_seg = ctx.deps.context.resolve_media_ref(image_ref, ImageSegment)
    except ValueError as e:
        logger.error(f"Failed to resolve image reference {image_ref}: {e}")
        return False
    if image_seg is None:
        logger.error(f"Image reference {image_ref} could not be found in context.")
        return False

    source = None
    if not image_seg.available:
        return False

    if image_seg.local_path:
        source = await image_seg.get_data_uri_async()
    elif image_seg.url:
        source = image_seg.url
    if source is None:
        logger.error(f"Image segment for {image_ref} does not contain a valid source.")
        return False

    url_or_base64 = await _modify_image_with_text(source, prompt)
    if url_or_base64 is None:
        logger.warning("Image modification failed, received None.")
        return False
    elif _is_valid_url(url_or_base64):
        _image = NBImage(url=url_or_base64, name="generated_image.webp")
        _image.desc = prompt  # pyright: ignore[reportAttributeAccessIssue]
        ctx.deps.reply_segments += _image
        logger.debug(f"Modified image URL: {url_or_base64}")
    else:
        webp_bytes = await asyncio.to_thread(
            convert_to_webp, url_or_base64, quality=85
        )
        _image = NBImage(raw=webp_bytes, name="modified_image.webp")
        _image.desc = prompt  # pyright: ignore[reportAttributeAccessIssue]
        ctx.deps.reply_segments += _image
        logger.debug("Modified image from base64 data.")
    return True


@image_generation_tool_group.tool(cost=4)
async def combine_images(ctx: Context, image_refs: list[str], prompt: str) -> bool:
    """
    基于多张图片和文本生成新图片
    生成的图片将被自动追加到回复中

    Args:
        image_refs (list[str]): 图片代号列表,如[IMG:1, IMG:2]
        prompt (str): 优化后的文本提示词, 应当使用英文描述

    Returns:
        bool : 成功生成返回True, 失败返回False
    """
    logger.debug(f"Combining images {image_refs} with prompt: {prompt}")

    reschedule_deadline(ctx, 30)

    sources = []

    for image_ref in image_refs:
        try:
            image_seg = ctx.deps.context.resolve_media_ref(image_ref, ImageSegment)
        except ValueError as e:
            logger.error(f"Failed to resolve image reference {image_ref}: {e}")
            continue
        if image_seg is None:
            logger.error(f"Image reference {image_ref} could not be found in context.")
            continue
        if not image_seg.available:
            logger.warning(f"Image reference {image_ref} has expired")
            continue
        elif image_seg.local_path:
            source = await image_seg.get_data_uri_async()
            sources.append(source)
        elif image_seg.url:
            sources.append(image_seg.url)
        else:
            logger.error(
                f"Image reference {image_ref} could not be found or has no valid source."
            )

    if not sources or (len(sources) != len(image_refs)):
        logger.error("Not enough valid image sources found for combination.")
        return False

    url_or_base64 = await _combine_images(sources, prompt)
    if url_or_base64 is None:
        logger.warning("Image combination failed, received None.")
        return False
    elif _is_valid_url(url_or_base64):
        _image = NBImage(url=url_or_base64, name="combined_image.webp")
        _image.desc = prompt  # pyright: ignore[reportAttributeAccessIssue]
        ctx.deps.reply_segments += _image
        logger.debug(f"Combined image URL: {url_or_base64}")
    else:
        webp_bytes = await asyncio.to_thread(
            convert_to_webp, url_or_base64, quality=85
        )
        _image = NBImage(raw=webp_bytes, name="combined_image.webp")
        _image.desc = prompt  # pyright: ignore[reportAttributeAccessIssue]
        ctx.deps.reply_segments += _image
        logger.debug("Combined image from base64 data.")
    return True
