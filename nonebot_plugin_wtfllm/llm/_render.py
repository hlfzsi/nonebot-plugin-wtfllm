import asyncio
from io import BytesIO

import pillowmd

from ..utils import RESOURCES_DIR

style = pillowmd.LoadMarkdownStyles(RESOURCES_DIR / "style")


async def render_markdown_to_image(markdown_text: str) -> bytes:
    """将Markdown文本渲染为图片，返回图片的字节内容。"""
    result = await asyncio.to_thread(style.Render, markdown_text)
    byte_io = BytesIO()
    await asyncio.to_thread(result.image.save, byte_io, format="WebP", quality=80, lossless=False)
    return byte_io.getvalue()
