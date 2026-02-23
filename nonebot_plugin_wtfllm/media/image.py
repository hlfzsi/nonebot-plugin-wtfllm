"""图片压缩、格式转换、缩放的统一工具函数"""

__all__ = ["convert_to_webp", "decode_image_base64"]

from io import BytesIO

import pybase64
from PIL import Image


def decode_image_base64(source: str) -> bytes:
    """将 base64 字符串或 data URI 解码为原始字节

    支持两种输入格式:
        - data URI: ``data:image/png;base64,iVBOR...``
        - 纯 base64 字符串: ``iVBOR...``

    Args:
        source: base64 编码的图片数据，可以是 data URI 或纯 base64 字符串

    Returns:
        解码后的原始图片二进制数据
    """
    if "," in source:
        _, raw_base64 = source.split(",", 1)
    else:
        raw_base64 = source
    return pybase64.b64decode(raw_base64)


def convert_to_webp(
    source: bytes | str,
    *,
    quality: int = 75,
    max_size: tuple[int, int] | None = None,
) -> bytes:
    """将图片转换为 WebP 格式，可选缩放

    支持三种输入:
        - ``bytes``: 原始图片二进制数据
        - ``str`` (data URI): ``data:image/...;base64,...`` 会自动解码
        - ``str`` (纯 base64): 自动 base64 解码

    Args:
        source: 图片数据，支持 bytes / data URI / 纯 base64 字符串
        quality: WebP 压缩质量 (0-100)，默认 75
        max_size: 最大尺寸 (width, height)，等比缩放，为 None 时不缩放

    Returns:
        转换后的 WebP 图片二进制数据

    Raises:
        IOError: 如果图片无法被识别或处理
    """
    if isinstance(source, str):
        image_bytes = decode_image_base64(source)
    else:
        image_bytes = source

    with Image.open(BytesIO(image_bytes)) as img:
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
        else:
            img = img.convert("RGB")

        if max_size is not None:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)

        output_buffer = BytesIO()
        img.save(output_buffer, format="WebP", quality=quality, lossless=False)
        return output_buffer.getvalue()
