"""media/image.py 单元测试"""

from io import BytesIO

import pybase64
import pytest
from PIL import Image

from nonebot_plugin_wtfllm.media.image import convert_to_webp, decode_image_base64


# ── helpers ──────────────────────────────────────────────────────────


def _make_png_bytes(width: int = 100, height: int = 80) -> bytes:
    img = Image.new("RGB", (width, height), color=(255, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_rgba_png_bytes(width: int = 100, height: int = 80) -> bytes:
    img = Image.new("RGBA", (width, height), color=(0, 0, 255, 128))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _webp_dimensions(data: bytes) -> tuple[int, int]:
    with Image.open(BytesIO(data)) as img:
        return img.size


# ── decode_image_base64 ─────────────────────────────────────────────


class TestDecodeImageBase64:
    def test_pure_base64(self):
        raw = b"hello"
        encoded = pybase64.b64encode(raw).decode()
        assert decode_image_base64(encoded) == raw

    def test_data_uri(self):
        raw = b"world"
        encoded = pybase64.b64encode(raw).decode()
        data_uri = f"data:image/png;base64,{encoded}"
        assert decode_image_base64(data_uri) == raw

    def test_data_uri_with_comma_in_base64(self):
        """data URI 只按第一个逗号拆分"""
        raw = _make_png_bytes(2, 2)
        encoded = pybase64.b64encode(raw).decode()
        data_uri = f"data:image/png;base64,{encoded}"
        result = decode_image_base64(data_uri)
        assert result == raw


# ── convert_to_webp ─────────────────────────────────────────────────


class TestConvertToWebp:
    # -- 基本转换 --

    def test_bytes_input_rgb(self):
        data = _make_png_bytes()
        result = convert_to_webp(data)
        assert result[:4] == b"RIFF"  # WebP magic

    def test_bytes_input_rgba(self):
        data = _make_rgba_png_bytes()
        result = convert_to_webp(data)
        assert result[:4] == b"RIFF"

    def test_base64_string_input(self):
        data = _make_png_bytes()
        encoded = pybase64.b64encode(data).decode()
        result = convert_to_webp(encoded)
        assert result[:4] == b"RIFF"

    def test_data_uri_input(self):
        data = _make_png_bytes()
        encoded = pybase64.b64encode(data).decode()
        data_uri = f"data:image/png;base64,{encoded}"
        result = convert_to_webp(data_uri)
        assert result[:4] == b"RIFF"

    # -- quality 参数 --

    def test_quality_affects_size(self):
        """高 quality 应该产生更大的文件"""
        # 使用渐变图片而非纯色，保证 quality 对文件大小有明显影响
        img = Image.new("RGB", (200, 200))
        for x in range(200):
            for y in range(200):
                img.putpixel((x, y), (x % 256, y % 256, (x + y) % 256))
        buf = BytesIO()
        img.save(buf, format="PNG")
        data = buf.getvalue()
        low = convert_to_webp(data, quality=1)
        high = convert_to_webp(data, quality=99)
        assert len(high) > len(low)

    # -- max_size 缩放 --

    def test_no_max_size_preserves_dimensions(self):
        data = _make_png_bytes(300, 200)
        result = convert_to_webp(data)
        assert _webp_dimensions(result) == (300, 200)

    def test_max_size_downscales(self):
        data = _make_png_bytes(300, 200)
        result = convert_to_webp(data, max_size=(150, 100))
        w, h = _webp_dimensions(result)
        assert w <= 150
        assert h <= 100

    def test_max_size_no_upscale(self):
        """小于 max_size 的图片不应被放大"""
        data = _make_png_bytes(50, 40)
        result = convert_to_webp(data, max_size=(800, 600))
        assert _webp_dimensions(result) == (50, 40)

    def test_max_size_keeps_aspect_ratio(self):
        data = _make_png_bytes(400, 200)
        result = convert_to_webp(data, max_size=(200, 200))
        w, h = _webp_dimensions(result)
        assert w == 200
        assert h == 100

    # -- thumbnail bug 修复验证 --

    def test_thumbnail_actually_resizes(self):
        """确认缩放后保存的图片尺寸确实缩小了（修复之前的 bug）"""
        data = _make_png_bytes(800, 600)
        result = convert_to_webp(data, max_size=(200, 150))
        w, h = _webp_dimensions(result)
        assert w <= 200 and h <= 150

    def test_thumbnail_rgba_resizes(self):
        """RGBA 图片的缩放同样应当生效"""
        data = _make_rgba_png_bytes(800, 600)
        result = convert_to_webp(data, max_size=(200, 150))
        w, h = _webp_dimensions(result)
        assert w <= 200 and h <= 150

    # -- 模式转换 --

    def test_palette_mode_image(self):
        """P (palette) 模式的图片应能正确转换"""
        img = Image.new("P", (50, 50))
        buf = BytesIO()
        img.save(buf, format="PNG")
        result = convert_to_webp(buf.getvalue())
        assert result[:4] == b"RIFF"

    def test_la_mode_image(self):
        """LA 模式的图片应能正确转换"""
        img = Image.new("LA", (50, 50))
        buf = BytesIO()
        img.save(buf, format="PNG")
        result = convert_to_webp(buf.getvalue())
        assert result[:4] == b"RIFF"
