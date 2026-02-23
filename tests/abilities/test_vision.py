"""abilities/vision.py 单元测试

覆盖:
- SingleImageDesc.to_string
- ImageDescResponse.__len__
- get_image_desc disabled 路径
"""

import pytest

from nonebot_plugin_wtfllm.abilities.vision import (
    ENABLE_VISION,
    get_image_desc,
    SingleImageDesc,
    ImageDescResponse,
)


class TestSingleImageDesc:
    def test_to_string_with_tags(self):
        desc = SingleImageDesc(type="表情包", text="一只猫", tags=["猫", "可爱"])
        s = desc.to_string()
        assert "表情包" in s
        assert "一只猫" in s
        assert "猫" in s
        assert "可爱" in s

    def test_to_string_empty_tags(self):
        desc = SingleImageDesc(type="照片", text="风景", tags=[])
        s = desc.to_string()
        assert "无" in s

    def test_model_fields(self):
        desc = SingleImageDesc(type="t", text="x", tags=["a"])
        assert desc.type == "t"
        assert desc.text == "x"
        assert desc.tags == ["a"]


class TestImageDescResponse:
    def test_len_zero(self):
        resp = ImageDescResponse(descriptions=[])
        assert len(resp) == 0

    def test_len_multiple(self):
        descs = [
            SingleImageDesc(type="a", text="b", tags=[]),
            SingleImageDesc(type="c", text="d", tags=["e"]),
        ]
        resp = ImageDescResponse(descriptions=descs)
        assert len(resp) == 2


class TestGetImageDescDisabled:
    """测试环境下 ENABLE_VISION=False，get_image_desc 应返回 None"""

    def test_vision_disabled(self):
        assert ENABLE_VISION is False

    @pytest.mark.asyncio
    async def test_returns_none_single(self):
        result = await get_image_desc("http://example.com/img.jpg")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_list(self):
        result = await get_image_desc(["http://a.com/1.jpg", "http://b.com/2.jpg"])
        assert result is None
