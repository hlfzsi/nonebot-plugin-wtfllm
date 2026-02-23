"""abilities/image_generation.py 单元测试

覆盖:
- ENABLE_IMAGE_GENERATION disabled 路径
- text_to_image / modify_image_with_text / combine_images 均返回 None
"""

import pytest

from nonebot_plugin_wtfllm.abilities.image_generation import (
    ENABLE_IMAGE_GENERATION,
    text_to_image,
    modify_image_with_text,
    combine_images,
)


class TestImageGenerationDisabled:
    """测试环境下 ENABLE_IMAGE_GENERATION=False"""

    def test_flag_is_false(self):
        assert ENABLE_IMAGE_GENERATION is False

    @pytest.mark.asyncio
    async def test_text_to_image_returns_none(self):
        result = await text_to_image("一只猫")
        assert result is None

    @pytest.mark.asyncio
    async def test_modify_image_returns_none(self):
        result = await modify_image_with_text("data:image/png;base64,abc", "变成狗")
        assert result is None

    @pytest.mark.asyncio
    async def test_combine_images_returns_none(self):
        result = await combine_images(["src1", "src2"], "合并")
        assert result is None
