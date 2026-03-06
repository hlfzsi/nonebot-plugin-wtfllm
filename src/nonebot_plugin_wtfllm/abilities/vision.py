__all__ = ["ENABLE_VISION", "get_image_desc"]

import asyncio
from typing import List, cast

import httpx
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import ImageUrl, BinaryImage
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIModelName
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from ..utils import APP_CONFIG, logger

_vision_model_config = APP_CONFIG.vision_model_config

ENABLE_VISION = bool(_vision_model_config)


class SingleImageDesc(BaseModel):
    type: str = Field(..., description="图片分类，如表情包、新闻等")
    text: str = Field(
        ...,
        description="图片简要的描述文本，若图片中包含文字请提取出来并放在这个字段, 或者图片文字的概要内容",
    )
    tags: List[str] = Field(
        ...,
        description="图片标签，如动漫人物名称、相关作品等",
    )
    # is_meme: bool = Field(..., description="是meme表情包/梗图吗？")

    def to_string(self) -> str:
        tags_str = ", ".join(self.tags) if self.tags else "无"
        return f"类型: {self.type}\n描述: {self.text}\n标签: {tags_str}\n"


class ImageDescResponse(BaseModel):
    descriptions: List[SingleImageDesc] = Field(
        ..., description="每个元素对应一张图片的描述内容"
    )

    def __len__(self):
        return len(self.descriptions)


if ENABLE_VISION:
    assert _vision_model_config is not None

    def is_url(source: str) -> bool:
        return source.startswith(("http://", "https://"))

    provider = OpenAIProvider(
        api_key=_vision_model_config.api_key, base_url=_vision_model_config.base_url
    )

    model = OpenAIChatModel(
        model_name=cast(OpenAIModelName, _vision_model_config.name), provider=provider
    )

    vision_agent = Agent(
        model,
        output_type=ImageDescResponse,
        model_settings=ModelSettings(
            extra_body={
                **_vision_model_config.extra_body,
            }
        ),
    )

    async def get_image_desc(
        sources: List[str] | str,
    ) -> List[SingleImageDesc] | None:
        """
        获取多张图片的描述。

        Args:
            sources: 可以是单图片(str)，也可以是图片列表(List[str])。
                        支持 URL 或带 Data URI 前缀的 Base64 字符串。
        """
        assert _vision_model_config is not None

        if isinstance(sources, str):
            sources = [sources]

        image_sources = [s for s in sources if s]

        if not image_sources:
            logger.warning("No image sources provided for description.")
            return None

        try:
            prompt_text = (
                "描述以下图片内容，提取其中的文字并进行概括，识别图片中的对象和场景，并根据需要添加相关标签。"
                "若单个图片中包含多张图片, 仍视为一个整体进行描述。 "
                "输出列表长度必须严格与输入图片数量一致，每个元素对应一张图片的描述内容。"
            )

            image_contents = [
                ImageUrl(url=source)
                if is_url(source)
                else BinaryImage.from_data_uri(source)
                for source in image_sources
            ]

            user_content = [prompt_text] + image_contents

            result = None
            try:
                async with asyncio.timeout(30):
                    result = await vision_agent.run(user_prompt=user_content)
            except asyncio.TimeoutError:
                logger.error("Vision model Agent timed out after 30 seconds.")
                return None
            finally:
                if result is not None:
                    logger.debug(f"Vision Agent cost tokens: {result.usage()}")

            assert result is not None, "Vision Agent did not return a result."

            descriptions = result.output

            if len(descriptions) != len(image_sources):
                logger.error(
                    f"Vision mismatch: expected {len(image_sources)} descs, got {len(descriptions)}."
                )
                return None

            return descriptions.descriptions

        except (httpx.HTTPError, ValueError, RuntimeError) as e:
            logger.error(f"Error calling vision model Agent: {e}")
            return None

else:

    async def get_image_desc(
        sources: List[str] | str,
    ) -> List[SingleImageDesc] | None:
        return None
