__all__ = [
    "ENABLE_IMAGE_GENERATION",
    "text_to_image",
    "modify_image_with_text",
    "combine_images",
]

from typing import List
from openai import AsyncOpenAI

from ..utils import APP_CONFIG, logger

_image_generation_config = APP_CONFIG.image_generation_model_config

ENABLE_IMAGE_GENERATION = bool(_image_generation_config)


if ENABLE_IMAGE_GENERATION:
    assert _image_generation_config is not None
    _image_generation_client = AsyncOpenAI(
        api_key=_image_generation_config.api_key,
        base_url=_image_generation_config.base_url,
    )

    async def text_to_image(prompt: str) -> str | None:
        assert _image_generation_config is not None

        response = await _image_generation_client.images.generate(
            model=_image_generation_config.name,
            prompt=prompt,
            n=1,
        )
        if response.data and len(response.data) > 0:
            return response.data[0].url or response.data[0].b64_json
        else:
            logger.warning("Image generation returned no data.")
            return None

    async def modify_image_with_text(source: str, prompt: str) -> str | None:
        assert _image_generation_config is not None

        response = await _image_generation_client.images.generate(
            model=_image_generation_config.name,
            prompt=prompt,
            n=1,
            extra_body={"image": source},
        )
        if response.data and len(response.data) > 0:
            return response.data[0].url or response.data[0].b64_json
        else:
            logger.warning("Image generation returned no data.")
            return None

    async def combine_images(sources: List[str], prompt: str) -> str | None:
        assert _image_generation_config is not None

        response = await _image_generation_client.images.generate(
            model=_image_generation_config.name,
            prompt=prompt,
            n=1,
            extra_body={"images": sources},
        )
        if response.data and len(response.data) > 0:
            return response.data[0].url or response.data[0].b64_json
        else:
            logger.warning("Image generation returned no data.")
            return None


else:

    async def text_to_image(prompt: str) -> str | None:
        return None

    async def modify_image_with_text(source: str, prompt: str) -> str | None:
        return None

    async def combine_images(sources: List[str], prompt: str) -> str | None:
        return None
