"""Config 的属性测试"""

import pytest
from hypothesis import given
from hypothesis import strategies as st


@pytest.mark.property
class TestModelConfigValidation:

    @given(
        name=st.text(min_size=1, max_size=50),
        base_url=st.from_regex(r"https?://[a-z]+\.[a-z]+", fullmatch=True),
        api_key=st.text(min_size=1, max_size=50),
    )
    def test_valid_model_config_always_constructs(self, name, base_url, api_key):
        from nonebot_plugin_wtfllm.config import ModelConfig

        mc = ModelConfig(name=name, base_url=base_url, api_key=api_key, extra_body={})
        assert mc.name == name
        assert mc.base_url == base_url
        assert mc.api_key == api_key


@pytest.mark.property
class TestCompressConfigFallback:

    @given(
        main_key=st.text(min_size=1, max_size=20),
        main_url=st.from_regex(r"https?://[a-z]+\.[a-z]+", fullmatch=True),
        main_name=st.text(min_size=1, max_size=20),
    )
    def test_compress_falls_back_to_main(self, main_key, main_url, main_name):
        from nonebot_plugin_wtfllm.config import Config

        cfg = Config(
            llm_api_key=main_key,
            llm_api_base_url=main_url,
            llm_model_name=main_name,
            huggingface_mirror_url="",
        )
        mc = cfg.compress_agent_model_config
        assert mc.name == main_name
        assert mc.api_key == main_key
        assert mc.base_url == main_url


@pytest.mark.property
class TestImageGenerationConfigReturnsNone:

    @given(
        main_key=st.text(min_size=1, max_size=20),
        main_url=st.from_regex(r"https?://[a-z]+\.[a-z]+", fullmatch=True),
        main_name=st.text(min_size=1, max_size=20),
    )
    def test_image_gen_none_when_fields_missing(self, main_key, main_url, main_name):
        from nonebot_plugin_wtfllm.config import Config

        cfg = Config(
            llm_api_key=main_key,
            llm_api_base_url=main_url,
            llm_model_name=main_name,
            huggingface_mirror_url="",
        )
        assert cfg.image_generation_model_config is None
