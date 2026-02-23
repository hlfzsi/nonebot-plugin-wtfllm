"""config.py 单元测试

覆盖:
- ModelConfig 创建
- Config 的各个 property（model_post_init, admin_users, model configs）
"""

import os
from unittest.mock import patch

import pytest

from nonebot_plugin_wtfllm.config import Config, ModelConfig


def _make_config(**overrides) -> Config:
    """构造一个最小可用的 Config 实例"""
    defaults = dict(
        llm_api_key="key-123",
        llm_api_base_url="http://api.test",
        llm_model_name="test-llm",
    )
    defaults.update(overrides)
    return Config(**defaults)


# ===========================================================================
# ModelConfig
# ===========================================================================


class TestModelConfig:
    def test_create_valid(self):
        mc = ModelConfig(
            name="m1", base_url="http://x", api_key="k", extra_body={}
        )
        assert mc.name == "m1"

    def test_missing_required_field_raises(self):
        with pytest.raises(Exception):
            ModelConfig(name="m1")  # type: ignore[call-arg]


# ===========================================================================
# Config.model_post_init
# ===========================================================================


class TestConfigPostInit:
    def test_sets_hf_endpoint_when_mirror_url(self):
        cfg = _make_config(huggingface_mirror_url="https://mirror.test")
        assert os.environ.get("HF_ENDPOINT") == "https://mirror.test"

    def test_no_hf_endpoint_when_empty(self):
        os.environ.pop("HF_ENDPOINT", None)
        _make_config(huggingface_mirror_url="")
        assert os.environ.get("HF_ENDPOINT") is None


# ===========================================================================
# admin_users
# ===========================================================================


class TestAdminUsers:
    def test_returns_superusers(self):
        cfg = _make_config(superusers=["u1", "u2"])
        assert cfg.admin_users == ["u1", "u2"]

    def test_default_empty(self):
        cfg = _make_config()
        assert cfg.admin_users == []


# ===========================================================================
# main_agent_model_config
# ===========================================================================


class TestMainAgentModelConfig:
    def test_returns_model_config(self):
        cfg = _make_config(
            llm_api_key="key",
            llm_api_base_url="http://main",
            llm_model_name="main-model",
            llm_extra_body={"k": "v"},
        )
        mc = cfg.main_agent_model_config
        assert isinstance(mc, ModelConfig)
        assert mc.name == "main-model"
        assert mc.api_key == "key"
        assert mc.base_url == "http://main"
        assert mc.extra_body == {"k": "v"}


# ===========================================================================
# compress_agent_model_config
# ===========================================================================


class TestCompressAgentModelConfig:
    def test_fallback_to_main(self):
        cfg = _make_config()
        mc = cfg.compress_agent_model_config
        assert mc.name == "test-llm"
        assert mc.api_key == "key-123"

    def test_custom_compress_config(self):
        cfg = _make_config(
            compress_model_name="compress-m",
            compress_api_base_url="http://compress",
            compress_api_key="ck",
            compress_extra_body={"t": 1},
        )
        mc = cfg.compress_agent_model_config
        assert mc.name == "compress-m"
        assert mc.api_key == "ck"
        assert mc.base_url == "http://compress"
        assert mc.extra_body == {"t": 1}


# ===========================================================================
# vision_model_config
# ===========================================================================


class TestVisionModelConfig:
    def test_none_when_fields_missing(self):
        cfg = _make_config()
        assert cfg.vision_model_config is None

    def test_returns_config_when_set(self):
        cfg = _make_config(
            vision_model_name="v-model",
            vision_model_base_url="http://vision",
            vision_api_key="vk",
            vision_extra_body={"x": 1},
        )
        mc = cfg.vision_model_config
        assert mc is not None
        assert mc.name == "v-model"


# ===========================================================================
# image_generation_model_config
# ===========================================================================


class TestImageGenerationModelConfig:
    def test_none_when_fields_missing(self):
        cfg = _make_config()
        assert cfg.image_generation_model_config is None

    def test_returns_config_when_set(self):
        cfg = _make_config(
            image_generation_model_name="ig-model",
            image_generation_model_base_url="http://ig",
            image_generation_api_key="igk",
            image_generation_extra_body={},
        )
        mc = cfg.image_generation_model_config
        assert mc is not None
        assert mc.name == "ig-model"
