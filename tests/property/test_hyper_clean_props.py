"""hyper_clean 模块的属性测试"""

import pytest
from hypothesis import given, assume, settings, HealthCheck
from hypothesis import strategies as st

import orjson


# ===== 核心属性：任意输入不崩溃 =====


@pytest.mark.property
class TestCleanHyperContentNeverCrashes:
    """clean_hyper_content 必须对任意输入返回字符串，永不抛异常"""

    @given(raw=st.text(max_size=2000))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_json_never_crashes(self, raw):
        from nonebot_plugin_wtfllm.stream_processing.hyper_clean import (
            clean_hyper_content,
        )

        result = clean_hyper_content(raw, "json")
        assert isinstance(result, str)
        assert len(result) > 0

    @given(raw=st.text(max_size=2000))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_xml_never_crashes(self, raw):
        from nonebot_plugin_wtfllm.stream_processing.hyper_clean import (
            clean_hyper_content,
        )

        result = clean_hyper_content(raw, "xml")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_none_json(self):
        from nonebot_plugin_wtfllm.stream_processing.hyper_clean import (
            clean_hyper_content,
        )

        result = clean_hyper_content(None, "json")
        parsed = orjson.loads(result)
        assert parsed["error"] == "parse_failed"

    def test_none_xml(self):
        from nonebot_plugin_wtfllm.stream_processing.hyper_clean import (
            clean_hyper_content,
        )

        result = clean_hyper_content(None, "xml")
        parsed = orjson.loads(result)
        assert parsed["error"] == "parse_failed"

    def test_empty_string_json(self):
        from nonebot_plugin_wtfllm.stream_processing.hyper_clean import (
            clean_hyper_content,
        )

        result = clean_hyper_content("", "json")
        parsed = orjson.loads(result)
        assert parsed["error"] == "parse_failed"

    def test_empty_string_xml(self):
        from nonebot_plugin_wtfllm.stream_processing.hyper_clean import (
            clean_hyper_content,
        )

        result = clean_hyper_content("", "xml")
        parsed = orjson.loads(result)
        assert parsed["error"] == "parse_failed"


# ===== 合法 JSON 输入 → 输出也是合法 JSON =====


@pytest.mark.property
class TestCleanJsonOutputIsValidJson:

    @given(
        data=st.recursive(
            st.one_of(
                st.none(),
                st.booleans(),
                st.integers(min_value=-10000, max_value=10000),
                st.text(max_size=100),
            ),
            lambda children: st.one_of(
                st.lists(children, max_size=5),
                st.dictionaries(
                    st.text(min_size=1, max_size=20), children, max_size=5
                ),
            ),
            max_leaves=20,
        )
    )
    def test_valid_json_in_valid_json_out(self, data):
        from nonebot_plugin_wtfllm.stream_processing.hyper_clean import (
            clean_hyper_content,
        )

        raw = orjson.dumps(data).decode("utf-8")
        result = clean_hyper_content(raw, "json")
        # 输出必须能被解析
        orjson.loads(result)


# ===== 噪声检测 =====


@pytest.mark.property
class TestNoiseDetection:

    @given(
        s=st.from_regex(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            fullmatch=True,
        )
    )
    def test_uuid_always_detected(self, s):
        from nonebot_plugin_wtfllm.stream_processing.hyper_clean import (
            _is_noisy_string,
        )

        assert _is_noisy_string(s) is True

    @given(s=st.from_regex(r"[0-9a-f]{32}", fullmatch=True))
    def test_md5_hash_always_detected(self, s):
        from nonebot_plugin_wtfllm.stream_processing.hyper_clean import (
            _is_noisy_string,
        )

        assert _is_noisy_string(s) is True

    @given(url=st.from_regex(r"https://[a-z]+\.[a-z]+/[a-z]+", fullmatch=True))
    def test_url_never_detected_as_noisy(self, url):
        from nonebot_plugin_wtfllm.stream_processing.hyper_clean import (
            _is_noisy_string,
        )

        assert _is_noisy_string(url) is False


# ===== URL 清洗 =====


@pytest.mark.property
class TestUrlCleaningPreservesStructure:

    @given(
        path=st.from_regex(r"/[a-z]{1,10}", fullmatch=True),
        param_key=st.from_regex(r"[a-z]{1,8}", fullmatch=True),
        param_val=st.from_regex(r"[a-z0-9]{1,10}", fullmatch=True),
    )
    def test_non_tracking_params_preserved(self, path, param_key, param_val):
        from nonebot_plugin_wtfllm.stream_processing.hyper_clean import (
            _clean_url,
            _TRACKING_PARAMS_KEYS,
        )

        assume(param_key not in _TRACKING_PARAMS_KEYS)
        assume(not param_key.startswith("utm_"))

        url = f"https://example.com{path}?{param_key}={param_val}"
        cleaned = _clean_url(url)
        assert param_key in cleaned
        assert param_val in cleaned

    @given(
        path=st.from_regex(r"/[a-z]{1,10}", fullmatch=True),
        tracking_key=st.sampled_from(["utm_source", "fbclid", "gclid", "traceid"]),
        tracking_val=st.from_regex(r"[a-z0-9]{5,15}", fullmatch=True),
    )
    def test_tracking_params_removed(self, path, tracking_key, tracking_val):
        from nonebot_plugin_wtfllm.stream_processing.hyper_clean import _clean_url

        url = f"https://example.com{path}?{tracking_key}={tracking_val}"
        cleaned = _clean_url(url)
        assert tracking_key not in cleaned
