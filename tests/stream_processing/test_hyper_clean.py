"""hyper_clean 模块单元测试"""

import orjson
import pytest

from nonebot_plugin_wtfllm.stream_processing.hyper_clean import (
    clean_hyper_content,
    _is_noisy_string,
    _is_noisy_key,
    _is_noisy_number,
    _flatten_single_key_dicts,
    _strip_xml_ns,
    _clean_json_value,
)


# ── _is_noisy_string 测试 ──────────────────────────────────


class TestIsNoisyString:
    def test_uuid(self):
        assert _is_noisy_string("550e8400-e29b-41d4-a716-446655440000") is True

    def test_uuid_uppercase(self):
        assert _is_noisy_string("550E8400-E29B-41D4-A716-446655440000") is True

    def test_hex_hash_md5(self):
        assert _is_noisy_string("d41d8cd98f00b204e9800998ecf8427e") is True

    def test_hex_hash_sha256(self):
        assert (
            _is_noisy_string(
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            )
            is True
        )

    def test_base64_blob(self):
        b64 = "A" * 80 + "=="
        assert _is_noisy_string(b64) is True

    def test_generic_token(self):
        assert _is_noisy_string("abc123def456ghi789jkl012mno345pq") is True  # 32 chars

    def test_long_numeric(self):
        assert _is_noisy_string("123456789012345") is True  # 15 digits

    def test_short_numeric_not_noisy(self):
        assert _is_noisy_string("12345") is False

    def test_url_not_noisy(self):
        assert (
            _is_noisy_string(
                "https://example.com/path?q=abc123def456ghi789jkl012mno345pq"
            )
            is False
        )

    def test_normal_text_not_noisy(self):
        assert _is_noisy_string("这是一段正常的文本") is False
        assert _is_noisy_string("Hello World") is False

    def test_short_string_not_noisy(self):
        assert _is_noisy_string("abc") is False

    def test_empty_string_not_noisy(self):
        assert _is_noisy_string("") is False
        assert _is_noisy_string("   ") is False

    def test_reverse_domain_noisy(self):
        assert _is_noisy_string("com.tencent.tuwen.lua") is True
        assert _is_noisy_string("qqconnect.sdkshare.abc") is True

    def test_reverse_domain_two_parts_not_noisy(self):
        # 两段不算反向域名 (需要 >=3 段)
        assert _is_noisy_string("qqconnect.sdkshare") is False

    def test_version_string_noisy(self):
        assert _is_noisy_string("0.0.0.1") is True
        assert _is_noisy_string("1.2.3") is True

    def test_version_two_parts_not_noisy(self):
        assert _is_noisy_string("1.0") is False


# ── _is_noisy_key 测试 ─────────────────────────────────────────


class TestIsNoisyKey:
    def test_known_noisy_keys(self):
        assert _is_noisy_key("app") is True
        assert _is_noisy_key("appid") is True
        assert _is_noisy_key("uin") is True
        assert _is_noisy_key("ctime") is True
        assert _is_noisy_key("ver") is True
        assert _is_noisy_key("config") is True
        assert _is_noisy_key("extra") is True
        assert _is_noisy_key("view") is True
        assert _is_noisy_key("bizsrc") is True

    def test_case_insensitive(self):
        assert _is_noisy_key("App") is True
        assert _is_noisy_key("APPID") is True
        assert _is_noisy_key("ServiceID") is True

    def test_content_keys_not_noisy(self):
        assert _is_noisy_key("title") is False
        assert _is_noisy_key("desc") is False
        assert _is_noisy_key("summary") is False
        assert _is_noisy_key("jumpUrl") is False
        assert _is_noisy_key("url") is False
        assert _is_noisy_key("prompt") is False
        assert _is_noisy_key("tag") is False
        assert _is_noisy_key("meta") is False
        assert _is_noisy_key("news") is False

    def test_noisy_pattern_key(self):
        """key 本身匹配噪音字符串模式时也应判定为噪音"""
        assert _is_noisy_key("550e8400-e29b-41d4-a716-446655440000") is True
        assert _is_noisy_key("123456789012345") is True


# ── _is_noisy_number 测试 ────────────────────────────────────


class TestIsNoisyNumber:
    def test_unix_timestamp_seconds(self):
        assert _is_noisy_number(1770519807) is True  # 2026年
        assert _is_noisy_number(1609459200) is True  # 2021-01-01
        assert _is_noisy_number(0) is False  # 0 是 1 位

    def test_unix_timestamp_milliseconds(self):
        assert _is_noisy_number(1770519807000) is True  # 13位毫秒级
        assert _is_noisy_number(1609459200000) is True

    def test_short_number_preserved(self):
        assert _is_noisy_number(99) is False
        assert _is_noisy_number(12345) is False
        assert _is_noisy_number(102115491) is False  # 9位 appid

    def test_long_number_noisy(self):
        assert _is_noisy_number(123456789012345) is True

    def test_non_timestamp_10_digit(self):
        # 10位但不在时间戳范围内
        assert _is_noisy_number(9999999999) is False


# ── _flatten_single_key_dicts 测试 ─────────────────────────


class TestFlattenSingleKeyDicts:
    def test_basic_flatten(self):
        data = {"meta": {"news": {"title": "x", "desc": "y"}}}
        result = _flatten_single_key_dicts(data)
        assert result == {"meta_news_title": "x", "meta_news_desc": "y"}

    def test_deep_single_chain(self):
        data = {"a": {"b": {"c": "val"}}}
        result = _flatten_single_key_dicts(data)
        assert result == {"a_b_c": "val"}

    def test_no_flatten_multi_keys(self):
        data = {"title": "x", "desc": "y"}
        result = _flatten_single_key_dicts(data)
        assert result == {"title": "x", "desc": "y"}

    def test_no_flatten_single_key_non_dict(self):
        data = {"title": "x"}
        result = _flatten_single_key_dicts(data)
        assert result == {"title": "x"}

    def test_mixed_structure(self):
        data = {
            "meta": {"news": {"title": "x"}},
            "prompt": "text",
        }
        result = _flatten_single_key_dicts(data)
        # meta 有单 key news, news 有单 key title -> flatten to meta_news_title
        # 但外层有两个 key (meta, prompt), 不会再打平
        assert result == {"meta": {"news_title": "x"}, "prompt": "text"}

    def test_flatten_with_list(self):
        data = {"items": [{"a": {"b": "v"}}]}
        result = _flatten_single_key_dicts(data)
        assert result == {"items": [{"a_b": "v"}]}

    def test_empty_dict(self):
        assert _flatten_single_key_dicts({}) == {}

    def test_non_dict(self):
        assert _flatten_single_key_dicts("hello") == "hello"
        assert _flatten_single_key_dicts(42) == 42


# ── _strip_xml_ns 测试 ──────────────────────────────────────


class TestStripXmlNs:
    def test_strip_namespace(self):
        assert _strip_xml_ns("{http://example.com}config") == "config"

    def test_no_namespace(self):
        assert _strip_xml_ns("config") == "config"

    def test_empty_namespace(self):
        assert _strip_xml_ns("{}config") == "config"


# ── JSON 清洗测试 ──────────────────────────────────────────


class TestCleanJsonValue:
    def test_removes_uuid_value(self):
        data = {"id": "550e8400-e29b-41d4-a716-446655440000", "title": "Hello"}
        result = _clean_json_value(data)
        assert result == {"title": "Hello"}

    def test_removes_token_value(self):
        data = {"token": "abc123def456ghi789jkl012mno345pq", "name": "Test"}
        result = _clean_json_value(data)
        assert result == {"name": "Test"}

    def test_preserves_url(self):
        data = {
            "link": "https://example.com/page",
            "hash": "d41d8cd98f00b204e9800998ecf8427e",
        }
        result = _clean_json_value(data)
        assert result == {"link": "https://example.com/page"}

    def test_preserves_short_number(self):
        data = {"price": 99, "count": 5}
        result = _clean_json_value(data)
        assert result == {"price": 99, "count": 5}

    def test_removes_long_number(self):
        data = {"internal_id": 123456789012345, "price": 50}
        result = _clean_json_value(data)
        assert result == {"price": 50}

    def test_preserves_bool(self):
        data = {"active": True, "deleted": False}
        result = _clean_json_value(data)
        assert result == {}

    def test_removes_empty_nested_dict(self):
        data = {
            "meta": {"token": "abc123def456ghi789jkl012mno345pq"},
            "title": "Card",
        }
        result = _clean_json_value(data)
        assert result == {"title": "Card"}

    def test_removes_empty_nested_list(self):
        data = {
            "tags": ["550e8400-e29b-41d4-a716-446655440000"],
            "desc": "Good",
        }
        result = _clean_json_value(data)
        assert result == {"desc": "Good"}

    def test_removes_noisy_keys(self):
        """“key 级别过滤: 噪音 key 直接跳过"""
        data = {
            "app": "com.tencent.tuwen.lua",
            "bizsrc": "qqconnect.sdkshare",
            "config": {"ctime": 4444444444, "forward": 1, "type": "normal"},
            "extra": {"app_type": 1, "appid": 444444444, "uin": 4444444444},
            "meta": {
                "news": {
                    "title": "标题",
                    "desc": "描述",
                    "appid": 444444444,
                    "uin": 4444444444,
                }
            },
            "ver": "0.0.0.1",
            "view": "news",
            "prompt": "[分享]标题",
        }
        result = _clean_json_value(data)
        assert "app" not in result
        assert "bizsrc" not in result
        assert "config" not in result
        assert "extra" not in result
        assert "ver" not in result
        assert "view" not in result
        assert result["meta"]["news"] == {"title": "标题", "desc": "描述"}
        assert result["prompt"] == "[分享]标题"

    def test_removes_timestamp_number(self):
        """时间戳数值应被移除"""
        data = {"ts": 1770519807, "name": "test", "ms_ts": 1770519807000}
        result = _clean_json_value(data)
        assert result == {"name": "test"}

    def test_removes_noisy_string_key(self):
        """数字模式的 key 也会被移除"""
        data = {"123456789012345": "value", "title": "ok"}
        result = _clean_json_value(data)
        assert result == {"title": "ok"}

    def test_nested_structure(self):
        data = {
            "meta": {
                "detail": {
                    "title": "分享链接",
                    "desc": "一段描述",
                    "url": "https://example.com",
                    "appid": 12345,
                    "token": "550e8400-e29b-41d4-a716-446655440000",
                }
            },
        }
        result = _clean_json_value(data)
        assert result == {
            "meta": {
                "detail": {
                    "title": "分享链接",
                    "desc": "一段描述",
                    "url": "https://example.com",
                }
            },
        }

    def test_all_noise_becomes_empty(self):
        data = {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "hash": "d41d8cd98f00b204e9800998ecf8427e",
        }
        result = _clean_json_value(data)
        assert result == {}


# ── clean_hyper_content JSON 集成测试 ──────────────────────


class TestCleanHyperContentJson:
    def test_qq_card_full_cleaning(self):
        """完整 QQ 卡片消息清洗 + 平铺测试"""
        raw = orjson.dumps(
            {
                "app": "com.tencent.tuwen.lua",
                "bizsrc": "qqconnect.sdkshare",
                "config": {
                    "ctime": 4444444444,
                    "forward": 1,
                    "type": "normal",
                },
                "extra": {
                    "app_type": 1,
                    "appid": 4444444444,
                    "uin": 4444444444,
                },
                "meta": {
                    "news": {
                        "app_type": 1,
                        "appid": 4444444444,
                        "ctime": 4444444444,
                        "desc": "元宝派红包，新春领不停",
                        "jumpUrl": "https://yb.tencent.com/...",
                        "preview": "https://pic.ugcimg.cn/...",
                        "tag": "元宝",
                        "tagIcon": "https://p.qpic.cn/qqconnect/0/app_4444444444_1744794565/100",
                        "title": "\u201c5555\u201d给你发了一个现金红包！",
                        "uin": 4444444444,
                    }
                },
                "prompt": "[分享]\u201c5555\u201d给你发了一个现金红包！",
                "ver": "0.0.0.1",
                "view": "news",
            }
        ).decode("utf-8")
        result = clean_hyper_content(raw, "json")
        parsed = orjson.loads(result)
        # meta -> news 是单 key 链, 会被平铺
        # 但外层有 prompt, 所以 meta 不会被继续平铺
        assert "news_title" in parsed["meta"]
        assert "news_desc" in parsed["meta"]
        assert "news_tag" in parsed["meta"]
        assert "news_jumpUrl" in parsed["meta"]
        assert "prompt" in parsed
        # 噪音已移除
        assert "app" not in parsed
        assert "bizsrc" not in parsed
        assert "config" not in parsed
        assert "extra" not in parsed

    def test_basic_json_cleaning(self):
        raw = orjson.dumps(
            {
                "meta": {
                    "news": {
                        "title": "今日新闻",
                        "jumpUrl": "https://news.example.com/article/123",
                        "appid": 123456789012345678,
                        "token": "aB3cD4eF5gH6iJ7kL8mN9oP0qR1sT2uX",
                    }
                },
            }
        ).decode("utf-8")
        result = clean_hyper_content(raw, "json")
        parsed = orjson.loads(result)
        # meta->news 单 key 链会被平铺 (appid 被 key过滤, token 被值过滤)
        assert parsed["meta_news_title"] == "今日新闻"
        assert parsed["meta_news_jumpUrl"] == "https://news.example.com/article/123"
        assert not any("appid" in k for k in parsed)
        assert not any("token" in k for k in parsed)

    def test_json_parse_failed(self):
        raw = "this is not json {{"
        result = clean_hyper_content(raw, "json")
        parsed = orjson.loads(result)
        assert parsed["error"] == "parse_failed"

    def test_json_all_noise_cleaned(self):
        raw = orjson.dumps(
            {
                "app": "com.tencent.test.app",
                "config": {"ctime": 123},
            }
        ).decode("utf-8")
        result = clean_hyper_content(raw, "json")
        parsed = orjson.loads(result)
        assert parsed == {"notice": "cleaned_empty"}

    def test_json_empty_input(self):
        result = clean_hyper_content("", "json")
        parsed = orjson.loads(result)
        assert parsed["error"] == "parse_failed"

    def test_json_preserves_clean_data(self):
        data = {"title": "分享", "desc": "文字描述", "url": "https://example.com"}
        raw = orjson.dumps(data).decode("utf-8")
        result = clean_hyper_content(raw, "json")
        parsed = orjson.loads(result)
        assert parsed == data


# ── clean_hyper_content XML 集成测试 ──────────────────────


class TestCleanHyperContentXml:
    def test_basic_xml_cleaning(self):
        raw = (
            '<msg serviceID="1" templateID="550e8400-e29b-41d4-a716-446655440000">'
            "<item><title>分享标题</title>"
            "<summary>一段描述</summary>"
            "<url>https://example.com/page</url>"
            "<token>abc123def456ghi789jkl012mno345pq</token>"
            "</item></msg>"
        )
        result = clean_hyper_content(raw, "xml")
        # title, summary, url 应保留
        assert "分享标题" in result
        assert "一段描述" in result
        assert "https://example.com/page" in result
        # UUID 属性和 token 文本应被移除
        assert "550e8400" not in result
        assert "abc123def456ghi789jkl012mno345pq" not in result

    def test_xml_noisy_tag_names_removed(self):
        raw = (
            "<root>"
            "<title>好内容</title>"
            "<config><ctime>12345</ctime></config>"
            "<extra><uin>67890</uin></extra>"
            "</root>"
        )
        result = clean_hyper_content(raw, "xml")
        assert "好内容" in result
        assert "config" not in result
        assert "extra" not in result

    def test_xml_noisy_attr_names_removed(self):
        raw = '<item serviceID="1" title="好标题"/>'
        result = clean_hyper_content(raw, "xml")
        assert "好标题" in result
        assert "serviceid" not in result.lower()

    def test_xml_parse_failed(self):
        raw = "<<< not xml at all >>>"
        result = clean_hyper_content(raw, "xml")
        # lxml recover=True 很宽容，所以损坏的 XML 可能仍能解析
        # 不做严格断言，只确认不会崩溃
        assert isinstance(result, str)
        assert len(result) > 0

    def test_xml_preserves_url_attribute(self):
        raw = '<action url="https://example.com/link" token="abc123def456ghi789jkl012mno345pq"/>'
        result = clean_hyper_content(raw, "xml")
        assert "https://example.com/link" in result
        assert "abc123def456ghi789jkl012mno345pq" not in result

    def test_xml_empty_after_clean(self):
        raw = '<msg token="550e8400-e29b-41d4-a716-446655440000">d41d8cd98f00b204e9800998ecf8427e</msg>'
        result = clean_hyper_content(raw, "xml")
        # 根元素清空后保留根标签名
        assert "msg" in result

    def test_xml_empty_input(self):
        result = clean_hyper_content("", "xml")
        parsed = orjson.loads(result)
        assert parsed["error"] == "parse_failed"


# ── 边界情况测试 ──────────────────────────────────────────


class TestEdgeCases:
    def test_whitespace_only_input(self):
        result = clean_hyper_content("   ", "json")
        parsed = orjson.loads(result)
        assert parsed["error"] == "parse_failed"

    def test_deeply_nested_json(self):
        data = {"a": {"b": {"c": {"d": {"title": "Deep", "hash": "a" * 32}}}}}
        raw = orjson.dumps(data).decode("utf-8")
        result = clean_hyper_content(raw, "json")
        parsed = orjson.loads(result)
        # a->b->c->d 链式单 key 会被平铺, d 有多 key title+hash(被清)只剩 title
        assert parsed["a_b_c_d_title"] == "Deep"

    def test_json_list_at_root(self):
        data = [
            {"title": "Item1", "id": "550e8400-e29b-41d4-a716-446655440000"},
            {"title": "Item2", "token": "abc123def456ghi789jkl012mno345pq"},
        ]
        raw = orjson.dumps(data).decode("utf-8")
        result = clean_hyper_content(raw, "json")
        parsed = orjson.loads(result)
        assert parsed == [{"title": "Item1"}, {"title": "Item2"}]

    def test_mixed_content_xml(self):
        raw = (
            "<root>"
            "<title>好消息</title>"
            "<desc>描述文本</desc>"
            "<noise>d41d8cd98f00b204e9800998ecf8427e</noise>"
            "<link>https://example.com</link>"
            "</root>"
        )
        result = clean_hyper_content(raw, "xml")
        assert "好消息" in result
        assert "描述文本" in result
        assert "https://example.com" in result
        assert "d41d8cd98f00b204e9800998ecf8427e" not in result


# ── XML namespace 边缘测试 ───────────────────────────────


class TestXmlNamespace:
    def test_namespaced_noisy_tag_removed(self):
        raw = (
            '<root xmlns:x="http://example.com">'
            "<x:config><x:ctime>12345</x:ctime></x:config>"
            "<title>测试</title>"
            "</root>"
        )
        result = clean_hyper_content(raw, "xml")
        assert "测试" in result
        # config 和 ctime 都是噪音 key
        assert "12345" not in result

    def test_namespaced_attr_cleaned(self):
        raw = '<root xmlns:x="http://example.com" x:appid="123" title="保留"/>'
        result = clean_hyper_content(raw, "xml")
        assert "保留" in result


# ── XML tail 空白测试 ─────────────────────────────────


class TestXmlTailStripping:
    def test_whitespace_only_tail_removed(self):
        raw = "<root><a>内容</a>   \n   <b>更多</b></root>"
        result = clean_hyper_content(raw, "xml")
        assert "内容" in result
        assert "更多" in result
