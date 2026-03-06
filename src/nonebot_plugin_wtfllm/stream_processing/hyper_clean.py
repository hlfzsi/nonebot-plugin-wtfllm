import re
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from typing import Any, Literal

import orjson
from lxml import etree

# UUID
_RE_UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)

# MD5 / SHA 系列 hex hash
_RE_HEX_HASH = re.compile(r"^[0-9a-f]{32,}$", re.IGNORECASE)

# 长 base64 blob
_RE_BASE64_BLOB = re.compile(r"^[A-Za-z0-9+/]{64,}={0,2}$")

# 通用 token / 密钥
_RE_GENERIC_TOKEN = re.compile(r"^[A-Za-z0-9_\-]{32,}$")

# 超长纯数字
_RE_LONG_NUMERIC = re.compile(r"^\d{15,}$")

# 反向域名标识符 (com.tencent.xxx.yyy 等)
_RE_REVERSE_DOMAIN = re.compile(
    r"^[a-z][a-z0-9]*(\.[a-z][a-z0-9_]*){2,}$", re.IGNORECASE
)

# 版本号字符串 (0.0.0.1, 1.2.3 等)
_RE_VERSION_STRING = re.compile(r"^\d+(\.\d+){2,}$")

# Unix 时间戳范围 (1970-01-01 ~ 2100-01-01)
_TIMESTAMP_MIN = 0
_TIMESTAMP_MAX = 4102444800  # 2100-01-01 00:00:00 UTC
_TIMESTAMP_MS_MIN = _TIMESTAMP_MIN * 1000
_TIMESTAMP_MS_MAX = _TIMESTAMP_MAX * 1000

_NOISE_PATTERNS = [
    _RE_UUID,
    _RE_HEX_HASH,
    _RE_BASE64_BLOB,
    _RE_GENERIC_TOKEN,
    _RE_LONG_NUMERIC,
    _RE_REVERSE_DOMAIN,
    _RE_VERSION_STRING,
]

# URL 追踪参数黑名单
_TRACKING_PARAMS_KEYS = {
    "sid",
    "spm",
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_content",
    "utm_term",
    "traceid",
    "trace_id",
    "clickid",
    "fbclid",
    "gclid",
    "from",
    "isappinstalled",
    "session_id",
    "share",
    "share_from",
    "tbpicau",
    "client_type",
    "client_version",
    "unique",
    "st",
    "source",
    "fr",
    "see_lz",
    "sfc",
    "share_medium",
    "is_video",
    "share_source",
    "bbid",
    "ts",
}

_NOISY_KEYS: set[str] = {
    # 应用 / 来源标识
    "app",
    "appid",
    "app_id",
    "app_type",
    "apptype",
    "bizsrc",
    # 用户 / 发送者 ID
    "uin",
    "uid",
    "senderuin",
    "sender_uin",
    # 时间戳
    "ctime",
    "mtime",
    "atime",
    "ftime",
    # 版本 / 构建元数据
    "ver",
    "version",
    # 布局 / 显示控制
    "view",
    "type",
    "forward",
    "autosize",
    "tagicon",
    # 技术 ID
    "seq",
    "flag",
    "fid",
    "serviceid",
    "templateid",
    "actiondata",
    "action_data",
    # 元数据块
    "config",
    "extra",
    "preview",
    "icon",
    "scene",
    "host",
}


def _is_url(s: str) -> bool:
    """简单判断是否为 URL"""
    return s.startswith(("http://", "https://"))


def _clean_url(url_str: str) -> str:
    """剥离 URL 中的追踪参数"""
    if not _is_url(url_str):
        return url_str

    try:
        parsed = urlparse(url_str)
        if not parsed.query:
            return url_str

        query_params = parse_qsl(parsed.query)

        cleaned_params = [
            (k, v)
            for k, v in query_params
            if k.lower() not in _TRACKING_PARAMS_KEYS
            and not k.lower().startswith("utm_")
        ]

        new_query = urlencode(cleaned_params)

        return urlunparse(parsed._replace(query=new_query))
    except (ValueError, UnicodeError):
        return url_str


def _is_noisy_string(s: str) -> bool:
    """判断一个字符串是否为噪音"""
    stripped = s.strip()
    if not stripped:
        return False
    if _is_url(stripped):
        return False
    return any(pat.match(stripped) for pat in _NOISE_PATTERNS)


def _is_noisy_key(key: str) -> bool:
    """判断 JSON key / XML tag / XML 属性名是否为噪音元数据 key

    同时对 key 本身跑噪音字符串检测,
    捕获无规律长数字 key 等边缘情况。
    """
    return key.lower() in _NOISY_KEYS or _is_noisy_string(key)


def _is_noisy_number(val: int | float) -> bool:
    """判断数值是否为噪音 (超长数字 / Unix 时间戳)"""
    abs_val = abs(int(val))
    digits = len(str(abs_val))
    # 超长数字 ID
    if digits >= 15:
        return True
    # 秒级 Unix 时间戳 (10 位, 1970~2100)
    if _TIMESTAMP_MIN <= abs_val <= _TIMESTAMP_MAX and digits == 10:
        return True
    # 毫秒级 Unix 时间戳 (13 位, 1970~2100)
    if _TIMESTAMP_MS_MIN <= abs_val <= _TIMESTAMP_MS_MAX and digits == 13:
        return True
    return False


def _strip_xml_ns(tag: str) -> str:
    """去除 XML 命名空间前缀 {http://...}localname -> localname"""
    if tag.startswith("{"):
        _, _, local = tag.partition("}")
        return local
    return tag


def _is_empty(val: Any) -> bool:
    """判断值在清洗语境下是否为空"""
    if val is None:
        return True
    if isinstance(val, str) and not val.strip():
        return True
    if isinstance(val, dict) and not val:
        return True
    if isinstance(val, list) and not val:
        return True
    return False


def _flatten_single_key_dicts(data: Any) -> Any:
    """将只有单个 key 的嵌套 dict 向上平铺, 用下划线拼接 key 路径

    例: {"meta": {"news": {"title": "x", "desc": "y"}}}
      -> {"news_title": "x", "news_desc": "y"}
    """
    if isinstance(data, dict):
        data = {k: _flatten_single_key_dicts(v) for k, v in data.items()}
        while len(data) == 1:
            only_key = next(iter(data))
            inner = data[only_key]
            if not isinstance(inner, dict):
                break
            data = {f"{only_key}_{ik}": iv for ik, iv in inner.items()}
        return data
    if isinstance(data, list):
        return [_flatten_single_key_dicts(item) for item in data]
    return data


def _clean_json_value(val: Any) -> Any:
    """递归清洗 JSON 值，返回清洗后的值，噪音返回 None"""
    if isinstance(val, dict):
        cleaned: dict[str, Any] = {}
        for k, v in val.items():
            if _is_noisy_key(k):
                continue
            cv = _clean_json_value(v)
            if not _is_empty(cv):
                cleaned[k] = cv
        return cleaned

    elif isinstance(val, list):
        result: list[Any] = []
        for item in val:
            ci = _clean_json_value(item)
            if not _is_empty(ci):
                result.append(ci)
        return result

    elif isinstance(val, str):
        if _is_url(val):
            return _clean_url(val)
        if _is_noisy_string(val):
            return None
        return val

    elif isinstance(val, bool):
        return None

    elif isinstance(val, (int, float)):
        if _is_noisy_number(val):
            return None
        return val

    return val


def _clean_xml_element(elem: etree._Element) -> bool:
    """递归清洗 XML 元素，返回 True 表示该元素有保留价值"""
    noisy_attr_keys: list[str] = []
    for attr_key, attr_val in elem.attrib.items():
        local_key = _strip_xml_ns(attr_key)
        if _is_url(attr_val):
            elem.attrib[attr_key] = _clean_url(attr_val)
        elif _is_noisy_key(local_key) or _is_noisy_string(attr_val):
            noisy_attr_keys.append(attr_key)

    for key in noisy_attr_keys:
        del elem.attrib[key]

    if elem.text:
        if _is_url(elem.text.strip()):
            elem.text = _clean_url(elem.text.strip())
        elif _is_noisy_string(elem.text.strip()):
            elem.text = None

    if elem.tail:
        tail_stripped = elem.tail.strip()
        if not tail_stripped:
            elem.tail = None
        elif _is_url(tail_stripped):
            elem.tail = _clean_url(tail_stripped)
        elif _is_noisy_string(tail_stripped):
            elem.tail = None

    children_to_remove: list[etree._Element] = []
    for child in elem:
        local_tag = _strip_xml_ns(child.tag) if isinstance(child.tag, str) else ""
        if _is_noisy_key(local_tag):
            children_to_remove.append(child)
            continue
        has_value = _clean_xml_element(child)
        if not has_value:
            children_to_remove.append(child)
    for child in children_to_remove:
        elem.remove(child)

    has_text = bool(elem.text and elem.text.strip())
    has_tail = bool(elem.tail and elem.tail.strip())
    has_attribs = bool(elem.attrib)
    has_children = len(elem) > 0
    return has_text or has_tail or has_attribs or has_children


def clean_hyper_content(raw: str | None, fmt: Literal["xml", "json"]) -> str:
    """清洗 Hyper 消息的原始内容

    Args:
        raw: 原始 XML 或 JSON 字符串
        fmt: 格式类型 ("xml" 或 "json")

    Returns:
        清洗后的字符串，保留原始 format 格式。
        若解析失败，返回 JSON 格式的错误描述。
    """
    if not raw or not raw.strip():
        return _make_parse_failed()

    if fmt == "json":
        return _clean_json_content(raw)
    elif fmt == "xml":
        return _clean_xml_content(raw)


def _clean_json_content(raw: str) -> str:
    """清洗 JSON 格式的 Hyper 内容"""
    try:
        data = orjson.loads(raw)
    except (orjson.JSONDecodeError, ValueError, UnicodeDecodeError):
        return _make_parse_failed()

    cleaned = _clean_json_value(data)

    if _is_empty(cleaned):
        return orjson.dumps({"notice": "cleaned_empty"}).decode("utf-8")

    cleaned = _flatten_single_key_dicts(cleaned)

    return orjson.dumps(cleaned).decode("utf-8")


def _clean_xml_content(raw: str) -> str:
    """清洗 XML 格式的 Hyper 内容"""
    try:
        parser = etree.XMLParser(recover=True)
        root = etree.fromstring(raw.encode("utf-8"), parser=parser)
    except (etree.XMLSyntaxError, UnicodeDecodeError, ValueError):
        return _make_parse_failed()

    if root is None:
        return _make_parse_failed()

    has_value = _clean_xml_element(root)
    if not has_value:
        return f"<{root.tag}/>"

    return etree.tostring(root, pretty_print=True, encoding="unicode").strip()


def _make_parse_failed() -> str:
    """生成解析失败的 JSON 描述"""
    return orjson.dumps(
        {
            "error": "parse_failed",
        }
    ).decode("utf-8")
