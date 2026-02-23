"""属性测试的 Hypothesis 自定义策略和配置"""

import string

from hypothesis import settings, HealthCheck
from hypothesis import strategies as st

# ===== Hypothesis profiles =====

settings.register_profile(
    "ci", max_examples=50, deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "dev", max_examples=20, deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "thorough", max_examples=200, deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("dev")

# ===== 原子值策略 =====

timestamps = st.integers(min_value=1_000_000_000, max_value=2_000_000_000)

non_empty_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S", "Z")),
    min_size=1,
    max_size=500,
)

safe_text = st.text(
    alphabet=string.printable,
    min_size=0,
    max_size=1000,
)

user_ids = st.from_regex(r"user_[a-z0-9]{3,10}", fullmatch=True)
group_ids = st.from_regex(r"group_[a-z0-9]{3,10}", fullmatch=True)
agent_ids = st.from_regex(r"agent_[a-z0-9]{3,10}", fullmatch=True)
message_ids = st.from_regex(r"msg_[a-z0-9]{5,15}", fullmatch=True)
urls = st.from_regex(r"https?://example\.com/[a-z0-9]{1,20}", fullmatch=True)


# ===== Segment 策略 =====


@st.composite
def text_segments(draw):
    from nonebot_plugin_wtfllm.memory.content.segments import TextSegment

    content = draw(non_empty_text)
    ts = draw(timestamps)
    return TextSegment(content=content, created_at=ts)


@st.composite
def emoji_segments(draw):
    from nonebot_plugin_wtfllm.memory.content.segments import EmojiSegment

    name = draw(st.text(min_size=1, max_size=20, alphabet=string.ascii_letters))
    url = draw(st.one_of(st.none(), urls))
    ts = draw(timestamps)
    return EmojiSegment(name=name, url=url, created_at=ts)


@st.composite
def mention_segments(draw):
    from nonebot_plugin_wtfllm.memory.content.segments import MentionSegment

    is_at_all = draw(st.booleans())
    ts = draw(timestamps)
    if is_at_all:
        return MentionSegment(at_all=True, created_at=ts)
    else:
        uid = draw(user_ids)
        return MentionSegment(user_id=uid, created_at=ts)


@st.composite
def image_segments(draw):
    from nonebot_plugin_wtfllm.memory.content.segments import ImageSegment

    url = draw(urls)
    ts = draw(timestamps)
    return ImageSegment(url=url, created_at=ts)


@st.composite
def video_segments(draw):
    from nonebot_plugin_wtfllm.memory.content.segments import VideoSegment

    url = draw(urls)
    ts = draw(timestamps)
    duration = draw(st.one_of(st.none(), st.integers(min_value=1, max_value=7200)))
    return VideoSegment(url=url, created_at=ts, duration=duration)


@st.composite
def file_segments(draw):
    from nonebot_plugin_wtfllm.memory.content.segments import FileSegment

    url = draw(urls)
    ts = draw(timestamps)
    filename = draw(
        st.from_regex(r"[a-z]{1,10}\.(pdf|txt|csv|doc)", fullmatch=True)
    )
    return FileSegment(url=url, filename=filename, created_at=ts)


@st.composite
def audio_segments(draw):
    from nonebot_plugin_wtfllm.memory.content.segments import AudioSegment

    url = draw(urls)
    ts = draw(timestamps)
    duration = draw(st.one_of(st.none(), st.integers(min_value=1, max_value=3600)))
    return AudioSegment(url=url, created_at=ts, duration=duration)


@st.composite
def unknown_segments(draw):
    from nonebot_plugin_wtfllm.memory.content.segments import UnknownSegment

    ts = draw(timestamps)
    orig = draw(st.text(min_size=1, max_size=30, alphabet=string.ascii_letters))
    return UnknownSegment(original_type=orig, created_at=ts)


@st.composite
def hyper_segments(draw):
    from nonebot_plugin_wtfllm.memory.content.segments import HyperSegment

    fmt = draw(st.sampled_from(["xml", "json"]))
    if fmt == "json":
        content = draw(
            st.from_regex(
                r'\{"[a-z]+":"[a-z]+"(,"[a-z]+":"[a-z]+")*\}', fullmatch=True
            )
        )
    else:
        content = draw(
            st.from_regex(
                r"<root><[a-z]+>[a-z]+</[a-z]+></root>", fullmatch=True
            )
        )
    ts = draw(timestamps)
    return HyperSegment(format=fmt, content=content, created_at=ts)


# 所有 leaf segment（不含 ForwardSegment，避免递归）
leaf_segments = st.one_of(
    text_segments(),
    emoji_segments(),
    mention_segments(),
    image_segments(),
    file_segments(),
    audio_segments(),
    unknown_segments(),
    hyper_segments(),
)

# 仅媒体 segment
media_segments = st.one_of(
    image_segments(),
    video_segments(),
    file_segments(),
    audio_segments(),
)


# ===== Message 策略 =====


@st.composite
def messages(draw, min_size=1, max_size=8):
    """生成包含 leaf segments 的 Message"""
    from nonebot_plugin_wtfllm.memory.content.message import Message

    segs = draw(st.lists(leaf_segments, min_size=min_size, max_size=max_size))
    return Message.create(segs)
