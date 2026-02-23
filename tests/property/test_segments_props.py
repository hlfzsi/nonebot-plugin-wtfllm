"""Segment 类型的属性测试"""

import time

import pytest
from hypothesis import given, assume
from hypothesis import strategies as st

from .conftest import (
    text_segments,
    emoji_segments,
    mention_segments,
    image_segments,
    video_segments,
    file_segments,
    audio_segments,
    unknown_segments,
    hyper_segments,
    leaf_segments,
    non_empty_text,
    urls,
    timestamps,
)


# ===== unique_key 稳定性 =====


@pytest.mark.property
class TestSegmentUniqueKeyStability:
    """unique_key 必须是确定性的：相同数据 -> 相同 key"""

    @given(seg=leaf_segments)
    def test_unique_key_deterministic(self, seg):
        """调用两次 unique_key 返回相同字符串"""
        assert seg.unique_key == seg.unique_key

    @given(seg=leaf_segments)
    def test_unique_key_is_nonempty_string(self, seg):
        """unique_key 不为空"""
        assert isinstance(seg.unique_key, str)
        assert len(seg.unique_key) > 0


# ===== __eq__ / __hash__ 一致性 =====


@pytest.mark.property
class TestSegmentEqHashConsistency:
    """如果 a == b，那么 hash(a) == hash(b)"""

    @given(seg=text_segments())
    def test_text_eq_hash(self, seg):
        from nonebot_plugin_wtfllm.memory.content.segments import TextSegment

        copy = TextSegment.model_validate(seg.model_dump())
        assert seg == copy
        assert hash(seg) == hash(copy)

    @given(seg=emoji_segments())
    def test_emoji_eq_hash(self, seg):
        from nonebot_plugin_wtfllm.memory.content.segments import EmojiSegment

        copy = EmojiSegment.model_validate(seg.model_dump())
        assert seg == copy
        assert hash(seg) == hash(copy)

    @given(seg=mention_segments())
    def test_mention_eq_hash(self, seg):
        from nonebot_plugin_wtfllm.memory.content.segments import MentionSegment

        copy = MentionSegment.model_validate(seg.model_dump())
        assert seg == copy
        assert hash(seg) == hash(copy)

    @given(seg=image_segments())
    def test_image_eq_hash(self, seg):
        from nonebot_plugin_wtfllm.memory.content.segments import ImageSegment

        copy = ImageSegment.model_validate(seg.model_dump())
        assert seg == copy
        assert hash(seg) == hash(copy)

    @given(seg=file_segments())
    def test_file_eq_hash(self, seg):
        from nonebot_plugin_wtfllm.memory.content.segments import FileSegment

        copy = FileSegment.model_validate(seg.model_dump())
        assert seg == copy
        assert hash(seg) == hash(copy)

    @given(seg=audio_segments())
    def test_audio_eq_hash(self, seg):
        from nonebot_plugin_wtfllm.memory.content.segments import AudioSegment

        copy = AudioSegment.model_validate(seg.model_dump())
        assert seg == copy
        assert hash(seg) == hash(copy)

    @given(seg=unknown_segments())
    def test_unknown_eq_hash(self, seg):
        from nonebot_plugin_wtfllm.memory.content.segments import UnknownSegment

        copy = UnknownSegment.model_validate(seg.model_dump())
        assert seg == copy
        assert hash(seg) == hash(copy)

    @given(seg=hyper_segments())
    def test_hyper_eq_hash(self, seg):
        from nonebot_plugin_wtfllm.memory.content.segments import HyperSegment

        copy = HyperSegment.model_validate(seg.model_dump())
        assert seg == copy
        assert hash(seg) == hash(copy)

    @given(seg=leaf_segments)
    def test_eq_reflexive(self, seg):
        """a == a 必须为 True"""
        assert seg == seg

    @given(seg=leaf_segments)
    def test_eq_different_type_is_false(self, seg):
        """不同 Python 类型比较返回 False"""
        assert seg != "not a segment"
        assert seg != 42


# ===== Pydantic 序列化往返 =====


@pytest.mark.property
class TestSegmentSerializationRoundtrip:
    """model_dump -> model_validate 往返保持等价"""

    @given(seg=text_segments())
    def test_text_roundtrip(self, seg):
        from nonebot_plugin_wtfllm.memory.content.segments import TextSegment

        data = seg.model_dump()
        restored = TextSegment.model_validate(data)
        assert seg == restored
        assert seg.content == restored.content
        assert seg.created_at == restored.created_at

    @given(seg=emoji_segments())
    def test_emoji_roundtrip(self, seg):
        from nonebot_plugin_wtfllm.memory.content.segments import EmojiSegment

        data = seg.model_dump()
        restored = EmojiSegment.model_validate(data)
        assert seg == restored

    @given(seg=mention_segments())
    def test_mention_roundtrip(self, seg):
        from nonebot_plugin_wtfllm.memory.content.segments import MentionSegment

        data = seg.model_dump()
        restored = MentionSegment.model_validate(data)
        assert seg == restored

    @given(seg=video_segments())
    def test_video_roundtrip(self, seg):
        from nonebot_plugin_wtfllm.memory.content.segments import VideoSegment

        data = seg.model_dump()
        restored = VideoSegment.model_validate(data)
        assert seg == restored

    @given(seg=unknown_segments())
    def test_unknown_roundtrip(self, seg):
        from nonebot_plugin_wtfllm.memory.content.segments import UnknownSegment

        data = seg.model_dump()
        restored = UnknownSegment.model_validate(data)
        assert seg == restored

    @given(seg=hyper_segments())
    def test_hyper_roundtrip(self, seg):
        from nonebot_plugin_wtfllm.memory.content.segments import HyperSegment

        data = seg.model_dump()
        restored = HyperSegment.model_validate(data)
        assert seg == restored


# ===== created_at 默认值 =====


@pytest.mark.property
class TestSegmentCreatedAtDefault:
    """created_at 应默认为当前时间"""

    @given(content=non_empty_text)
    def test_created_at_is_current(self, content):
        from nonebot_plugin_wtfllm.memory.content.segments import TextSegment

        before = int(time.time())
        seg = TextSegment(content=content)
        after = int(time.time())
        assert before <= seg.created_at <= after


# ===== MediaBaseSegment 校验 =====


@pytest.mark.property
class TestMediaBaseSegmentValidation:
    """MediaBaseSegment 必须有 url 或 local_path 或 expired=True"""

    def test_neither_url_nor_path_raises(self):
        from nonebot_plugin_wtfllm.memory.content.segments import ImageSegment

        with pytest.raises(ValueError):
            ImageSegment()

    @given(url=urls)
    def test_url_only_ok(self, url):
        from nonebot_plugin_wtfllm.memory.content.segments import ImageSegment

        seg = ImageSegment(url=url)
        assert seg.url == url

    def test_expired_only_ok(self):
        from nonebot_plugin_wtfllm.memory.content.segments import ImageSegment

        seg = ImageSegment(expired=True)
        assert seg.expired is True
        assert seg.available is False


# ===== MentionSegment 校验 =====


@pytest.mark.property
class TestMentionSegmentValidation:
    """MentionSegment 必须恰好设置 user_id 或 at_all 其一"""

    def test_neither_raises(self):
        from nonebot_plugin_wtfllm.memory.content.segments import MentionSegment

        with pytest.raises(ValueError):
            MentionSegment()

    def test_both_raises(self):
        from nonebot_plugin_wtfllm.memory.content.segments import MentionSegment

        with pytest.raises(ValueError):
            MentionSegment(user_id="u1", at_all=True)

    @given(uid=st.text(min_size=1, max_size=20))
    def test_user_id_only_ok(self, uid):
        from nonebot_plugin_wtfllm.memory.content.segments import MentionSegment

        seg = MentionSegment(user_id=uid)
        assert seg.user_id == uid
        assert seg.at_all is False

    def test_at_all_only_ok(self):
        from nonebot_plugin_wtfllm.memory.content.segments import MentionSegment

        seg = MentionSegment(at_all=True)
        assert seg.at_all is True
        assert seg.user_id is None
