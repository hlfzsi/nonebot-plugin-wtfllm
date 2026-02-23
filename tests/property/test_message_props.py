"""Message 容器的属性测试"""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from .conftest import text_segments, leaf_segments, messages, timestamps


# ===== len(Message) =====


@pytest.mark.property
class TestMessageLen:
    """len(Message) 计算 leaf segment 数量"""

    @given(segs=st.lists(text_segments(), min_size=0, max_size=20))
    def test_len_equals_segment_count_for_leaf_only(self, segs):
        from nonebot_plugin_wtfllm.memory.content.message import Message

        msg = Message.create(segs)
        assert len(msg) == len(segs)


# ===== created_at =====


@pytest.mark.property
class TestMessageCreatedAt:

    @given(msg=messages(min_size=1, max_size=10))
    def test_created_at_is_min_of_segments(self, msg):
        expected = min(seg.created_at for seg in msg.segments)
        assert msg.created_at == expected

    def test_empty_message_created_at_zero(self):
        from nonebot_plugin_wtfllm.memory.content.message import Message

        assert Message.create().created_at == 0


# ===== __add__ =====


@pytest.mark.property
class TestMessageAdd:

    @given(
        segs1=st.lists(text_segments(), min_size=0, max_size=5),
        segs2=st.lists(text_segments(), min_size=0, max_size=5),
    )
    def test_add_preserves_segment_count(self, segs1, segs2):
        from nonebot_plugin_wtfllm.memory.content.message import Message

        msg1 = Message.create(segs1)
        msg2 = Message.create(segs2)
        result = msg1 + msg2
        assert len(result.segments) == len(segs1) + len(segs2)

    @given(
        segs1=st.lists(text_segments(), min_size=1, max_size=5),
        segs2=st.lists(text_segments(), min_size=1, max_size=5),
    )
    def test_add_does_not_mutate_originals(self, segs1, segs2):
        from nonebot_plugin_wtfllm.memory.content.message import Message

        msg1 = Message.create(segs1)
        msg2 = Message.create(segs2)
        orig_len1 = len(msg1.segments)
        orig_len2 = len(msg2.segments)
        _ = msg1 + msg2
        assert len(msg1.segments) == orig_len1
        assert len(msg2.segments) == orig_len2


# ===== message_count =====


@pytest.mark.property
class TestMessageMessageCount:

    @given(segs=st.lists(text_segments(), min_size=0, max_size=10))
    def test_message_count_is_1_without_forward(self, segs):
        from nonebot_plugin_wtfllm.memory.content.message import Message

        msg = Message.create(segs)
        assert msg.message_count == 1


# ===== get_plain_text =====


@pytest.mark.property
class TestMessageGetPlainText:

    @given(
        texts=st.lists(
            st.text(min_size=1, max_size=50),
            min_size=1,
            max_size=5,
        )
    )
    def test_plain_text_concatenates_text_segments(self, texts):
        from nonebot_plugin_wtfllm.memory.content.message import Message

        msg = Message.create()
        for t in texts:
            msg.text(t)
        assert msg.get_plain_text() == "".join(texts)
