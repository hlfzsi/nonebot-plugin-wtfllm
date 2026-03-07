import numpy as np
import pytest

from nonebot_plugin_wtfllm.proactive import should_proactively_respond
from nonebot_plugin_wtfllm.proactive.topic_interest.judgment import (
    _compute_max_similarity,
    has_topic_interest_match,
)
from nonebot_plugin_wtfllm.vec import VECTORIZER
from nonebot_plugin_wtfllm.proactive.topic_interest.store import topic_interest_store


class _FakeVectorizer:
    def transform(self, text: str):
        if text == "火锅聚餐":
            return np.array([[1.0, 0.0]])
        return np.array([[0.0, 1.0]])

    def transform_batch(self, texts: list[str]):
        return np.array(
            [
                [1.0, 0.0] if text == "周末吃什么" else [0.0, 1.0]
                for text in texts
            ]
        )


@pytest.fixture(autouse=True)
def _clean_topic_interest_store():
    topic_interest_store.clear_all()
    yield
    topic_interest_store.clear_all()


class TestTopicInterestStore:
    def test_set_get_and_clear_topics(self):
        topic_interest_store.set_topics(
            agent_id="a1",
            user_id="u1",
            group_id="g1",
            topics=["天气", "天气", "  ", "周末"],
        )

        interest = topic_interest_store.get_interest(
            agent_id="a1",
            user_id="u1",
            group_id="g1",
        )

        assert interest is not None
        assert interest.topics == ("天气", "周末")

        topic_interest_store.clear_topics(
            agent_id="a1",
            user_id="u1",
            group_id="g1",
        )
        assert (
            topic_interest_store.get_interest(
                agent_id="a1",
                user_id="u1",
                group_id="g1",
            )
            is None
        )

    def test_group_interest_is_shared_by_group_session(self):
        topic_interest_store.set_topics(
            agent_id="a1",
            user_id="u1",
            group_id="g1",
            topics=["天气"],
        )

        interest = topic_interest_store.get_interest(
            agent_id="a1",
            user_id="u2",
            group_id="g1",
        )

        assert interest is not None
        assert interest.topics == ("天气",)

    def test_private_interest_is_isolated_by_user(self):
        topic_interest_store.set_topics(
            agent_id="a1",
            user_id="u1",
            group_id=None,
            topics=["天气"],
        )

        interest = topic_interest_store.get_interest(
            agent_id="a1",
            user_id="u2",
            group_id=None,
        )

        assert interest is None

    def test_expired_interest_is_removed(self):
        topic_interest_store.set_topics(
            agent_id="a1",
            user_id="u1",
            group_id=None,
            topics=["天气"],
            ttl_seconds=0,
        )

        assert (
            topic_interest_store.get_interest(
                agent_id="a1",
                user_id="u1",
                group_id=None,
            )
            is None
        )


class TestTopicInterestJudgment:
    @pytest.mark.asyncio
    async def test_direct_match_returns_true(self):
        matched = await has_topic_interest_match(
            plain_text="我们周末一起看电影吧",
            interested_topics=["周末", "吃火锅"],
        )

        assert matched is True

    @pytest.mark.asyncio
    async def test_embedding_match_returns_true(self):
        matched = await has_topic_interest_match(
            plain_text="火锅聚餐",
            interested_topics=["周末吃什么"],
            similarity_threshold=0.7,
            vectorizer=_FakeVectorizer(),
        )

        assert matched is True

    @pytest.mark.asyncio
    async def test_empty_topics_returns_false(self):
        matched = await has_topic_interest_match(
            plain_text="随便聊聊",
            interested_topics=None,
        )

        assert matched is False

    @pytest.mark.asyncio
    async def test_short_keyword_topic_similarity_threshold_can_discriminate(self):
        interested_topics = ["奶茶"]
        related_message = "下班一起去买奶茶吧"
        unrelated_message = "明天接口联调后记得补单测"

        related_similarity = await _compute_max_similarity(
            related_message,
            interested_topics,
            VECTORIZER,
        )
        unrelated_similarity = await _compute_max_similarity(
            unrelated_message,
            interested_topics,
            VECTORIZER,
        )

        assert related_similarity > unrelated_similarity, (
            f"expected related similarity > unrelated similarity, "
            f"got related={related_similarity:.4f}, "
            f"unrelated={unrelated_similarity:.4f}"
        )

        threshold = (related_similarity + unrelated_similarity) / 2

        related_matched = await has_topic_interest_match(
            plain_text=related_message,
            interested_topics=interested_topics,
            similarity_threshold=threshold,
            vectorizer=VECTORIZER,
        )
        unrelated_matched = await has_topic_interest_match(
            plain_text=unrelated_message,
            interested_topics=interested_topics,
            similarity_threshold=threshold,
            vectorizer=VECTORIZER,
        )

        assert related_matched is True, (
            f"threshold={threshold:.4f}, related_similarity={related_similarity:.4f}"
        )
        assert unrelated_matched is False, (
            f"threshold={threshold:.4f}, unrelated_similarity={unrelated_similarity:.4f}"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("interested_topics", "plain_text", "expected_at_055", "expected_at_060"),
        [
            (["系解学习", "医学知识", "学习困难"], "系解都有哪些", True, True),
            (["系解学习", "医学知识", "学习困难"], "系解这门课怎么复习", True, True),
            (["作业问题", "学习困难"], "作业不会做", True, False),
            (["系解学习", "医学知识", "学习困难"], "你会什么", False, False),
            (["作业问题", "学习困难"], "今天晚上吃什么", False, False),
            (["系解学习", "医学知识", "学习困难"], "这个观点我不同意", False, False),
        ],
    )
    async def test_real_model_threshold_examples(
        self,
        interested_topics,
        plain_text,
        expected_at_055,
        expected_at_060,
    ):
        matched_at_055 = await has_topic_interest_match(
            plain_text=plain_text,
            interested_topics=interested_topics,
            similarity_threshold=0.55,
            vectorizer=VECTORIZER,
        )
        matched_at_060 = await has_topic_interest_match(
            plain_text=plain_text,
            interested_topics=interested_topics,
            similarity_threshold=0.60,
            vectorizer=VECTORIZER,
        )

        assert matched_at_055 is expected_at_055
        assert matched_at_060 is expected_at_060


class TestProactiveJudge:
    @pytest.mark.asyncio
    async def test_should_proactively_respond_uses_active_interest(self):
        topic_interest_store.set_topics(
            agent_id="a1",
            user_id="u1",
            group_id="g1",
            topics=["周末", "天气"],
        )

        should_reply = await should_proactively_respond(
            agent_id="a1",
            user_id="u1",
            group_id="g1",
            plain_text="这周末你有空吗",
        )

        assert should_reply is True

    @pytest.mark.asyncio
    async def test_should_proactively_respond_without_interest_returns_false(self):
        should_reply = await should_proactively_respond(
            agent_id="a1",
            user_id="u1",
            group_id="g1",
            plain_text="这周末你有空吗",
        )

        assert should_reply is False