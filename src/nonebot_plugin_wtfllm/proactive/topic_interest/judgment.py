import asyncio
import re
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .store import topic_interest_store
from ...vec import TopicVectorizer, VECTORIZER
from ...utils import logger


_WHITESPACE_RE = re.compile(r"\s+")
_SIMILARITY_THRESHOLD = 0.45


def _normalize_text(text: str) -> str:
    collapsed = _WHITESPACE_RE.sub(" ", text).strip().lower()
    return collapsed


def _normalize_topics(topics: Sequence[str] | None) -> list[str]:
    if not topics:
        return []
    return [normalized for topic in topics if (normalized := _normalize_text(topic))]


def _has_direct_match(plain_text: str, interested_topics: Sequence[str]) -> bool:
    for topic in interested_topics:
        if topic in plain_text or plain_text in topic:
            return True
    return False


async def _compute_max_similarity(
    plain_text: str,
    interested_topics: Sequence[str],
    vectorizer: TopicVectorizer,
) -> float:
    query_vector, topic_vectors = await asyncio.gather(
        asyncio.to_thread(vectorizer.transform, plain_text),
        asyncio.to_thread(vectorizer.transform_batch, list(interested_topics)),
    )
    similarities: NDArray[np.floating] = topic_vectors @ query_vector.T
    return float(np.max(similarities))


async def has_topic_interest_match(
    *,
    plain_text: str,
    interested_topics: Sequence[str] | None,
    similarity_threshold: float = _SIMILARITY_THRESHOLD,
    vectorizer: TopicVectorizer | None = None,
) -> bool:
    normalized_text = _normalize_text(plain_text)
    normalized_topics = _normalize_topics(interested_topics)

    if not normalized_text or not normalized_topics:
        return False

    if _has_direct_match(normalized_text, normalized_topics):
        return True

    active_vectorizer = vectorizer
    if active_vectorizer is None:
        active_vectorizer = VECTORIZER

    max_similarity = await _compute_max_similarity(
        normalized_text,
        normalized_topics,
        active_vectorizer,
    )
    return max_similarity >= similarity_threshold


async def has_active_topic_interest_match(
    *,
    agent_id: str,
    user_id: str,
    group_id: str | None,
    plain_text: str,
) -> bool:
    interest = topic_interest_store.get_interest(
        agent_id=agent_id,
        user_id=user_id,
        group_id=group_id,
    )
    if interest is None:
        return False
    
    result = await has_topic_interest_match(
        plain_text=plain_text,
        interested_topics=interest.topics,
    )
    
    logger.debug(
        f"Checking topic interest match for user {user_id} with topics {interest.topics}. "
        f"Plain text: '{plain_text}'. Match result: {result}"
    )

    return result
