"""话题簇归档

将过期/淘汰的簇归档到 Qdrant 长期记忆。
"""

import asyncio
import time
from uuid import uuid4

from ..clustering.mmr import mmr_select
from .._types import ArchivalCandidate
from ...config import APP_CONFIG
from ...db import memory_item_repo
from ...v_db.models.topic_archive import TopicArchivePayload
from ...v_db import topic_archive_repo
from ...vec import VECTORIZER
from ...utils import logger

# 过滤后，最少需要的独立代表消息数；低于此值放弃归档
_MIN_UNIQUE_REPRESENTATIVES: int = 3


def _deduplicate_texts(
    indices: list[int],
    texts: list[str],
) -> tuple[list[int], list[str]]:
    """对 (indices, texts) 按 strip() 后文本去重，保留首次出现。"""
    seen: set[str] = set()
    dedup_indices: list[int] = []
    dedup_texts: list[str] = []
    for idx, text in zip(indices, texts):
        key = text.strip()
        if key not in seen:
            seen.add(key)
            dedup_indices.append(idx)
            dedup_texts.append(text)
    return dedup_indices, dedup_texts


def _remove_substrings(
    selected_indices: list[int],
    texts: list[str],
) -> tuple[list[int], list[str]]:
    """移除被其他更长文本包含的子串文本。

    Returns:
        过滤后的 (indices, texts)，顺序按长度降序。
    """
    paired = sorted(
        zip(selected_indices, texts),
        key=lambda p: len(p[1].strip()),
        reverse=True,
    )

    kept_indices: list[int] = []
    kept_texts: list[str] = []
    kept_stripped: list[str] = []

    for idx, text in paired:
        stripped = text.strip()
        if any(stripped in existing for existing in kept_stripped):
            continue
        kept_indices.append(idx)
        kept_texts.append(text)
        kept_stripped.append(stripped)

    return kept_indices, kept_texts


async def archive_cluster(
    candidate: ArchivalCandidate,
) -> None:
    """归档单个簇到 Qdrant 长期记忆。"""
    msg_ids = [mid for mid, _ts in candidate.cluster.message_entries]
    if not msg_ids:
        return

    items = await memory_item_repo.get_many_by_message_ids(msg_ids)
    if len(items) < APP_CONFIG.topic_archive_min_messages:
        logger.debug(
            f"簇归档跳过: 实际获取到 {len(items)} 条消息 "
            f"< 最低要求 {APP_CONFIG.topic_archive_min_messages}"
        )
        return

    texts = [item.get_plain_text() for item in items]
    valid = [(i, t) for i, t in enumerate(texts) if t.strip()]
    if len(valid) < 3:
        return

    valid_indices = [i for i, _ in valid]
    valid_texts = [t for _, t in valid]

    valid_indices, valid_texts = _deduplicate_texts(valid_indices, valid_texts)
    if len(valid_texts) < _MIN_UNIQUE_REPRESENTATIVES:
        logger.debug(
            f"簇归档跳过: 去重后仅 {len(valid_texts)} 条独立文本 "
            f"< 最低要求 {_MIN_UNIQUE_REPRESENTATIVES}"
        )
        return

    vectors = await asyncio.to_thread(VECTORIZER.transform_batch, valid_texts)

    centroid = candidate.centroid.flatten()
    k = min(APP_CONFIG.topic_archive_mmr_k, len(valid_indices))
    selected_local = mmr_select(
        vectors,
        centroid,
        k=k,
        lambda_param=APP_CONFIG.topic_archive_mmr_lambda,
    )

    representative_texts = [valid_texts[i] for i in selected_local]
    filtered_indices, representative_texts = _remove_substrings(
        selected_local, representative_texts
    )
    if len(representative_texts) < _MIN_UNIQUE_REPRESENTATIVES:
        logger.debug(
            f"簇归档跳过: 子串过滤后仅 {len(representative_texts)} 条独立代表消息 "
            f"< 最低要求 {_MIN_UNIQUE_REPRESENTATIVES}"
        )
        return

    # 拼接代表消息文本，作为 fastembed 嵌入的输入文档
    document = "\n".join(representative_texts)

    payload = TopicArchivePayload(
        archive_id=str(uuid4()),
        agent_id=candidate.session_key.agent_id,
        group_id=candidate.session_key.group_id,
        user_id=candidate.session_key.user_id,
        representative_message_ids=[
            str(items[valid_indices[i]].message_id) for i in filtered_indices
        ],
        message_count=candidate.cluster.message_count,
        created_at=int(time.time()),
    )

    await topic_archive_repo.upsert(payload, document=document)
    logger.debug(
        f"簇归档完成: agent={candidate.session_key.agent_id} "
        f"group={candidate.session_key.group_id} "
        f"messages={candidate.cluster.message_count} "
        f"selected={len(representative_texts)}"
    )
