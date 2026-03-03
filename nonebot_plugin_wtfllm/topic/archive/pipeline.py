"""话题簇归档

将过期/淘汰的簇归档到 Qdrant 长期记忆。
"""

import asyncio
import time
from uuid import uuid4

from ..clustering.vectorizer import vectorizer
from ..clustering.mmr import mmr_select
from .._types import ArchivalCandidate
from ...config import APP_CONFIG
from ...db import memory_item_repo
from ...v_db.models.topic_archive import TopicArchivePayload
from ...v_db import topic_archive_repo
from ...utils import logger


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
    # 过滤空文本
    valid = [(i, t) for i, t in enumerate(texts) if t.strip()]
    if len(valid) < 3:
        return

    valid_indices = [i for i, _ in valid]
    valid_texts = [t for _, t in valid]

    vectors = await asyncio.to_thread(vectorizer.transform_batch, valid_texts)

    centroid = candidate.centroid.flatten()
    k = min(APP_CONFIG.topic_archive_mmr_k, len(valid_indices))
    selected_local = mmr_select(
        vectors,
        centroid,
        k=k,
        lambda_param=APP_CONFIG.topic_archive_mmr_lambda,
    )

    # 拼接代表消息文本，作为 fastembed 嵌入的输入文档
    representative_texts = [valid_texts[i] for i in selected_local]
    document = "\n".join(representative_texts)

    payload = TopicArchivePayload(
        archive_id=str(uuid4()),
        agent_id=candidate.session_key.agent_id,
        group_id=candidate.session_key.group_id,
        user_id=candidate.session_key.user_id,
        representative_message_ids=[
            str(items[valid_indices[i]].message_id) for i in selected_local
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
