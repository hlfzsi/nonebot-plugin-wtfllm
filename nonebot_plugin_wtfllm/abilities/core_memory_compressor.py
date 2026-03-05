"""核心记忆压缩能力 — 后台自动压缩超限的核心记忆"""

import asyncio
import time
from typing import List, Set, Tuple, cast

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIModelName
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from ..utils import APP_CONFIG, logger, count_tokens
from ..memory.items.core_memory import CoreMemory
from ..memory._types import ID_PATTERN
from ..v_db import core_memory_repo

_compressing_sessions: Set[Tuple[str, str | None, str | None]] = set()

_compress_agent: Agent | None = None


def _get_compress_agent() -> Agent:
    """延迟初始化压缩 Agent，避免模块导入时创建 OpenAI 连接"""
    global _compress_agent
    if _compress_agent is None:
        config = APP_CONFIG.compress_agent_model_config
        provider = OpenAIProvider(api_key=config.api_key, base_url=config.base_url)
        _model = OpenAIChatModel(
            model_name=cast(OpenAIModelName, config.name), provider=provider
        )
        _compress_agent = Agent(
            _model,
            output_type=str,
            model_settings=ModelSettings(
                temperature=0.3,
                extra_body={
                    **config.extra_body,
                },
            ),
        )
    return _compress_agent


def schedule_compress(
    agent_id: str,
    group_id: str | None,
    user_id: str | None,
) -> None:
    """如果当前会话未在压缩中，则调度一个后台压缩任务"""
    session_key = (agent_id, group_id, user_id)
    if session_key in _compressing_sessions:
        logger.debug(
            f"Compression already in progress for session {session_key}, skipping"
        )
        return

    task = asyncio.create_task(_do_compress(agent_id, group_id, user_id))
    task.add_done_callback(lambda _: _compressing_sessions.discard(session_key))
    _compressing_sessions.add(session_key)


async def _do_compress(
    agent_id: str,
    group_id: str | None,
    user_id: str | None,
) -> None:
    """后台执行的核心记忆压缩逻辑"""
    session_key = (agent_id, group_id, user_id)
    try:
        all_memories = await core_memory_repo.get_by_session(
            agent_id=agent_id,
            group_id=group_id,
            user_id=user_id,
        )

        total_tokens = sum(m.token_count for m in all_memories)

        if total_tokens <= APP_CONFIG.core_memory_max_tokens:
            return

        target_tokens = int(
            APP_CONFIG.core_memory_max_tokens * APP_CONFIG.core_memory_compress_ratio
        )

        sorted_memories = sorted(all_memories, key=lambda m: m.updated_at)

        to_compress: List[CoreMemory] = []
        compress_tokens = 0
        for m in sorted_memories:
            if total_tokens - compress_tokens <= target_tokens:
                break
            to_compress.append(m)
            compress_tokens += m.token_count

        if len(to_compress) < 2:
            return

        logger.info(
            f"[Background] Compressing {len(to_compress)} core memories "
            f"(tokens: {compress_tokens} -> target: {target_tokens})"
        )

        compressed = await _compress_memories(to_compress, agent_id, group_id, user_id)

        await core_memory_repo.delete_by_storage_ids(
            [m.storage_id for m in to_compress]
        )
        await core_memory_repo.save_many_core_memories(compressed)

        logger.info(
            f"[Background] Compression complete: {len(to_compress)} -> {len(compressed)} memories, "
            f"tokens: {compress_tokens} -> {sum(m.token_count for m in compressed)}"
        )
    except (ValueError, RuntimeError, OSError):
        logger.exception(
            f"[Background] Core memory compression failed for session {session_key}"
        )


async def _compress_memories(
    memories: List[CoreMemory],
    agent_id: str,
    group_id: str | None,
    user_id: str | None,
) -> List[CoreMemory]:
    """用 LLM 将多条旧记忆压缩为更少的精炼记忆"""

    memory_texts = "\n".join(f"- {m.content}" for m in memories)

    prompt = (
        "你是一个记忆压缩助手。以下是一组旧的核心记忆条目，请将它们合并精炼为更少的条目。\n"
        "要求：\n"
        "1. 保留所有重要信息，合并重复或相似的内容, 剔除过时的内容\n"
        "2. 保持 {{entity_id}} 格式的实体引用无论如何都不要改变\n"
        "3. 每条记忆仍应聚焦一个主题\n"
        "4. 用换行分隔每条新记忆，每行一条\n"
        "5. 直接输出精炼后的记忆，不要有多余说明\n\n"
        f"需要压缩的记忆：\n{memory_texts}"
    )

    result = await _get_compress_agent().run(prompt)
    compressed_lines = [
        line.strip().lstrip("- ").strip()
        for line in result.output.strip().split("\n")
        if line.strip()
    ]

    # 收集原始记忆中所有已知的 entity_id
    all_source_entities: set[str] = set()
    for m in memories:
        all_source_entities.update(m.related_entities)

    now = int(time.time())
    compressed_memories = []
    for line in compressed_lines:
        token_count = count_tokens(line)
        # 从压缩内容中提取 entity_id，仅保留原始记忆中已知的
        found_entities = ID_PATTERN.findall(line)
        related = [eid for eid in found_entities if eid in all_source_entities]
        memory = CoreMemory(
            content=line,
            group_id=group_id,
            user_id=user_id,
            agent_id=agent_id,
            created_at=now,
            updated_at=now,
            source="compression",
            token_count=token_count,
            related_entities=related,
        )
        compressed_memories.append(memory)

    logger.debug(result.usage())

    return compressed_memories
