"""全局扫描入口"""

import asyncio
from time import time
from datetime import datetime

import numpy as np

from ...db import memory_item_repo
from ...utils import APP_CONFIG,logger
from ._types import SessionKey, ActivityCurve
from .prefilter import prefilter
from .curve import fit_activity_curve

GROUP_BATCH_SIZE = 3


def _bucket_range(minute: int, minute_bucket: int) -> tuple[int, int]:
    if minute_bucket <= 1:
        return minute, minute + 1
    start = (minute // minute_bucket) * minute_bucket
    end = min(start + minute_bucket, 1440)
    return start, end


def _process_rows(
    rows: list[tuple[str | None, str | None, str, int, int]],
    min_repeat_days: int,
    minute_bucket: int,
    quantile_lower: float,
    quantile_upper: float,
) -> list[ActivityCurve]:
    """将 DB 行分组 → 预筛选 → 拟合活跃曲线。"""
    grouped: dict[SessionKey, np.ndarray] = {}
    active_days_map: dict[SessionKey, int] = {}

    for group_id, user_id, sender, minute, active_days in rows:
        key = SessionKey(group_id=group_id, user_id=user_id, sender=sender)
        if key not in grouped:
            grouped[key] = np.zeros(1440, dtype=np.float32)
            active_days_map[key] = 0

        start, end = _bucket_range(minute, minute_bucket)
        grouped[key][start:end] = np.maximum(grouped[key][start:end], active_days)

        if active_days > active_days_map[key]:
            active_days_map[key] = active_days

    results: list[ActivityCurve] = []
    for session, bins in grouped.items():
        max_active_days = active_days_map[session]

        # 预筛选
        passed, quality = prefilter(bins, max_active_days, min_repeat_days)
        if not passed:
            continue

        # 曲线拟合
        curve = fit_activity_curve(
            session=session,
            bins=bins,
            data_quality=quality,
            quantile_lower=quantile_lower,
            quantile_upper=quantile_upper,
        )
        if curve is not None:
            results.append(curve)

    return results


async def scan_all_curves(agent_id: str) -> list[ActivityCurve]:
    """全局扫描所有会话，返回通过预筛选的用户活跃曲线列表。"""
    total_days = APP_CONFIG.inertia_observation_days
    min_repeat_days = APP_CONFIG.inertia_min_active_days
    minute_bucket = APP_CONFIG.inertia_minute_bucket

    quantile_lower = APP_CONFIG.inertia_quantile_lower
    quantile_upper = APP_CONFIG.inertia_quantile_upper
    since = int(time()) - total_days * 86400

    utc_offset_obj = datetime.now().astimezone().utcoffset()
    utc_offset = int(utc_offset_obj.total_seconds()) if utc_offset_obj else 0

    results: list[ActivityCurve] = []

    # 分批处理群聊
    group_ids = await memory_item_repo.get_active_group_ids(agent_id, since=since)

    for i in range(0, len(group_ids), GROUP_BATCH_SIZE):
        batch = group_ids[i : i + GROUP_BATCH_SIZE]
        rows = await memory_item_repo.get_group_activity_bins(
            agent_id,
            group_ids=batch,
            since=since,
            utc_offset=utc_offset,
            min_repeat_days=min_repeat_days,
        )
        result = await asyncio.to_thread(
            _process_rows,
            rows,
            min_repeat_days,
            minute_bucket,
            quantile_lower,
            quantile_upper,
        )
        results.extend(result)

    # 处理私聊
    private_rows = await memory_item_repo.get_private_activity_bins(
        agent_id,
        since=since,
        utc_offset=utc_offset,
        min_repeat_days=min_repeat_days,
    )
    result = await asyncio.to_thread(
        _process_rows,
        private_rows,
        min_repeat_days,
        minute_bucket,
        quantile_lower,
        quantile_upper,
    )
    results.extend(result)
    
    logger.debug(f"扫描完成：共 {len(results)} 条活跃曲线通过预筛选")
    logger.debug(f"样例曲线：{results[:3]}")

    return results
