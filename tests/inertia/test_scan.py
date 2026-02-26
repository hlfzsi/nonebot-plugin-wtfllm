"""scan.py 端到端管线测试（mock DB 层）"""

from unittest.mock import AsyncMock, patch
import pytest

from nonebot_plugin_wtfllm.proactive.inertia._types import ActivityCurve
from nonebot_plugin_wtfllm.proactive.inertia.scan import (
    _process_rows,
    _bucket_range,
)


# ===================== _bucket_range =====================


class TestBucketRange:
    def test_no_bucketing(self):
        assert _bucket_range(100, 1) == (100, 101)

    def test_15_min_bucket(self):
        assert _bucket_range(100, 15) == (90, 105)

    def test_end_of_day(self):
        start, end = _bucket_range(1430, 15)
        assert end <= 1440


# ===================== _process_rows =====================


class TestProcessRows:
    def _make_peaked_rows(
        self,
        group_id: str = "g1",
        sender: str = "u1",
        peak_minutes: range = range(540, 600),
        active_days: int = 10,
    ) -> list[tuple[str | None, str | None, str, int, int]]:
        """创建集中在 peak_minutes 范围的 DB 行。"""
        return [(group_id, None, sender, m, active_days) for m in peak_minutes]

    def test_peaked_user_produces_curve(self):
        """有明显峰值的用户应通过预筛选并产出曲线。"""
        rows = self._make_peaked_rows(active_days=10)
        results = _process_rows(
            rows,
            min_repeat_days=7,
            minute_bucket=1,
            quantile_lower=0.05,
            quantile_upper=0.95,
        )
        assert len(results) >= 1
        assert isinstance(results[0], ActivityCurve)
        assert results[0].session.group_id == "g1"

    def test_insufficient_data_filtered(self):
        """活跃天数不足的用户应被预筛选过滤。"""
        rows = self._make_peaked_rows(active_days=3)
        results = _process_rows(
            rows,
            min_repeat_days=7,
            minute_bucket=1,
            quantile_lower=0.05,
            quantile_upper=0.95,
        )
        assert len(results) == 0

    def test_uniform_user_filtered(self):
        """全天均匀活跃用户应被过滤。"""
        rows = [("g1", None, "u1", m, 10) for m in range(0, 1440)]
        results = _process_rows(
            rows,  # pyright: ignore[reportArgumentType]
            min_repeat_days=7,
            minute_bucket=1,
            quantile_lower=0.05,
            quantile_upper=0.95,
        )
        assert len(results) == 0

    def test_multiple_users(self):
        """多个用户应各自独立处理。"""
        rows = self._make_peaked_rows(
            sender="u1", active_days=10
        ) + self._make_peaked_rows(
            sender="u2", active_days=12, peak_minutes=range(600, 660)
        )
        results = _process_rows(
            rows,
            min_repeat_days=7,
            minute_bucket=1,
            quantile_lower=0.05,
            quantile_upper=0.95,
        )
        sessions = {r.session.sender for r in results}
        assert "u1" in sessions
        assert "u2" in sessions

    def test_bucket_aggregation(self):
        """分钟桶聚合应正确工作。"""
        rows = self._make_peaked_rows(
            peak_minutes=range(540, 600),
            active_days=10,
        )
        results = _process_rows(
            rows,
            min_repeat_days=7,
            minute_bucket=15,
            quantile_lower=0.05,
            quantile_upper=0.95,
        )
        assert len(results) >= 1


# ===================== scan_all_curves (integration) =====================


class TestScanAllCurves:
    @pytest.mark.asyncio
    async def test_empty_groups(self):
        """没有活跃群组时应返回空列表。"""
        with (
            patch(
                "nonebot_plugin_wtfllm.proactive.inertia.scan.memory_item_repo"
            ) as mock_repo,
            patch(
                "nonebot_plugin_wtfllm.proactive.inertia.scan.APP_CONFIG"
            ) as mock_config,
        ):
            mock_config.inertia_observation_days = 30
            mock_config.inertia_min_active_days = 7
            mock_config.inertia_minute_bucket = 15
            mock_config.inertia_quantile_lower = 0.05
            mock_config.inertia_quantile_upper = 0.95
            mock_repo.get_active_group_ids = AsyncMock(return_value=[])
            mock_repo.get_private_activity_bins = AsyncMock(return_value=[])

            from nonebot_plugin_wtfllm.proactive.inertia.scan import scan_all_curves

            results = await scan_all_curves("test_agent")
            assert results == []
