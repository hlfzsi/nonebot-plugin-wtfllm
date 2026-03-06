__all__ = [
    "DateTriggerConfig",
    "IntervalTriggerConfig",
    "CronTriggerConfig",
    "TriggerConfig",
]

from datetime import datetime, timezone
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, Tag


class DateTriggerConfig(BaseModel):
    """单次触发，对应 APScheduler ``trigger="date"``。"""

    type: Literal["date"] = "date"
    run_timestamp: int = Field(..., description="触发时刻的 Unix 时间戳 (UTC)")

    @property
    def run_date(self) -> datetime:
        return datetime.fromtimestamp(self.run_timestamp, tz=timezone.utc)


class IntervalTriggerConfig(BaseModel):
    """周期触发，对应 APScheduler ``trigger="interval"``。"""

    type: Literal["interval"] = "interval"
    seconds: int = 0
    minutes: int = 0
    hours: int = 0
    days: int = 0
    end_timestamp: int | None = Field(
        default=None, description="可选：周期结束时间戳"
    )


class CronTriggerConfig(BaseModel):
    """Cron 触发，对应 APScheduler ``trigger="cron"``。"""

    type: Literal["cron"] = "cron"
    minute: str = "*"
    hour: str = "*"
    day: str = "*"
    month: str = "*"
    day_of_week: str = "*"
    end_timestamp: int | None = Field(
        default=None, description="可选：Cron 结束时间戳"
    )


TriggerConfig = Annotated[
    Union[
        Annotated[DateTriggerConfig, Tag("date")],
        Annotated[IntervalTriggerConfig, Tag("interval")],
        Annotated[CronTriggerConfig, Tag("cron")],
    ],
    Field(discriminator="type"),
]
