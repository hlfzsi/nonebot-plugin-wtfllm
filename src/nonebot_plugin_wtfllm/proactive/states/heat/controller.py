import asyncio
from time import time
from typing import TYPE_CHECKING, Awaitable, Callable

from ._types import HeatSnapshot, MachineConfig, State, TransitionEvent
from .machine import HeatMachine

if TYPE_CHECKING:
    from ....memory.items.base import MemoryItem


TransitionCallback = Callable[[TransitionEvent], Awaitable[None]]


class MachineController:
    """管理单个会话的热度状态机。"""

    __slots__ = (
        "_machine",
        "_cfg",
        "_callback",
        "_timer_handle",
    )

    def __init__(
        self,
        config: MachineConfig,
        on_transition: TransitionCallback,
    ) -> None:
        self._machine = HeatMachine(config)
        self._cfg = config
        self._callback = on_transition
        self._timer_handle: asyncio.TimerHandle | None = None

    def feed(self, item: "MemoryItem", *, multiplier: float = 1.0) -> None:
        if item.is_from_agent:
            return

        increment = self._cfg.base_increment * multiplier
        event = self._machine.feed(
            timestamp=float(item.created_at),
            sender_id=item.sender,
            increment=increment,
        )

        if event is not None:
            self._fire_callback(event)

        self._reschedule_timer()

    def tick(self) -> None:
        event = self._machine.tick(time())
        if event is not None:
            self._fire_callback(event)
        self._reschedule_timer()

    def peek(self, timestamp: float | None = None) -> HeatSnapshot:
        return self._machine.peek(timestamp or time())

    def get_state(self, timestamp: float | None = None) -> State:
        return self._machine.peek(timestamp or time()).state

    def dispose(self) -> None:
        if self._timer_handle is not None:
            self._timer_handle.cancel()
            self._timer_handle = None

    def _fire_callback(self, event: TransitionEvent) -> None:
        asyncio.ensure_future(self._callback(event))

    def _reschedule_timer(self) -> None:
        if self._timer_handle is not None:
            self._timer_handle.cancel()
            self._timer_handle = None

        now = time()
        next_time = self._machine.predict_transition_time(now)
        if next_time is None:
            return

        delay = max(0.0, next_time - now)
        loop = asyncio.get_running_loop()
        self._timer_handle = loop.call_later(delay, self._on_timer_fire)

    def _on_timer_fire(self) -> None:
        self._timer_handle = None
        self.tick()
