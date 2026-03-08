from typing import TYPE_CHECKING, Awaitable, Callable

from cachetools import LRUCache

from ._types import HeatSnapshot, MachineConfig, State, TransitionEvent
from .controller import MachineController
from .utils import SessionKey

if TYPE_CHECKING:
    from ....memory.items.base import MemoryItem

TransitionCallback = Callable[[SessionKey, TransitionEvent], Awaitable[None]]


class MachinePool:
    """管理所有会话的热度状态机实例。"""

    def __init__(
        self,
        config: MachineConfig | None = None,
        *,
        max_sessions: int = 500,
    ) -> None:
        self._cfg = config or MachineConfig()
        self._controllers: LRUCache[tuple[str, str], MachineController] = LRUCache(
            maxsize=max_sessions,
        )
        self._callbacks: list[TransitionCallback] = []

    def feed(
        self,
        item: "MemoryItem",
        *,
        multiplier: float = 1.0,
    ) -> None:
        key = self._key_from_item(item)
        cache_key = key.available_id
        controller = self._controllers.get(cache_key)

        if controller is None:
            controller = MachineController(
                self._cfg,
                on_transition=self._make_dispatch(key),
            )
            self._controllers[cache_key] = controller

        controller.feed(item, multiplier=multiplier)

    def on_transition(self, callback: TransitionCallback) -> None:
        self._callbacks.append(callback)

    def get_snapshot(
        self,
        agent_id: str,
        user_id: str,
        group_id: str | None = None,
    ) -> HeatSnapshot:
        key = self._make_key(agent_id=agent_id, user_id=user_id, group_id=group_id)
        cache_key = key.available_id
        controller = self._controllers.get(cache_key)
        if controller is None:
            return HeatSnapshot(
                state=State.IDLE,
                heat=0.0,
                msg_ema=0.0,
                n_participants=0,
                velocity=0.0,
                last_update=0.0,
                state_entered_at=0.0,
            )
        return controller.peek()

    def get_state(
        self,
        agent_id: str,
        user_id: str,
        group_id: str | None = None,
    ) -> State:
        key = self._make_key(agent_id=agent_id, user_id=user_id, group_id=group_id)
        cache_key = key.available_id
        controller = self._controllers.get(cache_key)
        if controller is None:
            return State.IDLE
        return controller.get_state()

    def dispose(self) -> None:
        for controller in self._controllers.values():
            controller.dispose()
        self._controllers.clear()

    def _key_from_item(self, item: "MemoryItem") -> SessionKey:
        return self._make_key(
            agent_id=item.agent_id,
            user_id=item.sender,
            group_id=getattr(item, "group_id", None),
        )

    def _make_key(
        self,
        *,
        agent_id: str,
        user_id: str,
        group_id: str | None,
    ) -> SessionKey:
        return SessionKey(agent_id=agent_id, user_id=user_id, group_id=group_id)

    def _make_dispatch(
        self, key: SessionKey
    ) -> Callable[[TransitionEvent], Awaitable[None]]:
        async def dispatch(event: TransitionEvent) -> None:
            if event.new_state is State.IDLE:
                cache_key = key.available_id
                ctrl = self._controllers.pop(cache_key, None)
                if ctrl is not None:
                    ctrl.dispose()

            for callback in self._callbacks:
                await callback(key, event)

        return dispatch
