from functools import wraps
from typing import Awaitable, Callable

from ..heat import machine_pool
from ..heat._types import State, TransitionEvent
from ..heat.pool import MachinePool
from ..heat.utils import SessionKey


SilenceCallback = Callable[[SessionKey, TransitionEvent], Awaitable[None]]
SilenceDecorator = Callable[[SilenceCallback], SilenceCallback]


def is_silence_transition(event: TransitionEvent) -> bool:
	"""判断是否发生了会话静默结束转移。"""
	return (
		event.prev_state is State.INACTIVE
		and event.new_state is State.IDLE
	)


def silence_observer(pool: MachinePool = machine_pool) -> SilenceDecorator:
	"""返回一个会注册到状态机池的静默事件装饰器。"""

	def decorator(callback: SilenceCallback) -> SilenceCallback:
		@wraps(callback)
		async def wrapped(
			key: SessionKey,
			event: TransitionEvent,
		) -> None:
			if not is_silence_transition(event):
				return
			await callback(key, event)

		pool.on_transition(wrapped)
		return wrapped

	return decorator
