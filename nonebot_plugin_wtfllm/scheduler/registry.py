__all__ = [
    "scheduled_task",
    "get_task_handler",
    "get_task_params_model",
    "list_registered_tasks",
]

import inspect
from typing import Awaitable, Callable, Dict, Tuple, Type

from pydantic import BaseModel

HandleFuncType = Callable[..., Awaitable[None]]

# Registry: task_name -> (async handler, params BaseModel class)
_TASK_REGISTRY: Dict[str, Tuple[HandleFuncType, Type[BaseModel]]] = {}


def scheduled_task(name: str, params_model: Type[BaseModel]):
    """装饰器：注册异步函数为可调度任务类型。

    Usage::

        class MyParams(BaseModel):
            target_id: str

        @scheduled_task("my_task", MyParams)
        async def handle_my_task(params: MyParams) -> None:
            ...

    Args:
        name: 任务类型的唯一字符串标识，存入 DB 的 task_name 列
        params_model: 任务参数的 Pydantic BaseModel 类
    """

    def decorator(
        func: HandleFuncType,
    ) -> HandleFuncType:
        if name in _TASK_REGISTRY:
            raise ValueError(
                f"Scheduled task '{name}' is already registered "
                f"(existing: {_TASK_REGISTRY[name][0].__qualname__})"
            )
        if not inspect.iscoroutinefunction(func):
            raise ValueError("Scheduled task handler must be an async function")
        if not issubclass(params_model, BaseModel):
            raise ValueError(
                "Scheduled task params_model must be a subclass of pydantic.BaseModel"
            )

        sig = inspect.signature(func)
        params = [
            p
            for p in sig.parameters.values()
            if p.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        if len(params) != 1:
            raise ValueError(
                f"Scheduled task handler must accept exactly 1 parameter, "
                f"got {len(params)}: {[p.name for p in params]}"
            )
        ann = params[0].annotation
        if ann is not inspect.Parameter.empty and ann is not params_model:
            raise ValueError(
                f"Scheduled task handler parameter must be typed as "
                f"{params_model.__name__}, got {ann}"
            )

        _TASK_REGISTRY[name] = (func, params_model)
        return func

    return decorator


def get_task_handler(name: str) -> HandleFuncType:
    """根据注册名查找 handler 函数。

    Raises:
        KeyError: 未注册的任务名
    """
    if name not in _TASK_REGISTRY:
        raise KeyError(
            f"Unknown task type: '{name}'. Registered: {list(_TASK_REGISTRY.keys())}"
        )
    return _TASK_REGISTRY[name][0]


def get_task_params_model(name: str) -> Type[BaseModel]:
    """根据注册名查找参数 BaseModel 类。

    Raises:
        KeyError: 未注册的任务名
    """
    if name not in _TASK_REGISTRY:
        raise KeyError(
            f"Unknown task type: '{name}'. Registered: {list(_TASK_REGISTRY.keys())}"
        )
    return _TASK_REGISTRY[name][1]


def list_registered_tasks() -> list[str]:
    """返回所有已注册的任务名列表。"""
    return list(_TASK_REGISTRY.keys())
