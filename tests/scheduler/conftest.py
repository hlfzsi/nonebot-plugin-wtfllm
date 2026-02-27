"""
tests/scheduler 的 conftest

将 nonebot_plugin_wtfllm.scheduler 注册为裸包（不执行 __init__.py），
以便可以单独导入子模块进行测试。
"""
import sys
import types
import importlib as _il
from pathlib import Path
from unittest.mock import MagicMock

_SRC_DIR = Path(__file__).parent.parent.parent / "nonebot_plugin_wtfllm"


def _register_bare_package(dotted_name: str, directory: Path):
    if dotted_name in sys.modules:
        return
    pkg = types.ModuleType(dotted_name)
    pkg.__path__ = [str(directory)]
    pkg.__file__ = str(directory / "__init__.py")
    pkg.__package__ = dotted_name
    sys.modules[dotted_name] = pkg


# --- 清除可能被其他 conftest 注入的 scheduler mock ---
for _mod in [
    "nonebot_plugin_wtfllm.scheduler",
    "nonebot_plugin_wtfllm.scheduler.engine",
    "nonebot_plugin_wtfllm.scheduler.registry",
    "nonebot_plugin_wtfllm.scheduler.triggers",
    "nonebot_plugin_wtfllm.scheduler.executor",
    "nonebot_plugin_wtfllm.scheduler.recovery",
    "nonebot_plugin_wtfllm.scheduler.service",
    "nonebot_plugin_wtfllm.scheduler.tasks",
    "nonebot_plugin_wtfllm.scheduler.tasks.send_message",
    "nonebot_plugin_wtfllm.stream_processing",
    "nonebot_plugin_wtfllm.stream_processing.extract",
    "nonebot_plugin_wtfllm.stream_processing.store_flow",
]:
    sys.modules.pop(_mod, None)

# --- 将 scheduler 注册为裸包 ---
_register_bare_package(
    "nonebot_plugin_wtfllm.scheduler", _SRC_DIR / "scheduler"
)
_register_bare_package(
    "nonebot_plugin_wtfllm.scheduler.tasks", _SRC_DIR / "scheduler" / "tasks"
)

# --- 预注入 services mock（打断循环依赖）---
_services_mock = MagicMock()
for mod_name in [
    "nonebot_plugin_wtfllm.services",
    "nonebot_plugin_wtfllm.services.agent",
    "nonebot_plugin_wtfllm.services.store",
    "nonebot_plugin_wtfllm.services.summary",
    "nonebot_plugin_wtfllm.services.easy_ban",
    "nonebot_plugin_wtfllm.services.func",
    "nonebot_plugin_wtfllm.services.func.easy_ban",
    "nonebot_plugin_wtfllm.services.func.agent_cache",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = _services_mock

# --- 预注入 llm mock ---
for mod_name in [
    "nonebot_plugin_wtfllm.llm",
    "nonebot_plugin_wtfllm.llm.agents",
    "nonebot_plugin_wtfllm.llm.tools",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# --- 载入真实 stream_processing 包 ---
_il.import_module("nonebot_plugin_wtfllm.stream_processing")

# --- mock apscheduler ---
if "apscheduler" not in sys.modules:
    sys.modules["apscheduler"] = MagicMock()
if "apscheduler.schedulers" not in sys.modules:
    sys.modules["apscheduler.schedulers"] = MagicMock()
if "apscheduler.schedulers.asyncio" not in sys.modules:
    _aps_mock = MagicMock()
    sys.modules["apscheduler.schedulers.asyncio"] = _aps_mock

# Now import the actual scheduler submodules
_il.import_module("nonebot_plugin_wtfllm.scheduler.engine")
_il.import_module("nonebot_plugin_wtfllm.scheduler.registry")
_il.import_module("nonebot_plugin_wtfllm.scheduler.triggers")
_il.import_module("nonebot_plugin_wtfllm.scheduler.executor")
_il.import_module("nonebot_plugin_wtfllm.scheduler.recovery")
_il.import_module("nonebot_plugin_wtfllm.scheduler.service")
_il.import_module("nonebot_plugin_wtfllm.scheduler.tasks.send_static_message")
