"""
tests/llm 的 conftest

将 nonebot_plugin_wtfllm.llm 注册为裸包（不执行 __init__.py），
同时 mock services 子包，从而打断循环导入链：

    llm.__init__ → agents → tools → tool_group.chat
        → services.store → services.__init__ → services.agent
        → llm.agents  ← 循环!

策略：仿照根 conftest 的 _create_package_from_path，
给 llm（及 llm.tools、llm.tools.tool_group）创建空壳包，
使得 `import nonebot_plugin_wtfllm.llm.deps` 不触发 llm/__init__.py。
"""
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

_SRC_DIR = Path(__file__).parent.parent.parent / "nonebot_plugin_wtfllm"


def _register_bare_package(dotted_name: str, directory: Path):
    """注册一个不执行 __init__.py 的裸包到 sys.modules"""
    if dotted_name in sys.modules:
        return
    pkg = types.ModuleType(dotted_name)
    pkg.__path__ = [str(directory)]
    pkg.__file__ = str(directory / "__init__.py")
    pkg.__package__ = dotted_name
    sys.modules[dotted_name] = pkg


# --- 将 llm 及其子包注册为裸包，避免 __init__.py 执行 ---
_register_bare_package(
    "nonebot_plugin_wtfllm.llm", _SRC_DIR / "llm"
)
_register_bare_package(
    "nonebot_plugin_wtfllm.llm.tools", _SRC_DIR / "llm" / "tools"
)
_register_bare_package(
    "nonebot_plugin_wtfllm.llm.tools.tool_group",
    _SRC_DIR / "llm" / "tools" / "tool_group",
)

# --- 预注入 services mock（打断循环的回边）---
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

# --- 预注入 scheduler mock ---
_sched_mock = MagicMock()
for mod_name in [
    "nonebot_plugin_wtfllm.scheduler",
    "nonebot_plugin_wtfllm.scheduler.engine",
    "nonebot_plugin_wtfllm.scheduler.executor",
    "nonebot_plugin_wtfllm.scheduler.recovery",
    "nonebot_plugin_wtfllm.scheduler.service",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = _sched_mock

# --- 预注入 on_summary mock ---
if "nonebot_plugin_wtfllm.on_summary" not in sys.modules:
    sys.modules["nonebot_plugin_wtfllm.on_summary"] = MagicMock()
