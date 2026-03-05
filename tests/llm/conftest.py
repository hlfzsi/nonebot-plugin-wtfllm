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

# --- 将 services 及 func 注册为裸包，允许导入 memory_retrieval 等子模块 ---
_register_bare_package(
    "nonebot_plugin_wtfllm.services", _SRC_DIR / "services"
)
_register_bare_package(
    "nonebot_plugin_wtfllm.services.func", _SRC_DIR / "services" / "func"
)

# --- 预注入 services mock（打断循环的回边）---
_services_mock = MagicMock()
for mod_name in [
    "nonebot_plugin_wtfllm.services.agent",
    "nonebot_plugin_wtfllm.services.store",
    "nonebot_plugin_wtfllm.services.summary",
    "nonebot_plugin_wtfllm.services.easy_ban",
    "nonebot_plugin_wtfllm.services.func.easy_ban",
    "nonebot_plugin_wtfllm.services.func.agent_cache",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = _services_mock

# --- 预注入 scheduler ---
# scheduler 必须注册为裸包（有 __path__），否则子模块相对导入会失败
_register_bare_package(
    "nonebot_plugin_wtfllm.scheduler", _SRC_DIR / "scheduler"
)
_register_bare_package(
    "nonebot_plugin_wtfllm.scheduler.tasks", _SRC_DIR / "scheduler" / "tasks"
)

# schedule_message.py 的 `from ....scheduler import schedule_job, cancel_job`
# 需要这些属性存在于裸包上
_sched_pkg = sys.modules["nonebot_plugin_wtfllm.scheduler"]
_sched_pkg.schedule_job = MagicMock()
_sched_pkg.cancel_job = MagicMock()
_sched_pkg.scheduled_task = MagicMock()

# triggers / registry 是纯 pydantic，可以自然导入
# 但 engine / service / executor / recovery 有重依赖，需要 mock
for mod_name in [
    "nonebot_plugin_wtfllm.scheduler.engine",
    "nonebot_plugin_wtfllm.scheduler.executor",
    "nonebot_plugin_wtfllm.scheduler.recovery",
    "nonebot_plugin_wtfllm.scheduler.service",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# tasks/send_message.py 有重依赖 (stream_processing, msg_tracker 等)
# 创建轻量 stub 只提供 SendStaticMessageParams
_send_msg_stub = types.ModuleType(
    "nonebot_plugin_wtfllm.scheduler.tasks.send_message"
)

from pydantic import BaseModel as _BM, Field as _F
from typing import Any as _Any, Dict as _Dict, List as _List, Optional as _Opt

class _SendStaticMessageParams(_BM):
    target_data: _Dict[str, _Any] = _F(..., description="Target.dump() 输出")
    messages: _List[_Dict[str, _Any]] = _F(..., description="UniMessage.dump() 输出")
    user_id: str = _F(..., description="创建任务的用户ID")
    group_id: _Opt[str] = _F(default=None, description="关联群组ID")
    agent_id: str = _F(..., description="Bot/Agent ID")

_send_msg_stub.SendStaticMessageParams = _SendStaticMessageParams
_send_msg_stub.handle_send_static_message = MagicMock()
sys.modules["nonebot_plugin_wtfllm.scheduler.tasks.send_message"] = _send_msg_stub

# --- 预注入 on_summary mock ---
if "nonebot_plugin_wtfllm.on_summary" not in sys.modules:
    sys.modules["nonebot_plugin_wtfllm.on_summary"] = MagicMock()
