"""
tests/sdk 的 conftest

打断循环导入链，使 SDK 模块可安全导入而不触发重依赖初始化。

策略：
1. 将 llm / services / scheduler 注册为裸包（不执行 __init__.py）
2. 在裸包上挂载 agents.py 所需的属性（tool groups 和 register_tools_to_agent）
3. 让 agents.py 可以正常 import 并创建 CHAT_AGENT
"""
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

_SRC_DIR = Path(__file__).parent.parent.parent / "src" / "nonebot_plugin_wtfllm"


def _register_bare_package(dotted_name: str, directory: Path):
    """注册一个不执行 __init__.py 的裸包到 sys.modules。

    如果目标模块已被其他 conftest 注入为 MagicMock，则强制替换。
    """
    existing = sys.modules.get(dotted_name)
    if existing is not None and not isinstance(existing, MagicMock):
        return
    pkg = types.ModuleType(dotted_name)
    pkg.__path__ = [str(directory)]
    pkg.__file__ = str(directory / "__init__.py")
    pkg.__package__ = dotted_name
    sys.modules[dotted_name] = pkg


# --- 清除可能被 scheduler conftest 注入的 llm MagicMock ---
for _mod_name in [
    "nonebot_plugin_wtfllm.llm.agents",
    "nonebot_plugin_wtfllm.sdk.tools",
    "nonebot_plugin_wtfllm.sdk.agent",
    "nonebot_plugin_wtfllm.sdk.memory",
    "nonebot_plugin_wtfllm.sdk",
]:
    _existing = sys.modules.get(_mod_name)
    if isinstance(_existing, MagicMock):
        del sys.modules[_mod_name]

# --- 将 llm 及其子包注册为裸包 ---
_register_bare_package("nonebot_plugin_wtfllm.llm", _SRC_DIR / "llm")
_register_bare_package("nonebot_plugin_wtfllm.llm.tools", _SRC_DIR / "llm" / "tools")
_register_bare_package(
    "nonebot_plugin_wtfllm.llm.tools.tool_group",
    _SRC_DIR / "llm" / "tools" / "tool_group",
)

# --- 将 services 及 func 注册为裸包 ---
_register_bare_package("nonebot_plugin_wtfllm.services", _SRC_DIR / "services")
_register_bare_package(
    "nonebot_plugin_wtfllm.services.func", _SRC_DIR / "services" / "func"
)

# --- 预注入 services mock ---
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
_register_bare_package("nonebot_plugin_wtfllm.scheduler", _SRC_DIR / "scheduler")
_register_bare_package(
    "nonebot_plugin_wtfllm.scheduler.tasks", _SRC_DIR / "scheduler" / "tasks"
)

_sched_pkg = sys.modules["nonebot_plugin_wtfllm.scheduler"]
_sched_pkg.schedule_job = MagicMock()
_sched_pkg.cancel_job = MagicMock()
_sched_pkg.scheduled_task = MagicMock()

for mod_name in [
    "nonebot_plugin_wtfllm.scheduler.engine",
    "nonebot_plugin_wtfllm.scheduler.executor",
    "nonebot_plugin_wtfllm.scheduler.recovery",
    "nonebot_plugin_wtfllm.scheduler.service",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# tasks/send_message.py stub
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
if "nonebot_plugin_wtfllm.scheduler.tasks.send_message" not in sys.modules:
    sys.modules["nonebot_plugin_wtfllm.scheduler.tasks.send_message"] = _send_msg_stub

if "nonebot_plugin_wtfllm.on_summary" not in sys.modules:
    sys.modules["nonebot_plugin_wtfllm.on_summary"] = MagicMock()

# ── 补全 APP_CONFIG mock，确保 agents.py 模块级代码可执行 ──────────
# 根 conftest 的 _mock_config 没有设置 main_agent_model_config 和
# llm_use_responses_api，agents.py 在模块顶层使用这些属性创建 OpenAI provider。
from nonebot_plugin_wtfllm.utils import APP_CONFIG as _app_cfg

_main_cfg_mock = MagicMock()
_main_cfg_mock.api_key = "test_key"
_main_cfg_mock.base_url = "http://test-main"
_main_cfg_mock.name = "test-main-model"
_main_cfg_mock.extra_body = {}
_app_cfg.main_agent_model_config = _main_cfg_mock
_app_cfg.llm_use_responses_api = False


# ── 关键：让 agents.py 能从裸包 llm.tools 导入所需属性 ──────────────
# agents.py 执行 `from .tools import register_tools_to_agent, core_group, ...`
# 需要在裸包上挂载真实的 register_tools_to_agent 和 ToolGroupMeta 占位
from nonebot_plugin_wtfllm.llm.tools.tool_group.base import ToolGroupMeta
from nonebot_plugin_wtfllm.llm.tools.prepare import register_tools_to_agent

_tools_pkg = sys.modules["nonebot_plugin_wtfllm.llm.tools"]
_tools_pkg.register_tools_to_agent = register_tools_to_agent

# 为 agents.py 中导入的 9 个工具组创建轻量 ToolGroupMeta 实例
# 注意：如果业务代码已经创建过同名实例（进入了 ToolGroupMeta.mapping），
# 就直接复用；否则创建占位实例。
_BUILTIN_GROUP_NAMES = [
    "Core",
    "CoreMemory",
    "Chat",
    "UserPersona",
    "Memes",
    "WebSearch",
    "ImageGeneration",
    "ScheduleMessage",
    "KnowledgeBase",
]
_BUILTIN_ATTR_NAMES = [
    "core_group",
    "core_memory_group",
    "chat_tool_group",
    "user_tool_group",
    "memes_tool_group",
    "web_search_tool_group",
    "image_generation_tool_group",
    "schedule_message_group",
    "knowledge_base_group",
]

for _group_name, _attr_name in zip(_BUILTIN_GROUP_NAMES, _BUILTIN_ATTR_NAMES):
    if _group_name in ToolGroupMeta.mapping:
        _group_instance = ToolGroupMeta.mapping[_group_name]
    else:
        _group_instance = ToolGroupMeta(
            name=_group_name, description=f"{_group_name} (test stub)"
        )
    setattr(_tools_pkg, _attr_name, _group_instance)

# --- SDK 裸包注册 ---
_register_bare_package("nonebot_plugin_wtfllm.sdk", _SRC_DIR / "sdk")
