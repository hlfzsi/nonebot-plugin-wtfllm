"""
tests/services 的 conftest

将 nonebot_plugin_wtfllm.services 注册为裸包（不执行 __init__.py），
避免触发 services/__init__.py → services.agent → llm.agents 的链式导入。

测试 services.func 下的纯逻辑模块以及各 service handler。
"""
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

_SRC_DIR = Path(__file__).parent.parent.parent / "src" / "nonebot_plugin_wtfllm"

# easy_ban.py / summary.py / delete_media.py 从 arclet.alconna 导入
if "arclet.alconna" not in sys.modules:
    sys.modules["arclet.alconna"] = MagicMock()

# 配置 nonebot.on_message 和 nonebot_plugin_alconna.on_alconna 的 mock
# 使 @matcher.handle()、@matcher.assign() 等装饰器保持原函数不变
_nonebot_mod = sys.modules.get("nonebot")
if _nonebot_mod is not None:
    def _passthrough_decorator(*args, **kwargs):
        """返回一个透传装饰器，使被装饰函数保持原样"""
        def deco(func):
            return func
        return deco

    def _make_matcher(*args, **kwargs):
        """创建一个 mock matcher，其 handle/assign 是透传装饰器"""
        m = MagicMock()
        m.handle = _passthrough_decorator
        m.assign = _passthrough_decorator
        m.finish = AsyncMock()
        return m

    _nonebot_mod.on_message = _make_matcher

_alconna_mod = sys.modules.get("nonebot_plugin_alconna")
if _alconna_mod is not None:
    _alconna_mod.on_alconna = _make_matcher


def _register_bare_package(dotted_name: str, directory: Path):
    """注册一个不执行 __init__.py 的裸包到 sys.modules

    强制覆盖：tests/llm/conftest.py 可能已将 services.func 注入为 MagicMock，
    这里需要替换为真实的裸包才能让子模块正常导入。
    """
    pkg = types.ModuleType(dotted_name)
    pkg.__path__ = [str(directory)]
    pkg.__file__ = str(directory / "__init__.py")
    pkg.__package__ = dotted_name
    sys.modules[dotted_name] = pkg


# 将 services 注册为裸包，避免 __init__.py 导入 agent/store/summary 等重模块
_register_bare_package(
    "nonebot_plugin_wtfllm.services", _SRC_DIR / "services"
)
_register_bare_package(
    "nonebot_plugin_wtfllm.services.func", _SRC_DIR / "services" / "func"
)

# tests/llm/conftest.py 可能已将 agent_cache 等子模块注入为 MagicMock，
# 清除它们以便后续 import 能加载真实模块。
for _mod in [
    "nonebot_plugin_wtfllm.services.func.agent_cache",
    "nonebot_plugin_wtfllm.services.func.message_queue",
    "nonebot_plugin_wtfllm.services.func.easy_ban",
]:
    sys.modules.pop(_mod, None)

# 预导入真实模块，确保 sys.modules 中是文件系统上的实际模块
# 而非 tests/llm/conftest.py 遗留的 MagicMock
import importlib as _il
_il.import_module("nonebot_plugin_wtfllm.services.func.message_queue")
_il.import_module("nonebot_plugin_wtfllm.services.func.agent_cache")
_il.import_module("nonebot_plugin_wtfllm.services.func.easy_ban")

# 重新加载 func 包的 __init__.py，使包级名称（如 set_alias_to_cache）可用
# 这样 store.py 等模块的 `from .func import ...` 才能正常工作
sys.modules.pop("nonebot_plugin_wtfllm.services.func", None)
_il.import_module("nonebot_plugin_wtfllm.services.func")

# 清除可能被缓存的 service handler 模块，确保用新 matcher mock 重新导入
for _mod in [
    "nonebot_plugin_wtfllm.services.store",
    "nonebot_plugin_wtfllm.services.easy_ban",
    "nonebot_plugin_wtfllm.services.delete_media",
    "nonebot_plugin_wtfllm.services.summary",
    "nonebot_plugin_wtfllm.services.lifecycle",
]:
    sys.modules.pop(_mod, None)
