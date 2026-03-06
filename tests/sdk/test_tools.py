"""SDK 工具组注册与查询测试。

验证：
- 新工具组可以被后续显式补注册到 CHAT_AGENT
- 二次注册幂等
- 列表 / 查询接口正确
"""

import pytest
from unittest.mock import MagicMock, patch

from nonebot_plugin_wtfllm.llm.tools.tool_group.base import ToolGroupMeta

# 直接导入 agents 子模块避免 llm.__init__ 的循环导入
import nonebot_plugin_wtfllm.llm.agents as _agents_mod
import nonebot_plugin_wtfllm.sdk.tools as sdk_tools


# ===================== Fixtures =====================


@pytest.fixture(autouse=True)
def _clean_tool_group_mapping():
    """每个测试前保存 / 后恢复 ToolGroupMeta.mapping 和 _registered_group_names"""
    original_mapping = dict(ToolGroupMeta.mapping)
    original_registered = set(_agents_mod._registered_group_names)
    yield
    # 恢复
    to_remove = [k for k in ToolGroupMeta.mapping if k not in original_mapping]
    for k in to_remove:
        del ToolGroupMeta.mapping[k]
    _agents_mod._registered_group_names.clear()
    _agents_mod._registered_group_names.update(original_registered)


def _make_dummy_group(name: str) -> ToolGroupMeta:
    """创建一个带有虚拟工具函数的 ToolGroupMeta"""
    group = ToolGroupMeta(name=name, description=f"{name} 测试组")

    @group.tool
    def dummy_tool(ctx) -> str:
        """虚拟工具"""
        return "ok"

    return group


# ===================== register_tool_groups 测试 =====================


class TestRegisterToolGroups:

    def test_register_new_group(self):
        """新工具组可以被补注册"""
        group = _make_dummy_group("SDK_TestNew")

        with patch.object(_agents_mod, "register_tools_to_agent") as mock_reg:
            result = sdk_tools.register_tool_groups([group])

        assert result == ["SDK_TestNew"]
        assert "SDK_TestNew" in _agents_mod._registered_group_names
        mock_reg.assert_called_once()

    def test_register_idempotent(self):
        """二次注册同名工具组不会重复调用底层注册"""
        group = _make_dummy_group("SDK_TestIdem")

        with patch.object(_agents_mod, "register_tools_to_agent") as mock_reg:
            first = sdk_tools.register_tool_groups([group])
            second = sdk_tools.register_tool_groups([group])

        assert first == ["SDK_TestIdem"]
        assert second == []
        # 只被调用一次
        mock_reg.assert_called_once()

    def test_register_multiple_groups(self):
        """一次性注册多个新组"""
        g1 = _make_dummy_group("SDK_Multi_A")
        g2 = _make_dummy_group("SDK_Multi_B")

        with patch.object(_agents_mod, "register_tools_to_agent") as mock_reg:
            result = sdk_tools.register_tool_groups([g1, g2])

        assert sorted(result) == ["SDK_Multi_A", "SDK_Multi_B"]
        mock_reg.assert_called_once()

    def test_register_skips_existing(self):
        """已注册的组被跳过，只注册新组"""
        g1 = _make_dummy_group("SDK_Existing")
        # 手动标记为已注册
        _agents_mod._registered_group_names.add("SDK_Existing")

        g2 = _make_dummy_group("SDK_Brand_New")

        with patch.object(_agents_mod, "register_tools_to_agent") as mock_reg:
            result = sdk_tools.register_tool_groups([g1, g2])

        assert result == ["SDK_Brand_New"]
        # 只有 g2 被传入底层注册
        mock_reg.assert_called_once()
        registered_groups = mock_reg.call_args[0][1]
        assert len(registered_groups) == 1
        assert registered_groups[0].name == "SDK_Brand_New"


# ===================== list / get 测试 =====================


class TestListAndGet:

    def test_list_registered_groups(self):
        """list_registered_groups 返回已排序的组名列表"""
        names = sdk_tools.list_registered_groups()
        assert names == sorted(names)
        assert isinstance(names, list)

    def test_get_tool_group_found(self):
        """按名查找存在的工具组"""
        group = _make_dummy_group("SDK_GetTest")
        result = sdk_tools.get_tool_group("SDK_GetTest")
        assert result is group

    def test_get_tool_group_not_found(self):
        """按名查找不存在的工具组返回 None"""
        result = sdk_tools.get_tool_group("NonExistent_12345")
        assert result is None
