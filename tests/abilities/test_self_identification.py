# tests/abilities/test_self_identification.py
"""abilities/self_identification.py 单元测试"""

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import orjson

from nonebot_plugin_wtfllm.abilities.self_identification import LLMPersonaEvolution


class TestLLMPersonaEvolutionInit:
    """LLMPersonaEvolution 初始化测试"""

    def test_init_default_path(self, tmp_path: Path):
        """测试默认路径初始化"""
        persona = LLMPersonaEvolution(tmp_path)
        assert persona.path == tmp_path / "persona_evolution.json"
        assert persona._data is None

    def test_init_string_path(self, tmp_path: Path):
        """测试字符串路径初始化"""
        persona = LLMPersonaEvolution(str(tmp_path))
        assert persona.path == tmp_path / "persona_evolution.json"


class TestLLMPersonaEvolutionLoad:
    """LLMPersonaEvolution 加载测试"""

    @pytest.mark.asyncio
    async def test_lazy_load_creates_empty(self, tmp_path: Path):
        """测试懒加载创建空数据"""
        persona = LLMPersonaEvolution(tmp_path)
        data = await persona.get_all()

        assert data == {}
        assert persona._data is not None

    @pytest.mark.asyncio
    async def test_load_existing_file(self, tmp_path: Path):
        """测试加载已存在的文件"""
        existing_data = {"name": "TestBot", "traits": ["friendly"]}
        (tmp_path / "persona_evolution.json").write_bytes(
            orjson.dumps(existing_data)
        )

        persona = LLMPersonaEvolution(tmp_path)
        data = await persona.get_all()

        assert data == existing_data

    @pytest.mark.asyncio
    async def test_load_empty_file(self, tmp_path: Path):
        """测试加载空文件"""
        (tmp_path / "persona_evolution.json").write_text("")

        persona = LLMPersonaEvolution(tmp_path)
        data = await persona.get_all()

        assert data == {}

    @pytest.mark.asyncio
    async def test_load_corrupted_file(self, tmp_path: Path):
        """测试加载损坏的文件"""
        (tmp_path / "persona_evolution.json").write_text("not valid json {{{")

        persona = LLMPersonaEvolution(tmp_path)
        data = await persona.get_all()

        # 损坏文件应初始化为空
        assert data == {}


class TestLLMPersonaEvolutionGetAll:
    """LLMPersonaEvolution.get_all 测试"""

    @pytest.mark.asyncio
    async def test_get_all_returns_deep_copy(self, tmp_path: Path):
        """测试 get_all 返回深拷贝"""
        persona = LLMPersonaEvolution(tmp_path)
        await persona.update({"nested": {"key": "value"}})

        data1 = await persona.get_all()
        data1["nested"]["key"] = "modified"

        data2 = await persona.get_all()
        assert data2["nested"]["key"] == "value"  # 原始数据不应被修改


class TestLLMPersonaEvolutionGetAllJson:
    """LLMPersonaEvolution.get_all_json 测试"""

    @pytest.mark.asyncio
    async def test_get_all_json_format(self, tmp_path: Path):
        """测试 get_all_json 返回正确的 JSON 格式"""
        persona = LLMPersonaEvolution(tmp_path)
        await persona.update({"name": "Bot", "version": 1})

        json_str = await persona.get_all_json()
        data = json.loads(json_str)

        assert data["name"] == "Bot"
        assert data["version"] == 1

    @pytest.mark.asyncio
    async def test_get_all_json_empty(self, tmp_path: Path):
        """测试空数据的 JSON"""
        persona = LLMPersonaEvolution(tmp_path)
        json_str = await persona.get_all_json()

        assert json.loads(json_str) == {}


class TestLLMPersonaEvolutionUpdate:
    """LLMPersonaEvolution.update 测试"""

    @pytest.mark.asyncio
    async def test_update_add_new_field(self, tmp_path: Path):
        """测试添加新字段"""
        persona = LLMPersonaEvolution(tmp_path)
        await persona.update({"name": "TestBot"})

        data = await persona.get_all()
        assert data["name"] == "TestBot"

    @pytest.mark.asyncio
    async def test_update_overwrite_field(self, tmp_path: Path):
        """测试覆盖已存在的字段"""
        persona = LLMPersonaEvolution(tmp_path)
        await persona.update({"name": "OldName"})
        await persona.update({"name": "NewName"})

        data = await persona.get_all()
        assert data["name"] == "NewName"

    @pytest.mark.asyncio
    async def test_update_delete_field_with_none(self, tmp_path: Path):
        """测试使用 None 删除字段"""
        persona = LLMPersonaEvolution(tmp_path)
        await persona.update({"name": "Bot", "temp": "value"})
        await persona.update({"temp": None})

        data = await persona.get_all()
        assert "name" in data
        assert "temp" not in data

    @pytest.mark.asyncio
    async def test_update_nested_incremental(self, tmp_path: Path):
        """测试嵌套增量更新"""
        persona = LLMPersonaEvolution(tmp_path)
        await persona.update({
            "traits": {
                "personality": "friendly",
                "mood": "happy"
            }
        })
        await persona.update({
            "traits": {
                "mood": "excited"  # 只更新 mood
            }
        })

        data = await persona.get_all()
        assert data["traits"]["personality"] == "friendly"  # 保持不变
        assert data["traits"]["mood"] == "excited"  # 被更新

    @pytest.mark.asyncio
    async def test_update_nested_delete(self, tmp_path: Path):
        """测试嵌套字段删除"""
        persona = LLMPersonaEvolution(tmp_path)
        await persona.update({
            "traits": {
                "a": 1,
                "b": 2
            }
        })
        await persona.update({
            "traits": {
                "a": None  # 删除嵌套字段
            }
        })

        data = await persona.get_all()
        assert "a" not in data["traits"]
        assert data["traits"]["b"] == 2

    @pytest.mark.asyncio
    async def test_update_empty_delta_skips(self, tmp_path: Path):
        """测试空 delta 不执行操作"""
        persona = LLMPersonaEvolution(tmp_path)
        await persona.update({"initial": "data"})

        # 记录文件修改时间
        file_path = tmp_path / "persona_evolution.json"
        mtime_before = file_path.stat().st_mtime

        await asyncio.sleep(0.01)  # 确保时间差
        await persona.update({})

        # 文件不应被修改
        mtime_after = file_path.stat().st_mtime
        assert mtime_before == mtime_after

    @pytest.mark.asyncio
    async def test_update_persists_to_disk(self, tmp_path: Path):
        """测试更新持久化到磁盘"""
        persona = LLMPersonaEvolution(tmp_path)
        await persona.update({"key": "value"})

        # 创建新实例读取
        persona2 = LLMPersonaEvolution(tmp_path)
        data = await persona2.get_all()

        assert data["key"] == "value"

    @pytest.mark.asyncio
    async def test_atomic_write_no_temp_file_left(self, tmp_path: Path):
        """测试原子写入不留下临时文件"""
        persona = LLMPersonaEvolution(tmp_path)
        await persona.update({"key": "value"})

        temp_path = tmp_path / "persona_evolution.json.tmp"
        assert not temp_path.exists()


class TestLLMPersonaEvolutionClear:
    """LLMPersonaEvolution.clear 测试"""

    @pytest.mark.asyncio
    async def test_clear_removes_all_data(self, tmp_path: Path):
        """测试 clear 清空所有数据"""
        persona = LLMPersonaEvolution(tmp_path)
        await persona.update({"key1": "value1", "key2": "value2"})
        await persona.clear()

        data = await persona.get_all()
        assert data == {}

    @pytest.mark.asyncio
    async def test_clear_persists_to_disk(self, tmp_path: Path):
        """测试 clear 持久化到磁盘"""
        persona = LLMPersonaEvolution(tmp_path)
        await persona.update({"key": "value"})
        await persona.clear()

        # 创建新实例验证
        persona2 = LLMPersonaEvolution(tmp_path)
        data = await persona2.get_all()

        assert data == {}


class TestLLMPersonaEvolutionConcurrency:
    """LLMPersonaEvolution 并发测试"""

    @pytest.mark.asyncio
    async def test_concurrent_reads(self, tmp_path: Path):
        """测试并发读取"""
        persona = LLMPersonaEvolution(tmp_path)
        await persona.update({"initial": "data"})

        async def read_task():
            return await persona.get_all()

        results = await asyncio.gather(*[read_task() for _ in range(10)])

        for result in results:
            assert result == {"initial": "data"}

    @pytest.mark.asyncio
    async def test_concurrent_updates(self, tmp_path: Path):
        """测试并发更新"""
        persona = LLMPersonaEvolution(tmp_path)

        async def update_task(i: int):
            await persona.update({f"key_{i}": f"value_{i}"})

        await asyncio.gather(*[update_task(i) for i in range(5)])

        data = await persona.get_all()
        # 所有更新都应该成功（虽然顺序不确定）
        for i in range(5):
            assert f"key_{i}" in data


class TestIncrementalUpdate:
    """_incremental_update 内部方法测试"""

    def test_simple_add(self, tmp_path: Path):
        """测试简单添加"""
        persona = LLMPersonaEvolution(tmp_path)
        source = {}
        persona._incremental_update(source, {"a": 1})
        assert source == {"a": 1}

    def test_simple_overwrite(self, tmp_path: Path):
        """测试简单覆盖"""
        persona = LLMPersonaEvolution(tmp_path)
        source = {"a": 1}
        persona._incremental_update(source, {"a": 2})
        assert source == {"a": 2}

    def test_delete_with_none(self, tmp_path: Path):
        """测试 None 删除"""
        persona = LLMPersonaEvolution(tmp_path)
        source = {"a": 1, "b": 2}
        persona._incremental_update(source, {"a": None})
        assert source == {"b": 2}

    def test_delete_nonexistent_key(self, tmp_path: Path):
        """测试删除不存在的键（不报错）"""
        persona = LLMPersonaEvolution(tmp_path)
        source = {"a": 1}
        persona._incremental_update(source, {"nonexistent": None})
        assert source == {"a": 1}

    def test_nested_update(self, tmp_path: Path):
        """测试嵌套更新"""
        persona = LLMPersonaEvolution(tmp_path)
        source = {"nested": {"x": 1, "y": 2}}
        persona._incremental_update(source, {"nested": {"y": 3, "z": 4}})
        assert source == {"nested": {"x": 1, "y": 3, "z": 4}}

    def test_replace_non_dict_with_dict(self, tmp_path: Path):
        """测试用字典替换非字典值"""
        persona = LLMPersonaEvolution(tmp_path)
        source = {"a": "string"}
        persona._incremental_update(source, {"a": {"nested": "value"}})
        assert source == {"a": {"nested": "value"}}

    def test_replace_dict_with_non_dict(self, tmp_path: Path):
        """测试用非字典值替换字典"""
        persona = LLMPersonaEvolution(tmp_path)
        source = {"a": {"nested": "value"}}
        persona._incremental_update(source, {"a": "string"})
        assert source == {"a": "string"}
