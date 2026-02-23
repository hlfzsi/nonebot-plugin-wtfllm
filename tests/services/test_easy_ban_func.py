"""services/func/easy_ban.py 单元测试"""

import pytest
import orjson

import nonebot_plugin_wtfllm.services.func.easy_ban as easy_ban_mod


@pytest.fixture(autouse=True)
def reset_state(monkeypatch, tmp_path):
    """每个测试前重置模块级状态，并将文件路径指向临时目录"""
    monkeypatch.setattr(easy_ban_mod, "_data", None)
    monkeypatch.setattr(easy_ban_mod, "easy_ban_file", tmp_path / "easy_ban.json")
    # 重建锁避免跨测试共享
    import asyncio
    monkeypatch.setattr(easy_ban_mod, "_lock", asyncio.Lock())


class TestGetDefaultData:
    def test_returns_correct_structure(self):
        data = easy_ban_mod._get_default_data()
        assert "banned_user_ids" in data
        assert "banned_group_ids" in data
        assert isinstance(data["banned_user_ids"], set)
        assert isinstance(data["banned_group_ids"], set)
        assert "2854196310" in data["banned_user_ids"]


class TestEnsureData:
    @pytest.mark.asyncio
    async def test_file_not_exist_returns_default(self):
        data = await easy_ban_mod._ensure_data()
        assert "2854196310" in data["banned_user_ids"]
        assert data["banned_group_ids"] == set()

    @pytest.mark.asyncio
    async def test_loads_from_file(self, tmp_path, monkeypatch):
        file_path = tmp_path / "easy_ban.json"
        content = orjson.dumps({
            "banned_user_ids": ["u1", "u2"],
            "banned_group_ids": ["g1"],
        })
        file_path.write_bytes(content)
        monkeypatch.setattr(easy_ban_mod, "easy_ban_file", file_path)

        data = await easy_ban_mod._ensure_data()
        assert data["banned_user_ids"] == {"u1", "u2"}
        assert data["banned_group_ids"] == {"g1"}

    @pytest.mark.asyncio
    async def test_corrupt_json_returns_default(self, tmp_path, monkeypatch):
        file_path = tmp_path / "easy_ban.json"
        file_path.write_bytes(b"not valid json{{{")
        monkeypatch.setattr(easy_ban_mod, "easy_ban_file", file_path)

        data = await easy_ban_mod._ensure_data()
        assert "2854196310" in data["banned_user_ids"]

    @pytest.mark.asyncio
    async def test_caches_after_first_load(self):
        data1 = await easy_ban_mod._ensure_data()
        data2 = await easy_ban_mod._ensure_data()
        assert data1 is data2


class TestAddRemoveBannedUser:
    @pytest.mark.asyncio
    async def test_add_banned_user(self):
        await easy_ban_mod.add_banned_user("u_new")
        assert await easy_ban_mod.is_user_banned("u_new")

    @pytest.mark.asyncio
    async def test_add_banned_user_idempotent(self):
        await easy_ban_mod.add_banned_user("u_idem")
        await easy_ban_mod.add_banned_user("u_idem")
        users = await easy_ban_mod.get_banned_users()
        assert list(users).count("u_idem") == 1

    @pytest.mark.asyncio
    async def test_remove_banned_user(self):
        await easy_ban_mod.add_banned_user("u_rem")
        await easy_ban_mod.remove_banned_user("u_rem")
        assert not await easy_ban_mod.is_user_banned("u_rem")

    @pytest.mark.asyncio
    async def test_remove_nonexistent_user(self):
        await easy_ban_mod.remove_banned_user("ghost_user")


class TestAddRemoveBannedGroup:
    @pytest.mark.asyncio
    async def test_add_banned_group(self):
        await easy_ban_mod.add_banned_group("g_new")
        assert await easy_ban_mod.is_group_banned("g_new")

    @pytest.mark.asyncio
    async def test_remove_banned_group(self):
        await easy_ban_mod.add_banned_group("g_rem")
        await easy_ban_mod.remove_banned_group("g_rem")
        assert not await easy_ban_mod.is_group_banned("g_rem")


class TestIsBanned:
    @pytest.mark.asyncio
    async def test_user_banned(self):
        await easy_ban_mod.add_banned_user("u_ban")
        assert await easy_ban_mod.is_banned("u_ban", None)

    @pytest.mark.asyncio
    async def test_group_banned(self):
        await easy_ban_mod.add_banned_group("g_ban")
        assert await easy_ban_mod.is_banned("u_ok", "g_ban")

    @pytest.mark.asyncio
    async def test_neither_banned(self):
        assert not await easy_ban_mod.is_banned("u_clean", "g_clean")

    @pytest.mark.asyncio
    async def test_group_none_not_banned(self):
        assert not await easy_ban_mod.is_banned("u_clean", None)


class TestGetBannedSets:
    @pytest.mark.asyncio
    async def test_get_banned_users(self):
        users = await easy_ban_mod.get_banned_users()
        assert isinstance(users, set)

    @pytest.mark.asyncio
    async def test_get_banned_groups(self):
        groups = await easy_ban_mod.get_banned_groups()
        assert isinstance(groups, set)


class TestSaveData:
    @pytest.mark.asyncio
    async def test_persists_to_file(self, tmp_path, monkeypatch):
        file_path = tmp_path / "easy_ban.json"
        monkeypatch.setattr(easy_ban_mod, "easy_ban_file", file_path)

        await easy_ban_mod.add_banned_user("persist_user")
        assert file_path.exists()

        raw = orjson.loads(file_path.read_bytes())
        assert "persist_user" in raw["banned_user_ids"]

    @pytest.mark.asyncio
    async def test_get_easy_ban_data(self):
        data = await easy_ban_mod.get_easy_ban_data()
        assert "banned_user_ids" in data
        assert "banned_group_ids" in data
