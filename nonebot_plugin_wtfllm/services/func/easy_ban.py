import asyncio
import os
from typing import TypedDict

import orjson
import aiofiles

from ...utils import JSON_DIR

easy_ban_file = JSON_DIR / "easy_ban.json"


class EasyBanData(TypedDict):
    banned_user_ids: set[str]
    banned_group_ids: set[str]


_data: EasyBanData | None = None
_lock = asyncio.Lock()


async def _ensure_data() -> EasyBanData:
    global _data
    if _data is not None:
        return _data

    async with _lock:
        if _data is not None:
            return _data

        if easy_ban_file.exists():
            try:
                async with aiofiles.open(easy_ban_file, "rb") as f:
                    content = await f.read()
                    raw_data = orjson.loads(content)

                _data = {
                    "banned_user_ids": set(raw_data.get("banned_user_ids", [])),
                    "banned_group_ids": set(raw_data.get("banned_group_ids", [])),
                }
            except (orjson.JSONDecodeError, OSError, UnicodeDecodeError):
                _data = _get_default_data()
        else:
            _data = _get_default_data()

        return _data


def _get_default_data() -> EasyBanData:
    return {
        "banned_user_ids": {"2854196310"},
        "banned_group_ids": set(),
    }


async def _save_data():
    if _data is None:
        return

    async with _lock:
        export_data = {
            "banned_user_ids": list(_data["banned_user_ids"]),
            "banned_group_ids": list(_data["banned_group_ids"]),
        }

        temp_file = easy_ban_file.with_suffix(".tmp")
        try:
            async with aiofiles.open(temp_file, "wb") as f:
                await f.write(orjson.dumps(export_data, option=orjson.OPT_INDENT_2))
            os.replace(temp_file, easy_ban_file)
        except OSError:
            if temp_file.exists():
                os.remove(temp_file)
            raise


async def get_easy_ban_data() -> EasyBanData:
    return await _ensure_data()


async def add_banned_user(user_id: str):
    data = await _ensure_data()
    if user_id not in data["banned_user_ids"]:
        data["banned_user_ids"].add(user_id)
        await _save_data()


async def remove_banned_user(user_id: str):
    data = await _ensure_data()
    if user_id in data["banned_user_ids"]:
        data["banned_user_ids"].remove(user_id)
        await _save_data()


async def add_banned_group(group_id: str):
    data = await _ensure_data()
    if group_id not in data["banned_group_ids"]:
        data["banned_group_ids"].add(group_id)
        await _save_data()


async def remove_banned_group(group_id: str):
    data = await _ensure_data()
    if group_id in data["banned_group_ids"]:
        data["banned_group_ids"].remove(group_id)
        await _save_data()


async def is_user_banned(user_id: str) -> bool:
    data = await _ensure_data()
    return user_id in data["banned_user_ids"]


async def is_group_banned(group_id: str) -> bool:
    data = await _ensure_data()
    return group_id in data["banned_group_ids"]


async def is_banned(user_id: str, group_id: str | None) -> bool:
    data = await _ensure_data()
    if user_id in data["banned_user_ids"]:
        return True
    if group_id and group_id in data["banned_group_ids"]:
        return True
    return False


async def get_banned_users() -> set[str]:
    return (await _ensure_data())["banned_user_ids"]


async def get_banned_groups() -> set[str]:
    return (await _ensure_data())["banned_group_ids"]
