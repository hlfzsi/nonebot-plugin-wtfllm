import asyncio
import copy
from pathlib import Path
from typing import Any, Dict, Optional, Union

import aiofiles
import orjson

from ..utils import logger


class LLMPersonaEvolution:
    def __init__(self, storage_path: Union[str, Path] = "./"):
        self.path = Path(storage_path) / "persona_evolution.json"
        self._data: Optional[Dict[str, Any]] = None
        self._lock = asyncio.Lock()

    async def _load_if_needed(self) -> Dict[str, Any]:
        """懒加载并确保内存缓存已初始化"""
        if self._data is not None:
            return self._data

        async with self._lock:
            if self._data is not None:
                return self._data

            if self.path.exists():
                try:
                    async with aiofiles.open(self.path, mode="rb") as f:
                        content = await f.read()
                        if content:
                            self._data = orjson.loads(content)
                            logger.debug(f"已从 {self.path} 加载演进人设数据")
                        else:
                            self._data = {}
                except (OSError, orjson.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.error(f"解析人设文件 {self.path} 失败，初始化为空数据: {e}")
                    self._data = {}
            else:
                logger.info(f"人设文件 {self.path} 不存在，将创建新的人设层")
                self._data = {}

            assert self._data is not None
            return self._data

    async def _save_to_disk(self):
        if self._data is None:
            return

        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")

        try:
            binary_data = orjson.dumps(
                self._data, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS
            )

            async with aiofiles.open(temp_path, mode="wb") as f:
                await f.write(binary_data)
            temp_path.replace(self.path)
        except (OSError, TypeError, ValueError) as e:
            logger.error(f"保存人设文件至 {self.path} 失败: {e}")
            if "temp_path" in locals() and temp_path.exists():
                temp_path.unlink()
            raise e

    async def get_all(self) -> Dict[str, Any]:
        """获取全部人设数据的深拷贝"""
        data = await self._load_if_needed()
        async with self._lock:
            return copy.deepcopy(data)

    async def get_all_json(self) -> str:
        """获取包含全部人设数据的JSON"""
        data = await self._load_if_needed()
        return orjson.dumps(
            data, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS
        ).decode("utf-8")

    def _incremental_update(self, source: Dict[str, Any], delta: Dict[str, Any]):
        """
        增量更新
        """
        for key, value in delta.items():
            if value is None:
                # 传入 None 则删除该字段
                if key in source:
                    source.pop(key)
                    logger.debug(f"删除人设字段: {key}")
            elif (
                isinstance(value, dict)
                and key in source
                and isinstance(source[key], dict)
            ):
                self._incremental_update(source[key], value)
            else:
                source[key] = value

    async def update(self, delta: Dict[str, Any]):
        """增量更新人设并自动持久化"""
        if not delta:
            return

        await self._load_if_needed()

        async with self._lock:
            if self._data is None:
                self._data = {}

            logger.info(f"正在增量更新人设数据，变更字段: {list(delta.keys())}")
            self._incremental_update(self._data, delta)
            await self._save_to_disk()

    async def clear(self):
        """清空人设演进数据"""
        async with self._lock:
            self._data = {}
            await self._save_to_disk()
            logger.warning(f"人设演进数据已重置: {self.path}")
