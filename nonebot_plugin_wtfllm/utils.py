__all__ = [
    "logger",
    "DATABASE_DATA_DIR",
    "APP_CONFIG",
    "VECTOR_DATABASE_DIR",
    "MEMES_DIR",
    "MODELS_DIR",
    "MEDIA_DIR",
    "RESOURCES_DIR",
    "JSON_DIR",
    "SCHEDULED_MESSAGE_CACHE_DIR",
    "get_agent_id_from_bot",
    "ensure_msgid_from_receipt",
    "get_http_client",
    "init_http_client",
    "shutdown_http_client",
    "count_tokens",
]
import uuid
from pathlib import Path
from importlib import resources
from typing import Final

import httpx
import tiktoken

from nonebot import logger
from nonebot.adapters import Bot
from nonebot_plugin_uninfo import Session
from nonebot_plugin_alconna.uniseg import Receipt
from nonebot_plugin_localstore import get_plugin_data_dir, get_plugin_cache_dir

from .config import APP_CONFIG

_tiktoken_encoding = tiktoken.get_encoding("cl100k_base")


BASE_DATABASE_DIR: Final[Path] = get_plugin_data_dir() / "database"
DATABASE_DATA_DIR: Final[Path] = BASE_DATABASE_DIR / "db"
VECTOR_DATABASE_DIR: Final[Path] = BASE_DATABASE_DIR / "vector_db"
JSON_DIR: Final[Path] = get_plugin_data_dir() / "json"
MEMES_DIR: Final[Path] = get_plugin_data_dir() / "memes"
MEDIA_DIR: Final[Path] = get_plugin_data_dir() / "media"
SCHEDULED_MESSAGE_CACHE_DIR: Final[Path] = get_plugin_cache_dir() / "scheduled_messages"


MODELS_DIR: Final[Path] = get_plugin_data_dir() / "models"  # 无用。Qdrant只能读默认路径
RESOURCES_DIR: Final[Path] = (
    Path(str(resources.files(__package__).joinpath("resources")))
    if __package__
    else Path(__file__).parent / "resources"
)

_paths = [
    BASE_DATABASE_DIR,
    DATABASE_DATA_DIR,
    VECTOR_DATABASE_DIR,
    JSON_DIR,
    MEMES_DIR,
    SCHEDULED_MESSAGE_CACHE_DIR,
    MEDIA_DIR,
    # MODELS_DIR,
]

for path in _paths:
    path.mkdir(parents=True, exist_ok=True)


def get_agent_id_from_bot(bot_or_session: Bot | Session) -> str:
    """从 Bot ID 中提取 Agent ID"""
    if isinstance(bot_or_session, Session):
        return bot_or_session.self_id.removeprefix("llonebot:").removeprefix("napcat:")
    else:
        return bot_or_session.self_id.removeprefix("llonebot:").removeprefix("napcat:")


def ensure_msgid_from_receipt(
    receipt: Receipt, session: Session | None = None
) -> str:  # session备用
    """从发送回执中提取消息 ID, 不保证返回真实消息 ID"""
    reply = receipt.get_reply(-1)
    if reply is None:
        msg_id = f"fake_{uuid.uuid4().hex}"
    else:
        msg_id = reply.id

    return msg_id


_http_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    """获取全局 HTTP 客户端"""
    if _http_client is None:
        raise RuntimeError(
            "HTTP client is not initialized. Ensure init_http_client() was called on startup."
        )
    return _http_client


async def init_http_client() -> None:
    """初始化全局 HTTP 客户端"""
    global _http_client
    _http_client = httpx.AsyncClient(follow_redirects=True, timeout=60)


async def shutdown_http_client() -> None:
    """关闭全局 HTTP 客户端"""
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


def count_tokens(text: str) -> int:
    """计算文本的 token 数"""
    return len(_tiktoken_encoding.encode(text))
