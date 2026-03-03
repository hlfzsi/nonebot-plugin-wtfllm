import asyncio
from importlib.metadata import version, PackageNotFoundError

from nonebot import get_driver, require
from nonebot.plugin import PluginMetadata, inherit_supported_adapters

require("nonebot_plugin_localstore")
require("nonebot_plugin_alconna")
require("nonebot_plugin_uninfo")

from .services import setup_lifecycle_tasks, shutdown_lifecycle_tasks  # noqa: E402, F401
from .config import Config  # noqa: E402
from .db import init_db as rdb_init_db, shutdown_db as rdb_shutdown_db  # noqa: E402
from .v_db import on_startup as vdb_on_startup, on_shutdown as vdb_on_shutdown  # noqa: E402
from .scheduler import init_scheduler, shutdown_scheduler  # noqa: E402
from .utils import init_http_client, shutdown_http_client  # noqa: E402
from .topic import topic_manager  # noqa: E402
from .topic.archive.pipeline import archive_cluster  # noqa: E402

try:
    __version__ = version(__package__) if __package__ else None
except PackageNotFoundError:
    __version__ = None

__author__ = "hlfzsi"

__plugin_meta__ = PluginMetadata(
    name="WtfLLM",
    description="一个十分甚至有九分重量级的Agent实现",
    usage="详见项目 README 文档",
    config=Config,
    homepage="https://github.com/hlfzsi/nonebot-plugin-wtfllm",
    supported_adapters=inherit_supported_adapters(
        "nonebot_plugin_localstore", "nonebot_plugin_alconna", "nonebot_plugin_uninfo"
    ),
    extra={
        "author": __author__,
        "version": __version__,
    },
)
driver = get_driver()


@driver.on_startup
async def on_startup():
    async with asyncio.TaskGroup() as tg:
        tg.create_task(init_http_client())
        tg.create_task(rdb_init_db())
        tg.create_task(vdb_on_startup())

    async with asyncio.TaskGroup() as tg:
        tg.create_task(init_scheduler())

    setup_lifecycle_tasks()
    topic_manager.start(archive_cluster)


@driver.on_shutdown
async def on_shutdown():
    shutdown_lifecycle_tasks()

    await topic_manager.stop()

    async with asyncio.TaskGroup() as tg:
        tg.create_task(shutdown_scheduler())

    async with asyncio.TaskGroup() as tg:
        tg.create_task(vdb_on_shutdown())
        tg.create_task(rdb_shutdown_db())
        tg.create_task(shutdown_http_client())
