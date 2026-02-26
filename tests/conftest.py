"""
全局测试配置和 fixtures

提供所有测试共用的 mock 和 fixture。
"""

import sys
import asyncio
import importlib.util
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

import pytest

# ===== 环境设置 =====

# 将项目源码目录添加到 sys.path
_PROJECT_ROOT = Path(__file__).parent.parent
_SRC_DIR = _PROJECT_ROOT / "nonebot_plugin_wtfllm"

# 创建临时目录用于测试
_temp_dir = tempfile.mkdtemp()

# ===== NoneBot Mock（必须在导入项目模块之前执行）=====

# Mock nonebot 核心模块
_mock_nonebot = MagicMock()
_mock_config = MagicMock()
_mock_config.no_reply_groups = []
_mock_config.admin_users = []
_mock_config.token_quota_enable = False
_mock_config.huggingface_mirror_url = ""
_mock_config.bot_name = "TestBot"
_mock_config.llm_api_key = "test_key"
_mock_config.llm_api_base_url = "http://test"
_mock_config.llm_model_name = "test-model"
_mock_config.llm_role_setting = "agent"
_mock_config.embedding_model_name = "test/model"
_mock_config.sparse_model_name = "test/sparse"
_mock_config.vision_model_config = None
_mock_config.image_generation_model_config = None
_mock_config.core_memory_max_tokens = 2048
_mock_config.core_memory_compress_ratio = 0.6
_mock_config.knowledge_base_max_results = 5
_mock_config.knowledge_base_max_tokens = 1024
_mock_config.memory_item_max_chars = 60
_mock_config.short_memory_time_minutes = 15
_mock_config.short_memory_max_count = 10
_mock_config.database_url = None

# Mock compress_agent_model_config
_mock_compress_config = MagicMock()
_mock_compress_config.api_key = "test_compress_key"
_mock_compress_config.base_url = "http://test-compress"
_mock_compress_config.name = "test-compress-model"
_mock_compress_config.extra_body = {}
_mock_config.compress_agent_model_config = _mock_compress_config

_mock_nonebot.get_plugin_config = MagicMock(return_value=_mock_config)
_mock_nonebot.logger = MagicMock()
_mock_nonebot.get_driver = MagicMock()
_mock_nonebot.require = MagicMock()

# Mock 所有 nonebot 相关模块
sys.modules["nonebot"] = _mock_nonebot
sys.modules["nonebot.adapters"] = MagicMock()
sys.modules["nonebot.adapters"].Bot = MagicMock()
sys.modules["nonebot.internal"] = MagicMock()
sys.modules["nonebot.internal.matcher"] = MagicMock()
sys.modules["nonebot.params"] = MagicMock()

# Mock nonebot_plugin_localstore
_mock_localstore = MagicMock()
_mock_localstore.get_plugin_data_dir = MagicMock(return_value=Path(_temp_dir))
sys.modules["nonebot_plugin_localstore"] = _mock_localstore

# Mock nonebot_plugin_alconna 及其子模块
_mock_alconna = MagicMock()
_mock_alconna.UniMessage = MagicMock()
sys.modules["nonebot_plugin_alconna"] = _mock_alconna
sys.modules["nonebot_plugin_alconna.uniseg"] = MagicMock()
sys.modules["nonebot_plugin_alconna.uniseg.segment"] = MagicMock()

# Mock nonebot_plugin_uninfo
sys.modules["nonebot_plugin_uninfo"] = MagicMock()

# Mock nonebot_plugin_waiter
sys.modules["nonebot_plugin_waiter"] = MagicMock()


# ===== 将源码目录注册为 nonebot_plugin_wtfllm 包 =====
# 这是关键：创建一个可以正确处理相对导入的包


def _create_package_from_path(package_name: str, package_path: Path):
    """将文件系统目录注册为 Python 包"""
    if str(package_path) not in sys.path:
        sys.path.insert(0, str(package_path.parent))

    # 创建包模块
    spec = importlib.util.spec_from_file_location(
        package_name,
        str(package_path / "__init__.py"),
        submodule_search_locations=[str(package_path)],
    )
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_path)]
    package.__file__ = str(package_path / "__init__.py")
    package.__package__ = package_name
    if spec:
        package.__spec__ = spec
        package.__loader__ = spec.loader
    sys.modules[package_name] = package
    return package


# 注册主包
_create_package_from_path("nonebot_plugin_wtfllm", _SRC_DIR)

# 现在可以正常导入了，只需要更新测试文件中的导入语句


# ===== pytest 配置 =====


def pytest_configure(config):
    """pytest 配置钩子"""
    config.addinivalue_line("markers", "asyncio: mark test as async")


# ===== 基础 Fixtures =====


@pytest.fixture
def temp_dir():
    """创建临时目录用于文件测试"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def event_loop():
    """提供事件循环"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ===== 数据库 Mock =====


@pytest.fixture
def mock_db_session():
    """Mock SQLModel 数据库会话"""
    session = AsyncMock()
    session.get = AsyncMock()
    session.exec = AsyncMock()
    session.add = MagicMock()
    session.add_all = MagicMock()
    session.commit = AsyncMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    session.delete = AsyncMock()
    session.begin = MagicMock(return_value=AsyncMock())
    return session


@pytest.fixture
def mock_session_maker(mock_db_session):
    """Mock SESSION_MAKER"""
    context_manager = AsyncMock()
    context_manager.__aenter__ = AsyncMock(return_value=mock_db_session)
    context_manager.__aexit__ = AsyncMock(return_value=None)

    maker = MagicMock(return_value=context_manager)
    return maker


@pytest.fixture
def mock_write_lock():
    """Mock WRITE_LOCK"""
    lock = AsyncMock()
    lock.__aenter__ = AsyncMock()
    lock.__aexit__ = AsyncMock(return_value=None)
    return lock


# ===== Qdrant Mock =====


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant 客户端"""
    client = AsyncMock()
    client.collection_exists = AsyncMock(return_value=True)
    client.retrieve = AsyncMock(return_value=[])
    client.upsert = AsyncMock()
    client.delete = AsyncMock()
    client.query_points = AsyncMock()
    client.scroll = AsyncMock(return_value=([], None))
    client.count = AsyncMock()
    client.get_fastembed_vector_params = MagicMock(return_value={"dense": {}})
    client.get_fastembed_sparse_vector_params = MagicMock(return_value={"sparse": {}})
    return client


# ===== 配置 Mock =====


@pytest.fixture
def mock_app_config():
    """Mock APP_CONFIG"""
    return _mock_config
