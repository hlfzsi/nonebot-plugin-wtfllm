__all__ = ["get_global_qdrant_deployer"]


import os
from threading import Lock

from .platforms import WindowsQdrantDeployer, UnixQdrantDeployer
from .utils import BaseQdrantDeployer
from ...utils import VECTOR_DATABASE_DIR, MODELS_DIR  # noqa: F401

_lock = Lock()
_instance: BaseQdrantDeployer | None = None


def get_global_qdrant_deployer() -> BaseQdrantDeployer:
    """获取 Qdrant 部署器"""
    global _instance

    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = (
                    WindowsQdrantDeployer(
                        base_dir=VECTOR_DATABASE_DIR,  # model_dir=MODELS_DIR
                    )
                    if os.name == "nt"
                    else UnixQdrantDeployer(
                        base_dir=VECTOR_DATABASE_DIR,  # model_dir=MODELS_DIR
                    )
                )
    return _instance
