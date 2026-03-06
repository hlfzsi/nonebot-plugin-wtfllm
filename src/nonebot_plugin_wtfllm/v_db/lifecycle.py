"""向量数据库生命周期管理"""

from ..deploy import get_global_qdrant_deployer
from .models import MODELS


async def on_startup() -> None:
    """启动时初始化 Qdrant"""
    qdrant_deployer = get_global_qdrant_deployer()
    await qdrant_deployer.download_if_needed()
    await qdrant_deployer.start()
    await qdrant_deployer.post_start()
    for model in MODELS:
        await model.init_collection(
            collection_name=model.collection_name,
            payload_indexes=model.indexes if model.indexes else None,
        )
        if model.indexes:
            for field_name, field_schema in model.indexes.items():
                await get_global_qdrant_deployer().client.create_payload_index(
                    collection_name=model.collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )


async def on_shutdown() -> None:
    """关闭时清理 Qdrant 连接"""
    qdrant_deployer = get_global_qdrant_deployer()
    await qdrant_deployer.stop()
