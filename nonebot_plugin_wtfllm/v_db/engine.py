__all__ = ["get_qdrant_client"]

from ..deploy import get_global_qdrant_deployer


def get_qdrant_client():
    qdrant_deployer = get_global_qdrant_deployer()
    return qdrant_deployer.client
