from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import normalize
from model2vec import StaticModel

from ..utils import RESOURCES_DIR

# 输出向量为512维
STATICMODEL_PATH = (RESOURCES_DIR / "acge-m2v-distilled").resolve()

class TopicVectorizer:
    def __init__(self, model_path: str | Path = STATICMODEL_PATH) -> None:
        self._model = StaticModel.from_pretrained(model_path)

    def transform(self, text: str) -> NDArray[np.floating]:
        """向量化单条文本，返回 1×D 特征向量"""
        raw = self._model.encode(text).reshape(1, -1)
        return normalize(raw, norm="l2")

    def transform_batch(self, texts: list[str]) -> NDArray[np.floating]:
        """批量向量化，返回 M×D 特征矩阵"""
        raw = self._model.encode(texts)
        return normalize(raw, norm="l2")
