import numpy as np
from numpy.typing import NDArray


def mmr_select(
    candidate_vectors: NDArray[np.floating],
    query_vector: NDArray[np.floating],
    k: int = 5,
    lambda_param: float = 0.5,
) -> list[int]:
    """MMR 选择。

    Args:
        candidate_vectors: (N, D) 候选向量矩阵，已归一化
        query_vector: (D,) 查询向量，已归一化
        k: 选择数量
        lambda_param: MMR λ 参数，平衡相关性和多样性

    Returns:
        选中的候选索引列表（长度 min(k, N)）
    """
    n = candidate_vectors.shape[0]
    if n == 0:
        return []
    k = min(k, n)

    query = query_vector.flatten()
    sim_to_query = candidate_vectors @ query

    selected: list[int] = []
    max_sim_to_selected = np.full(n, -np.inf, dtype=np.float64)

    for _ in range(k):
        mmr_scores = lambda_param * sim_to_query - (1.0 - lambda_param) * np.maximum(
            max_sim_to_selected, 0.0
        )
        for idx in selected:
            mmr_scores[idx] = -np.inf

        best = int(np.argmax(mmr_scores))
        selected.append(best)

        sim_to_best = candidate_vectors @ candidate_vectors[best]
        np.maximum(max_sim_to_selected, sim_to_best, out=max_sim_to_selected)

    return selected
