"""
Adaptive-k for KET-QA Retriever (triple 단위).
논문: "Efficient Context Selection for Long-Context QA: No Tuning, No Iteration, Just Adaptive-k"
- 원 논문은 k 단위가 토큰 수; KET-QA는 triple 개수로 적용.
- 정렬된 유사도(점수) 분포에서 largest gap 위치를 찾아 그 앞까지를 k로 사용.
"""

import numpy as np
from typing import List, Union


def adaptive_k_largest_gap(
    scores: np.ndarray,
    top_frac: float = 0.9,
    buffer: int = 5,
    min_k: int = 1,
    max_k: int = 200,
) -> int:
    """
    정렬된 점수(내림차순)에서 가장 큰 폭의 하락이 나는 인덱스를 찾고,
    그 위치를 k로 하여 top k개 triple을 가져오는 방식의 adaptive k 계산.

    Args:
        scores: 이미 내림차순 정렬된 점수 배열 (길이 N)
        top_frac: largest gap을 찾을 구간. 상위 (top_frac*100)% 구간만 검사 (논문: top 90%)
        buffer: k에 더해 줄 문서 수 (논문: B=5)
        min_k: 반환 k의 하한
        max_k: 반환 k의 상한

    Returns:
        k: 이번 쿼리에서 가져올 triple 개수
    """
    if len(scores) < 2:
        return min(1, max_k)
    scores = np.asarray(scores, dtype=np.float64)
    # gaps: scores[i] - scores[i+1]
    gaps = np.diff(scores)
    # 상위 top_frac 구간에서만 gap 검사 (하위 (1-top_frac) 제외)
    n = len(gaps)
    cut_tail = int(n * (1 - top_frac))
    if cut_tail > 0:
        search_gaps = gaps[:-cut_tail]
    else:
        search_gaps = gaps
    if len(search_gaps) == 0:
        k = 1
    else:
        # largest gap의 인덱스 (0-based). 그 인덱스까지 포함하므로 가져올 개수는 idx+1
        idx = np.argmax(search_gaps)
        k = idx + 1
    k = k + buffer
    k = max(min_k, min(max_k, k))
    return int(k)


def apply_adaptive_k(
    ranked_triples: List[dict],
    score_key: str = "score",
    top_frac: float = 0.9,
    buffer: int = 5,
    min_k: int = 1,
    max_k: int = 200,
) -> List[dict]:
    """
    ranked_triples는 이미 점수 기준 내림차순 정렬된 리스트.
    adaptive-k로 k를 정한 뒤 상위 k개만 반환.
    """
    if not ranked_triples:
        return []
    scores = np.array([t[score_key] for t in ranked_triples])
    k = adaptive_k_largest_gap(
        scores,
        top_frac=top_frac,
        buffer=buffer,
        min_k=min_k,
        max_k=max_k,
    )
    return ranked_triples[:k]
