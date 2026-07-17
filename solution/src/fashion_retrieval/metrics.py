from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def precision_at_k(ranked_ids: Sequence[str], relevance: dict[str, int], k: int = 1) -> float:
    chosen = ranked_ids[:k]
    return sum(relevance.get(i, 0) == 2 for i in chosen) / max(len(chosen), 1)


def recall_at_k(ranked_ids: Sequence[str], relevance: dict[str, int], k: int) -> float:
    relevant = sum(v == 2 for v in relevance.values())
    return sum(relevance.get(i, 0) == 2 for i in ranked_ids[:k]) / max(relevant, 1)


def average_precision_at_k(ranked_ids: Sequence[str], relevance: dict[str, int], k: int = 10) -> float:
    hits, score = 0, 0.0
    for rank, image_id in enumerate(ranked_ids[:k], 1):
        if relevance.get(image_id, 0) == 2:
            hits += 1
            score += hits / rank
    return score / max(min(sum(v == 2 for v in relevance.values()), k), 1)


def ndcg_at_k(ranked_ids: Sequence[str], relevance: dict[str, int], k: int = 10) -> float:
    def dcg(grades: Sequence[int]) -> float:
        return sum((2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(grades))
    actual = [relevance.get(i, 0) for i in ranked_ids[:k]]
    ideal = sorted(relevance.values(), reverse=True)[:k]
    return dcg(actual) / max(dcg(ideal), 1e-12)


def evaluate_rankings(rankings: list[list[str]], judgments: list[dict[str, int]]) -> dict[str, float]:
    if len(rankings) != len(judgments) or not rankings:
        raise ValueError("rankings and judgments must be non-empty and aligned")
    return {
        "precision@1": float(np.mean([precision_at_k(r, j, 1) for r, j in zip(rankings, judgments)])),
        "recall@5": float(np.mean([recall_at_k(r, j, 5) for r, j in zip(rankings, judgments)])),
        "recall@10": float(np.mean([recall_at_k(r, j, 10) for r, j in zip(rankings, judgments)])),
        "map@10": float(np.mean([average_precision_at_k(r, j, 10) for r, j in zip(rankings, judgments)])),
        "ndcg@10": float(np.mean([ndcg_at_k(r, j, 10) for r, j in zip(rankings, judgments)])),
    }

