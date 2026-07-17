from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .index import DenseIndex
from .parser import QueryConstraints, garment_matches, parse_query
from .schema import ImageRecord


@dataclass
class SearchResult:
    record: ImageRecord
    score: float
    dense_score: float
    binding_score: float
    context_score: float
    explanation: str
    sid_score: float = 0.0


def _fraction(required: set | frozenset, present: set) -> float:
    return 1.0 if not required else len(set(required) & present) / len(required)


def structured_scores(constraints: QueryConstraints, record: ImageRecord) -> tuple[float, float, float, str]:
    pairs = {(g.type, g.color) for g in record.garments}
    garments = {g.type for g in record.garments}
    colors = {g.color for g in record.garments}
    binding = float(np.mean([
        any(garment_matches(requested_garment, garment.type) and requested_color == garment.color for garment in record.garments)
        for requested_garment, requested_color in constraints.bindings
    ])) if constraints.bindings else 0.0
    if not constraints.bindings:
        garment_score = float(np.mean([
            any(garment_matches(requested, garment.type) for garment in record.garments)
            for requested in constraints.garments
        ])) if constraints.garments else 1.0
        binding = 0.5 * garment_score + 0.5 * _fraction(constraints.colors, colors)

    checks: list[float] = []
    labels: list[str] = []
    if constraints.environment:
        checks.append(float(record.environment == constraints.environment)); labels.append("setting")
    if constraints.activity:
        checks.append(float(record.activity == constraints.activity)); labels.append("activity")
    if constraints.objects:
        checks.append(_fraction(constraints.objects, set(record.objects))); labels.append("objects")
    context = float(np.mean(checks)) if checks else 1.0
    style = _fraction(constraints.styles, set(record.style))
    matched = [name for name, value in (("garment-color", binding), ("context", context), ("style", style)) if value >= 0.99]
    return binding, context, style, "matched " + ", ".join(matched or ["semantic content"])


class HybridRetriever:
    def __init__(self, index: DenseIndex, encoder, weights: dict[str, float] | None = None, candidate_pool: int = 200):
        self.index = index
        self.encoder = encoder
        self.weights = weights or {"dense": 0.55, "binding": 0.25, "context": 0.15, "style": 0.05}
        self.candidate_pool = candidate_pool

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        qvec = self.encoder.encode_texts([query])[0]
        ids, dense = self.index.search(qvec, max(k, self.candidate_pool))
        constraints = parse_query(query)
        results: list[SearchResult] = []
        for idx, cosine in zip(ids, dense):
            record = self.index.records[int(idx)]
            binding, context, style, explanation = structured_scores(constraints, record)
            # Cosine is mapped from [-1, 1] to [0, 1] before score fusion.
            dense01 = float((cosine + 1.0) / 2.0)
            total = (self.weights["dense"] * dense01 + self.weights["binding"] * binding +
                     self.weights["context"] * context + self.weights["style"] * style)
            results.append(SearchResult(record, total, float(cosine), binding, context, explanation))
        return sorted(results, key=lambda r: (-r.score, r.record.image_id))[:k]
