from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .index import DenseIndex
from .metrics import evaluate_rankings
from .models import ResidualQuantizer
from .parser import parse_query
from .retriever import SearchResult, structured_scores
from .schema import JudgedQuery


def semantic_id_relation(query_sid: list[int] | tuple[int, ...], candidate_sid: list[int] | tuple[int, ...]) -> float:
    """Map hierarchical code agreement to a simple retrieval prior."""
    if not query_sid or not candidate_sid:
        return 0.0
    depth = min(len(query_sid), len(candidate_sid))
    matches = 0
    for i in range(depth):
        if query_sid[i] != candidate_sid[i]:
            break
        matches += 1
    if matches == depth:
        return 1.0
    if matches >= 2:
        return 0.70
    if matches >= 1:
        return 0.35
    return 0.0


class SemanticIDHybridRetriever:
    """Direct Qwen-to-RQ-VAE query quantization plus structured reranking."""

    def __init__(
        self,
        index: DenseIndex,
        encoder,
        semantic_dir: str | Path,
        weights: dict[str, float] | None = None,
        candidate_pool: int = 100,
        sid_pool: int = 100,
    ):
        self.index = index
        self.encoder = encoder
        self.candidate_pool = candidate_pool
        self.sid_pool = sid_pool
        self.weights = weights or {"dense": 0.40, "binding": 0.20, "context": 0.12, "sid": 0.10, "style": 0.18}

        target = Path(semantic_dir)
        payload = torch.load(target / "rqvae.pt", map_location="cpu", weights_only=True)
        self.model = ResidualQuantizer(
            payload["input_dim"], payload["latent_dim"], payload["levels"], payload["codebook_size"]
        )
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.image_to_sid = {
            image_id: [int(code) for code in codes]
            for image_id, codes in json.loads((target / "image_to_sid.json").read_text(encoding="utf-8")).items()
        }
        self.prefix_to_images = {
            key: [str(image_id) for image_id in image_ids]
            for key, image_ids in json.loads((target / "prefix_to_images.json").read_text(encoding="utf-8")).items()
        }
        self.id_to_index = {record.image_id: i for i, record in enumerate(index.records)}

    def predict_semantic_id(self, query: str) -> list[int]:
        qvec = self.encoder.encode_texts([query])[0]
        with torch.no_grad():
            return self.model.semantic_ids(torch.from_numpy(qvec[None]).float())[0].tolist()

    def _sid_candidates(self, query_sid: list[int]) -> list[int]:
        ordered: list[int] = []
        seen: set[int] = set()
        for depth in range(len(query_sid), 0, -1):
            key = "/".join(map(str, query_sid[:depth]))
            for image_id in self.prefix_to_images.get(key, []):
                idx = self.id_to_index.get(image_id)
                if idx is None or idx in seen:
                    continue
                ordered.append(idx)
                seen.add(idx)
                if len(ordered) >= self.sid_pool:
                    return ordered
        return ordered

    def retrieve(self, query: str, k: int = 10) -> tuple[list[SearchResult], list[int], int]:
        qvec = self.encoder.encode_texts([query])[0]
        query_sid = self.predict_semantic_id(query)
        dense_ids, dense_scores = self.index.search(qvec, max(k, self.candidate_pool))
        dense_lookup = {int(idx): float(score) for idx, score in zip(dense_ids, dense_scores)}
        sid_candidates = self._sid_candidates(query_sid)
        candidate_ids = list(dict.fromkeys([int(idx) for idx in dense_ids] + sid_candidates))
        constraints = parse_query(query)
        results: list[SearchResult] = []
        for idx in candidate_ids:
            record = self.index.records[idx]
            cosine = dense_lookup.get(idx, float(self.index.vectors[idx] @ qvec))
            binding, context, style, explanation = structured_scores(constraints, record)
            sid_score = semantic_id_relation(query_sid, self.image_to_sid.get(record.image_id, ()))
            dense01 = float((cosine + 1.0) / 2.0)
            total = (
                self.weights["dense"] * dense01
                + self.weights["binding"] * binding
                + self.weights["context"] * context
                + self.weights["sid"] * sid_score
                + self.weights["style"] * style
            )
            results.append(
                SearchResult(
                    record,
                    total,
                    float(cosine),
                    binding,
                    context,
                    f"{explanation}; semantic-id={sid_score:.2f}",
                    sid_score=sid_score,
                )
            )
        return sorted(results, key=lambda r: (-r.score, r.record.image_id))[:k], query_sid, len(sid_candidates)

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        return self.retrieve(query, k)[0]


def evaluate_semantic_ids(
    index: DenseIndex,
    encoder,
    queries: list[JudgedQuery],
    semantic_dir: str | Path,
    output: str | Path,
    candidate_pool: int = 100,
    sid_pool: int = 100,
    k: int = 10,
    weights: dict[str, float] | None = None,
) -> dict:
    retriever = SemanticIDHybridRetriever(index, encoder, semantic_dir, weights=weights, candidate_pool=candidate_pool, sid_pool=sid_pool)
    rankings, judgments, latencies = [], [], []
    binding_hits = binding_total = context_hits = context_total = 0
    sid_coverages: list[float] = []
    sid_candidate_counts: list[int] = []
    diagnostics = []
    for query in queries:
        start = time.perf_counter()
        results, query_sid, sid_candidates = retriever.retrieve(query.text, k)
        latencies.append((time.perf_counter() - start) * 1000)
        rankings.append([r.record.image_id for r in results])
        judgments.append(query.relevance)
        sid_candidate_counts.append(sid_candidates)
        sid_coverages.append(float(any(r.sid_score > 0 for r in results)))
        constraints = parse_query(query.text)
        if constraints.bindings:
            binding_total += 1
            binding_hits += int(bool(results) and query.relevance.get(results[0].record.image_id, 0) == 2)
        if constraints.environment or constraints.activity or constraints.objects:
            context_total += 1
            context_hits += int(bool(results) and results[0].context_score >= 0.999)
        if len(diagnostics) < 5 and results:
            diagnostics.append(
                {
                    "query": query.text,
                    "predicted_semantic_id": query_sid,
                    "semantic_id_candidates": sid_candidates,
                    "top_result": {
                        "image_id": results[0].record.image_id,
                        "score": results[0].score,
                        "binding_score": results[0].binding_score,
                        "context_score": results[0].context_score,
                        "sid_score": results[0].sid_score,
                        "explanation": results[0].explanation,
                    },
                }
            )
    metrics = evaluate_rankings(rankings, judgments)
    metrics.update(
        {
            "binding_accuracy": binding_hits / max(binding_total, 1),
            "context_satisfaction": context_hits / max(context_total, 1),
            "semantic_id_hit_rate@10": float(np.mean(sid_coverages)),
            "semantic_id_candidate_mean": float(np.mean(sid_candidate_counts)),
            "latency_ms_mean": float(np.mean(latencies)),
            "latency_ms_p95": float(np.percentile(latencies, 95)),
        }
    )
    result = {"semantic_id_hybrid": metrics, "diagnostics": diagnostics}
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def tune_semantic_id_weights(
    index: DenseIndex,
    encoder,
    queries: list[JudgedQuery],
    semantic_dir: str | Path,
    output: str | Path,
    candidate_pool: int = 100,
    sid_pool: int = 100,
    k: int = 10,
    trials: int = 1200,
    seed: int = 42,
) -> dict:
    """Choose hybrid score weights on validation data without touching test queries."""
    names = ("dense", "binding", "context", "sid", "style")
    default = np.asarray([0.40, 0.20, 0.12, 0.10, 0.18], dtype=np.float32)
    retriever = SemanticIDHybridRetriever(index, encoder, semantic_dir, candidate_pool=candidate_pool, sid_pool=sid_pool)
    cached: list[tuple[list[str], np.ndarray]] = []
    for query in queries:
        qvec = encoder.encode_texts([query.text])[0]
        query_sid = retriever.predict_semantic_id(query.text)
        dense_ids, dense_scores = index.search(qvec, max(k, candidate_pool))
        dense_lookup = {int(idx): float(score) for idx, score in zip(dense_ids, dense_scores)}
        sid_candidates = retriever._sid_candidates(query_sid)
        candidate_ids = list(dict.fromkeys([int(idx) for idx in dense_ids] + sid_candidates))
        constraints = parse_query(query.text)
        ids, features = [], []
        for idx in candidate_ids:
            record = index.records[idx]
            cosine = dense_lookup.get(idx, float(index.vectors[idx] @ qvec))
            binding, context, style, _ = structured_scores(constraints, record)
            sid_score = semantic_id_relation(query_sid, retriever.image_to_sid.get(record.image_id, ()))
            ids.append(record.image_id)
            features.append([(cosine + 1.0) / 2.0, binding, context, sid_score, style])
        cached.append((ids, np.asarray(features, dtype=np.float32)))

    rng = np.random.default_rng(seed)
    candidates = np.vstack([default[None], rng.dirichlet(np.ones(len(names)), size=trials)]).astype(np.float32)
    judgments = [query.relevance for query in queries]
    best_metrics: dict[str, float] | None = None
    best_weights: np.ndarray | None = None
    best_key: tuple[float, float, float] | None = None
    for weights in candidates:
        rankings = []
        for ids, features in cached:
            order = np.argsort(-(features @ weights), kind="stable")[:k]
            rankings.append([ids[int(position)] for position in order])
        metrics = evaluate_rankings(rankings, judgments)
        # P@1 is the requested accuracy measure; ranking quality resolves ties.
        key = (metrics["precision@1"], metrics["ndcg@10"], metrics["map@10"])
        if best_key is None or key > best_key:
            best_key, best_metrics, best_weights = key, metrics, weights.copy()
    assert best_metrics is not None and best_weights is not None
    result = {
        "selection_split": "validation",
        "objective": "precision@1, then nDCG@10, then mAP@10",
        "trials": len(candidates),
        "default_weights": dict(zip(names, map(float, default))),
        "selected_weights": dict(zip(names, map(float, best_weights))),
        "selected_validation": best_metrics,
        "candidate_pool": candidate_pool,
        "sid_pool": sid_pool,
    }
    target = Path(output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def sample_semantic_id_retrievals(
    index: DenseIndex,
    encoder,
    queries: list[JudgedQuery],
    semantic_dir: str | Path,
    output: str | Path,
    weights: dict[str, float] | None = None,
    samples: int = 5,
) -> dict:
    """Run a small, environment-diverse qualitative sample from judged queries."""
    retriever = SemanticIDHybridRetriever(index, encoder, semantic_dir, weights=weights)
    source_records = {record.image_id: record for record in index.records}
    selected: list[JudgedQuery] = []
    used: set[str] = set()
    for environment in ("office", "urban_street", "park", "home"):
        query = next((item for item in queries if item.query_id.removeprefix("auto_") in source_records
                      and source_records[item.query_id.removeprefix("auto_")].environment == environment), None)
        if query is not None:
            selected.append(query); used.add(query.query_id)
    for query in queries:
        if len(selected) >= samples:
            break
        if query.query_id not in used:
            selected.append(query); used.add(query.query_id)

    rows = []
    for query in selected[:samples]:
        results = retriever.search(query.text, 1)
        source_id = query.query_id.removeprefix("auto_")
        rows.append({
            "query_id": query.query_id,
            "query": query.text,
            "source_image_id": source_id,
            "source_environment": source_records.get(source_id).environment if source_id in source_records else "unknown",
            "top_result": None if not results else {
                "image_id": results[0].record.image_id,
                "image_path": results[0].record.image_path,
                "caption": results[0].record.caption,
                "garments": [garment.model_dump() for garment in results[0].record.garments],
                "environment": results[0].record.environment,
                "score": results[0].score,
                "binding_score": results[0].binding_score,
                "context_score": results[0].context_score,
                "relevance_grade": query.relevance.get(results[0].record.image_id, 0),
            },
        })
    result = {"selection": "first available query from each environment, then original order", "samples": rows}
    target = Path(output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
