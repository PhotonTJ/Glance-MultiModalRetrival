from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from .encoder import l2_normalize
from .schema import ImageRecord


DEFAULT_FAISS_CONFIG: dict[str, int | str] = {
    "index_type": "ivf_pq",
    "nlist": 16,
    "m": 16,
    "nbits": 4,
    "nprobe": 16,
    "rerank_factor": 6,
    "min_ivfpq_vectors": 64,
}


def _compatible_subquantizers(dimension: int, requested: int) -> int:
    """Return the largest requested-or-smaller PQ block count dividing dimension."""
    return next(value for value in range(min(requested, dimension), 0, -1) if dimension % value == 0)


class DenseIndex:
    """Persisted FAISS IVF-PQ candidate index with exact candidate rescoring.

    ``vectors`` remains available because RQ-VAE/siamese training and the final
    hybrid reranker require the normalized float vectors. Serving candidate
    generation is performed by the persisted FAISS index, not a NumPy scan.
    """

    def __init__(
        self,
        vectors: np.ndarray,
        records: list[ImageRecord],
        faiss_config: dict[str, Any] | None = None,
        faiss_index: Any | None = None,
    ):
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or not len(vectors):
            raise ValueError("vectors must be a non-empty 2D array")
        if len(vectors) != len(records):
            raise ValueError("vectors and records must have equal length")
        self.vectors = np.ascontiguousarray(l2_normalize(vectors), dtype=np.float32)
        self.records = records
        self.faiss_config = {**DEFAULT_FAISS_CONFIG, **(faiss_config or {})}
        self.faiss_index = faiss_index or self._build_faiss_index()
        if hasattr(self.faiss_index, "nprobe"):
            self.faiss_index.nprobe = min(int(self.faiss_config["nprobe"]), int(self.faiss_index.nlist))

    def _build_faiss_index(self):
        count, dimension = self.vectors.shape
        requested_type = str(self.faiss_config["index_type"]).lower()
        if requested_type != "ivf_pq":
            raise ValueError(f"Unsupported FAISS index type: {requested_type}")

        if count < int(self.faiss_config["min_ivfpq_vectors"]):
            index = faiss.IndexFlatIP(dimension)
            index.add(self.vectors)
            return index

        nlist = min(int(self.faiss_config["nlist"]), count)
        m = _compatible_subquantizers(dimension, int(self.faiss_config["m"]))
        nbits = int(self.faiss_config["nbits"])
        if count < 2**nbits:
            raise ValueError(f"IVF-PQ needs at least {2**nbits} vectors for {nbits}-bit codebooks")
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
        index.train(self.vectors)
        index.add(self.vectors)
        index.nprobe = min(int(self.faiss_config["nprobe"]), nlist)
        return index

    @property
    def serving_index_type(self) -> str:
        return "IndexIVFPQ" if isinstance(self.faiss_index, faiss.IndexIVFPQ) else "IndexFlatIP"

    def search(self, query_vector: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        if k <= 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        query = np.ascontiguousarray(l2_normalize(np.asarray(query_vector).reshape(1, -1)), dtype=np.float32)
        k = min(k, len(self.records))
        search_k = min(len(self.records), max(k, k * int(self.faiss_config["rerank_factor"])))
        _, candidate_matrix = self.faiss_index.search(query, search_k)
        candidates = candidate_matrix[0]
        candidates = candidates[candidates >= 0]

        # Sparse probed lists can occasionally return fewer than k candidates.
        # Retry all IVF lists before rescoring; candidate generation remains FAISS-backed.
        if len(candidates) < k and hasattr(self.faiss_index, "nprobe"):
            old_nprobe = int(self.faiss_index.nprobe)
            self.faiss_index.nprobe = int(self.faiss_index.nlist)
            _, candidate_matrix = self.faiss_index.search(query, search_k)
            self.faiss_index.nprobe = old_nprobe
            candidates = candidate_matrix[0]
            candidates = candidates[candidates >= 0]

        candidates = np.unique(candidates)
        scores = self.vectors[candidates] @ query[0]
        order = np.argsort(-scores, kind="stable")[:k]
        return candidates[order], scores[order]

    def save(self, directory: str | Path) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss_index, str(target / "index.faiss"))
        # Raw vectors are an offline/training and candidate-rescoring artifact;
        # index.faiss is the persisted serving index.
        np.save(target / "vectors.npy", self.vectors)
        (target / "records.json").write_text(
            json.dumps([r.model_dump() for r in self.records], ensure_ascii=False), encoding="utf-8"
        )
        metadata = {
            **self.faiss_config,
            "serving_index": self.serving_index_type,
            "metric": "cosine_via_normalized_inner_product",
            "dimension": int(self.vectors.shape[1]),
            "vectors": int(self.vectors.shape[0]),
            "raw_vectors_purpose": "offline training and exact candidate rescoring",
        }
        (target / "index_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "DenseIndex":
        target = Path(directory)
        records = [ImageRecord.model_validate(r) for r in json.loads((target / "records.json").read_text(encoding="utf-8"))]
        config_path = target / "index_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else None
        index_path = target / "index.faiss"
        faiss_index = faiss.read_index(str(index_path)) if index_path.exists() else None
        return cls(np.load(target / "vectors.npy"), records, config, faiss_index)
