from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)


class QwenEncoder:
    """Thin wrapper around the official SentenceTransformers Qwen3-VL interface."""

    def __init__(self, model_name: str = "Qwen/Qwen3-VL-Embedding-2B", dimension: int = 256, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device, truncate_dim=dimension)

    def encode_texts(self, texts: Sequence[str], batch_size: int = 8, prompt: str | None = None) -> np.ndarray:
        return l2_normalize(self.model.encode(list(texts), batch_size=batch_size, prompt=prompt, convert_to_numpy=True))

    def encode_images(self, paths: Sequence[str | Path], batch_size: int = 8) -> np.ndarray:
        # Qwen's official SentenceTransformers API accepts local image paths directly.
        return l2_normalize(self.model.encode([str(Path(p).resolve()) for p in paths], batch_size=batch_size, convert_to_numpy=True))


class HashEncoder:
    """Deterministic, dependency-light encoder for tests only; never use for reported ML results."""

    def __init__(self, dimension: int = 256):
        self.dimension = dimension

    def _one(self, value: str) -> np.ndarray:
        tokens = value.lower().replace("_", " ").split()
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in tokens:
            digest = hashlib.sha256(token.encode()).digest()
            vector[int.from_bytes(digest[:4], "little") % self.dimension] += 1.0
        return vector

    def encode_texts(self, texts: Sequence[str], **_: object) -> np.ndarray:
        return l2_normalize(np.stack([self._one(t) for t in texts]))

    def encode_images(self, paths: Sequence[str | Path], **_: object) -> np.ndarray:
        return l2_normalize(np.stack([self._one(Path(p).stem) for p in paths]))


class ProjectedEncoder:
    """Apply a trained shared Siamese head to any compatible base encoder."""

    def __init__(self, base_encoder, checkpoint: str | Path):
        import torch
        from .models import SiameseProjection

        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.base_encoder = base_encoder
        self.model = SiameseProjection(payload["input_dim"], max(512, payload["output_dim"] * 2), payload["output_dim"])
        self.model.load_state_dict(payload["state_dict"]); self.model.eval()

    def _project(self, vectors: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            return self.model(torch.from_numpy(vectors).float()).numpy()

    def encode_texts(self, texts: Sequence[str], **kwargs: object) -> np.ndarray:
        return self._project(self.base_encoder.encode_texts(texts, **kwargs))

    def encode_images(self, paths: Sequence[str | Path], **kwargs: object) -> np.ndarray:
        return self._project(self.base_encoder.encode_images(paths, **kwargs))
