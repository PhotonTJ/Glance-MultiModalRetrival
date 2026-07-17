"""Build the dense-vector index used by the baseline workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def read_records(metadata_path: Path) -> list[dict]:
    with metadata_path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_image_path(image_path: str, metadata_path: Path) -> Path:
    path = Path(image_path)
    if path.is_absolute() or path.exists():
        return path

    # Metadata stores paths from the workflow root (for example data/raw/...).
    for parent in metadata_path.resolve().parents:
        candidate = parent / path
        if candidate.exists():
            return candidate
    return path


def build_index(metadata: Path, output: Path, model_name: str, dimension: int, batch_size: int) -> None:
    records = read_records(metadata)
    if not records:
        raise ValueError(f"No image records found in {metadata}")

    image_paths = [str(resolve_image_path(row["image_path"], metadata).resolve()) for row in records]
    missing = [path for path in image_paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Could not find {len(missing)} images; first missing path: {missing[0]}")

    model = SentenceTransformer(model_name, truncate_dim=dimension)
    vectors = model.encode(image_paths, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    output.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output / "index.faiss"))
    (output / "records.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    (output / "index_config.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "dimension": int(vectors.shape[1]),
                "images": len(records),
                "metric": "cosine similarity",
                "index": "FAISS IndexFlatIP",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Indexed {len(records)} images in {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the baseline image index")
    parser.add_argument("--metadata", type=Path, required=True, help="JSONL image manifest")
    parser.add_argument("--output", type=Path, default=Path("baseline/index"))
    parser.add_argument("--model", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    build_index(args.metadata, args.output, args.model, args.dimension, args.batch_size)


if __name__ == "__main__":
    main()
