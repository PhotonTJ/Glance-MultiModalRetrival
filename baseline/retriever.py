"""Run natural-language search against the baseline dense-vector index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


def search(query: str, index_dir: Path, k: int) -> list[dict]:
    config = json.loads((index_dir / "index_config.json").read_text(encoding="utf-8"))
    records = json.loads((index_dir / "records.json").read_text(encoding="utf-8"))
    index = faiss.read_index(str(index_dir / "index.faiss"))
    model = SentenceTransformer(config["model"], truncate_dim=config["dimension"])
    query_vector = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, ids = index.search(query_vector, min(k, len(records)))

    return [
        {
            "rank": rank,
            "image_id": records[row_id]["image_id"],
            "image_path": records[row_id]["image_path"],
            "score": round(float(score), 6),
        }
        for rank, (row_id, score) in enumerate(zip(ids[0], scores[0]), start=1)
        if row_id >= 0
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the baseline image index")
    parser.add_argument("query", help="Natural-language image description")
    parser.add_argument("--index", type=Path, default=Path("baseline/index"))
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(search(args.query, args.index, args.k), indent=2))


if __name__ == "__main__":
    main()
