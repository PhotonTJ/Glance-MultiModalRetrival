"""Part B entry point: retrieve images for a multi-attribute text query."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from fashion_retrieval.encoder import QwenEncoder
from fashion_retrieval.index import DenseIndex
from fashion_retrieval.semantic_id import SemanticIDHybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Search with the Semantic-ID hybrid retriever")
    parser.add_argument("query", help="Natural-language image description")
    parser.add_argument("--index", type=Path, default=Path("artifacts/indexes/fashionpedia_qwen"))
    parser.add_argument("--semantic-ids", type=Path, default=Path("artifacts/checkpoints/rqvae_direct_qwen"))
    parser.add_argument("--weights", type=Path, default=Path("artifacts/results/direct_semantic_binding_weight_tuning.json"))
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    weight_payload = json.loads(args.weights.read_text(encoding="utf-8"))
    weights = weight_payload.get(
        "selected_weights",
        weight_payload.get("best_weights", weight_payload.get("weights", weight_payload)),
    )
    index = DenseIndex.load(args.index)
    encoder = QwenEncoder(config["encoder"]["model"], config["encoder"]["dimension"])
    retriever = SemanticIDHybridRetriever(index, encoder, args.semantic_ids, weights=weights)
    results = retriever.search(args.query, args.k)
    print(
        json.dumps(
            [
                {
                    "rank": rank,
                    "image_id": result.record.image_id,
                    "image_path": result.record.image_path,
                    "score": round(result.score, 6),
                    "explanation": result.explanation,
                }
                for rank, result in enumerate(results, start=1)
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
