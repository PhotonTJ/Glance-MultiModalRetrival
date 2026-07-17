"""Part A entry point: extract image features and persist the FAISS index."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from fashion_retrieval.encoder import QwenEncoder
from fashion_retrieval.index import DenseIndex
from fashion_retrieval.schema import ImageRecord, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the solution image index")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/indexes/main"))
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    records = load_jsonl(args.metadata, ImageRecord)
    encoder = QwenEncoder(config["encoder"]["model"], config["encoder"]["dimension"])
    vectors = encoder.encode_images(
        [record.image_path for record in records],
        batch_size=config["encoder"]["batch_size"],
    )
    index = DenseIndex(vectors, records, config["faiss"])
    index.save(args.output)
    print(f"Indexed {len(records)} images with {index.serving_index_type} in {args.output}")


if __name__ == "__main__":
    main()
