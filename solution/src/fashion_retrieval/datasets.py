from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

from .schema import Garment, ImageRecord, write_jsonl


def validate_image(path: Path, min_side: int = 224) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.mode in {"RGB", "RGBA"} and min(image.size) >= min_side
    except (OSError, ValueError):
        return False


def _stable_split(image_id: str) -> str:
    bucket = int(hashlib.sha1(image_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def fashionpedia_manifest(
    annotation_file: str | Path,
    image_dir: str | Path,
    output: str | Path,
    max_images: int = 1000,
    min_side: int = 224,
    seed: int = 42,
) -> list[ImageRecord]:
    """Convert official Fashionpedia COCO+attribute JSON into the project schema.

    Fashionpedia has garment-localized categories and attributes but not the four assignment
    environments. Those remain ``unknown`` until manually or VLM annotated; this is explicit
    to prevent fabricated context labels.
    """
    payload = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
    categories = {item["id"]: item["name"].lower().replace(" ", "_") for item in payload["categories"]}
    attributes = {item["id"]: item["name"].lower().replace(" ", "_") for item in payload.get("attributes", [])}
    by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in payload["annotations"]:
        by_image[annotation["image_id"]].append(annotation)

    candidates = list(payload["images"])
    random.Random(seed).shuffle(candidates)
    records: list[ImageRecord] = []
    for item in candidates:
        path = Path(image_dir) / item["file_name"]
        if not validate_image(path, min_side):
            continue
        garments = []
        for annotation in by_image.get(item["id"], []):
            labels = [attributes.get(i, "") for i in annotation.get("attribute_ids", [])]
            # Attribute names vary by release; retain any literal color label, else unknown.
            color = next((label for label in labels if label in {"black", "white", "gray", "red", "orange", "yellow", "green", "blue", "navy", "purple", "pink", "brown", "beige"}), "unknown")
            garments.append(Garment(type=categories[annotation["category_id"]], color=color))
        record = ImageRecord(
            image_id=f"fashionpedia_{item['id']}", image_path=str(path), garments=garments,
            source="fashionpedia", split=_stable_split(str(item["id"])),
        )
        records.append(record)
        if len(records) >= max_images:
            break
    write_jsonl(output, records)
    return records


DATASET_GUIDE = {
    "Fashionpedia": "Primary: real-world images, garment masks, 294 localized attributes; add scene labels.",
    "DeepFashion2": "Optional supplement: 491K images and 801K clothing items with masks/landmarks.",
    "FashionIQ": "Evaluation supplement: fine-grained human language, but weak environment coverage.",
}

