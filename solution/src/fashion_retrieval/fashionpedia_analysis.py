from __future__ import annotations

import colorsys
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .schema import Garment, ImageRecord, write_jsonl


MAIN_CATEGORY_MAX_ID = 26
ENVIRONMENT_PROMPTS = {
    "office": "a person inside a modern office or workplace",
    "urban_street": "a person outdoors on an urban city street or sidewalk",
    "park": "a person outdoors in a green public park or garden",
    "home": "a person inside a home, bedroom, kitchen, or living room",
}


def load_fashionpedia(annotation_paths: list[str | Path]) -> tuple[list[dict], list[dict], dict[int, dict]]:
    images, annotations, categories = [], [], {}
    for split_index, path in enumerate(annotation_paths):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        split = "train" if "train" in Path(path).name else "validation"
        categories.update({item["id"]: item for item in payload["categories"]})
        images.extend([{**item, "split": split} for item in payload["images"]])
        annotations.extend([{**item, "split": split} for item in payload["annotations"]])
    return images, annotations, categories


def native_clothing_summary(annotation_paths: list[str | Path], output: str | Path | None = None) -> dict:
    images, annotations, categories = load_fashionpedia(annotation_paths)
    primary = [a for a in annotations if a["category_id"] <= MAIN_CATEGORY_MAX_ID]
    per_image = Counter(a["image_id"] for a in primary)
    summary = {
        "scope": "Fashionpedia train + validation (full annotated corpus)",
        "images": len(images),
        "primary_garment_instances": len(primary),
        "all_annotations": len(annotations),
        "clothing_categories": dict(Counter(categories[a["category_id"]]["name"] for a in primary)),
        "clothing_families": dict(Counter(categories[a["category_id"]]["supercategory"] for a in primary)),
        "garments_per_image": {
            "mean": float(np.mean(list(per_image.values()))),
            "median": float(np.median(list(per_image.values()))),
            "p95": float(np.percentile(list(per_image.values()), 95)),
        },
        "native_axis_availability": {
            "clothing_type": 1.0,
            "garment_color": 0.0,
            "environment": 0.0,
        },
    }
    if output:
        target = Path(output); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def rgb_to_color_name(rgb: np.ndarray) -> str:
    """Map a robust median garment RGB value to the assignment's color vocabulary."""
    r, g, b = np.clip(rgb / 255.0, 0, 1)
    hue, saturation, value = colorsys.rgb_to_hsv(float(r), float(g), float(b))
    degrees = hue * 360
    if value < 0.18: return "black"
    if saturation < 0.12 and value > 0.84: return "white"
    if saturation < 0.18: return "gray"
    if 18 <= degrees < 48 and value < 0.62: return "brown"
    if 25 <= degrees < 55 and saturation < 0.42 and value > 0.65: return "beige"
    if (degrees < 15 or degrees >= 345): return "red"
    if degrees < 42: return "orange"
    if degrees < 70: return "yellow"
    if degrees < 165: return "green"
    if degrees < 255: return "navy" if value < 0.42 else "blue"
    if degrees < 315: return "purple"
    return "pink"


def _polygon_pixels(image: Image.Image, segmentation: list, bbox: list[float], max_samples: int = 4000) -> np.ndarray:
    x, y, width, height = [int(round(value)) for value in bbox]
    x, y = max(0, x), max(0, y)
    right, bottom = min(image.width, x + max(width, 1)), min(image.height, y + max(height, 1))
    crop = image.crop((x, y, right, bottom)).convert("RGB")
    mask = Image.new("1", crop.size)
    draw = ImageDraw.Draw(mask)
    if isinstance(segmentation, list):
        for polygon in segmentation:
            points = [(polygon[i] - x, polygon[i + 1] - y) for i in range(0, len(polygon) - 1, 2)]
            if len(points) >= 3: draw.polygon(points, fill=1)
    pixels = np.asarray(crop)[np.asarray(mask, dtype=bool)]
    if len(pixels) > max_samples:
        pixels = pixels[np.linspace(0, len(pixels) - 1, max_samples).astype(int)]
    return pixels


def extract_mask_colors(
    annotation_path: str | Path,
    image_dir: str | Path,
    output: str | Path,
    min_area: int = 500,
) -> dict:
    """Extract a robust median color inside each native garment polygon mask."""
    payload = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    images = {item["id"]: item for item in payload["images"]}
    categories = {item["id"]: item["name"] for item in payload["categories"]}
    by_image = defaultdict(list)
    for annotation in payload["annotations"]:
        if annotation["category_id"] <= MAIN_CATEGORY_MAX_ID and annotation.get("area", 0) >= min_area:
            by_image[annotation["image_id"]].append(annotation)
    target = Path(output); target.parent.mkdir(parents=True, exist_ok=True)
    color_counts, processed, skipped = Counter(), 0, 0
    with target.open("w", encoding="utf-8") as handle:
        for image_id, annotations in by_image.items():
            path = Path(image_dir) / images[image_id]["file_name"]
            try:
                with Image.open(path) as image:
                    for annotation in annotations:
                        pixels = _polygon_pixels(image, annotation.get("segmentation", []), annotation["bbox"])
                        if len(pixels) < 25:
                            skipped += 1; continue
                        # Median limits highlights/background leakage; discard near-transparent mask edges by construction.
                        rgb = np.median(pixels, axis=0)
                        color = rgb_to_color_name(rgb); color_counts[color] += 1; processed += 1
                        handle.write(json.dumps({"annotation_id": annotation["id"], "image_id": image_id,
                                                 "garment": categories[annotation["category_id"]], "color": color,
                                                 "median_rgb": [round(float(x), 1) for x in rgb]}) + "\n")
            except OSError:
                skipped += len(annotations)
    return {"processed_instances": processed, "skipped_instances": skipped, "colors": dict(color_counts)}


def classify_environments(
    annotation_path: str | Path,
    image_dir: str | Path,
    output: str | Path,
    batch_size: int = 32,
    confidence_threshold: float = 0.40,
    margin_threshold: float = 0.08,
) -> dict:
    """Four-way CLIP scene classification with an uncertainty-aware ``unknown`` class."""
    import torch
    from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

    payload = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    items = [(item["id"], Path(image_dir) / item["file_name"]) for item in payload["images"]]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "openai/clip-vit-base-patch32"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForZeroShotImageClassification.from_pretrained(model_name).to(device).eval()
    labels, prompts = list(ENVIRONMENT_PROMPTS), list(ENVIRONMENT_PROMPTS.values())
    counts = Counter(); target = Path(output); target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            valid = []
            for image_id, path in batch:
                try:
                    with Image.open(path) as image: valid.append((image_id, image.convert("RGB").copy()))
                except OSError: continue
            if not valid: continue
            inputs = processor(text=prompts, images=[x[1] for x in valid], return_tensors="pt", padding=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            probabilities = model(**inputs).logits_per_image.softmax(dim=1).cpu().numpy()
            for (image_id, _), scores in zip(valid, probabilities):
                order = np.argsort(-scores); best, second = int(order[0]), int(order[1])
                label = labels[best] if scores[best] >= confidence_threshold and scores[best] - scores[second] >= margin_threshold else "unknown"
                counts[label] += 1
                handle.write(json.dumps({"image_id": image_id, "environment": label,
                                         "confidence": round(float(scores[best]), 4),
                                         "margin": round(float(scores[best] - scores[second]), 4)}) + "\n")
    return {"model": model_name, "counts": dict(counts), "images": sum(counts.values())}


def build_balanced_subset(
    annotation_path: str | Path,
    image_dir: str | Path,
    colors_path: str | Path,
    environments_path: str | Path,
    output: str | Path,
    size: int = 1000,
    seed: int = 42,
) -> dict:
    """Build an environment-first subset, then retain clothing/color diversity."""
    import hashlib
    import random

    payload = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    categories = {item["id"]: item for item in payload["categories"]}
    image_info = {item["id"]: item for item in payload["images"]}
    color_by_annotation = {}
    with Path(colors_path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line); color_by_annotation[row["annotation_id"]] = row["color"]
    environment_by_image = {}
    with Path(environments_path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line); environment_by_image[row["image_id"]] = row["environment"]
    garments_by_image = defaultdict(list)
    for annotation in payload["annotations"]:
        if annotation["category_id"] <= MAIN_CATEGORY_MAX_ID:
            category = categories[annotation["category_id"]]["name"].replace(",", " /")
            garments_by_image[annotation["image_id"]].append(
                Garment(type=category, color=color_by_annotation.get(annotation["id"], "unknown"))
            )

    records = []
    for image_id, item in image_info.items():
        garments = garments_by_image[image_id]
        environment = environment_by_image.get(image_id, "unknown")
        description = ", ".join(f"{g.color} {g.type}" for g in garments[:6]) or "fashion clothing"
        bucket = int(hashlib.sha1(str(image_id).encode()).hexdigest()[:8], 16) % 100
        split = "train" if bucket < 70 else "validation" if bucket < 85 else "test"
        records.append(ImageRecord(image_id=f"fashionpedia_{image_id}", image_path=str(Path(image_dir) / item["file_name"]),
                                   caption=f"A person wearing {description} in a {environment.replace('_', ' ')} setting.",
                                   garments=garments, environment=environment, source="fashionpedia_val2020", split=split))

    rng = random.Random(seed)
    groups = defaultdict(list)
    for record in records: groups[record.environment].append(record)
    for group in groups.values(): rng.shuffle(group)
    selected = []; quota = size // 4
    for label in ENVIRONMENT_PROMPTS:
        selected.extend(groups[label][:quota])
        groups[label] = groups[label][quota:]
    remaining = [record for group in groups.values() for record in group]
    # Prefer records with known mask-derived colors and more garment variety when filling shortages.
    rng.shuffle(remaining)
    remaining.sort(key=lambda r: (sum(g.color != "unknown" for g in r.garments), len({g.type for g in r.garments})), reverse=True)
    selected.extend(remaining[: max(0, size - len(selected))])
    selected = selected[:size]
    write_jsonl(output, selected)
    return {
        "available_images": len(records), "selected_images": len(selected),
        "environment_counts": dict(Counter(r.environment for r in selected)),
        "garment_instances": sum(len(r.garments) for r in selected),
        "known_color_instances": sum(g.color != "unknown" for r in selected for g in r.garments),
        "selection_rule": "up to 250 per requested environment, then fill by known color and garment diversity",
    }
