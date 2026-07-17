from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

from .index import DenseIndex
from .models import ResidualQuantizer, SiameseProjection, distance_regression_loss
from .schema import ImageRecord


def record_text(record: ImageRecord) -> str:
    if record.caption.strip():
        return record.caption
    garments = ", ".join(f"{g.color} {g.type}" for g in record.garments)
    extras = ", ".join(x for x in [record.environment, record.activity, " ".join(record.style)] if x != "unknown")
    return f"A person wearing {garments}. {extras}.".strip()


def _hard_negative_indices(records: list[ImageRecord], seed: int = 42) -> np.ndarray:
    """Prefer swapped bindings or same garments in the wrong context."""
    rng = random.Random(seed)
    negatives = []
    for i, anchor in enumerate(records):
        anchor_pairs = {(g.type, g.color) for g in anchor.garments}
        anchor_types = {g.type for g in anchor.garments}
        best_score, candidates = -1, []
        for j, item in enumerate(records):
            if i == j:
                continue
            item_pairs = {(g.type, g.color) for g in item.garments}
            item_types = {g.type for g in item.garments}
            score = 3 * len(anchor_types & item_types) - 2 * len(anchor_pairs & item_pairs)
            score += int(anchor.environment != item.environment)
            if score > best_score:
                best_score, candidates = score, [j]
            elif score == best_score:
                candidates.append(j)
        negatives.append(rng.choice(candidates) if candidates else i)
    return np.asarray(negatives, dtype=np.int64)


def train_siamese(
    index: DenseIndex,
    encoder,
    output_dir: str | Path,
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    output_dim: int = 256,
    seed: int = 42,
) -> dict[str, list[float]]:
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_image_vectors = torch.from_numpy(index.vectors).float()
    train_ids = [i for i, record in enumerate(index.records) if record.split == "train"]
    if not train_ids:  # Enables the explicitly synthetic smoke set; real experiments must have train splits.
        train_ids = list(range(len(index.records)))
    train_records = [index.records[i] for i in train_ids]
    image_vectors = all_image_vectors[train_ids]
    query_vectors = torch.from_numpy(encoder.encode_texts([record_text(r) for r in train_records])).float()
    negatives = _hard_negative_indices(train_records, seed)
    model = SiameseProjection(all_image_vectors.shape[1], max(512, output_dim * 2), output_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history = defaultdict(list)

    for _ in range(epochs):
        permutation = torch.randperm(len(train_records))
        totals = defaultdict(float); batches = 0
        model.train()
        for start in range(0, len(permutation), batch_size):
            ids = permutation[start:start + batch_size]
            q = model(query_vectors[ids].to(device))
            positive = model(image_vectors[ids].to(device))
            negative_ids = torch.from_numpy(negatives[ids.numpy()])
            negative = model(image_vectors[negative_ids].to(device))
            target = torch.ones(len(ids), device=device)
            loss, parts = distance_regression_loss(q, positive, target, negative)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["total"] += float(loss.detach())
            for name, value in parts.items(): totals[name] += float(value.detach())
            batches += 1
        for name, value in totals.items(): history[name].append(value / max(batches, 1))

    target = Path(output_dir); target.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "input_dim": image_vectors.shape[1], "output_dim": output_dim}, target / "siamese.pt")
    model.eval()
    with torch.no_grad():
        projected = model(all_image_vectors.to(device)).cpu().numpy()
    DenseIndex(projected, index.records).save(target / "index")
    (target / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return dict(history)


@torch.no_grad()
def _kmeans_codebooks(model: ResidualQuantizer, vectors: torch.Tensor, iterations: int = 12) -> None:
    latent = model.encoder(vectors)
    residual = latent.clone()
    for level, codebook in enumerate(model.codebooks):
        count = codebook.shape[0]
        seeds = residual[torch.linspace(0, len(residual) - 1, count).long() % len(residual)].clone()
        for _ in range(iterations):
            assignments = torch.cdist(residual, seeds).argmin(dim=1)
            for code in range(count):
                members = residual[assignments == code]
                if len(members): seeds[code] = members.mean(dim=0)
        model.codebooks[level].copy_(seeds)
        residual -= seeds[torch.cdist(residual, seeds).argmin(dim=1)]


def _attribute_supervision(records: list[ImageRecord], device: torch.device) -> tuple[dict[str, list[str]], dict[str, torch.Tensor]]:
    vocab = {
        "garment": sorted({g.type for record in records for g in record.garments if g.type != "unknown"}),
        "binding": sorted({f"{g.type}|{g.color}" for record in records for g in record.garments if g.type != "unknown" and g.color != "unknown"}),
        "environment": sorted({record.environment for record in records if record.environment != "unknown"}),
        "style": sorted({style for record in records for style in record.style if style != "unknown"}),
    }

    def multi_hot(values: list[set[str]], names: list[str]) -> torch.Tensor:
        positions = {name: idx for idx, name in enumerate(names)}
        result = torch.zeros((len(values), len(names)), device=device)
        for row, row_values in enumerate(values):
            for value in row_values:
                if value in positions:
                    result[row, positions[value]] = 1.0
        return result

    targets = {
        "garment": multi_hot([{g.type for g in record.garments if g.type != "unknown"} for record in records], vocab["garment"]),
        "binding": multi_hot([
            {f"{g.type}|{g.color}" for g in record.garments if g.type != "unknown" and g.color != "unknown"}
            for record in records
        ], vocab["binding"]),
        "style": multi_hot([set(record.style) for record in records], vocab["style"]),
    }
    environment_positions = {name: idx for idx, name in enumerate(vocab["environment"])}
    targets["environment"] = torch.tensor(
        [environment_positions.get(record.environment, -100) for record in records], dtype=torch.long, device=device
    )
    return vocab, targets


def _positive_weights(target: torch.Tensor) -> torch.Tensor:
    positives = target.sum(dim=0).clamp_min(1.0)
    negatives = target.shape[0] - positives
    return (negatives / positives).clamp(1.0, 10.0)


def train_rqvae(
    index: DenseIndex,
    output_dir: str | Path,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    latent_dim: int = 64,
    levels: int = 3,
    codebook_size: int = 16,
    attribute_weight: float = 0.0,
    seed: int = 42,
) -> dict[str, list[float]]:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_vectors = torch.from_numpy(index.vectors).float().to(device)
    train_ids = [i for i, record in enumerate(index.records) if record.split == "train"]
    if not train_ids:
        train_ids = list(range(len(index.records)))
    vectors = all_vectors[train_ids]
    train_records = [index.records[i] for i in train_ids]
    model = ResidualQuantizer(all_vectors.shape[1], min(latent_dim, all_vectors.shape[1]), levels, codebook_size).to(device)
    # Brief reconstruction warm-up gives k-means a meaningful latent space.
    warm_optimizer = AdamW(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=learning_rate)
    for _ in range(min(5, epochs)):
        latent = model.encoder(vectors); loss = torch.nn.functional.mse_loss(model.decoder(latent), vectors)
        warm_optimizer.zero_grad(); loss.backward(); warm_optimizer.step()
    _kmeans_codebooks(model, vectors)
    attribute_vocab, attribute_targets = _attribute_supervision(train_records, device)
    heads = torch.nn.ModuleDict({
        name: torch.nn.Linear(min(latent_dim, all_vectors.shape[1]), len(values))
        for name, values in attribute_vocab.items() if values
    }).to(device)
    optimizer = AdamW(list(model.parameters()) + list(heads.parameters()), lr=learning_rate)
    history = defaultdict(list)
    for _ in range(epochs):
        _, _, losses = model(vectors)
        if attribute_weight > 0 and heads:
            prefixes = model.soft_quantized_prefixes(vectors)
            attribute_parts = []
            if "garment" in heads:
                garment = torch.nn.functional.binary_cross_entropy_with_logits(
                    heads["garment"](prefixes[:, 0]), attribute_targets["garment"],
                    pos_weight=_positive_weights(attribute_targets["garment"]),
                )
                attribute_parts.append(0.25 * garment)
                losses["attribute_garment"] = garment
            if "binding" in heads:
                binding_level = min(1, levels - 1)
                binding = torch.nn.functional.binary_cross_entropy_with_logits(
                    heads["binding"](prefixes[:, binding_level]), attribute_targets["binding"],
                    pos_weight=_positive_weights(attribute_targets["binding"]),
                )
                attribute_parts.append(0.45 * binding)
                losses["attribute_binding"] = binding
            if "environment" in heads and bool((attribute_targets["environment"] != -100).any()):
                environment = torch.nn.functional.cross_entropy(
                    heads["environment"](prefixes[:, -1]), attribute_targets["environment"], ignore_index=-100
                )
                attribute_parts.append(0.20 * environment)
                losses["attribute_environment"] = environment
            if "style" in heads:
                style = torch.nn.functional.binary_cross_entropy_with_logits(
                    heads["style"](prefixes[:, -1]), attribute_targets["style"],
                    pos_weight=_positive_weights(attribute_targets["style"]),
                )
                attribute_parts.append(0.10 * style)
                losses["attribute_style"] = style
            attribute_loss = torch.stack(attribute_parts).sum()
            losses["attribute"] = attribute_loss
            losses["total"] = losses["total"] + attribute_weight * attribute_loss
        optimizer.zero_grad(); losses["total"].backward(); optimizer.step()
        for name, value in losses.items(): history[name].append(float(value.detach()))

    target = Path(output_dir); target.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "input_dim": all_vectors.shape[1], "latent_dim": min(latent_dim, all_vectors.shape[1]),
                "levels": levels, "codebook_size": codebook_size}, target / "rqvae.pt")
    model.eval()
    codes = model.semantic_ids(all_vectors).cpu().tolist()
    image_to_sid = {record.image_id: code for record, code in zip(index.records, codes)}
    prefixes: dict[str, list[str]] = defaultdict(list)
    for image_id, code in image_to_sid.items():
        for depth in range(1, len(code) + 1): prefixes["/".join(map(str, code[:depth]))].append(image_id)
    (target / "image_to_sid.json").write_text(json.dumps(image_to_sid, indent=2), encoding="utf-8")
    (target / "prefix_to_images.json").write_text(json.dumps(prefixes, indent=2), encoding="utf-8")
    diagnostics = {"used_codes": [len({row[level] for row in codes}) for level in range(levels)],
                   "collision_rate": 1.0 - len({tuple(row) for row in codes}) / len(codes),
                   "attribute_weight": attribute_weight,
                   "attribute_vocabulary_sizes": {name: len(values) for name, values in attribute_vocab.items()}}
    if attribute_weight > 0:
        torch.save({"state_dict": heads.state_dict(), "vocabulary": attribute_vocab}, target / "attribute_heads.pt")
    (target / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    (target / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return dict(history)
