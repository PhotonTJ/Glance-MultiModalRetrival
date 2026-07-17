from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from .classical import _garment_label_matches, _human_garment
from .index import DenseIndex
from .metrics import evaluate_rankings
from .models import AttentionGRUComposer
from .parser import parse_query
from .schema import ImageRecord, JudgedQuery


HEAD_WEIGHTS = {"garment": 0.20, "color": 0.15, "binding": 0.45, "environment": 0.20}


@dataclass(frozen=True)
class AttributeExample:
    components: tuple[str, ...]
    garments: tuple[str, ...]
    colors: tuple[str, ...]
    bindings: tuple[str, ...]
    environment: str | None


def _pair_label(garment: str, color: str) -> str:
    return f"{garment}|||{color}"


def _record_examples(record: ImageRecord, max_single: int = 4) -> list[AttributeExample]:
    known = []
    seen = set()
    for garment in record.garments:
        key = (garment.type, garment.color)
        if garment.color == "unknown" or key in seen:
            continue
        seen.add(key); known.append(garment)
    known.sort(key=lambda g: (g.type, g.color))
    environment = record.environment if record.environment != "unknown" else None
    suffix = (environment.replace("_", " "),) if environment else ()
    output = []
    for garment in known[:max_single]:
        phrase = f"{garment.color} {_human_garment(garment.type)}"
        output.append(AttributeExample(
            (phrase,) + suffix,
            (garment.type,),
            (garment.color,),
            (_pair_label(garment.type, garment.color),),
            environment,
        ))
    if len(known) >= 2:
        selected = known[:2]
        output.append(AttributeExample(
            tuple(f"{g.color} {_human_garment(g.type)}" for g in selected) + suffix,
            tuple(g.type for g in selected),
            tuple(g.color for g in selected),
            tuple(_pair_label(g.type, g.color) for g in selected),
            environment,
        ))
    return output


def _query_components(text: str) -> list[str]:
    constraints = parse_query(text)
    components = [f"{color} {garment.replace('_', ' ')}" for garment, color in constraints.bindings]
    if not components:
        components.extend(sorted(g.replace("_", " ") for g in constraints.garments))
        components.extend(sorted(constraints.colors))
    if constraints.environment:
        components.append(constraints.environment.replace("_", " "))
    return components or [text]


def _labels(records: list[ImageRecord]) -> dict[str, list[str]]:
    train = [record for record in records if record.split == "train"]
    if not train:
        raise ValueError("no train records found")
    garments = sorted({g.type for r in train for g in r.garments if g.color != "unknown"})
    colors = sorted({g.color for r in train for g in r.garments if g.color != "unknown"})
    bindings = sorted({_pair_label(g.type, g.color) for r in train for g in r.garments if g.color != "unknown"})
    environments = sorted({r.environment for r in train if r.environment != "unknown"})
    if not all((garments, colors, bindings, environments)):
        raise ValueError("training data must contain known garment, color, binding, and environment labels")
    return {"garment": garments, "color": colors, "binding": bindings, "environment": environments}


def _encode_examples(examples: list[AttributeExample], encoder, labels: dict[str, list[str]]) -> tuple[torch.Tensor, ...]:
    unique_components = sorted({component for example in examples for component in example.components})
    encoded = encoder.encode_texts(unique_components, batch_size=32)
    lookup = dict(zip(unique_components, encoded))
    max_length = max(len(example.components) for example in examples)
    input_dim = encoded.shape[1]
    inputs = np.zeros((len(examples), max_length, input_dim), dtype=np.float32)
    masks = np.zeros((len(examples), max_length), dtype=bool)
    targets = {
        name: np.zeros((len(examples), len(values)), dtype=np.float32)
        for name, values in labels.items() if name != "environment"
    }
    environment = np.full(len(examples), -100, dtype=np.int64)
    indices = {name: {value: i for i, value in enumerate(values)} for name, values in labels.items()}
    for row, example in enumerate(examples):
        for column, component in enumerate(example.components):
            inputs[row, column] = lookup[component]; masks[row, column] = True
        for value in example.garments:
            if value in indices["garment"]: targets["garment"][row, indices["garment"][value]] = 1
        for value in example.colors:
            if value in indices["color"]: targets["color"][row, indices["color"][value]] = 1
        for value in example.bindings:
            if value in indices["binding"]: targets["binding"][row, indices["binding"][value]] = 1
        if example.environment in indices["environment"]:
            environment[row] = indices["environment"][example.environment]
    return (
        torch.from_numpy(inputs), torch.from_numpy(masks),
        torch.from_numpy(targets["garment"]), torch.from_numpy(targets["color"]),
        torch.from_numpy(targets["binding"]), torch.from_numpy(environment),
    )


def _positive_weights(target: torch.Tensor) -> torch.Tensor:
    positive = target.sum(dim=0)
    negative = len(target) - positive
    return (negative / positive.clamp_min(1)).clamp(1, 20)


def train_attention_gru(
    index: DenseIndex,
    encoder,
    output_dir: str | Path,
    epochs: int = 40,
    batch_size: int = 64,
    hidden_dim: int = 128,
    learning_rate: float = 3e-4,
    patience: int = 6,
    seed: int = 42,
) -> dict:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    labels = _labels(index.records)
    train_examples = [e for r in index.records if r.split == "train" for e in _record_examples(r)]
    validation_examples = [e for r in index.records if r.split == "validation" for e in _record_examples(r)]
    if not train_examples or not validation_examples:
        raise ValueError("attention-GRU training requires non-empty train and validation splits")
    all_tensors = _encode_examples(train_examples + validation_examples, encoder, labels)
    split = len(train_examples)
    train_tensors = tuple(t[:split] for t in all_tensors)
    validation_tensors = tuple(t[split:] for t in all_tensors)
    train_loader = DataLoader(TensorDataset(*train_tensors), batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(TensorDataset(*validation_tensors), batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_sizes = {name: len(values) for name, values in labels.items()}
    model = AttentionGRUComposer(all_tensors[0].shape[-1], hidden_dim, output_sizes).to(device)
    losses = {
        name: nn.BCEWithLogitsLoss(pos_weight=_positive_weights(train_tensors[i]).to(device))
        for name, i in (("garment", 2), ("color", 3), ("binding", 4))
    }
    environment_loss = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history = {"train": [], "validation": []}; best_loss = float("inf"); best_state = None; stale = 0

    def run(loader: DataLoader, training: bool) -> float:
        model.train(training); total = count = 0
        for batch in loader:
            x, mask, garment, color, binding, environment = [value.to(device) for value in batch]
            with torch.set_grad_enabled(training):
                logits, _ = model(x, mask)
                parts = {
                    "garment": losses["garment"](logits["garment"], garment),
                    "color": losses["color"](logits["color"], color),
                    "binding": losses["binding"](logits["binding"], binding),
                    "environment": environment_loss(logits["environment"], environment) if (environment != -100).any()
                    else logits["environment"].sum() * 0,
                }
                loss = sum(HEAD_WEIGHTS[name] * value for name, value in parts.items())
                if training:
                    optimizer.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss.detach()) * len(x); count += len(x)
        return total / max(count, 1)

    for _ in range(epochs):
        train_loss = run(train_loader, True); validation_loss = run(validation_loader, False)
        history["train"].append(train_loss); history["validation"].append(validation_loss)
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}; stale = 0
        else:
            stale += 1
            if stale >= patience: break
    if best_state is None: raise RuntimeError("attention-GRU training did not produce a checkpoint")
    model.load_state_dict(best_state)
    target = Path(output_dir); target.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state, "input_dim": all_tensors[0].shape[-1], "hidden_dim": hidden_dim,
        "labels": labels, "head_weights": HEAD_WEIGHTS,
    }, target / "attention_gru.pt")
    summary = {
        "device": str(device), "train_examples": len(train_examples), "validation_examples": len(validation_examples),
        "epochs_completed": len(history["train"]), "best_validation_loss": best_loss, "history": history,
    }
    (target / "history.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


class AttentionGRUReranker:
    def __init__(self, index: DenseIndex, encoder, checkpoint: str | Path, dense_weight: float = 0.60, candidate_pool: int = 200):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.index = index; self.encoder = encoder; self.labels = payload["labels"]
        self.head_weights = payload.get("head_weights", HEAD_WEIGHTS)
        self.dense_weight = dense_weight; self.candidate_pool = candidate_pool
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AttentionGRUComposer(payload["input_dim"], payload["hidden_dim"], {k: len(v) for k, v in self.labels.items()}).to(self.device)
        self.model.load_state_dict(payload["state_dict"]); self.model.eval()
        self.record_targets = self._record_targets(index.records)

    def _record_targets(self, records: list[ImageRecord]) -> dict[str, np.ndarray]:
        indices = {name: {value: i for i, value in enumerate(values)} for name, values in self.labels.items()}
        output = {name: np.zeros((len(records), len(values)), dtype=np.float32) for name, values in self.labels.items()}
        for row, record in enumerate(records):
            for garment in record.garments:
                if garment.type in indices["garment"]: output["garment"][row, indices["garment"][garment.type]] = 1
                if garment.color in indices["color"]: output["color"][row, indices["color"][garment.color]] = 1
                pair = _pair_label(garment.type, garment.color)
                if pair in indices["binding"]: output["binding"][row, indices["binding"][pair]] = 1
            if record.environment in indices["environment"]: output["environment"][row, indices["environment"][record.environment]] = 1
        return output

    @staticmethod
    def _top_compatibility(probabilities: np.ndarray, targets: np.ndarray, count: int) -> np.ndarray:
        count = max(1, min(count, len(probabilities)))
        chosen = np.argsort(-probabilities, kind="stable")[:count]
        weights = probabilities[chosen]
        return (targets[:, chosen] * weights).sum(axis=1) / max(float(weights.sum()), 1e-8)

    def search(self, query: str, k: int = 10) -> tuple[list[int], dict]:
        components = _query_components(query)
        component_vectors = self.encoder.encode_texts(components, batch_size=32)
        x = torch.from_numpy(component_vectors).unsqueeze(0).to(self.device)
        mask = torch.ones((1, len(components)), dtype=torch.bool, device=self.device)
        with torch.no_grad():
            logits, attentions = self.model(x, mask)
            probabilities = {
                name: (torch.softmax(value, dim=-1) if name == "environment" else torch.sigmoid(value))[0].cpu().numpy()
                for name, value in logits.items()
            }
        qvec = self.encoder.encode_texts([query])[0]
        dense = self.index.vectors @ qvec
        candidates = np.argsort(-dense, kind="stable")[:max(k, self.candidate_pool)]
        constraints = parse_query(query)
        requested = {
            "garment": max(len(constraints.garments), len(constraints.bindings), 1),
            "color": max(len(constraints.colors), len(constraints.bindings), 1),
            "binding": max(len(constraints.bindings), 1),
        }
        scores = {}
        for name in ("garment", "color", "binding"):
            scores[name] = self._top_compatibility(probabilities[name], self.record_targets[name][candidates], requested[name])
        scores["environment"] = self.record_targets["environment"][candidates] @ probabilities["environment"]
        attribute = sum(self.head_weights[name] * scores[name] for name in self.head_weights)
        dense01 = (dense[candidates] + 1) / 2
        total = self.dense_weight * dense01 + (1 - self.dense_weight) * attribute
        order = np.argsort(-total, kind="stable")
        diagnostics = {
            "components": components,
            "attention": {
                name: [float(v) for v in values[0].cpu().numpy()]
                for name, values in attentions.items()
            },
            "top_predictions": {
                name: self.labels[name][int(np.argmax(values))] for name, values in probabilities.items()
            },
        }
        return [int(candidates[i]) for i in order[:k]], diagnostics


def evaluate_attention_gru(
    index: DenseIndex,
    encoder,
    queries: list[JudgedQuery],
    checkpoint: str | Path,
    output: str | Path,
    candidate_pool: int = 200,
    dense_weight: float = 0.60,
    k: int = 10,
) -> dict:
    reranker = AttentionGRUReranker(index, encoder, checkpoint, dense_weight, candidate_pool)
    rankings = []; latencies = []; diagnostics = []
    classification_hits = {name: 0 for name in ("garment", "color", "binding", "environment")}
    classification_totals = {name: 0 for name in classification_hits}
    for query in queries:
        start = time.perf_counter(); ids, detail = reranker.search(query.text, k)
        latencies.append((time.perf_counter() - start) * 1000)
        rankings.append([index.records[i].image_id for i in ids])
        constraints = parse_query(query.text); predictions = detail["top_predictions"]
        if constraints.garments:
            classification_totals["garment"] += 1
            classification_hits["garment"] += int(any(_garment_label_matches(g, predictions["garment"]) for g in constraints.garments))
        if constraints.colors:
            classification_totals["color"] += 1
            classification_hits["color"] += int(predictions["color"] in constraints.colors)
        if constraints.bindings:
            classification_totals["binding"] += 1
            predicted_garment, predicted_color = predictions["binding"].split("|||", 1)
            classification_hits["binding"] += int(any(
                _garment_label_matches(g, predicted_garment) and c == predicted_color for g, c in constraints.bindings
            ))
        if constraints.environment:
            classification_totals["environment"] += 1
            classification_hits["environment"] += int(predictions["environment"] == constraints.environment)
        if len(diagnostics) < 5: diagnostics.append({"query": query.text, **detail})
    metrics = evaluate_rankings(rankings, [query.relevance for query in queries])
    metrics.update({
        "queries": len(queries), "candidate_pool": candidate_pool, "dense_weight": dense_weight,
        "latency_ms_mean": float(np.mean(latencies)), "latency_ms_p95": float(np.percentile(latencies, 95)),
    })
    metrics.update({
        f"{name}_top1_accuracy": classification_hits[name] / max(classification_totals[name], 1)
        for name in classification_hits
    })
    result = {"attention_gru": metrics, "diagnostics": diagnostics}
    target = Path(output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
