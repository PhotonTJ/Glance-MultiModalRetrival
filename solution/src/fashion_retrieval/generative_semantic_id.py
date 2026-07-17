"""TIGER-style text-to-Semantic-ID generative retrieval.

The RQ-VAE assigns each gallery image a short hierarchical code.  A pretrained
encoder--decoder model is then fine-tuned to emit that code from a natural
language query.  Constrained beam search ensures every generated code exists in
the gallery; its image bucket is subsequently reranked using dense and
structured evidence.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .index import DenseIndex
from .metrics import evaluate_rankings
from .parser import parse_query
from .retriever import SearchResult, structured_scores
from .schema import ImageRecord, JudgedQuery
from .training import record_text


def sid_token(level: int, code: int) -> str:
    """Use level-specific tokens: code 5 at level 0 is not code 5 at level 1."""
    return f"<SID_L{level}_{code}>"


def sid_target(sid: list[int] | tuple[int, ...]) -> str:
    return " ".join(sid_token(level, code) for level, code in enumerate(sid))


def _record_queries(record: ImageRecord) -> list[str]:
    """Small metadata-only paraphrase set; it never uses held-out query text."""
    garments = list(dict.fromkeys(f"{g.color} {g.type}" for g in record.garments if g.type != "unknown"))
    outfit = " and ".join(garments) or "fashion clothing"
    scene = f" in a {record.environment} setting" if record.environment != "unknown" else ""
    style = f" with a {' '.join(record.style)} style" if record.style else ""
    values = [
        record_text(record),
        f"A person wearing {outfit}{scene}.",
        f"Find a fashion image with {outfit}{scene}{style}.",
        f"Retrieve an outfit featuring {outfit}{scene}.",
    ]
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


class _Pairs(Dataset):
    def __init__(self, pairs: list[tuple[str, str]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[str, str]:
        return self.pairs[index]


def _load_semantic_ids(directory: str | Path) -> dict[str, tuple[int, ...]]:
    values = json.loads((Path(directory) / "image_to_sid.json").read_text(encoding="utf-8"))
    return {image_id: tuple(int(code) for code in codes) for image_id, codes in values.items()}


def _semantic_vocabulary(image_to_sid: dict[str, tuple[int, ...]]) -> list[str]:
    return sorted({sid_token(level, code) for sid in image_to_sid.values() for level, code in enumerate(sid)})


def train_generative_semantic_ids(
    index: DenseIndex,
    semantic_dir: str | Path,
    output_dir: str | Path,
    model_name: str = "google/flan-t5-small",
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    max_source_length: int = 64,
    seed: int = 42,
) -> dict[str, object]:
    """Fine-tune a pretrained T5 encoder--decoder on train-image text -> RQ codes."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    image_to_sid = _load_semantic_ids(semantic_dir)
    pairs = [
        (f"generate fashion semantic id: {text}", sid_target(image_to_sid[record.image_id]))
        for record in index.records
        if record.split == "train" and record.image_id in image_to_sid
        for text in _record_queries(record)
    ]
    if not pairs:
        raise ValueError("No training records with Semantic IDs were found in the index.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": _semantic_vocabulary(image_to_sid)})
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if added_tokens:
        model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    model.to(device)

    def collate(rows: list[tuple[str, str]]) -> dict[str, torch.Tensor]:
        sources, targets = zip(*rows)
        encoded = tokenizer(list(sources), padding=True, truncation=True, max_length=max_source_length, return_tensors="pt")
        labels = tokenizer(text_target=list(targets), padding=True, truncation=True, max_length=max(len(next(iter(image_to_sid.values()))) + 1, 8), return_tensors="pt").input_ids
        labels[labels == tokenizer.pad_token_id] = -100
        encoded["labels"] = labels
        return encoded

    loader = DataLoader(_Pairs(pairs), batch_size=batch_size, shuffle=True, collate_fn=collate)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history: list[float] = []
    model.train()
    for _ in range(epochs):
        total = 0.0
        for batch in loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            loss = model(**batch).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
        history.append(total / len(loader))

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(target)
    tokenizer.save_pretrained(target)
    metadata = {
        "model_name": model_name,
        "levels": len(next(iter(image_to_sid.values()))),
        "training_examples": len(pairs),
        "training_images": sum(record.split == "train" for record in index.records),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "semantic_dir": str(Path(semantic_dir)),
        "loss_history": history,
    }
    (target / "generative_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


class GenerativeSemanticIDRetriever:
    """Query -> constrained Semantic-ID beams -> Semantic-ID bucket reranking."""

    def __init__(
        self,
        index: DenseIndex,
        encoder,
        semantic_dir: str | Path,
        generator_dir: str | Path,
        weights: dict[str, float] | None = None,
        beam_size: int = 20,
    ):
        self.index = index
        self.encoder = encoder
        self.beam_size = beam_size
        self.weights = weights or {"dense": 0.40, "binding": 0.20, "context": 0.12, "sid": 0.15, "style": 0.13}
        self.image_to_sid = _load_semantic_ids(semantic_dir)
        self.sid_to_indices: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for idx, record in enumerate(index.records):
            if record.image_id in self.image_to_sid:
                self.sid_to_indices[self.image_to_sid[record.image_id]].append(idx)
        self.levels = len(next(iter(self.sid_to_indices)))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(generator_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(generator_dir).to(self.device).eval()
        self.decoder_start = self.model.config.decoder_start_token_id or self.tokenizer.pad_token_id
        self.eos = self.model.config.eos_token_id or self.tokenizer.eos_token_id
        self.ignored_tokens = {self.decoder_start, self.tokenizer.pad_token_id, self.eos}
        self.token_to_code = {
            self.tokenizer.convert_tokens_to_ids(sid_token(level, code)): (level, code)
            for sid in self.sid_to_indices
            for level, code in enumerate(sid)
        }
        self.prefix_to_next: dict[tuple[int, ...], set[int]] = defaultdict(set)
        for sid in self.sid_to_indices:
            ids = tuple(self.tokenizer.convert_tokens_to_ids(sid_token(level, code)) for level, code in enumerate(sid))
            for depth, token in enumerate(ids):
                self.prefix_to_next[ids[:depth]].add(token)

    def _allowed_tokens(self, _batch_id: int, generated: torch.Tensor) -> list[int]:
        prefix = tuple(token for token in generated.tolist() if token not in self.ignored_tokens)
        if len(prefix) >= self.levels:
            return [self.eos]
        return sorted(self.prefix_to_next.get(prefix, {self.eos}))

    @torch.no_grad()
    def predict_semantic_ids(self, query: str) -> list[tuple[tuple[int, ...], float]]:
        encoded = self.tokenizer(
            f"generate fashion semantic id: {query}", return_tensors="pt", truncation=True, max_length=64
        ).to(self.device)
        beams = min(self.beam_size, len(self.sid_to_indices))
        output = self.model.generate(
            **encoded,
            num_beams=beams,
            num_return_sequences=beams,
            max_new_tokens=self.levels + 1,
            prefix_allowed_tokens_fn=self._allowed_tokens,
            return_dict_in_generate=True,
            output_scores=True,
        )
        sequence_scores = getattr(output, "sequences_scores", None)
        # Greedy decoding (one beam) does not expose beam-normalized scores.
        raw_scores = sequence_scores.tolist() if sequence_scores is not None else [0.0] * len(output.sequences)
        result: list[tuple[tuple[int, ...], float]] = []
        seen: set[tuple[int, ...]] = set()
        for sequence, score in zip(output.sequences.tolist(), raw_scores):
            generated = [token for token in sequence if token not in self.ignored_tokens]
            if len(generated) != self.levels or any(token not in self.token_to_code for token in generated):
                continue
            sid = tuple(self.token_to_code[token][1] for token in generated)
            if sid in self.sid_to_indices and sid not in seen:
                result.append((sid, float(score)))
                seen.add(sid)
        return result

    def retrieve(self, query: str, k: int = 10) -> tuple[list[SearchResult], list[tuple[int, ...]], int]:
        predictions = self.predict_semantic_ids(query)
        generated_sids = [sid for sid, _ in predictions]
        if not predictions:
            return [], [], 0
        peak = max(score for _, score in predictions)
        sid_confidence = {sid: math.exp(score - peak) for sid, score in predictions}
        candidates = list(dict.fromkeys(idx for sid in generated_sids for idx in self.sid_to_indices[sid]))
        qvec = self.encoder.encode_texts([query])[0]
        constraints = parse_query(query)
        results: list[SearchResult] = []
        for idx in candidates:
            record = self.index.records[idx]
            cosine = float(self.index.vectors[idx] @ qvec)
            binding, context, style, explanation = structured_scores(constraints, record)
            sid_score = sid_confidence[self.image_to_sid[record.image_id]]
            total = (
                self.weights["dense"] * ((cosine + 1.0) / 2.0)
                + self.weights["binding"] * binding
                + self.weights["context"] * context
                + self.weights["sid"] * sid_score
                + self.weights["style"] * style
            )
            results.append(SearchResult(record, total, cosine, binding, context, f"{explanation}; generated-semantic-id={sid_score:.2f}", sid_score))
        return sorted(results, key=lambda item: (-item.score, item.record.image_id))[:k], generated_sids, len(candidates)

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        return self.retrieve(query, k)[0]


def evaluate_generative_semantic_ids(
    index: DenseIndex,
    encoder,
    queries: list[JudgedQuery],
    semantic_dir: str | Path,
    generator_dir: str | Path,
    output: str | Path,
    beam_size: int = 20,
    k: int = 10,
) -> dict[str, object]:
    retriever = GenerativeSemanticIDRetriever(index, encoder, semantic_dir, generator_dir, beam_size=beam_size)
    rankings: list[list[str]] = []
    judgments: list[dict[str, int]] = []
    latencies: list[float] = []
    candidate_counts: list[int] = []
    sid_hits: list[float] = []
    candidate_hits: list[float] = []
    source_sid_correct = 0
    source_sid_total = 0
    level_correct = np.zeros(retriever.levels, dtype=int)
    level_total = np.zeros(retriever.levels, dtype=int)
    codebook_size = max(code for sid in retriever.sid_to_indices for code in sid) + 1
    confusion = np.zeros((retriever.levels, codebook_size, codebook_size), dtype=int)
    binding_hits = binding_total = context_hits = context_total = 0
    diagnostics = []
    for query in queries:
        start = time.perf_counter()
        results, generated_sids, candidate_count = retriever.retrieve(query.text, k)
        latencies.append((time.perf_counter() - start) * 1000)
        rankings.append([item.record.image_id for item in results])
        judgments.append(query.relevance)
        candidate_counts.append(candidate_count)
        target_sids = {retriever.image_to_sid[image_id] for image_id, grade in query.relevance.items() if grade == 2 and image_id in retriever.image_to_sid}
        sid_hits.append(float(any(sid in target_sids for sid in generated_sids)))
        candidate_hits.append(float(any(image_id in query.relevance for sid in generated_sids for image_id in (retriever.index.records[idx].image_id for idx in retriever.sid_to_indices[sid]))))
        # Auto-query IDs preserve their source image, giving an unambiguous
        # code-classification target in addition to multi-relevance retrieval.
        source_image_id = query.query_id.removeprefix("auto_")
        if source_image_id in retriever.image_to_sid and generated_sids:
            expected, predicted = retriever.image_to_sid[source_image_id], generated_sids[0]
            source_sid_total += 1
            source_sid_correct += int(expected == predicted)
            for level, (actual_code, predicted_code) in enumerate(zip(expected, predicted)):
                level_total[level] += 1
                level_correct[level] += int(actual_code == predicted_code)
                confusion[level, actual_code, predicted_code] += 1
        constraints = parse_query(query.text)
        if constraints.bindings:
            binding_total += 1
            binding_hits += int(bool(results) and query.relevance.get(results[0].record.image_id, 0) == 2)
        if constraints.environment or constraints.activity or constraints.objects:
            context_total += 1
            context_hits += int(bool(results) and results[0].context_score >= 0.999)
        if len(diagnostics) < 5:
            diagnostics.append({
                "query": query.text,
                "generated_semantic_ids": [list(sid) for sid in generated_sids],
                "semantic_id_candidates": candidate_count,
                "top_result": None if not results else {"image_id": results[0].record.image_id, "score": results[0].score, "sid_score": results[0].sid_score, "explanation": results[0].explanation},
            })
    metrics = evaluate_rankings(rankings, judgments)
    metrics.update({
        "beam_size": beam_size,
        "binding_accuracy": binding_hits / max(binding_total, 1),
        "context_satisfaction": context_hits / max(context_total, 1),
        "semantic_id_exact_hit@beam": float(np.mean(sid_hits)),
        "semantic_candidate_hit@beam": float(np.mean(candidate_hits)),
        "semantic_id_candidate_mean": float(np.mean(candidate_counts)),
        "semantic_id_source_exact_accuracy@1": source_sid_correct / max(source_sid_total, 1),
        "semantic_id_level_accuracy@1": (level_correct / np.maximum(level_total, 1)).tolist(),
        "semantic_id_confusion_matrices": confusion.tolist(),
        "latency_ms_mean": float(np.mean(latencies)),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
    })
    result = {"generative_semantic_id": metrics, "diagnostics": diagnostics}
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
