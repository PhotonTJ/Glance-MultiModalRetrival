from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from .datasets import fashionpedia_manifest
from .encoder import HashEncoder, ProjectedEncoder, QwenEncoder
from .index import DenseIndex
from .metrics import evaluate_rankings
from .parser import parse_query
from .plots import plot_attention_gru_results, plot_direct_semantic_hybrid_method, plot_generative_semantic_id_results, plot_qualitative_retrievals, plot_semantic_id_confusion, plot_coverage, plot_dataset_analysis, plot_eda_report_figures, plot_results
from .retriever import HybridRetriever
from .schema import ImageRecord, JudgedQuery, load_jsonl
from .training import train_rqvae, train_siamese
from .fashionpedia_analysis import build_balanced_subset, classify_environments, extract_mask_colors, native_clothing_summary
from .classical import build_automatic_queries, run_classical_baselines
from .attention_gru import evaluate_attention_gru, train_attention_gru
from .semantic_id import SemanticIDHybridRetriever, evaluate_semantic_ids, sample_semantic_id_retrievals, tune_semantic_id_weights
from .generative_semantic_id import GenerativeSemanticIDRetriever, evaluate_generative_semantic_ids, train_generative_semantic_ids


def _config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _encoder(config: dict, backend: str):
    settings = config["encoder"]
    if backend == "hash":
        return HashEncoder(settings["dimension"])
    return QwenEncoder(settings["model"], settings["dimension"])


def _search_encoder(config: dict, backend: str, projection: str | None):
    encoder = _encoder(config, backend)
    return ProjectedEncoder(encoder, projection) if projection else encoder


def command_index(args: argparse.Namespace) -> None:
    config = _config(args.config)
    records = load_jsonl(args.metadata, ImageRecord)
    encoder = _search_encoder(config, args.backend, args.projection)
    vectors = encoder.encode_images([r.image_path for r in records], batch_size=config["encoder"]["batch_size"])
    index = DenseIndex(vectors, records, config.get("faiss"))
    index.save(args.output)
    print(f"Indexed {len(records)} images into {args.output} with FAISS {index.serving_index_type}")


def command_search(args: argparse.Namespace) -> None:
    config = _config(args.config)
    index = DenseIndex.load(args.index)
    encoder = _search_encoder(config, args.backend, args.projection)
    retriever = HybridRetriever(index, encoder, config["retrieval"]["weights"], config["retrieval"]["candidate_pool"])
    for rank, result in enumerate(retriever.search(args.query, args.k), 1):
        print(json.dumps({"rank": rank, "image_id": result.record.image_id, "path": result.record.image_path,
                          "score": round(result.score, 4), "explanation": result.explanation}))


def command_evaluate(args: argparse.Namespace) -> None:
    config = _config(args.config)
    index = DenseIndex.load(args.index)
    queries = load_jsonl(args.queries, JudgedQuery)
    weights = config["retrieval"]["weights"] if args.mode == "hybrid" else {"dense": 1.0, "binding": 0.0, "context": 0.0, "style": 0.0}
    retriever = HybridRetriever(index, _search_encoder(config, args.backend, args.projection), weights, config["retrieval"]["candidate_pool"])
    rankings, judgments, latencies = [], [], []
    binding_hits = binding_total = context_hits = context_total = 0
    for query in queries:
        start = time.perf_counter()
        results = retriever.search(query.text, args.k)
        latencies.append((time.perf_counter() - start) * 1000)
        rankings.append([r.record.image_id for r in results]); judgments.append(query.relevance)
        constraints = parse_query(query.text)
        if constraints.bindings:
            binding_total += 1
            binding_hits += int(bool(results) and query.relevance.get(results[0].record.image_id, 0) == 2)
        if constraints.environment or constraints.activity or constraints.objects:
            context_total += 1
            context_hits += int(bool(results) and results[0].context_score >= 0.999)
    metrics = evaluate_rankings(rankings, judgments)
    metrics.update({
        "binding_accuracy": binding_hits / max(binding_total, 1),
        "context_satisfaction": context_hits / max(context_total, 1),
        "latency_ms_mean": float(np.mean(latencies)),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
    })
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
    existing[args.method] = metrics
    target.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def command_search_semantic_ids(args: argparse.Namespace) -> None:
    config = _config(args.config)
    index = DenseIndex.load(args.index)
    retriever = SemanticIDHybridRetriever(index, _encoder(config, args.backend), args.semantic_dir, weights=_weights_from_file(args.weights_json), candidate_pool=args.candidate_pool, sid_pool=args.sid_pool)
    for rank, result in enumerate(retriever.search(args.query, args.k), 1):
        print(json.dumps({"rank": rank, "image_id": result.record.image_id, "path": result.record.image_path,
                          "score": round(result.score, 4), "explanation": result.explanation, "sid_score": round(result.sid_score, 4)}))


def _weights_from_file(path: str | None) -> dict[str, float] | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload.get("selected_weights", payload)


def command_search_generative_semantic_ids(args: argparse.Namespace) -> None:
    config = _config(args.config)
    index = DenseIndex.load(args.index)
    retriever = GenerativeSemanticIDRetriever(index, _encoder(config, args.backend), args.semantic_dir, args.generator_dir, beam_size=args.beam_size)
    for rank, result in enumerate(retriever.search(args.query, args.k), 1):
        print(json.dumps({"rank": rank, "image_id": result.record.image_id, "path": result.record.image_path,
                          "score": round(result.score, 4), "explanation": result.explanation, "generated_sid_score": round(result.sid_score, 4)}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fashion and context retrieval")
    parser.add_argument("--config", default="config/default.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-fashionpedia")
    prepare.add_argument("--annotations", required=True); prepare.add_argument("--images", required=True)
    prepare.add_argument("--output", default="data/processed/metadata.jsonl"); prepare.add_argument("--max-images", type=int, default=1000)
    prepare.set_defaults(func=lambda a: fashionpedia_manifest(a.annotations, a.images, a.output, a.max_images))
    index = sub.add_parser("index")
    index.add_argument("--metadata", required=True); index.add_argument("--output", default="artifacts/indexes/main")
    index.add_argument("--backend", choices=("qwen", "hash"), default="qwen"); index.add_argument("--projection"); index.set_defaults(func=command_index)
    search = sub.add_parser("search")
    search.add_argument("query"); search.add_argument("--index", default="artifacts/indexes/main"); search.add_argument("-k", type=int, default=10)
    search.add_argument("--backend", choices=("qwen", "hash"), default="qwen"); search.add_argument("--projection"); search.set_defaults(func=command_search)
    semantic_search = sub.add_parser("search-semantic-ids")
    semantic_search.add_argument("query"); semantic_search.add_argument("--index", required=True); semantic_search.add_argument("--semantic-dir", required=True)
    semantic_search.add_argument("-k", type=int, default=10); semantic_search.add_argument("--candidate-pool", type=int, default=100)
    semantic_search.add_argument("--sid-pool", type=int, default=100)
    semantic_search.add_argument("--backend", choices=("qwen", "hash"), default="qwen"); semantic_search.add_argument("--weights-json"); semantic_search.set_defaults(func=command_search_semantic_ids)
    generative_search = sub.add_parser("search-generative-semantic-ids")
    generative_search.add_argument("query"); generative_search.add_argument("--index", required=True)
    generative_search.add_argument("--semantic-dir", required=True); generative_search.add_argument("--generator-dir", required=True)
    generative_search.add_argument("-k", type=int, default=10); generative_search.add_argument("--beam-size", type=int, default=20)
    generative_search.add_argument("--backend", choices=("qwen", "hash"), default="qwen"); generative_search.set_defaults(func=command_search_generative_semantic_ids)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--index", default="artifacts/indexes/main"); evaluate.add_argument("--queries", required=True)
    evaluate.add_argument("--output", default="artifacts/results/results.json"); evaluate.add_argument("--method", default="hybrid")
    evaluate.add_argument("--mode", choices=("dense", "hybrid"), default="hybrid")
    evaluate.add_argument("--backend", choices=("qwen", "hash"), default="qwen"); evaluate.add_argument("-k", type=int, default=10)
    evaluate.add_argument("--projection")
    evaluate.set_defaults(func=command_evaluate)
    coverage = sub.add_parser("plot-coverage"); coverage.add_argument("--metadata", required=True); coverage.add_argument("--output", default="artifacts/figures/coverage.png")
    coverage.set_defaults(func=lambda a: plot_coverage(a.metadata, a.output))
    results = sub.add_parser("plot-results"); results.add_argument("--results", required=True); results.add_argument("--output", default="artifacts/figures/results.png")
    results.set_defaults(func=lambda a: plot_results(a.results, a.output))
    attention_plot = sub.add_parser("plot-attention-gru-results")
    attention_plot.add_argument("--baseline1", required=True); attention_plot.add_argument("--baseline2", required=True)
    attention_plot.add_argument("--output", default="../report/figures/baseline2_attention_gru.png")
    attention_plot.set_defaults(func=lambda a: plot_attention_gru_results(a.baseline1, a.baseline2, a.output))
    generative_plot = sub.add_parser("plot-generative-semantic-id-results")
    generative_plot.add_argument("--baseline1", required=True); generative_plot.add_argument("--baseline2", required=True)
    generative_plot.add_argument("--direct-semantic", required=True); generative_plot.add_argument("--generative", required=True)
    generative_plot.add_argument("--output", default="../report/figures/generative_semantic_id.png")
    generative_plot.set_defaults(func=lambda a: plot_generative_semantic_id_results(a.baseline1, a.baseline2, a.direct_semantic, a.generative, a.output))
    confusion_plot = sub.add_parser("plot-generative-semantic-id-confusion")
    confusion_plot.add_argument("--results", required=True); confusion_plot.add_argument("--output", default="../report/figures/generative_semantic_id_confusion.png")
    confusion_plot.set_defaults(func=lambda a: plot_semantic_id_confusion(a.results, a.output))
    direct_method_plot = sub.add_parser("plot-direct-semantic-method")
    direct_method_plot.add_argument("--output", default="../report/figures/direct_semantic_hybrid_method.png")
    direct_method_plot.set_defaults(func=lambda a: plot_direct_semantic_hybrid_method(a.output))
    qualitative_plot = sub.add_parser("plot-qualitative-retrievals")
    qualitative_plot.add_argument("--samples", required=True); qualitative_plot.add_argument("--output", default="../report/figures/qualitative_retrievals.png")
    qualitative_plot.set_defaults(func=lambda a: plot_qualitative_retrievals(a.samples, a.output))
    analysis = sub.add_parser("analyze-data"); analysis.add_argument("--metadata", required=True)
    analysis.add_argument("--output", default="artifacts/figures/data_analysis.png")
    analysis.add_argument("--summary", default="artifacts/results/data_summary.json")
    analysis.set_defaults(func=lambda a: print(json.dumps(plot_dataset_analysis(a.metadata, a.output, a.summary), indent=2)))
    report_figures = sub.add_parser("plot-eda-report")
    report_figures.add_argument("--metadata", required=True); report_figures.add_argument("--output-dir", default="../report/figures")
    report_figures.set_defaults(func=lambda a: print("\n".join(plot_eda_report_figures(a.metadata, a.output_dir))))
    native = sub.add_parser("analyze-fashionpedia-native")
    native.add_argument("--annotations", nargs="+", required=True); native.add_argument("--output", default="artifacts/results/fashionpedia_native.json")
    native.set_defaults(func=lambda a: print(json.dumps(native_clothing_summary(a.annotations, a.output), indent=2)))
    colors = sub.add_parser("extract-fashionpedia-colors")
    colors.add_argument("--annotations", required=True); colors.add_argument("--images", required=True); colors.add_argument("--output", required=True)
    colors.set_defaults(func=lambda a: print(json.dumps(extract_mask_colors(a.annotations, a.images, a.output), indent=2)))
    scenes = sub.add_parser("classify-fashionpedia-scenes")
    scenes.add_argument("--annotations", required=True); scenes.add_argument("--images", required=True); scenes.add_argument("--output", required=True)
    scenes.add_argument("--batch-size", type=int, default=32)
    scenes.set_defaults(func=lambda a: print(json.dumps(classify_environments(a.annotations, a.images, a.output, a.batch_size), indent=2)))
    subset = sub.add_parser("build-fashionpedia-subset")
    subset.add_argument("--annotations", required=True); subset.add_argument("--images", required=True)
    subset.add_argument("--colors", required=True); subset.add_argument("--environments", required=True)
    subset.add_argument("--output", default="data/processed/fashionpedia_1000/metadata.jsonl"); subset.add_argument("--size", type=int, default=1000)
    subset.set_defaults(func=lambda a: print(json.dumps(build_balanced_subset(a.annotations, a.images, a.colors, a.environments, a.output, a.size), indent=2)))
    queries = sub.add_parser("build-auto-queries")
    queries.add_argument("--metadata", required=True); queries.add_argument("--output", required=True); queries.add_argument("--limit", type=int, default=120)
    queries.add_argument("--source-split", choices=("train", "validation", "test"), default="test")
    queries.set_defaults(func=lambda a: print(f"Wrote {len(build_automatic_queries(load_jsonl(a.metadata, ImageRecord), a.output, a.limit, a.source_split))} queries"))
    classical = sub.add_parser("run-classical-baselines")
    classical.add_argument("--index", required=True); classical.add_argument("--queries", required=True); classical.add_argument("--output", default="artifacts/results/baseline1.json")
    classical.add_argument("--backend", choices=("qwen", "hash"), default="qwen"); classical.add_argument("--candidate-pool", type=int, default=200)
    classical.set_defaults(func=lambda a: print(json.dumps(run_classical_baselines(DenseIndex.load(a.index), _encoder(_config(a.config), a.backend), load_jsonl(a.queries, JudgedQuery), a.output, a.candidate_pool), indent=2)))
    attention_train = sub.add_parser("train-attention-gru")
    attention_train.add_argument("--index", required=True); attention_train.add_argument("--output", default="artifacts/checkpoints/attention_gru")
    attention_train.add_argument("--backend", choices=("qwen", "hash"), default="qwen")
    attention_train.add_argument("--epochs", type=int, default=40); attention_train.add_argument("--batch-size", type=int, default=64)
    attention_train.add_argument("--hidden-dim", type=int, default=128); attention_train.add_argument("--learning-rate", type=float, default=3e-4)
    attention_train.add_argument("--patience", type=int, default=6)
    attention_train.set_defaults(func=lambda a: print(json.dumps(train_attention_gru(
        DenseIndex.load(a.index), _encoder(_config(a.config), a.backend), a.output, a.epochs,
        a.batch_size, a.hidden_dim, a.learning_rate, a.patience), indent=2)))
    attention_eval = sub.add_parser("evaluate-attention-gru")
    attention_eval.add_argument("--index", required=True); attention_eval.add_argument("--queries", required=True)
    attention_eval.add_argument("--checkpoint", required=True); attention_eval.add_argument("--output", default="artifacts/results/baseline2_attention_gru.json")
    attention_eval.add_argument("--backend", choices=("qwen", "hash"), default="qwen")
    attention_eval.add_argument("--candidate-pool", type=int, default=200); attention_eval.add_argument("--dense-weight", type=float, default=0.60)
    attention_eval.add_argument("-k", type=int, default=10)
    attention_eval.set_defaults(func=lambda a: print(json.dumps(evaluate_attention_gru(
        DenseIndex.load(a.index), _encoder(_config(a.config), a.backend), load_jsonl(a.queries, JudgedQuery),
        a.checkpoint, a.output, a.candidate_pool, a.dense_weight, a.k), indent=2)))
    semantic_eval = sub.add_parser("evaluate-semantic-ids")
    semantic_eval.add_argument("--index", required=True); semantic_eval.add_argument("--queries", required=True)
    semantic_eval.add_argument("--semantic-dir", required=True); semantic_eval.add_argument("--output", default="artifacts/results/semantic_id_hybrid.json")
    semantic_eval.add_argument("--backend", choices=("qwen", "hash"), default="qwen")
    semantic_eval.add_argument("--candidate-pool", type=int, default=100); semantic_eval.add_argument("--sid-pool", type=int, default=100)
    semantic_eval.add_argument("-k", type=int, default=10)
    semantic_eval.add_argument("--weights-json")
    semantic_eval.set_defaults(func=lambda a: print(json.dumps(evaluate_semantic_ids(
        DenseIndex.load(a.index), _encoder(_config(a.config), a.backend), load_jsonl(a.queries, JudgedQuery),
        a.semantic_dir, a.output, a.candidate_pool, a.sid_pool, a.k, _weights_from_file(a.weights_json)), indent=2)))
    semantic_tune = sub.add_parser("tune-semantic-id-weights")
    semantic_tune.add_argument("--index", required=True); semantic_tune.add_argument("--queries", required=True); semantic_tune.add_argument("--semantic-dir", required=True)
    semantic_tune.add_argument("--output", default="artifacts/results/semantic_id_weight_tuning.json")
    semantic_tune.add_argument("--backend", choices=("qwen", "hash"), default="qwen")
    semantic_tune.add_argument("--candidate-pool", type=int, default=100); semantic_tune.add_argument("--sid-pool", type=int, default=100); semantic_tune.add_argument("-k", type=int, default=10)
    semantic_tune.add_argument("--trials", type=int, default=1200)
    semantic_tune.set_defaults(func=lambda a: print(json.dumps(tune_semantic_id_weights(
        DenseIndex.load(a.index), _encoder(_config(a.config), a.backend), load_jsonl(a.queries, JudgedQuery),
        a.semantic_dir, a.output, a.candidate_pool, a.sid_pool, a.k, a.trials), indent=2)))
    semantic_samples = sub.add_parser("sample-semantic-id-retrievals")
    semantic_samples.add_argument("--index", required=True); semantic_samples.add_argument("--queries", required=True); semantic_samples.add_argument("--semantic-dir", required=True)
    semantic_samples.add_argument("--output", default="artifacts/results/qualitative_retrievals.json"); semantic_samples.add_argument("--weights-json")
    semantic_samples.add_argument("--backend", choices=("qwen", "hash"), default="qwen"); semantic_samples.add_argument("--samples", type=int, default=5)
    semantic_samples.set_defaults(func=lambda a: print(json.dumps(sample_semantic_id_retrievals(
        DenseIndex.load(a.index), _encoder(_config(a.config), a.backend), load_jsonl(a.queries, JudgedQuery),
        a.semantic_dir, a.output, _weights_from_file(a.weights_json), a.samples), indent=2)))
    generative_train = sub.add_parser("train-generative-semantic-ids")
    generative_train.add_argument("--index", required=True); generative_train.add_argument("--semantic-dir", required=True)
    generative_train.add_argument("--output", default="artifacts/checkpoints/generative_semantic_ids")
    generative_train.add_argument("--model", default="google/flan-t5-small"); generative_train.add_argument("--epochs", type=int, default=20)
    generative_train.add_argument("--batch-size", type=int, default=32); generative_train.add_argument("--learning-rate", type=float, default=3e-4)
    generative_train.set_defaults(func=lambda a: print(json.dumps(train_generative_semantic_ids(
        DenseIndex.load(a.index), a.semantic_dir, a.output, a.model, a.epochs, a.batch_size, a.learning_rate), indent=2)))
    generative_eval = sub.add_parser("evaluate-generative-semantic-ids")
    generative_eval.add_argument("--index", required=True); generative_eval.add_argument("--queries", required=True)
    generative_eval.add_argument("--semantic-dir", required=True); generative_eval.add_argument("--generator-dir", required=True)
    generative_eval.add_argument("--output", default="artifacts/results/generative_semantic_id.json")
    generative_eval.add_argument("--backend", choices=("qwen", "hash"), default="qwen")
    generative_eval.add_argument("--beam-size", type=int, default=20); generative_eval.add_argument("-k", type=int, default=10)
    generative_eval.set_defaults(func=lambda a: print(json.dumps(evaluate_generative_semantic_ids(
        DenseIndex.load(a.index), _encoder(_config(a.config), a.backend), load_jsonl(a.queries, JudgedQuery),
        a.semantic_dir, a.generator_dir, a.output, a.beam_size, a.k), indent=2)))
    siamese = sub.add_parser("train-siamese")
    siamese.add_argument("--index", required=True); siamese.add_argument("--output", default="artifacts/checkpoints/siamese")
    siamese.add_argument("--backend", choices=("qwen", "hash"), default="qwen"); siamese.add_argument("--epochs", type=int, default=30)
    siamese.add_argument("--batch-size", type=int, default=64); siamese.add_argument("--output-dim", type=int, default=256)
    siamese.set_defaults(func=lambda a: train_siamese(DenseIndex.load(a.index), _encoder(_config(a.config), a.backend), a.output, a.epochs, a.batch_size, output_dim=a.output_dim))
    rqvae = sub.add_parser("train-rqvae")
    rqvae.add_argument("--index", required=True); rqvae.add_argument("--output", default="artifacts/checkpoints/rqvae")
    rqvae.add_argument("--epochs", type=int, default=50); rqvae.add_argument("--levels", type=int, default=3); rqvae.add_argument("--codebook-size", type=int, default=16)
    rqvae.add_argument("--attribute-weight", type=float, default=0.0)
    rqvae.set_defaults(func=lambda a: train_rqvae(DenseIndex.load(a.index), a.output, a.epochs, levels=a.levels, codebook_size=a.codebook_size, attribute_weight=a.attribute_weight))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
