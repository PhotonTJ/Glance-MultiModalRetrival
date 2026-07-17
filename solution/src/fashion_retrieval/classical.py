from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .index import DenseIndex
from .metrics import evaluate_rankings
from .parser import garment_matches, parse_query
from .schema import ImageRecord, JudgedQuery, write_jsonl


def _human_garment(value: str) -> str:
    return value.replace("_", " ").split(" / ")[0].strip()


def _garment_label_matches(requested: str, label: str) -> bool:
    return garment_matches(requested, label)


def build_automatic_queries(records: list[ImageRecord], output: str | Path, limit: int = 120, source_split: str = "test") -> list[JudgedQuery]:
    """Create reproducible held-out queries; these are automatic judgments, not final human labels."""
    candidates = []
    for record in records:
        if record.split != source_split or record.environment == "unknown":
            continue
        known = [g for g in record.garments if g.color != "unknown"]
        if not known:
            continue
        # Prefer apparel over accessories when possible.
        accessory = {"shoe", "bag", "belt", "glasses", "watch", "sock"}
        known.sort(key=lambda g: (_human_garment(g.type) in accessory, g.type, g.color))
        garment = known[0]
        garment_name = _human_garment(garment.type)
        article = "an" if record.environment == "office" else "a"
        text = f"A person wearing a {garment.color} {garment_name} in {article} {record.environment.replace('_', ' ')} setting."
        relevance = {}
        for item in records:
            exact_binding = any(g.color == garment.color and _garment_label_matches(garment_name, g.type) for g in item.garments)
            garment_match = any(_garment_label_matches(garment_name, g.type) for g in item.garments)
            environment_match = item.environment == record.environment
            grade = 2 if exact_binding and environment_match else 1 if garment_match or environment_match else 0
            if grade: relevance[item.image_id] = grade
        candidates.append(JudgedQuery(query_id=f"auto_{record.image_id}", text=text, relevance=relevance))
    candidates = candidates[:limit]
    write_jsonl(output, candidates)
    return candidates


class BinaryBank:
    def __init__(self, estimator, labels: list[str], score_kind: str = "probability"):
        self.estimator = estimator; self.labels = labels; self.score_kind = score_kind
        self.models = []; self.constants = []

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "BinaryBank":
        for column in range(targets.shape[1]):
            values = targets[:, column]
            if len(np.unique(values)) < 2:
                self.models.append(None); self.constants.append(float(values[0]))
            else:
                model = clone(self.estimator).fit(features, values)
                self.models.append(model); self.constants.append(0.0)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        output = np.zeros((len(features), len(self.labels)), dtype=np.float32)
        for column, model in enumerate(self.models):
            if model is None: output[:, column] = self.constants[column]
            elif hasattr(model, "predict_proba"): output[:, column] = model.predict_proba(features)[:, 1]
            else: output[:, column] = 1.0 / (1.0 + np.exp(-np.clip(model.decision_function(features), -20, 20)))
        return output


class EnvironmentModel:
    def __init__(self, estimator, labels: list[str]): self.model = clone(estimator); self.labels = labels
    def fit(self, features, targets): self.model.fit(features, targets); return self
    def predict(self, features):
        output = np.zeros((len(features), len(self.labels)), dtype=np.float32)
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features)
        else:
            decision = self.model.decision_function(features)
            decision = decision - decision.max(axis=1, keepdims=True); probabilities = np.exp(decision) / np.exp(decision).sum(axis=1, keepdims=True)
        for source, label in enumerate(self.model.classes_): output[:, self.labels.index(label)] = probabilities[:, source]
        return output


def _requested_score(requested: set, labels: list[str], predictions: np.ndarray, garment: bool = False) -> np.ndarray:
    if not requested: return np.ones(len(predictions), dtype=np.float32)
    columns = []
    for value in requested:
        matches = [i for i, label in enumerate(labels) if (_garment_label_matches(value, label) if garment else value == label)]
        if matches: columns.append(predictions[:, matches].max(axis=1))
    return np.mean(columns, axis=0) if columns else np.zeros(len(predictions), dtype=np.float32)


def run_classical_baselines(index: DenseIndex, encoder, queries: list[JudgedQuery], output: str | Path, candidate_pool: int = 200, seed: int = 42) -> dict:
    records = index.records; train_ids = np.array([i for i, r in enumerate(records) if r.split == "train"])
    if not len(train_ids): raise ValueError("no train records found")
    garment_labels = sorted({g.type for r in records for g in r.garments})
    color_labels = sorted({g.color for r in records for g in r.garments if g.color != "unknown"})
    environment_labels = sorted({r.environment for r in records if r.environment != "unknown"})
    garment_targets = np.array([[int(any(g.type == label for g in r.garments)) for label in garment_labels] for r in records])
    color_targets = np.array([[int(any(g.color == label for g in r.garments)) for label in color_labels] for r in records])
    environment_targets = np.array([r.environment for r in records])

    components = min(256, len(train_ids) - 1, index.vectors.shape[1])
    pca = PCA(n_components=components, random_state=seed).fit(index.vectors[train_ids])
    scaler = StandardScaler().fit(pca.transform(index.vectors[train_ids]))
    features = scaler.transform(pca.transform(index.vectors))
    estimators = {
        "naive_bayes": GaussianNB(),
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        "linear_svm": LinearSVC(class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=80, class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
    }
    predictions = {}
    known_environment_train = train_ids[environment_targets[train_ids] != "unknown"]
    for name, estimator in estimators.items():
        garment_bank = BinaryBank(estimator, garment_labels).fit(features[train_ids], garment_targets[train_ids])
        color_bank = BinaryBank(estimator, color_labels).fit(features[train_ids], color_targets[train_ids])
        environment_model = EnvironmentModel(estimator, environment_labels).fit(features[known_environment_train], environment_targets[known_environment_train])
        predictions[name] = (garment_bank.predict(features), color_bank.predict(features), environment_model.predict(features))

    kmeans = KMeans(n_clusters=min(24, len(train_ids)), random_state=seed, n_init=10).fit(features[train_ids])
    image_clusters = kmeans.predict(features)
    query_vectors = encoder.encode_texts([q.text for q in queries])
    query_features = scaler.transform(pca.transform(query_vectors))
    query_clusters = kmeans.predict(query_features)
    methods = ["raw_cosine", "cosine_naive_bayes", "cosine_logistic_regression", "cosine_linear_svm", "cosine_random_forest", "cosine_kmeans", "full_classical_ensemble"]
    rankings = {method: [] for method in methods}; latencies = defaultdict(list)

    for query_index, (query, qvec) in enumerate(zip(queries, query_vectors)):
        constraints = parse_query(query.text)
        dense = index.vectors @ qvec; candidate_ids = np.argsort(-dense, kind="stable")[:candidate_pool]
        dense01 = (dense[candidate_ids] + 1) / 2
        rankings["raw_cosine"].append([records[i].image_id for i in candidate_ids])
        method_components = {}
        for name in estimators:
            start = time.perf_counter(); garment_pred, color_pred, environment_pred = predictions[name]
            garment_score = _requested_score(set(constraints.garments), garment_labels, garment_pred[candidate_ids], garment=True)
            color_score = _requested_score(set(constraints.colors), color_labels, color_pred[candidate_ids])
            env_score = _requested_score({constraints.environment} if constraints.environment else set(), environment_labels, environment_pred[candidate_ids])
            total = 0.65 * dense01 + 0.20 * garment_score + 0.10 * color_score + 0.05 * env_score
            order = np.argsort(-total, kind="stable")
            key = "cosine_" + name; rankings[key].append([records[candidate_ids[i]].image_id for i in order]); latencies[key].append((time.perf_counter()-start)*1000)
            method_components[name] = (garment_score, color_score, env_score)
        cluster_score = (image_clusters[candidate_ids] == query_clusters[query_index]).astype(float)
        cluster_total = 0.85 * dense01 + 0.15 * cluster_score
        order = np.argsort(-cluster_total, kind="stable"); rankings["cosine_kmeans"].append([records[candidate_ids[i]].image_id for i in order])
        garment_score = np.mean([x[0] for x in method_components.values()], axis=0)
        color_score = np.mean([x[1] for x in method_components.values()], axis=0)
        env_score = np.mean([x[2] for x in method_components.values()], axis=0)
        binding_score = np.array([np.mean([any(_garment_label_matches(g, item.type) and item.color == c for item in records[i].garments) for g, c in constraints.bindings]) if constraints.bindings else 1.0 for i in candidate_ids])
        ensemble = 0.50*dense01 + 0.18*garment_score + 0.12*color_score + 0.10*env_score + 0.05*cluster_score + 0.05*binding_score
        order = np.argsort(-ensemble, kind="stable"); rankings["full_classical_ensemble"].append([records[candidate_ids[i]].image_id for i in order])

    judgments = [q.relevance for q in queries]; results = {}
    for method in methods:
        metrics = evaluate_rankings(rankings[method], judgments)
        metrics["queries"] = len(queries); metrics["candidate_pool"] = candidate_pool
        if latencies[method]: metrics["rerank_latency_ms_mean"] = float(np.mean(latencies[method]))
        results[method] = metrics
    target = Path(output); target.parent.mkdir(parents=True, exist_ok=True); target.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
