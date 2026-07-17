from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from .schema import ImageRecord, load_jsonl


def _style():
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 180, "axes.titleweight": "bold"})
    return plt


def plot_coverage(metadata_path: str | Path, output: str | Path) -> None:
    plt = _style()
    records = load_jsonl(metadata_path, ImageRecord)
    environments = Counter(r.environment for r in records)
    garments = Counter(g.type for r in records for g in r.garments)
    colors = Counter(g.color for r in records for g in r.garments)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    palette = ["#31572c", "#4f772d", "#90a955", "#ecf39e", "#bc6c25", "#dda15e"]
    for ax, (title, counts) in zip(axes, [("Environment", environments), ("Top garments", garments), ("Top colors", colors)]):
        items = counts.most_common(8)[::-1]
        ax.barh([x[0].replace("_", " ") for x in items], [x[1] for x in items], color=palette[: len(items)])
        ax.set_title(title); ax.set_xlabel("Images / instances"); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Dataset coverage audit · n={len(records):,}", fontsize=18)
    target = Path(output); target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, bbox_inches="tight", facecolor="white"); plt.close(fig)


def plot_results(results_path: str | Path, output: str | Path) -> None:
    plt = _style()
    payload = json.loads(Path(results_path).read_text(encoding="utf-8"))
    methods = [name for name, metrics in payload.items() if isinstance(metrics, dict)]
    metrics = [m for m in ("precision@1", "recall@5", "map@10", "ndcg@10", "binding_accuracy") if any(m in payload[x] for x in methods)]
    if not metrics:
        raise ValueError("results JSON contains no supported metrics")
    x = np.arange(len(methods)); width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(max(9, len(methods) * 1.7), 6), constrained_layout=True)
    colors = ["#003049", "#669bbc", "#fcbf49", "#f77f00", "#d62828"]
    display = {"precision@1": "P@1", "recall@5": "R@5", "map@10": "mAP@10", "ndcg@10": "nDCG@10", "binding_accuracy": "Binding acc."}
    for i, metric in enumerate(metrics):
        values = [payload[m].get(metric, 0.0) for m in methods]
        bars = ax.bar(x + (i - (len(metrics) - 1) / 2) * width, values, width, label=display[metric], color=colors[i])
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x, [m.replace("_", "\n") for m in methods]); ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score"); ax.set_title("Retrieval quality (higher is better)"); ax.legend(ncols=min(3, len(metrics)), loc="upper center", bbox_to_anchor=(0.5, 1.20), frameon=False, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    target = Path(output); target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, bbox_inches="tight", facecolor="white"); plt.close(fig)


def plot_attention_gru_results(baseline1_path: str | Path, baseline2_path: str | Path, output: str | Path) -> None:
    """Compare the learned query composer and visualize its head-specific attention."""
    import seaborn as sns

    plt = _style()
    baseline1 = json.loads(Path(baseline1_path).read_text(encoding="utf-8"))
    baseline2 = json.loads(Path(baseline2_path).read_text(encoding="utf-8"))
    methods = {
        "Raw Qwen cosine": baseline1["raw_cosine"],
        "Classical ensemble": baseline1["full_classical_ensemble"],
        "Attention-GRU": baseline2["attention_gru"],
    }
    metrics = [("precision@1", "P@1"), ("recall@5", "R@5"), ("recall@10", "R@10"),
               ("map@10", "mAP@10"), ("ndcg@10", "nDCG@10")]
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.2), constrained_layout=True, gridspec_kw={"width_ratios": [1.45, 1]})
    x = np.arange(len(methods)); width = 0.15
    palette = ["#003049", "#669bbc", "#fcbf49", "#f77f00", "#d62828"]
    for column, (metric, label) in enumerate(metrics):
        values = [result[metric] for result in methods.values()]
        bars = axes[0].bar(x + (column - 2) * width, values, width, label=label, color=palette[column])
        axes[0].bar_label(bars, fmt="%.2f", fontsize=8, padding=2, rotation=90)
    axes[0].set_xticks(x, methods); axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("Score")
    axes[0].set_title("Baseline 2 retrieval cross-check"); axes[0].legend(ncols=5, fontsize=9, frameon=False, loc="upper center")
    axes[0].spines[["top", "right"]].set_visible(False)

    # The final stored diagnostic is a successful example with all four top predictions correct.
    example = baseline2["diagnostics"][-1]
    heads = ["garment", "color", "binding", "environment"]
    matrix = np.asarray([example["attention"][head] for head in heads])
    sns.heatmap(matrix, annot=True, fmt=".3f", vmin=0, vmax=1, cmap="YlGnBu", cbar_kws={"label": "Attention weight"},
                xticklabels=example["components"], yticklabels=heads, ax=axes[1])
    axes[1].set_title("Head-specific attention example")
    axes[1].set_xlabel("Ordered query component"); axes[1].set_ylabel("Classification head")
    target = Path(output); target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, bbox_inches="tight", facecolor="white"); plt.close(fig)


def plot_generative_semantic_id_results(
    baseline1_path: str | Path,
    baseline2_path: str | Path,
    direct_semantic_path: str | Path,
    generative_path: str | Path,
    output: str | Path,
) -> None:
    """Compare the proposed generator with the established retrieval baselines."""
    plt = _style()
    baseline1 = json.loads(Path(baseline1_path).read_text(encoding="utf-8"))
    baseline2 = json.loads(Path(baseline2_path).read_text(encoding="utf-8"))
    direct = json.loads(Path(direct_semantic_path).read_text(encoding="utf-8"))
    generated = json.loads(Path(generative_path).read_text(encoding="utf-8"))
    methods = {
        "Classical\nensemble": baseline1["full_classical_ensemble"],
        "Attention-GRU": baseline2["attention_gru"],
        "Direct\nSemantic-ID": direct["semantic_id_hybrid"],
        "Generative\nSemantic-ID": generated["generative_semantic_id"],
    }
    metrics = [("precision@1", "P@1"), ("recall@10", "R@10"), ("map@10", "mAP@10"), ("ndcg@10", "nDCG@10")]
    fig, axes = plt.subplots(1, 2, figsize=(17, 6), constrained_layout=True, gridspec_kw={"width_ratios": [1.45, 1]})
    x = np.arange(len(methods)); width = 0.18; colors = ["#003049", "#669bbc", "#fcbf49", "#d62828"]
    for column, (metric, label) in enumerate(metrics):
        bars = axes[0].bar(x + (column - 1.5) * width, [values[metric] for values in methods.values()], width, label=label, color=colors[column])
        axes[0].bar_label(bars, fmt="%.2f", fontsize=8, padding=2, rotation=90)
    axes[0].set_xticks(x, methods); axes[0].set_ylim(0, 0.92); axes[0].set_ylabel("Score")
    axes[0].set_title("Held-out retrieval quality"); axes[0].legend(ncols=4, fontsize=9, frameon=False, loc="upper center")
    axes[0].spines[["top", "right"]].set_visible(False)

    g = generated["generative_semantic_id"]
    diagnostics = [("Exact Semantic-ID hit@beam", g["semantic_id_exact_hit@beam"]), ("Any relevant bucket hit@beam", g["semantic_candidate_hit@beam"]),
                   ("Binding accuracy", g["binding_accuracy"]), ("Context satisfaction", g["context_satisfaction"])]
    bars = axes[1].barh([name for name, _ in diagnostics][::-1], [value for _, value in diagnostics][::-1], color=["#2a9d8f", "#457b9d", "#e9c46a", "#e76f51"][::-1])
    axes[1].bar_label(bars, fmt="%.3f", padding=3); axes[1].set_xlim(0, 1.08); axes[1].set_xlabel("Rate")
    axes[1].set_title("Generator candidate quality")
    axes[1].spines[["top", "right"]].set_visible(False)
    target = Path(output); target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, bbox_inches="tight", facecolor="white"); plt.close(fig)


def plot_semantic_id_confusion(results_path: str | Path, output: str | Path) -> None:
    """Visualize source-image Semantic-ID classification by quantization level."""
    import seaborn as sns

    plt = _style()
    metrics = json.loads(Path(results_path).read_text(encoding="utf-8"))["generative_semantic_id"]
    matrices = np.asarray(metrics["semantic_id_confusion_matrices"])
    accuracies = metrics["semantic_id_level_accuracy@1"]
    columns = 2
    rows = int(np.ceil(len(matrices) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(12, 5.3 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for level, (matrix, accuracy) in enumerate(zip(matrices, accuracies)):
        active = np.where((matrix.sum(axis=0) + matrix.sum(axis=1)) > 0)[0]
        shown = matrix[np.ix_(active, active)]
        sns.heatmap(shown, annot=True, fmt="d", cmap="Blues", cbar=False, square=True,
                    xticklabels=active, yticklabels=active, ax=axes[level])
        axes[level].set_title(f"Semantic-ID level {level + 1}: accuracy={accuracy:.3f}")
        axes[level].set_xlabel("Predicted code"); axes[level].set_ylabel("True code")
    for axis in axes[len(matrices):]:
        axis.axis("off")
    fig.suptitle(f"Top-1 Semantic-ID code confusion on held-out source images (exact ID accuracy={metrics['semantic_id_source_exact_accuracy@1']:.3f})", fontweight="bold")
    target = Path(output); target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, bbox_inches="tight", facecolor="white"); plt.close(fig)


def plot_direct_semantic_hybrid_method(output: str | Path) -> None:
    """One-page plain-language diagram of the final direct hybrid method."""
    from matplotlib.patches import FancyBboxPatch

    plt = _style()
    fig, ax = plt.subplots(figsize=(17, 6.5), constrained_layout=True)
    ax.set_axis_off(); ax.set_xlim(0, 17); ax.set_ylim(0, 6.5)
    boxes = [
        (0.3, 3.8, 2.4, 1.25, "Text query\n‘red shoes on street’", "#e9c46a"),
        (3.4, 4.65, 2.8, 1.1, "Frozen Qwen encoder\ntext embedding", "#457b9d"),
        (3.4, 2.25, 2.8, 1.1, "RQ-VAE quantizer\nSemantic-ID prefixes", "#2a9d8f"),
        (7.0, 4.65, 2.6, 1.1, "Dense candidates\nTop 100 by cosine", "#a8dadc"),
        (7.0, 2.25, 2.6, 1.1, "Semantic-ID candidates\nprefix matches", "#a8dadc"),
        (10.5, 3.4, 2.9, 1.7, "Binding-aware reranking\nplural normalization\ncompound-label matching", "#f4a261"),
        (14.2, 3.8, 2.4, 1.25, "Best image\nrank 1", "#90be6d"),
    ]
    for x, y, width, height, text, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.12,rounding_size=0.12", linewidth=1.5, edgecolor="#264653", facecolor=color))
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=11, fontweight="bold")
    arrows = [((2.7, 4.45), (3.4, 5.15)), ((2.7, 4.25), (3.4, 2.8)), ((6.2, 5.2), (7.0, 5.2)),
              ((6.2, 2.8), (7.0, 2.8)), ((9.6, 5.2), (10.5, 4.6)), ((9.6, 2.8), (10.5, 3.9)), ((13.4, 4.25), (14.2, 4.4))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 2, "color": "#264653"})
    ax.text(11.95, 2.65, "Final score: 0.204 dense + 0.198 binding +\n0.439 context + 0.011 Semantic-ID + 0.147 style", ha="center", fontsize=10)
    ax.text(8.5, 0.65, "Held-out result: P@1 = 92.5%  |  Binding accuracy = 94.1%  |  Context satisfaction = 100%", ha="center", fontsize=13, fontweight="bold", color="#264653")
    target = Path(output); target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, bbox_inches="tight", facecolor="white"); plt.close(fig)


def plot_qualitative_retrievals(samples_path: str | Path, output: str | Path) -> None:
    """Show the top retrieved image for each stored qualitative query sample."""
    from PIL import Image
    import textwrap

    plt = _style()
    samples = json.loads(Path(samples_path).read_text(encoding="utf-8"))["samples"]
    fig, axes = plt.subplots(1, len(samples), figsize=(4.2 * len(samples), 5.9), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, sample in zip(axes, samples):
        top = sample["top_result"]
        if top is None or not Path(top["image_path"]).exists():
            axis.text(0.5, 0.5, "Image unavailable", ha="center", va="center")
            axis.axis("off"); continue
        with Image.open(top["image_path"]) as image:
            axis.imshow(image.convert("RGB"))
        requested = textwrap.fill(sample["query"].replace("A person wearing ", ""), width=24)
        garments = ", ".join(f"{item['color']} {item['type'].replace('_/_', '/') }" for item in top["garments"][:3])
        result = textwrap.fill(f"Top: {garments}\nScene: {top['environment']}\nGrade: {top['relevance_grade']}/2", width=26)
        axis.set_title(requested, fontsize=10, fontweight="bold", pad=10)
        axis.set_xlabel(result, fontsize=9, labelpad=8)
        color = "#2a9d8f" if top["relevance_grade"] == 2 else "#e9c46a" if top["relevance_grade"] == 1 else "#e76f51"
        for spine in axis.spines.values():
            spine.set_visible(True); spine.set_linewidth(5); spine.set_edgecolor(color)
        axis.set_xticks([]); axis.set_yticks([])
    fig.suptitle("Five held-out test queries: top retrieved image", fontsize=17, fontweight="bold")
    target = Path(output); target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, bbox_inches="tight", facecolor="white"); plt.close(fig)


def dataset_summary(metadata_path: str | Path) -> dict:
    records = load_jsonl(metadata_path, ImageRecord)
    garments = [g for record in records for g in record.garments]
    known = lambda value: bool(value and value != "unknown")
    return {
        "images": len(records),
        "garment_instances": len(garments),
        "sources": dict(Counter(r.source for r in records)),
        "splits": dict(Counter(r.split for r in records)),
        "environments": dict(Counter(r.environment for r in records)),
        "activities": dict(Counter(r.activity for r in records)),
        "garments": dict(Counter(g.type for g in garments)),
        "colors": dict(Counter(g.color for g in garments)),
        "annotation_completeness": {
            "caption": sum(bool(r.caption.strip()) for r in records) / max(len(records), 1),
            "garments": sum(bool(r.garments) for r in records) / max(len(records), 1),
            "garment_color": sum(bool(r.garments) and all(known(g.color) for g in r.garments) for r in records) / max(len(records), 1),
            "environment": sum(known(r.environment) for r in records) / max(len(records), 1),
            "activity": sum(known(r.activity) for r in records) / max(len(records), 1),
            "style": sum(bool(r.style) for r in records) / max(len(records), 1),
            "objects": sum(bool(r.objects) for r in records) / max(len(records), 1),
        },
    }


def plot_dataset_analysis(metadata_path: str | Path, output: str | Path, summary_output: str | Path | None = None) -> dict:
    """Create one compact EDA dashboard from validated project metadata."""
    import seaborn as sns

    plt = _style()
    records = load_jsonl(metadata_path, ImageRecord)
    summary = dataset_summary(metadata_path)
    garments = [g for record in records for g in record.garments]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    main = "#264653"; accent = "#e76f51"; secondary = "#2a9d8f"

    def horizontal_counts(ax, counts: dict, title: str, limit: int = 10):
        items = Counter(counts).most_common(limit)[::-1]
        ax.barh([name.replace("_", " ") for name, _ in items], [value for _, value in items], color=main)
        ax.set_title(title); ax.set_xlabel("Count"); ax.spines[["top", "right"]].set_visible(False)
        for container in ax.containers: ax.bar_label(container, padding=3, fontsize=9)

    horizontal_counts(axes[0, 0], summary["environments"], "Environment coverage")
    horizontal_counts(axes[0, 1], summary["garments"], "Garment instances")
    horizontal_counts(axes[0, 2], summary["colors"], "Garment colors")

    garment_names = [name for name, _ in Counter(g.type for g in garments).most_common(8)]
    color_names = [name for name, _ in Counter(g.color for g in garments).most_common(8)]
    matrix = np.zeros((len(garment_names), len(color_names)), dtype=int)
    garment_pos = {name: i for i, name in enumerate(garment_names)}
    color_pos = {name: i for i, name in enumerate(color_names)}
    for garment in garments:
        if garment.type in garment_pos and garment.color in color_pos:
            matrix[garment_pos[garment.type], color_pos[garment.color]] += 1
    if matrix.size:
        sns.heatmap(matrix, annot=True, fmt="d", cmap="YlGnBu", cbar_kws={"label": "Instances"},
                    xticklabels=[x.replace("_", " ") for x in color_names],
                    yticklabels=[x.replace("_", " ") for x in garment_names], ax=axes[1, 0])
    axes[1, 0].set_title("Garment × color coverage"); axes[1, 0].set_xlabel("Color"); axes[1, 0].set_ylabel("Garment")

    environments = [name for name, _ in Counter(r.environment for r in records).most_common(5)]
    top_garments = [name for name, _ in Counter(g.type for g in garments).most_common(8)]
    environment_garment = np.zeros((len(environments), len(top_garments)), dtype=int)
    environment_pos = {name: i for i, name in enumerate(environments)}
    top_garment_pos = {name: i for i, name in enumerate(top_garments)}
    for record in records:
        for garment in record.garments:
            if garment.type in top_garment_pos:
                environment_garment[environment_pos[record.environment], top_garment_pos[garment.type]] += 1
    short_names = []
    for name in top_garments:
        parts = name.replace("_", " ").split(" / ")
        short_names.append("/".join(parts[:2]))
    if environment_garment.size:
        sns.heatmap(environment_garment, annot=True, fmt="d", cmap="OrRd", cbar_kws={"label": "Instances"},
                    xticklabels=short_names,
                    yticklabels=[x.replace("_", " ") for x in environments], ax=axes[1, 1])
    axes[1, 1].set_title("Environment × clothing type")
    axes[1, 1].set_xlabel("Clothing type"); axes[1, 1].set_ylabel("Environment")
    axes[1, 1].tick_params(axis="x", rotation=45, labelsize=9)
    completeness = summary["annotation_completeness"]
    names = [name.replace("_", " ") for name in completeness]
    values = [100 * value for value in completeness.values()]
    colors = [secondary if value >= 80 else "#e9c46a" if value >= 50 else accent for value in values]
    axes[1, 2].barh(names[::-1], values[::-1], color=colors[::-1])
    axes[1, 2].set_xlim(0, 105); axes[1, 2].set_xlabel("Records complete (%)"); axes[1, 2].set_title("Annotation completeness")
    axes[1, 2].axvline(80, color="#555555", linestyle="--", linewidth=1)
    for container in axes[1, 2].containers: axes[1, 2].bar_label(container, fmt="%.0f%%", padding=3, fontsize=9)
    axes[1, 2].spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Fashion retrieval dataset analysis · {len(records):,} images · {len(garments):,} garment instances", fontsize=18, fontweight="bold", y=1.045)
    target = Path(output); target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, bbox_inches="tight", facecolor="white"); plt.close(fig)
    if summary_output:
        summary_path = Path(summary_output); summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def plot_eda_report_figures(metadata_path: str | Path, output_dir: str | Path) -> list[str]:
    """Write focused, publication-ready EDA figures for the LaTeX report."""
    import seaborn as sns

    plt = _style()
    records = load_jsonl(metadata_path, ImageRecord)
    garments = [g for record in records for g in record.garments]
    target = Path(output_dir); target.mkdir(parents=True, exist_ok=True)
    outputs = []

    # Figure 1: the three required axes, using counts rather than percentages to expose imbalance.
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)
    panels = [
        ("Environment", Counter(r.environment for r in records), 8),
        ("Clothing type", Counter(g.type for g in garments), 10),
        ("Garment color", Counter(g.color for g in garments), 10),
    ]
    for ax, (title, counts, limit) in zip(axes, panels):
        items = counts.most_common(limit)[::-1]
        labels = [name.replace("_", " ") for name, _ in items]
        values = [value for _, value in items]
        colors = ["#e9c46a" if label == "unknown" else "#264653" for label in labels]
        bars = ax.barh(labels, values, color=colors)
        ax.bar_label(bars, padding=3, fontsize=9); ax.set_title(title); ax.set_xlabel("Count")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Fashionpedia-1K: distributions along the three primary axes", fontweight="bold", fontsize=17)
    path = target / "eda_primary_axes.png"; fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig); outputs.append(str(path))

    top_garments = [name for name, _ in Counter(g.type for g in garments).most_common(8)]
    known_colors = [name for name, _ in Counter(g.color for g in garments if g.color != "unknown").most_common(8)]

    # Figure 2: row-normalized garment/color composition so shoes do not hide rare garments.
    matrix = np.zeros((len(top_garments), len(known_colors)), dtype=float)
    garment_pos = {name: i for i, name in enumerate(top_garments)}; color_pos = {name: i for i, name in enumerate(known_colors)}
    for garment in garments:
        if garment.type in garment_pos and garment.color in color_pos:
            matrix[garment_pos[garment.type], color_pos[garment.color]] += 1
    matrix = 100 * matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(11, 6.8), constrained_layout=True)
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="YlGnBu", vmin=0, cbar_kws={"label": "Share within clothing type (%)"},
                xticklabels=[x.replace("_", " ") for x in known_colors],
                yticklabels=[x.replace("_", " ") for x in top_garments], ax=ax)
    ax.set_title("Garment–color composition", fontweight="bold"); ax.set_xlabel("Dominant mask-derived color"); ax.set_ylabel("Clothing type")
    path = target / "eda_garment_color.png"; fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig); outputs.append(str(path))

    # Figure 3: row-normalized environment/clothing composition to compare unequal scene groups.
    environments = [name for name, _ in Counter(r.environment for r in records).most_common()]
    matrix = np.zeros((len(environments), len(top_garments)), dtype=float)
    env_pos = {name: i for i, name in enumerate(environments)}
    for record in records:
        for garment in record.garments:
            if garment.type in garment_pos: matrix[env_pos[record.environment], garment_pos[garment.type]] += 1
    matrix = 100 * matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    short = ["/".join(x.replace("_", " ").split(" / ")[:2]) for x in top_garments]
    fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="OrRd", vmin=0, cbar_kws={"label": "Share within environment (%)"},
                xticklabels=short, yticklabels=[x.replace("_", " ") for x in environments], ax=ax)
    ax.set_title("Environment–clothing composition", fontweight="bold"); ax.set_xlabel("Clothing type"); ax.set_ylabel("Environment")
    ax.tick_params(axis="x", rotation=35); ax.tick_params(axis="y", rotation=0)
    path = target / "eda_environment_clothing.png"; fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig); outputs.append(str(path))

    # Figure 4: completeness of labels used by the retrieval design.
    completeness = dataset_summary(metadata_path)["annotation_completeness"]
    names = [name.replace("_", " ") for name in completeness]
    values = [100 * value for value in completeness.values()]
    colors = ["#2a9d8f" if value >= 80 else "#e9c46a" if value >= 50 else "#e76f51" for value in values]
    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    bars = ax.barh(names[::-1], values[::-1], color=colors[::-1]); ax.bar_label(bars, fmt="%.0f%%", padding=3)
    ax.axvline(80, color="#555555", linestyle="--", linewidth=1); ax.set_xlim(0, 105)
    ax.set_xlabel("Records complete (%)"); ax.set_title("Annotation completeness", fontweight="bold"); ax.spines[["top", "right"]].set_visible(False)
    path = target / "eda_annotation_completeness.png"; fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig); outputs.append(str(path))
    return outputs
