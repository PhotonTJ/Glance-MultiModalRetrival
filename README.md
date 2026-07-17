# Fashion Search That Understands the Whole Sentence

Natural-language image retrieval over fashion, color, and context — with a dense-vector baseline and a compositional Semantic-ID solution.

> **Best retained result:** 92.5% Precision@1, 94.1% garment–color binding accuracy, and 100% context satisfaction on 120 held-out automatic queries.

## The problem

Finding an image of a *red shirt* is easy. Finding *a red tie with a white shirt in an office* is harder.

A global image embedding often recognizes all the right concepts but loses the relationships between them. It may return a white tie with a red shirt, or match the clothes while ignoring the requested location. This project tests how much explicit structure is needed on top of multimodal vector search to keep those details together.

The retriever handles queries that mix:

- garment type and color;
- several garment–color pairs in the same sentence;
- environment, such as office, park, home, or street;
- activity and visible objects when labels are available; and
- broad style cues such as formal, professional, or casual.

Examples include:

```text
a bright yellow raincoat
professional business attire inside a modern office
a blue shirt while sitting on a park bench
a casual weekend outfit for a city walk
a red tie and a white shirt in a formal setting
```

## What is in the repository

The submission is organized as two complete workflows. Each has a separate indexer (Part A) and retriever (Part B), and each keeps its own results.

```text
.
├── baseline/
│   ├── indexer.py                 # Part A: Qwen image embeddings + flat FAISS index
│   ├── retriever.py               # Part B: text embedding + cosine top-k search
│   ├── checkpoints/               # retained baseline model state
│   ├── results/                   # baseline evaluation JSON
│   └── README.md
│
├── solution/
│   ├── indexer.py                 # Part A: Qwen features + FAISS IVF-PQ storage
│   ├── retriever.py               # Part B: Semantic-ID hybrid retrieval
│   ├── app.py                     # optional Streamlit interface
│   ├── config/default.yaml        # model, index, and scoring settings
│   ├── data/                      # smoke data, manifests, and local dataset files
│   ├── artifacts/
│   │   ├── indexes/
│   │   ├── checkpoints/
│   │   ├── results/
│   │   └── figures/
│   ├── src/fashion_retrieval/     # reusable implementation
│   ├── tests/
│   ├── DATASET_CARD.md
│   └── README.md
│
└── report/
    ├── report.md                  # complete method-by-method technical report
    ├── main.tex                   # earlier LaTeX report source
    └── figures/                   # evaluation plots and qualitative outputs
```

The `report/` directory is deliberately isolated. The complete Markdown report is available at [`report/report.md`](report/report.md). The folder can be deleted after the final report is exported without affecting either workflow.

## Method 1: dense-vector baseline

The baseline answers a simple question: how far can a strong shared embedding space get on its own?

### Part A — Indexer

1. Load the image manifest from JSONL.
2. Encode every image with `Qwen/Qwen3-VL-Embedding-2B`.
3. Truncate the Matryoshka representation to 256 dimensions.
4. L2-normalize every vector.
5. Store the vectors in a FAISS inner-product index.

For normalized vectors, inner product is cosine similarity. Retrieval is therefore semantic vector search, not filename or keyword matching.

### Part B — Retriever

1. Accept a free-form query string.
2. Encode and normalize the complete sentence with the same model.
3. Search the FAISS index.
4. Return the top `k` image IDs, paths, and similarity scores.

Because the whole sentence is embedded, the baseline can respond to multiple attributes. Its weakness is that it has no explicit representation of *which color belongs to which garment*.

### Baseline experiments

The saved baseline results also include two attempts to improve dense rankings:

- **Classical ensemble:** PCA-reduced Qwen features with Naive Bayes, logistic regression, linear SVM, random forest, and K-means evidence.
- **Attention-GRU:** composes query components with learned attention and predicts garment, color, garment–color binding, and environment signals before reranking dense candidates.

These experiments are retained as honest comparisons. The classical ensemble improves the first result substantially, while the Attention-GRU improves deeper ranking quality but still struggles with binding.

## Method 2: Direct Semantic-ID Hybrid

The proposed solution keeps Qwen as the open-vocabulary recall layer and adds explicit evidence for the relationships that a single vector tends to blur.

### Part A — Feature extraction and vector storage

```text
image
  └── Qwen3-VL embedding (256D, normalized)
        ├── FAISS IVF-PQ index
        └── residual quantizer → Semantic ID (c1, c2, c3)
```

The retained 1,000-image index uses:

| Setting | Value |
|---|---:|
| FAISS index | IVF-PQ |
| Inverted lists (`nlist`) | 16 |
| PQ subquantizers (`m`) | 16 |
| Bits per subquantizer | 4 |
| Probed lists (`nprobe`) | 16 |
| Candidate expansion | 6× |
| Embedding dimension | 256 |

Each index directory contains:

```text
index.faiss        persisted serving index
index_config.json  model and FAISS settings
records.json       structured image metadata
vectors.npy        normalized vectors for training and exact candidate rescoring
```

The FAISS index generates candidates. The stored float vectors are only used for training and exact rescoring of that smaller candidate set; serving does not begin with a full NumPy scan.

The residual quantizer assigns each image a three-level Semantic ID. Prefix tables map coarse and fine Semantic-ID prefixes back to gallery images, giving the retriever a second candidate route beyond dense similarity.

### Part B — Context-aware retrieval

```text
natural-language query
        │
        ├── Qwen text embedding ──→ FAISS candidates ──────┐
        ├── residual quantizer ───→ Semantic-ID candidates ├──→ hybrid score → top k
        └── query parser ─────────→ structured constraints ┘
```

The parser extracts garment–color pairs and contextual constraints. It also normalizes plurals and Fashionpedia compound labels, so `shoes` matches `shoe`, and `shirt` matches `shirt_/_blouse`.

Every candidate receives five signals:

- dense Qwen similarity;
- garment–color binding agreement;
- environment, activity, and object agreement;
- Semantic-ID prefix relationship; and
- style agreement.

The final weights were selected on validation queries and then frozen:

```text
score = 0.204 × dense similarity
      + 0.198 × garment–color binding
      + 0.439 × context
      + 0.011 × Semantic-ID relation
      + 0.147 × style
```

This is the recommended method. Semantic IDs help with routing, but no single code is trusted to make the decision; dense and structured evidence still determine the final ranking.

## Generative Semantic-ID experiment

The repository also contains a research extension that asks `google/flan-t5-small` to generate Semantic IDs directly from the raw query.

An attribute-aware RQ-VAE shapes the hierarchy before the generator is trained:

```text
level 1 prefix → garment category
level 2 prefix → garment–color binding
level 3 prefix → environment and style
```

The generator uses trie-constrained beam search, so it can emit only Semantic-ID sequences that exist in the gallery. Images from the generated buckets are then reranked with dense and structured evidence.

At beam size 5, this method reaches 86.7% Precision@1. It is a useful experiment, but it opens about 327 candidates per query and trails the simpler Direct Hybrid on accuracy, ranking quality, binding, and candidate efficiency.

## Results

All figures below come from the JSON files committed under `baseline/results/` and `solution/artifacts/results/`.

### Main comparison

| Method | P@1 | R@5 | R@10 | mAP@10 | nDCG@10 | Binding | Context | Mean candidates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw Qwen cosine | 21.7% | 9.7% | 15.5% | 8.4% | 48.5% | — | — | 200 |
| Classical ensemble | 72.5% | 32.5% | 42.1% | 42.5% | 75.3% | — | — | 200 |
| Attention-GRU | 67.5% | 33.8% | 49.7% | 57.5% | 78.7% | 58.5% | 100.0% | 200 |
| Generative Semantic IDs, beam 5 | 86.7% | 36.8% | 41.9% | 55.9% | 80.9% | 88.1% | 94.2% | 326.5 |
| **Direct Semantic-ID Hybrid** | **92.5%** | **47.5%** | **54.6%** | **68.6%** | **86.5%** | **94.1%** | **100.0%** | **95.2** |

The final method gets the first result right for **111 of 120 queries**. It also has a 98.3% Semantic-ID hit rate at 10 and a measured mean query latency of 127.9 ms in the retained run.

### What the comparison shows

- Dense embeddings provide useful recall but are weak on exact multi-attribute matches.
- Classical reranking produces a large first-result improvement without solving binding directly.
- The Attention-GRU improves mAP and nDCG, but its 58.5% binding accuracy exposes the remaining compositional problem.
- Attribute-aware generation improves Semantic-ID retrieval, but broad generated buckets reduce efficiency.
- The Direct Hybrid gives the strongest balance of top-result accuracy, deeper ranking quality, interpretability, and candidate count.

## Dataset and evaluation protocol

The working subset contains 1,000 Fashionpedia validation images and 4,337 garment instances.

| Split | Images |
|---|---:|
| Train | 700 |
| Validation | 151 |
| Test | 149 |

The structured metadata contains garment masks and categories, mask-derived colors, and environment labels for office, urban street, park, and home. Garment color is populated for 72.7% of images, and environment is populated for 90.7%. Activity, style, and object fields remain unpopulated in this particular 1,000-image run.

The held-out evaluation file contains 120 automatically generated queries. Relevance uses three grades:

- `2` — exact match;
- `1` — partial match;
- `0` — irrelevant.

Precision, recall, and mAP treat grade 2 as relevant. nDCG uses the graded labels.

There is an important limitation: the queries and judgments were generated from the same ontology used by the structured reranker. The results are valid for controlled engineering comparison, but they are not a replacement for human-written queries and independent relevance judgments. See [`solution/DATASET_CARD.md`](solution/DATASET_CARD.md) for the full dataset notes.

## Quick start

### 1. Create the environment

Python 3.10 or newer is required. The repository includes both `pyproject.toml` and `uv.lock`.

```bash
uv sync --all-extras
```

The first real indexing or search run downloads the Qwen model through Sentence Transformers. Large dataset files, indexes, and trained checkpoints are ignored by Git.

### 2. Run the baseline

From the repository root:

```bash
uv run python baseline/indexer.py \
  --metadata solution/data/processed/fashionpedia_1000/metadata.jsonl \
  --output baseline/index

uv run python baseline/retriever.py \
  "a blue shirt while sitting on a park bench" \
  --index baseline/index \
  -k 5
```

### 3. Run the proposed solution

The solution manifest stores image paths relative to `solution/`, so run these commands from that directory:

```bash
cd solution

uv run python indexer.py \
  --metadata data/processed/fashionpedia_1000/metadata.jsonl \
  --output artifacts/indexes/fashionpedia_qwen

uv run python retriever.py \
  "a red tie and a white shirt in a formal office" \
  --index artifacts/indexes/fashionpedia_qwen \
  --semantic-ids artifacts/checkpoints/rqvae_direct_qwen \
  --weights artifacts/results/direct_semantic_binding_weight_tuning.json \
  -k 5
```

The retriever prints JSON containing rank, image ID, image path, final score, and a short explanation of the matched evidence.

### 4. Launch the interface

From `solution/`:

```bash
uv run streamlit run app.py
```

## Reproduce the retained evaluations

The package exposes a larger experiment CLI as `fashion-retrieval`.

### Direct Semantic-ID Hybrid

```bash
cd solution

uv run fashion-retrieval evaluate-semantic-ids \
  --index artifacts/indexes/fashionpedia_qwen \
  --queries data/processed/fashionpedia_1000/queries_auto_test.jsonl \
  --semantic-dir artifacts/checkpoints/rqvae_direct_qwen \
  --weights-json artifacts/results/direct_semantic_binding_weight_tuning.json \
  --output artifacts/results/semantic_id_hybrid_binding_fixed_qwen.json
```

### Attribute-aware generative Semantic IDs

```bash
cd solution

uv run fashion-retrieval train-rqvae \
  --index artifacts/indexes/fashionpedia_qwen \
  --output artifacts/checkpoints/rqvae_attribute_qwen \
  --epochs 50 \
  --levels 3 \
  --codebook-size 16 \
  --attribute-weight 0.1

uv run fashion-retrieval train-generative-semantic-ids \
  --index artifacts/indexes/fashionpedia_qwen \
  --semantic-dir artifacts/checkpoints/rqvae_attribute_qwen \
  --output artifacts/checkpoints/generative_attribute_flan_t5 \
  --epochs 20 \
  --batch-size 32 \
  --learning-rate 3e-4

uv run fashion-retrieval evaluate-generative-semantic-ids \
  --index artifacts/indexes/fashionpedia_qwen \
  --queries data/processed/fashionpedia_1000/queries_auto_test.jsonl \
  --semantic-dir artifacts/checkpoints/rqvae_attribute_qwen \
  --generator-dir artifacts/checkpoints/generative_attribute_flan_t5 \
  --beam-size 5 \
  --output artifacts/results/generative_attribute_flan_t5.json
```

## Tests

From the repository root:

```bash
uv run pytest -q
```

The test suite covers query parsing, garment-label normalization, structured scoring, FAISS persistence, Semantic-ID routing, RQ-VAE training, and model shapes. A deterministic hash encoder and six-image smoke gallery test the wiring without presenting smoke scores as model quality.

## Key implementation modules

| Module | Responsibility |
|---|---|
| `encoder.py` | Qwen, hash-test, and projected encoders |
| `index.py` | FAISS IVF-PQ construction, persistence, and candidate rescoring |
| `parser.py` | query parsing and garment-label normalization |
| `retriever.py` | dense and structured hybrid scoring |
| `semantic_id.py` | Direct Semantic-ID routing and evaluation |
| `generative_semantic_id.py` | text-to-ID training, constrained generation, and retrieval |
| `training.py` | Siamese projection and RQ-VAE training |
| `classical.py` | classical baseline models and ensemble |
| `attention_gru.py` | Attention-GRU baseline training and reranking |
| `metrics.py` | Precision, Recall, mAP, and nDCG |
| `fashionpedia_analysis.py` | subset preparation, masks, colors, and environment labels |

## Practical notes

- The six-image smoke set proves that code paths work; it is not evidence of retrieval quality.
- The retained IVF-PQ settings favor recall on a small gallery. A million-image deployment would need new training data and joint tuning of `nlist`, `nprobe`, PQ width, candidate expansion, latency, and recall.
- `vectors.npy` is retained because the training code needs float embeddings and the candidate set receives an exact cosine rescore.
- Checkpoints and full image data are local artifacts and are excluded from Git because of their size.
- The recommended production path is the Direct Semantic-ID Hybrid. The generator remains an experimental extension.
