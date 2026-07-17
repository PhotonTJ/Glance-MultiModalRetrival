# Multi-Modal Fashion Retrieval: Submission Report

## Abstract

This project builds and evaluates a natural-language image retrieval system for fashion images. The goal is to retrieve images that satisfy not only single visual concepts, such as "red shirt", but also compositional queries such as "a red tie and a white shirt in an office". The system compares a dense multimodal retrieval baseline against a proposed Semantic-ID hybrid retriever that combines Qwen image-text embeddings, FAISS indexing, structured garment-color/context scoring, and hierarchical Semantic-ID candidate routing.

The best retained system is the Direct Semantic-ID Hybrid. On 120 held-out automatically generated Fashionpedia queries, it achieved 92.5% Precision@1, 54.6% Recall@10, 68.6% mAP@10, 86.5% nDCG@10, 94.1% garment-color binding accuracy, and 100% context satisfaction. These results show that explicit structure improves retrieval when a query contains multiple garments, colors, and scene constraints.

## 1. Problem Statement

Standard image retrieval systems often embed the whole image and query into a shared vector space, then rank images by cosine similarity. This works well for broad semantic similarity, but it can fail when a query contains multiple bound attributes. For example, the query "a red tie with a white shirt in an office" is different from "a white tie with a red shirt in an office". A single global embedding may detect the words "red", "white", "shirt", "tie", and "office", but still return an image where the colors are attached to the wrong garments.

The project therefore focuses on multi-attribute fashion retrieval with three main requirements:

- Match garment categories, such as shirt, dress, pants, tie, shoe, and jacket.
- Preserve garment-color bindings, such as red tie versus red shirt.
- Respect contextual constraints, such as office, park, home, or urban street.

## 2. Objectives

The project objectives were:

1. Build a baseline multimodal retrieval system using dense Qwen image and text embeddings.
2. Build a stronger retrieval system that keeps dense semantic recall while adding explicit structured reasoning.
3. Evaluate both systems on top-k retrieval metrics and compositional correctness metrics.
4. Provide reproducible scripts for indexing, retrieval, testing, and evaluation.
5. Document dataset limitations clearly so results are interpreted correctly.

## 3. Dataset

The project uses a 1,000-image subset selected from the Fashionpedia 2020 validation split. The official Fashionpedia test-only images were not used because they do not include garment annotations.

Dataset summary:

| Property | Value |
|---|---:|
| Source | Fashionpedia 2020 validation split |
| Images | 1,000 |
| Garment/accessory instances | 4,337 |
| Train images | 700 |
| Validation images | 151 |
| Test images | 149 |
| Seed | 42 |
| Metadata file | `solution/data/processed/fashionpedia_1000/metadata.jsonl` |

The metadata includes Fashionpedia garment categories and segmentation masks. Garment colors are derived from mask pixels using a fixed 13-color vocabulary: black, white, gray, red, orange, yellow, green, blue, navy, purple, pink, brown, and beige.

Environment labels are not native to Fashionpedia. They are derived using zero-shot CLIP scene classification into office, urban street, park, home, or unknown. This makes the dataset useful for controlled retrieval experiments, but environment labels should be treated as automatically generated labels rather than human annotations.

Important dataset limitations:

- The subset is skewed toward urban street images.
- Shoes are overrepresented compared with rare categories such as ties.
- Color labels reduce patterned or multicolored garments to one dominant color.
- Activity, object, and style annotations are limited in the retained 1,000-image run.
- Evaluation queries and judgments use the same ontology as the structured reranker, so results are best read as controlled engineering comparisons.

## 4. Baseline Method

The baseline tests how far dense multimodal retrieval can go without explicit structure.

### 4.1 Baseline Indexer

The baseline indexer performs the following steps:

1. Load image metadata from JSONL.
2. Encode each image using `Qwen/Qwen3-VL-Embedding-2B`.
3. Truncate the embedding to 256 dimensions.
4. L2-normalize the vector.
5. Store vectors in a FAISS inner-product index.

For normalized vectors, inner product is equivalent to cosine similarity. The baseline therefore performs semantic vector search rather than keyword or filename matching.

### 4.2 Baseline Retriever

The baseline retriever:

1. Accepts a natural-language query.
2. Encodes the full query sentence using the same embedding model.
3. Normalizes the query vector.
4. Searches the FAISS index.
5. Returns the top-k image IDs, paths, and similarity scores.

This approach can recognize broad query meaning, but it has no explicit representation of which color belongs to which garment.

## 5. Proposed Method: Direct Semantic-ID Hybrid

The proposed solution keeps dense Qwen search as a recall layer and adds structured reranking for compositional correctness.

### 5.1 Indexing

Each image is represented with:

- A 256-dimensional normalized Qwen embedding.
- A persisted FAISS IVF-PQ index for candidate retrieval.
- Structured metadata for garment categories, garment colors, and environment.
- A three-level Semantic ID produced by a residual quantizer.

Retained FAISS settings:

| Setting | Value |
|---|---:|
| Index type | IVF-PQ |
| Embedding dimension | 256 |
| Inverted lists | 16 |
| PQ subquantizers | 16 |
| Bits per subquantizer | 4 |
| Probed lists | 16 |
| Candidate expansion | 6x |

The FAISS index provides dense candidates. Semantic-ID prefix tables provide an additional route to candidates that share coarse or fine quantized structure with the query.

### 5.2 Query Processing

For each natural-language query, the system:

1. Encodes the query with Qwen.
2. Parses garment-color pairs and contextual constraints.
3. Normalizes labels, such as matching "shirt" to Fashionpedia's `shirt_/_blouse`.
4. Retrieves candidates from FAISS.
5. Retrieves candidates from Semantic-ID prefix tables.
6. Reranks the merged candidate set using dense and structured evidence.

### 5.3 Hybrid Scoring

Each candidate receives five scoring signals:

- Dense Qwen similarity.
- Garment-color binding agreement.
- Context agreement.
- Semantic-ID relation.
- Style agreement.

The retained validation-tuned scoring formula is:

```text
score = 0.204 * dense_similarity
      + 0.198 * garment_color_binding
      + 0.439 * context
      + 0.011 * semantic_id_relation
      + 0.147 * style
```

This weighting makes context and garment-color structure important while still preserving dense semantic recall.

## 6. Additional Experiment: Generative Semantic IDs

The repository also includes a generative Semantic-ID experiment. In this version, `google/flan-t5-small` is trained to generate Semantic IDs directly from a query. Trie-constrained beam search ensures the generator emits only Semantic-ID sequences that exist in the gallery. Generated buckets are then reranked with dense and structured evidence.

This method reached 86.7% Precision@1 with beam size 5. It is useful as a research extension, but it opened a larger mean candidate set than the Direct Semantic-ID Hybrid and performed worse on final ranking metrics.

## 7. Implementation Structure

The repository is organized into a baseline workflow and a proposed solution workflow.

| Path | Purpose |
|---|---|
| `baseline/indexer.py` | Baseline Part A: image embedding and FAISS indexing |
| `baseline/retriever.py` | Baseline Part B: text query retrieval |
| `baseline/results/` | Baseline evaluation results |
| `solution/indexer.py` | Proposed Part A: Qwen features and IVF-PQ storage |
| `solution/retriever.py` | Proposed Part B: Semantic-ID hybrid retrieval |
| `solution/app.py` | Optional Streamlit interface |
| `solution/src/fashion_retrieval/` | Reusable package implementation |
| `solution/tests/` | Unit and integration tests |
| `solution/artifacts/results/` | Saved evaluation JSON files |
| `solution/artifacts/figures/` | Saved plots and visual outputs |
| `solution/DATASET_CARD.md` | Dataset scope and limitations |

Key implementation modules:

| Module | Responsibility |
|---|---|
| `encoder.py` | Qwen, test, and projected encoders |
| `index.py` | FAISS index construction, persistence, and rescoring |
| `parser.py` | Query parsing and label normalization |
| `retriever.py` | Dense and structured hybrid scoring |
| `semantic_id.py` | Semantic-ID routing and evaluation |
| `generative_semantic_id.py` | Text-to-ID generation and constrained retrieval |
| `training.py` | Siamese projection and RQ-VAE training |
| `metrics.py` | Precision, Recall, mAP, and nDCG |

## 8. Evaluation Protocol

The retained evaluation uses 120 automatically generated held-out queries. Relevance is graded:

| Grade | Meaning |
|---:|---|
| 2 | Exact match |
| 1 | Partial match |
| 0 | Irrelevant |

Precision, Recall, and mAP treat grade 2 as relevant. nDCG uses the graded labels. Additional compositional metrics measure garment-color binding accuracy and context satisfaction.

Main metrics:

- Precision@1: whether the first result is exactly relevant.
- Recall@5 and Recall@10: how many exact relevant images appear in the top results.
- mAP@10: ranking quality among the top 10.
- nDCG@10: graded ranking quality among the top 10.
- Binding accuracy: whether requested garment-color pairs are correctly matched.
- Context satisfaction: whether requested environment/context is satisfied.
- Mean candidates: average number of candidates reranked per query.

## 9. Results

The Direct Semantic-ID Hybrid produced the best retained result.

| Method | P@1 | R@5 | R@10 | mAP@10 | nDCG@10 | Binding | Context | Mean candidates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw Qwen cosine | 21.7% | 9.7% | 15.5% | 8.4% | 48.5% | N/A | N/A | 200.0 |
| Classical ensemble | 72.5% | 32.5% | 42.1% | 42.5% | 75.3% | N/A | N/A | 200.0 |
| Attention-GRU | 67.5% | 33.8% | 49.7% | 57.5% | 78.7% | 58.5% | 100.0% | 200.0 |
| Generative Semantic IDs, beam 5 | 86.7% | 36.8% | 41.9% | 55.9% | 80.9% | 88.1% | 94.2% | 326.5 |
| Direct Semantic-ID Hybrid | 92.5% | 47.5% | 54.6% | 68.6% | 86.5% | 94.1% | 100.0% | 95.2 |

The Direct Semantic-ID Hybrid retrieved the correct first result for 111 out of 120 held-out queries. It also achieved a 98.3% Semantic-ID hit rate at 10 and a retained mean latency of 127.9 ms.

## 10. Discussion

The results show that dense retrieval is a useful recall mechanism but is not enough for precise compositional retrieval. Raw Qwen cosine search reached only 21.7% Precision@1 on the retained evaluation, which confirms that a global embedding can miss exact attribute relationships.

Classical reranking improved top-result accuracy substantially, reaching 72.5% Precision@1, but it still did not explicitly solve garment-color binding. The Attention-GRU improved deeper ranking quality and context prediction, but its 58.5% binding accuracy showed that learned query-component attention still confused some garment-color relationships.

The Direct Semantic-ID Hybrid worked best because it combined three complementary strengths:

1. Dense embeddings provided open-vocabulary semantic recall.
2. Structured parsing and scoring enforced garment-color and context constraints.
3. Semantic IDs provided an additional compact candidate route without replacing final evidence-based reranking.

The generative Semantic-ID experiment showed that text-to-ID generation is feasible, but the generated candidate buckets were broader and less efficient than direct hybrid routing.

## 11. Limitations

The main limitations are:

- Query and relevance labels are automatically generated from the same ontology used by the reranker.
- Human-written search queries may contain ambiguity, synonyms, and fashion terms not covered by the parser.
- Environment labels are zero-shot predictions, not manually verified annotations.
- The 1,000-image subset is small and imbalanced.
- Rare categories such as ties and yellow raincoats are not represented enough for strong category-specific claims.
- The retained numbers should not be interpreted as production-scale retrieval performance.

## 12. Reproducibility

Install dependencies from the repository root:

```bash
uv sync --all-extras
```

Run tests:

```bash
uv run pytest -q
```

Run the baseline:

```bash
uv run python baseline/indexer.py ^
  --metadata solution/data/processed/fashionpedia_1000/metadata.jsonl ^
  --output baseline/index

uv run python baseline/retriever.py ^
  "a blue shirt while sitting on a park bench" ^
  --index baseline/index ^
  -k 5
```

Run the proposed solution from the `solution/` directory:

```bash
uv run python indexer.py ^
  --metadata data/processed/fashionpedia_1000/metadata.jsonl ^
  --output artifacts/indexes/fashionpedia_qwen

uv run python retriever.py ^
  "a red tie and a white shirt in a formal office" ^
  --index artifacts/indexes/fashionpedia_qwen ^
  --semantic-ids artifacts/checkpoints/rqvae_direct_qwen ^
  --weights artifacts/results/direct_semantic_binding_weight_tuning.json ^
  -k 5
```

Launch the optional interface:

```bash
cd solution
uv run streamlit run app.py
```

## 13. Conclusion

This project demonstrates that fashion retrieval benefits from combining multimodal embeddings with explicit compositional structure. Dense Qwen embeddings provide broad semantic recall, but the strongest performance comes from reranking candidates with garment-color binding, environment agreement, style evidence, and Semantic-ID routing.

The Direct Semantic-ID Hybrid is the recommended final method. It achieved the best overall accuracy, the highest binding accuracy, perfect context satisfaction on the retained evaluation, and a smaller candidate set than the generative Semantic-ID approach. The most important future work is to evaluate on human-written queries with independently judged relevance labels and to expand the dataset for rare garment-color-context combinations.

