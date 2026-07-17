# Proposed solution

The solution keeps dense multimodal search for broad recall, then reranks a small candidate set using garment–color bindings, environment, activity, style, and hierarchical Semantic IDs. This makes queries such as “a red tie and a white shirt in an office” different from “a white tie and a red shirt in an office.”

- `indexer.py` is Part A: image feature extraction and FAISS IVF-PQ storage.
- `retriever.py` is Part B: natural-language top-k search with explicit multi-attribute scoring.
- `src/fashion_retrieval/` contains the implementation behind those two entry points.
- `artifacts/results/` contains the saved evaluation output for the proposed methods.

Run commands from this directory so dataset paths in the manifest resolve correctly:

```bash
python indexer.py \
  --metadata data/processed/fashionpedia_1000/metadata.jsonl \
  --output artifacts/indexes/fashionpedia_qwen

python retriever.py \
  "a red tie and a white shirt in a formal office" \
  --index artifacts/indexes/fashionpedia_qwen \
  --semantic-ids artifacts/checkpoints/rqvae_direct_qwen \
  -k 5
```

The retained Direct Semantic-ID Hybrid result is 92.5% Precision@1 (111 of 120 queries) on the automatic held-out test set. Because the queries and judgments come from the same structured ontology used by the reranker, that number is best read as a controlled engineering comparison, not as a claim about unrestricted real-world search.
