# Baseline workflow

This is the reference system. It encodes every image with Qwen3-VL, stores normalized vectors in a FAISS index, and ranks images by cosine similarity to the query. It does not use filenames or keyword lookup.

`indexer.py` is Part A. `retriever.py` is Part B. A query such as “a red tie with a white shirt in an office” is encoded as one sentence, so the baseline can represent several attributes, although it has no explicit mechanism for keeping each color attached to the correct garment.

From the repository root:

```bash
python baseline/indexer.py --metadata solution/data/processed/fashionpedia_1000/metadata.jsonl
python baseline/retriever.py "a blue shirt while sitting on a park bench" -k 5
```

The retained full-run measurements are in `results/`. The first baseline JSON compares raw cosine search with classical rerankers; the second records the Attention-GRU cross-check. These are kept here as experimental references and are not the proposed solution.
