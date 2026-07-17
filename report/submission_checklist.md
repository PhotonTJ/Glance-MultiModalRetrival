# Submission Checklist

## Primary Files

- `report/submission_report.md` - final written report.
- `README.md` - project overview, commands, and retained results.
- `solution/DATASET_CARD.md` - dataset scope, labels, and limitations.

## Code To Include

- `baseline/indexer.py`
- `baseline/retriever.py`
- `solution/indexer.py`
- `solution/retriever.py`
- `solution/src/fashion_retrieval/`
- `solution/tests/`
- `pyproject.toml`
- `requirements.txt`

## Results To Include

- `baseline/results/baseline1_fashionpedia_qwen.json`
- `baseline/results/baseline2_attention_gru.json`
- `solution/artifacts/results/semantic_id_hybrid_binding_fixed_qwen.json`
- `solution/artifacts/results/generative_attribute_flan_t5.json`
- `solution/artifacts/results/direct_semantic_binding_weight_tuning.json`
- `solution/artifacts/figures/`

## Usually Excluded From Submission

These are large generated artifacts and are already ignored by Git:

- `solution/data/raw/`
- `solution/artifacts/indexes/`
- `solution/artifacts/checkpoints/`
- `solution/artifacts/embeddings/`
- `baseline/index/`
- `baseline/checkpoints/`

## Suggested Final Zip Layout

```text
Multi-modalRetrival/
  README.md
  pyproject.toml
  requirements.txt
  baseline/
  solution/
  report/
```

Before submitting, run:

```bash
uv run pytest -q
```

