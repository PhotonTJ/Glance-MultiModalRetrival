import json
import numpy as np

from fashion_retrieval.encoder import HashEncoder, ProjectedEncoder
from fashion_retrieval.index import DenseIndex
from fashion_retrieval.metrics import evaluate_rankings
from fashion_retrieval.parser import garment_matches, parse_query
from fashion_retrieval.retriever import HybridRetriever, structured_scores
from fashion_retrieval.schema import Garment, ImageRecord
from fashion_retrieval.semantic_id import SemanticIDHybridRetriever, semantic_id_relation
from fashion_retrieval.generative_semantic_id import _record_queries, sid_target, sid_token
from fashion_retrieval.models import AttentionGRUComposer, ResidualQuantizer, SiameseProjection
from fashion_retrieval.training import train_rqvae
import torch


def test_parser_preserves_garment_color_binding():
    parsed = parse_query("A red tie and a white shirt in a formal setting")
    assert set(parsed.bindings) == {("tie", "red"), ("shirt", "white")}
    assert "professional" in parsed.styles


def test_binding_normalizes_plural_queries_and_compound_fashionpedia_labels():
    parsed = parse_query("red shoes and a black shirt")
    assert set(parsed.bindings) == {("shoe", "red"), ("shirt", "black")}
    assert garment_matches("shirt", "shirt_/_blouse")
    record = ImageRecord(image_id="shirt", image_path="shirt", garments=[Garment(type="shirt_/_blouse", color="black")])
    binding, _, _, _ = structured_scores(parse_query("black shirt"), record)
    assert binding == 1.0


def test_binding_reranker_rejects_swapped_colors():
    correct = ImageRecord(image_id="correct", image_path="correct", garments=[Garment(type="tie", color="red"), Garment(type="shirt", color="white")], style=["professional"])
    swapped = ImageRecord(image_id="swapped", image_path="swapped", garments=[Garment(type="tie", color="white"), Garment(type="shirt", color="red")], style=["professional"])
    vectors = np.ones((2, 8), dtype=np.float32)  # force an exact dense tie
    result = HybridRetriever(DenseIndex(vectors, [swapped, correct]), HashEncoder(8)).search("red tie and white shirt", 2)
    assert [r.record.image_id for r in result] == ["correct", "swapped"]


def test_metrics_exact_ranking():
    values = evaluate_rankings([["a", "b"]], [{"a": 2, "b": 1}])
    assert values == {"precision@1": 1.0, "recall@5": 1.0, "recall@10": 1.0, "map@10": 1.0, "ndcg@10": 1.0}


def test_faiss_ivfpq_is_persisted_and_loaded(tmp_path):
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(1000, 16)).astype(np.float32)
    records = [ImageRecord(image_id=str(i), image_path=str(i)) for i in range(len(vectors))]
    index = DenseIndex(vectors, records, {"nlist": 16, "m": 4, "nbits": 4, "nprobe": 8})
    target = tmp_path / "index"
    index.save(target)

    loaded = DenseIndex.load(target)
    ids, _ = loaded.search(vectors[123], 10)
    metadata = json.loads((target / "index_config.json").read_text())
    assert (target / "index.faiss").exists()
    assert loaded.serving_index_type == "IndexIVFPQ"
    assert metadata["metric"] == "cosine_via_normalized_inner_product"
    assert 123 in ids


def test_learned_modules_have_expected_shapes():
    projected = SiameseProjection(input_dim=16, hidden_dim=8, output_dim=4)(torch.randn(3, 16))
    assert projected.shape == (3, 4)
    assert torch.allclose(projected.norm(dim=1), torch.ones(3), atol=1e-5)
    rq = ResidualQuantizer(input_dim=4, latent_dim=2, levels=3, codebook_size=4)
    reconstruction, codes, losses = rq(projected)
    assert reconstruction.shape == projected.shape and codes.shape == (3, 3)
    assert losses["total"].ndim == 0
    assert rq.soft_quantized_prefixes(projected).shape == (3, 3, 2)

    gru = AttentionGRUComposer(16, 8, {"garment": 3, "color": 2, "binding": 4, "environment": 2})
    logits, attention = gru(torch.randn(3, 4, 16), torch.tensor([
        [True, True, False, False], [True, True, True, False], [True, True, True, True]
    ]))
    assert logits["binding"].shape == (3, 4)
    assert attention["binding"].shape == (3, 4)
    for values in attention.values():
        assert torch.allclose(values.sum(dim=1), torch.ones(3), atol=1e-5)
        assert values[0, 2:].sum() == 0


def test_projected_encoder_loads_shared_head(tmp_path):
    model = SiameseProjection(input_dim=8, hidden_dim=512, output_dim=4)
    path = tmp_path / "head.pt"
    torch.save({"state_dict": model.state_dict(), "input_dim": 8, "output_dim": 4}, path)
    vectors = ProjectedEncoder(HashEncoder(8), path).encode_texts(["red tie", "blue shirt"])
    assert vectors.shape == (2, 4)


def test_semantic_id_hybrid_uses_prefix_matches(tmp_path):
    records = [
        ImageRecord(image_id="exact", image_path="exact", garments=[Garment(type="tie", color="red")]),
        ImageRecord(image_id="prefix", image_path="prefix", garments=[Garment(type="shirt", color="white")]),
    ]
    index = DenseIndex(np.eye(2, dtype=np.float32), records)
    semantic_dir = tmp_path / "rq"
    semantic_dir.mkdir()
    rq = ResidualQuantizer(input_dim=2, latent_dim=2, levels=3, codebook_size=2)
    torch.save(
        {"state_dict": rq.state_dict(), "input_dim": 2, "latent_dim": 2, "levels": 3, "codebook_size": 2},
        semantic_dir / "rqvae.pt",
    )
    (semantic_dir / "image_to_sid.json").write_text(json.dumps({"exact": [0, 0, 0], "prefix": [0, 1, 1]}), encoding="utf-8")
    (semantic_dir / "prefix_to_images.json").write_text(json.dumps({"0/0/0": ["exact"], "0": ["exact", "prefix"]}), encoding="utf-8")
    retriever = SemanticIDHybridRetriever(index, HashEncoder(2), semantic_dir, candidate_pool=1, sid_pool=5)
    assert retriever._sid_candidates([0, 0, 0]) == [0, 1]
    assert semantic_id_relation([0, 0, 0], [0, 1, 1]) == 0.35


def test_generative_semantic_id_tokens_preserve_codebook_level():
    assert sid_token(0, 5) != sid_token(1, 5)
    assert sid_target((5, 2, 1)) == "<SID_L0_5> <SID_L1_2> <SID_L2_1>"
    record = ImageRecord(image_id="x", image_path="x", garments=[Garment(type="dress", color="gray")], environment="office")
    assert "A person wearing gray dress in a office setting." in _record_queries(record)


def test_attribute_aware_rqvae_training_saves_supervision_artifacts(tmp_path):
    records = [
        ImageRecord(image_id="a", image_path="a", garments=[Garment(type="tie", color="red")], environment="office", style=["professional"], split="train"),
        ImageRecord(image_id="b", image_path="b", garments=[Garment(type="shirt", color="blue")], environment="park", style=["casual"], split="train"),
        ImageRecord(image_id="c", image_path="c", garments=[Garment(type="shirt", color="white")], environment="office", style=["professional"], split="train"),
    ]
    output = tmp_path / "attribute_rqvae"
    history = train_rqvae(DenseIndex(np.eye(3, dtype=np.float32), records), output, epochs=1, latent_dim=2,
                           levels=3, codebook_size=2, attribute_weight=0.5)
    assert len(history["attribute"]) == 1
    assert (output / "attribute_heads.pt").exists()
    diagnostics = json.loads((output / "diagnostics.json").read_text())
    assert diagnostics["attribute_weight"] == 0.5
    assert diagnostics["attribute_vocabulary_sizes"]["binding"] == 3
