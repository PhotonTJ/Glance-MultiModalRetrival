from pathlib import Path

import streamlit as st
import yaml

from fashion_retrieval.encoder import QwenEncoder
from fashion_retrieval.index import DenseIndex
from fashion_retrieval.retriever import HybridRetriever

st.set_page_config(page_title="Fashion Context Search", layout="wide")
st.title("Fashion & Context Retrieval")
st.caption("Dense Qwen3-VL recall with garment–color and context reranking")

index_path = st.sidebar.text_input("Index directory", "artifacts/indexes/main")
query = st.text_input("Describe the image", "A red tie and a white shirt in a formal setting.")
k = st.slider("Results", 1, 12, 6)

if st.button("Search", type="primary"):
    if not Path(index_path, "index.faiss").exists():
        st.error("Index not found. Run the index command from the solution README first.")
    else:
        with st.spinner("Loading model and searching…"):
            config = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
            index = DenseIndex.load(index_path)
            encoder = QwenEncoder(config["encoder"]["model"], config["encoder"]["dimension"])
            retriever = HybridRetriever(index, encoder, config["retrieval"]["weights"])
            results = retriever.search(query, k)
        columns = st.columns(3)
        for rank, result in enumerate(results):
            with columns[rank % 3]:
                st.image(result.record.image_path, use_container_width=True)
                st.markdown(f"**{rank + 1}. {result.record.image_id}** · {result.score:.3f}")
                st.caption(result.explanation)
