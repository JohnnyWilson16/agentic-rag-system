"""
Lightweight ONNX Embeddings Module
====================================
Provides high-performance, ultra-low-memory semantic embeddings using ONNX Runtime
and Hugging Face Tokenizers for `sentence-transformers/all-MiniLM-L6-v2`.

Replaces the 450MB+ PyTorch/torch dependency with a ~60MB ONNX Runtime execution engine,
ensuring the Agentic RAG service runs seamlessly on memory-constrained platforms like
Render Free Tier (512MB RAM cap).
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, List, Optional
import numpy as np
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class OnnxMiniLMEmbeddings(Embeddings):
    """
    LangChain-compatible Embeddings implementation using ONNX Runtime.
    Produces identical 384-dimensional normalized vectors to
    `sentence-transformers/all-MiniLM-L6-v2` with 75% less RAM than PyTorch.
    """

    def __init__(
        self,
        model_repo: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: Optional[str] = None,
        max_seq_length: int = 256,
        batch_size: int = 16,
    ):
        self.model_repo = model_repo
        self.cache_dir = cache_dir
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self._session: Any = None
        self._tokenizer: Any = None

    def _ensure_loaded(self) -> None:
        """Lazily initializes ONNX runtime session and fast tokenizer on first call."""
        if self._session is not None:
            return

        try:
            from huggingface_hub import hf_hub_download
            import onnxruntime as ort
            from tokenizers import Tokenizer

            model_path = hf_hub_download(
                repo_id=self.model_repo,
                filename="onnx/model.onnx",
                cache_dir=self.cache_dir,
            )
            tokenizer_path = hf_hub_download(
                repo_id=self.model_repo,
                filename="tokenizer.json",
                cache_dir=self.cache_dir,
            )

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self._session = ort.InferenceSession(
                model_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )

            self._tokenizer = Tokenizer.from_file(tokenizer_path)
            self._tokenizer.enable_padding(length=None)  # Dynamic padding per batch
            self._tokenizer.enable_truncation(max_length=self.max_seq_length)
            logger.info("Successfully initialized ONNX MiniLM-L6-v2 embedding session.")
        except Exception as e:
            logger.error(f"Failed to initialize ONNX embeddings: {e}")
            raise RuntimeError(
                f"Could not load ONNX model '{self.model_repo}': {str(e)}"
            ) from e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates 384-dimensional normalized vector embeddings for a list of texts."""
        self._ensure_loaded()
        if not texts:
            return []

        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            encoded_list = self._tokenizer.encode_batch(batch)

            input_ids = np.array([e.ids for e in encoded_list], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encoded_list], dtype=np.int64)
            token_type_ids = np.array([e.type_ids for e in encoded_list], dtype=np.int64)

            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }

            outputs = self._session.run(None, inputs)
            token_embeddings = outputs[0]

            # Mean pooling with attention mask weighting
            mask_expanded = np.broadcast_to(
                np.expand_dims(attention_mask, -1), token_embeddings.shape
            )
            sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
            sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
            batch_emb = sum_embeddings / sum_mask

            # L2 Normalization
            norms = np.linalg.norm(batch_emb, axis=1, keepdims=True)
            batch_emb = batch_emb / np.maximum(norms, 1e-9)

            all_embeddings.extend(batch_emb.tolist())

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Generates 384-dimensional normalized vector embedding for a single search query."""
        results = self.embed_documents([text])
        return results[0] if results else [0.0] * 384
