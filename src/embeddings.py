"""
Embeddings Factory Module.
Provides local HuggingFace embeddings (zero-cost) with device auto-detection (MPS/CUDA/CPU).
"""

import logging
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


def get_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None,
) -> Embeddings:
    """
    Initializes and returns a HuggingFace Embeddings instance.
    Auto-detects MPS (Apple Silicon), CUDA, or CPU if device is not specified.
    """
    if device is None:
        try:
            import torch

            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        except Exception:
            device = "cpu"

    logger.info(f"Loading embedding model '{model_name}' on device '{device}'...")

    encode_kwargs = {"normalize_embeddings": True}
    model_kwargs = {"device": device}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return embeddings
