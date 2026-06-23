"""
indexing.py — Build and save FAISS vector index from chunked documents
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def build_index(
    chunks: List[Dict[str, Any]],
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    faiss_store_dir: str = "faiss_store",
    csv_path: str | None = None,
) -> tuple[pd.DataFrame, faiss.Index, list]:
    """
    Encode chunks, build a cosine-similarity FAISS index, and persist artefacts.

    Returns
    -------
    df         : DataFrame of all chunks (with list columns serialised)
    index      : faiss.IndexFlatIP
    metadata   : list of dicts (same rows as df)
    """

    df = pd.DataFrame(chunks)

    # Serialise list columns for CSV
    df["brand_name"] = df["brand_name"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else x
    )
    df["policy_param"] = df["policy_param"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else x
    )

    if csv_path:
        os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"  ✓ Chunks CSV saved → {csv_path}")

    # ── Embeddings ──────────────────────────────────────────────────────────
    print(f"  Encoding {len(df)} chunks with {embedding_model_name} …")
    model = SentenceTransformer(embedding_model_name)
    texts = df["content"].tolist()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # enables cosine via inner product
    ).astype("float32")
    print(f"  ✓ Embedding shape: {embeddings.shape}")

    # ── FAISS ───────────────────────────────────────────────────────────────
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"  ✓ FAISS index: {index.ntotal} vectors")

    # ── Persist ─────────────────────────────────────────────────────────────
    os.makedirs(faiss_store_dir, exist_ok=True)
    index_path = os.path.join(faiss_store_dir, "policy_index.faiss")
    meta_path  = os.path.join(faiss_store_dir, "policy_metadata.pkl")

    faiss.write_index(index, index_path)

    metadata = df.to_dict("records")
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"  ✓ FAISS index saved  → {index_path}")
    print(f"  ✓ Metadata saved     → {meta_path}")

    return df, index, metadata


def load_index(
    faiss_store_dir: str = "faiss_store",
) -> tuple[faiss.Index, list, np.ndarray]:
    """Load a previously saved FAISS index + metadata + raw embeddings."""
    index_path = os.path.join(faiss_store_dir, "policy_index.faiss")
    meta_path  = os.path.join(faiss_store_dir, "policy_metadata.pkl")

    index = faiss.read_index(index_path)

    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    # Reconstruct embedding matrix
    embeddings = np.vstack([
        index.reconstruct(i) for i in range(index.ntotal)
    ]).astype("float32")

    print(f"  ✓ FAISS loaded: {index.ntotal} vectors from {faiss_store_dir}")
    return index, metadata, embeddings
