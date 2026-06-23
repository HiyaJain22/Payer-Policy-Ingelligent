#!/usr/bin/env python3
"""
run_pipeline.py — Main driver for the PA Policy Extraction Pipeline
────────────────────────────────────────────────────────────────────
Usage:
    python run_pipeline.py

Steps:
    1. Reads credentials and paths from .env
    2. Extracts + chunks all PDFs in INPUT_PDF_FOLDER
    3. Builds (or reloads) FAISS index
    4. For each row in INPUT_CSV, retrieves relevant chunks and calls the LLM
    5. Writes results to OUTPUT_CSV  (default: output/result.csv)
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from pathlib import Path

# ── Load .env before importing project modules ──────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed; env vars must be set externally
    pass

import pandas as pd

from chunking import PolicyChunkingPipeline, load_all_pdfs
from indexing  import build_index, load_index
from retrieval import HybridRetriever
from extraction import (
    GroqExtractor,
    build_prompt,
    compute_access_score,
    derive_reauthorization_required,
)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration (from .env or defaults)
# ──────────────────────────────────────────────────────────────────────────────

def cfg(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


GROQ_API_KEY       = cfg("GROQ_API_KEY")
GROQ_MODEL         = cfg("GROQ_MODEL",         "llama-3.3-70b-versatile")
INPUT_PDF_FOLDER   = cfg("INPUT_PDF_FOLDER",   "input_pdfs")
INPUT_CSV          = cfg("INPUT_CSV",          "input.csv")
OUTPUT_CSV         = cfg("OUTPUT_CSV",         "output/result.csv")
FAISS_STORE        = cfg("FAISS_STORE",        "faiss_store")
EMBEDDING_MODEL    = cfg("EMBEDDING_MODEL",    "sentence-transformers/all-MiniLM-L6-v2")

MIN_CHUNK_SIZE     = int(cfg("MIN_CHUNK_SIZE",            "250"))
MAX_CHUNK_SIZE     = int(cfg("MAX_CHUNK_SIZE",            "800"))
OVERLAP_TOKENS     = int(cfg("OVERLAP_TOKENS",            "150"))
MERGE_SIMILARITY   = float(cfg("MERGE_SIMILARITY_THRESHOLD", "0.80"))

DENSE_TOP_K        = int(cfg("DENSE_TOP_K",   "30"))
BM25_TOP_K         = int(cfg("BM25_TOP_K",    "30"))
RRF_K              = int(cfg("RRF_K",         "60"))
MMR_TOP_K          = int(cfg("MMR_TOP_K",     "3"))
MMR_LAMBDA         = float(cfg("MMR_LAMBDA",  "0.7"))
GLOBAL_TOP_CHUNKS  = int(cfg("GLOBAL_TOP_CHUNKS", "5"))
SLEEP_BETWEEN_ROWS = float(cfg("SLEEP_BETWEEN_ROWS", "2"))

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def validate_env():
    errors = []
    if not GROQ_API_KEY:
        errors.append("GROQ_API_KEY is not set in .env")
    if not Path(INPUT_PDF_FOLDER).is_dir():
        errors.append(f"INPUT_PDF_FOLDER '{INPUT_PDF_FOLDER}' does not exist")
    if not Path(INPUT_CSV).is_file():
        errors.append(f"INPUT_CSV '{INPUT_CSV}' does not exist")
    if errors:
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)


def index_is_stale(pdf_folder: str, faiss_store: str) -> bool:
    """Return True if any PDF is newer than the saved index."""
    index_path = Path(faiss_store) / "policy_index.faiss"
    meta_path  = Path(faiss_store) / "policy_metadata.pkl"
    if not index_path.exists() or not meta_path.exists():
        return True
    index_mtime = min(index_path.stat().st_mtime, meta_path.stat().st_mtime)
    for f in Path(pdf_folder).glob("*.pdf"):
        if f.stat().st_mtime > index_mtime:
            return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 — Build / reload index
# ──────────────────────────────────────────────────────────────────────────────

def phase_index(force_rebuild: bool = False):
    stale = force_rebuild or index_is_stale(INPUT_PDF_FOLDER, FAISS_STORE)

    if not stale:
        print("\n[Phase 1] Index is up-to-date — loading from disk …")
        index, metadata, embeddings = load_index(FAISS_STORE)
        return index, metadata, embeddings

    print("\n[Phase 1] Building index from PDFs …")
    pdf_docs = load_all_pdfs(INPUT_PDF_FOLDER)
    if not pdf_docs:
        print("  ✗ No PDFs found in", INPUT_PDF_FOLDER)
        sys.exit(1)

    pipeline = PolicyChunkingPipeline(
        min_chunk_size=MIN_CHUNK_SIZE,
        max_chunk_size=MAX_CHUNK_SIZE,
        merge_similarity_threshold=MERGE_SIMILARITY,
        overlap_tokens=OVERLAP_TOKENS,
        embedding_model_name=EMBEDDING_MODEL,
    )

    all_chunks = []
    for i, doc in enumerate(pdf_docs):
        print(f"  Chunking [{i+1}/{len(pdf_docs)}]: {doc['pdf_name']}")
        chunks = pipeline.run(
            markdown_text=doc["text"],
            doc_id=f"doc_{i}",
            pdf_name=doc["pdf_name"],
        )
        all_chunks.extend(chunks)

    print(f"\n  Total chunks: {len(all_chunks)}")

    chunks_csv = str(Path(FAISS_STORE) / "all_policy_chunks.csv")
    _, index, metadata = build_index(
        chunks=all_chunks,
        embedding_model_name=EMBEDDING_MODEL,
        faiss_store_dir=FAISS_STORE,
        csv_path=chunks_csv,
    )

    # Reload with embeddings matrix
    index, metadata, embeddings = load_index(FAISS_STORE)
    return index, metadata, embeddings


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 — Extract PA parameters for each (filename, brand) row
# ──────────────────────────────────────────────────────────────────────────────

def phase_extract(metadata, embeddings):
    print(f"\n[Phase 2] Extracting PA parameters …")
    print(f"  Input CSV  : {INPUT_CSV}")
    print(f"  Output CSV : {OUTPUT_CSV}")

    df_input = pd.read_csv(INPUT_CSV)
    required_cols = {"Filename", "Brand"}
    if not required_cols.issubset(df_input.columns):
        print(f"  ✗ INPUT_CSV must have columns: {required_cols}")
        sys.exit(1)

    retriever = HybridRetriever(
        metadata=metadata,
        embeddings=embeddings,
        embedding_model_name=EMBEDDING_MODEL,
        dense_top_k=DENSE_TOP_K,
        bm25_top_k=BM25_TOP_K,
        rrf_k=RRF_K,
        mmr_top_k=MMR_TOP_K,
        mmr_lambda=MMR_LAMBDA,
        global_top_chunks=GLOBAL_TOP_CHUNKS,
    )

    extractor = GroqExtractor(api_key=GROQ_API_KEY, model=GROQ_MODEL)

    results = []
    total = len(df_input)

    for idx, row in df_input.iterrows():
        filename = str(row["Filename"]).strip()
        brand    = str(row["Brand"]).strip()

        print(f"\n  [{idx+1}/{total}] {filename} | {brand}")

        # ── Retrieval ──────────────────────────────────────────────────────
        try:
            chunks, param_to_chunk_ids = retriever.retrieve(filename, brand)
        except Exception as e:
            print(f"    ✗ Retrieval error: {e}")
            chunks, param_to_chunk_ids = [], {}

        # ── LLM extraction ─────────────────────────────────────────────────
        parsed, raw_text, status = {}, "", "skipped"
        if chunks:
            prompt = build_prompt(filename, brand, chunks, param_to_chunk_ids)
            parsed, raw_text, status = extractor.extract(prompt)
            parsed = parsed or {}
            print(f"    {'✓' if status == 'success' else '⚠'} {status}")
        else:
            print("    ⚠ No chunks retrieved — skipping LLM call")

        # ── Post-processing ────────────────────────────────────────────────
        reauth_required = derive_reauthorization_required(parsed)
        access_score    = compute_access_score(parsed) if parsed else ""

        chunks_used_content = "\n\n---\n\n".join(
            f"[{c.get('chunk_id', '')}]\n{c.get('content', '')}" for c in chunks
        )
        chunk_ids_used = ", ".join(str(c.get("chunk_id", "")) for c in chunks)

        results.append({
            "Filename": filename,
            "Brand":    brand,
            "Age":
                parsed.get("Age", ""),
            "Step Therapy Requirements Documented in Policy":
                parsed.get("Step Therapy Requirements Documented in Policy", ""),
            "Number of Steps through Brands":
                parsed.get("Number of Steps through Brands", ""),
            "Number of Steps through Generic":
                parsed.get("Number of Steps through Generic", ""),
            "Step through-Phototherapy":
                parsed.get("Step through-Phototherapy", ""),
            "TB Test required":
                parsed.get("TB Test required", ""),
            "Quantity Limits":
                parsed.get("Quantity Limits", ""),
            "Specialist Types":
                parsed.get("Specialist Types", ""),
            "Initial Authorization Duration(in-months)":
                parsed.get("Initial Authorization Duration(in-months)", ""),
            "Reauthorization Duration(in-months)":
                parsed.get("Reauthorization Duration(in-months)", ""),
            "Reauthorization Required":
                reauth_required,
            "Reauthorization Requirements Documented in Policy":
                parsed.get("Reauthorization Requirements Documented in Policy", ""),
            "Access Score":
                access_score,
            "Chunk IDs Used":
                chunk_ids_used,
            "Chunks Used (content)":
                chunks_used_content,
            "Raw Response":
                raw_text,
            "Status":
                status,
        })

        # ── Checkpoint ────────────────────────────────────────────────────
        os.makedirs(os.path.dirname(OUTPUT_CSV) if os.path.dirname(OUTPUT_CSV) else ".", exist_ok=True)
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

        time.sleep(SLEEP_BETWEEN_ROWS)

    # Final save
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"\n  ✓ Done — results saved to {OUTPUT_CSV}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PA Policy Extraction Pipeline")
    parser.add_argument(
        "--rebuild-index", action="store_true",
        help="Force rebuild of the FAISS index even if it appears up-to-date",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  PA Policy Extraction Pipeline")
    print("=" * 60)

    validate_env()

    # Phase 1: index
    _, metadata, embeddings = phase_index(force_rebuild=args.rebuild_index)

    # Phase 2: extract
    phase_extract(metadata, embeddings)

    print("\n✅  Pipeline complete.")


if __name__ == "__main__":
    main()
