#!/usr/bin/env python3
"""
Generate golden (reference) embeddings for queries and documents using PyLate.
Outputs a .npz file with numeric embeddings and a .json file with metadata.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

try:
    import torch
except ImportError:
    print("Error: torch package is required for this script.", file=sys.stderr)
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PyLate golden embeddings.")
    parser.add_argument("--model-name", type=str, default="VAGOsolutions/SauerkrautLM-Multi-ModernColBERT",
                        help="HuggingFace model ID or path to local model directory")
    parser.add_argument("--outdir", type=str, default="tests/fixtures",
                        help="Directory to save the golden files")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on (e.g. 'cuda', 'cpu'). Auto-detected if not specified.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # We import pylate here to avoid import issues before dependencies are installed
    try:
        from pylate import models
    except ImportError:
        print("Error: pylate package is not installed. Install via pip.", file=sys.stderr)
        sys.exit(1)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Determine device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading PyLate model '{args.model_name}' on device '{device}'...")

    try:
        model = models.ColBERT(model_name_or_path=args.model_name, device=device)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    # Define standard evaluation items
    queries = [
        "Which planet is known as the Red Planet?",
        "Welcher Planet ist als der Rote Planet bekannt?"
    ]
    documents = [
        "Mars is the Red Planet.",
        "Mars ist als der Rote Planet bekannt."
    ]

    print("Encoding queries...")
    # PyLate ColBERT encode generates late-interaction embeddings.
    # We want to make sure we keep the token-level embeddings (not pooling).
    # In pylate, encode() returns a numpy array or list of numpy arrays representing the embeddings for each token.
    # E.g. shape [num_docs, seq_len, embed_dim] or list of [seq_len, embed_dim].
    # Let's inspect the shapes.
    q_embs = model.encode(queries, is_query=True, batch_size=1)
    
    print("Encoding documents...")
    d_embs = model.encode(documents, is_query=False, batch_size=1)

    # In pylate, the output of encode might be a list of arrays (due to variable token lengths),
    # or a single padded array. Let's handle list of arrays.
    print("\nEncoding results:")
    for i, (q, q_emb) in enumerate(zip(queries, q_embs)):
        print(f"Query {i}: '{q}' -> Shape: {q_emb.shape}")
    for i, (d, d_emb) in enumerate(zip(documents, d_embs)):
        print(f"Doc {i}: '{d}' -> Shape: {d_emb.shape}")

    # Prepare outputs
    npz_path = outdir / "pylate_golden.npz"
    json_path = outdir / "pylate_golden.json"

    # We will save the embeddings as separate keys in NPZ
    npz_data = {}
    json_metadata = {
        "model_name": args.model_name,
        "device": device,
        "queries": [],
        "documents": []
    }

    for i, (q, q_emb) in enumerate(zip(queries, q_embs)):
        key = f"query_{i}"
        npz_data[key] = q_emb
        json_metadata["queries"].append({
            "index": i,
            "text": q,
            "tensor_key": key,
            "shape": list(q_emb.shape)
        })

    for i, (d, d_emb) in enumerate(zip(documents, d_embs)):
        key = f"doc_{i}"
        npz_data[key] = d_emb
        json_metadata["documents"].append({
            "index": i,
            "text": d,
            "tensor_key": key,
            "shape": list(d_emb.shape)
        })

    # Save files
    print(f"\nSaving embeddings to {npz_path}...")
    np.savez(npz_path, **npz_data)

    print(f"Saving metadata to {json_path}...")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_metadata, f, indent=2)

    print("Success! Golden generation completed.")


if __name__ == "__main__":
    main()
