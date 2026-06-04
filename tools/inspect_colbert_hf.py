#!/usr/bin/env python3
"""
Diagnostic tool to inspect a ColBERT HuggingFace repository (local or downloaded from Hub).
Prints metadata, module structures, configurations, and safetensors tensor shapes.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: huggingface_hub package is not installed.", file=sys.stderr)
    sys.exit(1)

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("inspect_colbert_hf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a HuggingFace ColBERT repository structure and tensors.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-id", type=str, help="Hugging Face model repository ID")
    group.add_argument("--model-dir", type=str, help="Path to local Hugging Face repository directory")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory for Hugging Face downloads")
    parser.add_argument("--revision", type=str, default="main", help="Git revision/branch name")
    parser.add_argument("--no-download", action="store_true", help="Do not download if model-id is specified")
    return parser.parse_args()


def inspect_safetensors(path: Path) -> None:
    if not path.exists():
        print(f"  Safetensors file not found at: {path}")
        return
    if safe_open is None:
        print(f"  Safetensors package not available to inspect: {path}")
        return
    
    print(f"  Tensors in {path.name}:")
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in sorted(f.keys()):
                tensor_slice = f.get_slice(key)
                shape = tensor_slice.get_shape()
                # safetensors get_slice might not expose dtype directly in some versions,
                # let's try getting dtype or fall back if not available.
                dtype_str = "unknown"
                try:
                    # Load a minimal piece to get the dtype or check direct attribute if exists
                    dtype_str = str(f.get_tensor(key).dtype)
                except Exception:
                    pass
                print(f"    - {key}: shape={shape}, dtype={dtype_str}")
    except Exception as e:
        print(f"  Error reading safetensors: {e}")


def main() -> None:
    args = parse_args()

    model_path: Optional[Path] = None
    if args.model_dir:
        model_path = Path(args.model_dir)
        if not model_path.exists() or not model_path.is_dir():
            logger.error(f"Local model directory does not exist: {model_path}")
            sys.exit(1)
        print(f"Inspecting local model directory: {model_path.resolve()}")
    else:
        # Use snapshot download
        if args.no_download:
            logger.error("Cannot resolve --model-id when --no-download is specified without local path.")
            sys.exit(1)
        
        print(f"Downloading/resolving model snapshot for '{args.model_id}' (revision: {args.revision})...")
        try:
            download_dir = snapshot_download(
                repo_id=args.model_id,
                revision=args.revision,
                cache_dir=args.cache_dir,
                ignore_patterns=["*.bin", "*.h5", "*.ot", "*.msgpack"]  # we prefer safetensors and configs
            )
            model_path = Path(download_dir)
            print(f"Resolved model to snapshot cache: {model_path.resolve()}")
        except Exception as e:
            logger.error(f"Failed to download/snapshot model: {e}")
            sys.exit(1)

    assert model_path is not None

    # 1. Modules config
    modules_json_path = model_path / "modules.json"
    if modules_json_path.exists():
        print("\n--- Modules Config (modules.json) ---")
        try:
            with open(modules_json_path, "r", encoding="utf-8") as f:
                modules = json.load(f)
                print(json.dumps(modules, indent=2))
        except Exception as e:
            print(f"Error reading modules.json: {e}")
    else:
        print(f"\nWARNING: modules.json not found in {model_path}")

    # 2. Main Backbone Config
    config_json_path = model_path / "config.json"
    if config_json_path.exists():
        print("\n--- Backbone Config (config.json) ---")
        try:
            with open(config_json_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                # print some vital architecture info
                vital_keys = [
                    "model_type", "hidden_size", "intermediate_size", 
                    "num_hidden_layers", "num_attention_heads", 
                    "max_position_embeddings", "layer_norm_eps"
                ]
                print("Vital stats:")
                for k in vital_keys:
                    if k in cfg:
                        print(f"  {k}: {cfg[k]}")
                print("\nFull config subset:")
                # print first 30 keys to avoid cluttering
                subset = {k: v for i, (k, v) in enumerate(cfg.items()) if i < 40}
                print(json.dumps(subset, indent=2))
                if len(cfg) > 40:
                    print(f"  ... and {len(cfg) - 40} more keys.")
        except Exception as e:
            print(f"Error reading config.json: {e}")
    else:
        print(f"\nWARNING: config.json not found in {model_path}")

    # 3. Dense Projection Config
    dense_dir = model_path / "1_Dense"
    dense_cfg_path = dense_dir / "config.json"
    if dense_cfg_path.exists():
        print("\n--- Dense Projection Config (1_Dense/config.json) ---")
        try:
            with open(dense_cfg_path, "r", encoding="utf-8") as f:
                dense_cfg = json.load(f)
                print(json.dumps(dense_cfg, indent=2))
        except Exception as e:
            print(f"Error reading 1_Dense/config.json: {e}")
    else:
        print(f"\nWARNING: 1_Dense/config.json not found in {model_path}")

    # 4. SentenceTransformers configs
    st_cfg_path = model_path / "config_sentence_transformers.json"
    if st_cfg_path.exists():
        print("\n--- SentenceTransformers Config ---")
        try:
            with open(st_cfg_path, "r", encoding="utf-8") as f:
                st_cfg = json.load(f)
                print(json.dumps(st_cfg, indent=2))
        except Exception as e:
            print(f"Error reading config_sentence_transformers.json: {e}")

    # 5. Tokenizer configs
    tok_cfg_path = model_path / "tokenizer_config.json"
    if tok_cfg_path.exists():
        print("\n--- Tokenizer Config Summary ---")
        try:
            with open(tok_cfg_path, "r", encoding="utf-8") as f:
                tok_cfg = json.load(f)
                # print some interesting keys
                for k in ["tokenizer_class", "model_max_length", "clean_up_tokenization_spaces", "query_prefix", "doc_prefix", "document_prefix"]:
                    if k in tok_cfg:
                        print(f"  {k}: {tok_cfg[k]}")
        except Exception as e:
            print(f"Error reading tokenizer_config.json: {e}")

    # 6. Tensor Shapes (Backbone)
    print("\n--- Safetensors Inspection ---")
    # check model.safetensors or model-XXXXX-of-XXXXX.safetensors
    safetensors_files = list(model_path.glob("*.safetensors"))
    if not safetensors_files:
        print("  No safetensors found in root.")
    else:
        for sf in safetensors_files:
            inspect_safetensors(sf)

    # 7. Tensor Shapes (Dense)
    if dense_dir.exists():
        dense_safetensors = list(dense_dir.glob("*.safetensors"))
        if not dense_safetensors:
            print("  No safetensors found in 1_Dense/.")
        else:
            for sf in dense_safetensors:
                inspect_safetensors(sf)


if __name__ == "__main__":
    main()
