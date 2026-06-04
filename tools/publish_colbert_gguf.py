#!/usr/bin/env python3
"""
Publish a converted ColBERT GGUF model to HuggingFace Hub.
Downloads a source ColBERT model, runs the converter, generates a model card (README.md),
and uploads all files to a specified target HF Hub repository.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import HfApi, login
except ImportError:
    print("Error: huggingface_hub package is not installed. Run 'pip install huggingface_hub'.", file=sys.stderr)
    sys.exit(1)

# Import convert main or function to run conversion programmatically
sys.path.append(str(Path(__file__).parent.resolve()))
import convert_colbert_hf_to_gguf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert and publish a ColBERT model to Hugging Face Hub as GGUF.")
    parser.add_argument("--model-id", type=str, required=True,
                        help="Source Hugging Face model repository ID (e.g. VAGOsolutions/SauerkrautLM-Multi-ModernColBERT)")
    parser.add_argument("--target-repo-id", type=str, required=True,
                        help="Target Hugging Face model repository ID (e.g. username/SauerkrautLM-Multi-ModernColBERT-GGUF)")
    parser.add_argument("--outtype", type=str, choices=["f32", "f16"], default="f16",
                        help="Conversion precision: f16 or f32 (default: f16)")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace API write token (or set HF_TOKEN environment variable)")
    parser.add_argument("--private", action="store_true", help="Create the target repository as private")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose conversion logs")
    parser.add_argument("--tolerance", type=float, default=1e-4,
                        help="Numerical tolerance for model output verification (default: 1e-4)")
    parser.add_argument("--no-validation", action="store_true",
                        help="Skip numerical validation check before uploading")
    return parser.parse_args()


def generate_model_card(model_id: str, target_repo_id: str, outtype: str, license_str: str = "apache-2.0") -> str:
    """Generate markdown README.md model card for the GGUF model."""
    model_name = model_id.split("/")[-1]
    
    # Strip username if exists in target_repo_id
    target_name = target_repo_id.split("/")[-1]

    card = f"""---
tags:
- colbert
- pg_colbert
- gguf
- sentence-transformers
- pylate
- text-embeddings
- modernbert
library_name: sentence-transformers
license: {license_str}
---

# {target_name} (ColBERT GGUF)

This repository contains GGUF format weights for the ColBERT retrieval model [{model_id}](https://huggingface.co/{model_id}), converted specifically for our GGML-based ColBERT C++ runtime.

This GGUF file is **NOT** a standard llama.cpp decoder model. It uses the `pg_colbert_v1` layout containing backbone transformer weights, tokenizer metadata, ColBERT dense projection layers, and similarity metrics.

## Model Summary

* **Source Model**: [{model_id}](https://huggingface.co/{model_id})
* **Format**: GGUF (`pg_colbert_v1` schema)
* **Precision**: `{outtype.upper()}`
* **Encoder Backbone**: ModernBERT / BERT
* **ColBERT Projection Dimension**: 128

## Usage with ColBERT GGML C++ Runtime

This GGUF model is designed to be loaded directly by our custom C++ ColBERT runtime (which uses GGML for the ModernBERT/BERT backbone forward pass and executes the ColBERT late-interaction dense projection).

### Loading the Model in C++

```cpp
// Initialize the ColBERT GGML model context
colbert_model model = colbert_model_load("{model_name}.{outtype}.gguf");

// Tokenize and encode queries into late-interaction token embeddings
std::vector<float> query_embeddings = colbert_encode_query(model, "Which planet is known as the Red Planet?");

// Tokenize and encode documents into late-interaction token embeddings
std::vector<float> doc_embeddings = colbert_encode_doc(model, "Mars is the Red Planet.");
```

## GGUF Conversion Info

Generated using the `convert_colbert_hf_to_gguf.py` utility.

* Command: `python tools/convert_colbert_hf_to_gguf.py --model-id {model_id} --outfile {model_name}.{outtype}.gguf --outtype {outtype}`
* Converter Version: `1.0.0`
"""
    return card


def main() -> None:
    args = parse_args()
    
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("Warning: No Hugging Face token specified. You must be logged in via 'huggingface-cli login' or supply a token.", file=sys.stderr)

    api = HfApi(token=token)
    
    # Verify authentication if token provided
    if token:
        try:
            user_info = api.whoami()
            print(f"Authenticated as Hugging Face user: {user_info['name']}")
        except Exception as e:
            print(f"Authentication failed: {e}", file=sys.stderr)
            sys.exit(1)

    # 1. Download and convert in a temporary workflow
    model_name = args.model_id.replace("/", "_").replace("-", "_")
    gguf_filename = f"{model_name}.{args.outtype}.gguf"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        local_gguf_path = tmp_path / gguf_filename
        
        print(f"\nStep 1: Converting model '{args.model_id}' to {local_gguf_path}...")
        
        # Override sys.argv to trigger conversion logic
        import sys
        old_argv = sys.argv
        
        # Check if the model ID is a local directory (common in tests/local runs)
        model_is_dir = Path(args.model_id).exists() and Path(args.model_id).is_dir()
        
        sys.argv = [
            "convert_colbert_hf_to_gguf.py",
            "--outfile", str(local_gguf_path),
            "--outtype", args.outtype,
        ]
        if model_is_dir:
            sys.argv.extend(["--model-dir", args.model_id])
        else:
            sys.argv.extend(["--model-id", args.model_id])
            
        if args.verbose:
            sys.argv.append("--verbose")
            
        try:
            convert_colbert_hf_to_gguf.main()
        except Exception as e:
            print(f"Error during GGUF conversion: {e}", file=sys.stderr)
            sys.exit(1)
        finally:
            sys.argv = old_argv

        # 1.b. Test and Validate the model outputs
        if not args.no_validation:
            print(f"\nStep 1.b: Testing and validating GGUF model numerical inference...")
            import validate_colbert_gguf_inference
            
            old_argv = sys.argv
            sys.argv = [
                "validate_colbert_gguf_inference.py",
                str(local_gguf_path),
                "--hf-model", args.model_id,
                "--tolerance", str(args.tolerance)
            ]
            try:
                validate_colbert_gguf_inference.main()
                print("Validation: SUCCESS (GGUF model output matches reference model outputs!)")
            except SystemExit as se:
                if se.code != 0:
                    print("Error: GGUF model validation failed! Aborting publication.", file=sys.stderr)
                    sys.exit(se.code)
            except Exception as e:
                print(f"Error during GGUF model validation: {e}", file=sys.stderr)
                sys.exit(1)
            finally:
                sys.argv = old_argv

        # 2. Write model card (README.md)
        readme_path = tmp_path / "README.md"
        print(f"\nStep 2: Generating GGUF Model Card at {readme_path}...")
        card_content = generate_model_card(args.model_id, args.target_repo_id, args.outtype)
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(card_content)

        # 3. Create repo on Hub if it doesn't exist
        print(f"\nStep 3: Creating target repository '{args.target_repo_id}' on HuggingFace Hub (if not exists)...")
        try:
            api.create_repo(
                repo_id=args.target_repo_id,
                repo_type="model",
                private=args.private,
                exist_ok=True
            )
        except Exception as e:
            print(f"Failed to create repository on Hub: {e}", file=sys.stderr)
            sys.exit(1)

        # 4. Upload GGUF model
        print(f"\nStep 4: Uploading GGUF file to Hugging Face Hub: {gguf_filename}...")
        try:
            api.upload_file(
                path_or_fileobj=str(local_gguf_path),
                path_in_repo=gguf_filename,
                repo_id=args.target_repo_id,
                repo_type="model"
            )
            print("Successfully uploaded GGUF model!")
        except Exception as e:
            print(f"Failed to upload GGUF model to Hub: {e}", file=sys.stderr)
            sys.exit(1)

        # 5. Upload README.md model card
        print("Uploading Model Card (README.md) to Hugging Face Hub...")
        try:
            api.upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=args.target_repo_id,
                repo_type="model"
            )
            print("Successfully uploaded Model Card README.md!")
        except Exception as e:
            print(f"Failed to upload README.md to Hub: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"\nPublication process finished! View your repository here: https://huggingface.co/{args.target_repo_id}")


if __name__ == "__main__":
    main()
