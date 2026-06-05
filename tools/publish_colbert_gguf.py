#!/usr/bin/env python3
"""
Publish a converted ColBERT GGUF model to HuggingFace Hub.
Downloads a source ColBERT model, runs the converter, generates a model card (README.md),
and uploads all files to a specified target HF Hub repository.
"""

import argparse
import json
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
    parser.add_argument("--outtype", type=str,
                        choices=["f32", "f16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q2_k", "q3_k_s", "q3_k_m", "q3_k_l", "q4_k_s", "q4_k_m", "q5_k_s", "q5_k_m", "q6_k"],
                        default="f16",
                        help="Conversion precision (default: f16)")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace API write token (or set HF_TOKEN environment variable)")
    parser.add_argument("--private", action="store_true", help="Create the target repository as private")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose conversion logs")
    parser.add_argument("--tolerance", type=float, default=1e-4,
                        help="Numerical tolerance for model output verification (default: 1e-4)")
    parser.add_argument("--target-runtime", type=str, choices=["pg_colbert", "llama_cpp", "both"], default="pg_colbert",
                        help="Target GGUF format layout (default: pg_colbert)")
    parser.add_argument("--texts-file", type=str, default=None,
                        help="Optional path to a text file containing one text per line for verification")
    parser.add_argument("--no-validation", action="store_true",
                        help="Skip numerical validation check before uploading")
    return parser.parse_args()


def generate_model_card(
    model_id: str,
    target_repo_id: str,
    outtype: str,
    target_runtime: str,
    profile_schema: str = "pg_colbert_profile_v1",
    token_plan_parity_passed: Optional[bool] = None,
    vector_parity_checked: bool = False,
    vector_parity_passed: Optional[bool] = None,
    cli_command: str = "",
    license_str: str = "apache-2.0"
) -> str:
    """Generate markdown README.md model card for the GGUF model."""
    model_name = model_id.split("/")[-1]
    
    # Strip username if exists in target_repo_id
    target_name = target_repo_id.split("/")[-1]

    token_plan_status = "PASSED" if token_plan_parity_passed else "FAILED" if token_plan_parity_passed is not None else "NOT CHECKED"
    vector_parity_checked_str = "YES" if vector_parity_checked else "NO"
    vector_parity_status = "PASSED" if vector_parity_passed else "FAILED" if vector_parity_passed is not None else "NOT CHECKED"
    
    vector_parity_claim = (
        "PASSED (verified numerical equivalence with reference PyLate embeddings)"
        if vector_parity_passed
        else "Not strictly verified or verification failed (do not trust for strict parity)"
    )

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
* **Profile Schema**: `{profile_schema}`
* **Target Runtime**: `{target_runtime}`

## Parity & Verification Report

* **Strict PyLate Token-Plan Parity**: `{token_plan_status}`
* **Vector Parity Checked**: `{vector_parity_checked_str}`
* **Strict Vector Parity Status**: `{vector_parity_status}`
* **Vector Parity Claim**: {vector_parity_claim}

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

* **CLI Command**: `{cli_command}`
* **Converter Version**: `1.0.0`
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
    
    import subprocess
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        local_gguf_path = tmp_path / gguf_filename
        
        # 1.a. Set up texts file for parity checks
        if args.texts_file:
            texts_file_path = Path(args.texts_file)
        else:
            texts_file_path = tmp_path / "default_validation_texts.txt"
            with open(texts_file_path, "w", encoding="utf-8") as f:
                f.write("Mars is the Red Planet.\n")
                f.write("How do you convert PyTorch to GGUF format?\n")
                f.write("PostgreSQL pg_colbert extension validation checks.\n")
                
        print(f"\nStep 1: Converting model '{args.model_id}' to {local_gguf_path}...")
        
        # Override sys.argv to trigger conversion logic
        old_argv = sys.argv
        model_is_dir = Path(args.model_id).exists() and Path(args.model_id).is_dir()
        
        sys.argv = [
            "convert_colbert_hf_to_gguf.py",
            "--outfile", str(local_gguf_path),
            "--outtype", args.outtype,
            "--target-runtime", args.target_runtime,
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

        # Identify files created
        gguf_files = []
        if args.target_runtime == "both":
            if local_gguf_path.suffix == ".gguf":
                pg_colbert_outfile = local_gguf_path.with_name(local_gguf_path.stem + ".pg_colbert.gguf")
                llama_outfile = local_gguf_path.with_name(local_gguf_path.stem + ".llama.gguf")
            else:
                pg_colbert_outfile = Path(str(local_gguf_path) + ".pg_colbert.gguf")
                llama_outfile = Path(str(local_gguf_path) + ".llama.gguf")
            gguf_files.append(pg_colbert_outfile)
            gguf_files.append(llama_outfile)
        else:
            gguf_files.append(local_gguf_path)

        # We will keep track of overall parity report results
        token_plan_parity_passed = None
        vector_parity_checked = False
        vector_parity_passed = None

        for g_file in gguf_files:
            if not g_file.exists():
                print(f"Warning: Expected GGUF file does not exist: {g_file}", file=sys.stderr)
                continue

            is_llama_cpp = g_file.name.endswith(".llama.gguf") or args.target_runtime == "llama_cpp"

            # 1.b. Test and Validate the GGUF model outputs
            if not args.no_validation and not is_llama_cpp:
                print(f"\nStep 1.b: Testing and validating GGUF model numerical inference: {g_file.name}...")
                import validate_colbert_gguf_inference
                
                old_argv = sys.argv
                sys.argv = [
                    "validate_colbert_gguf_inference.py",
                    str(g_file),
                    "--hf-model", args.model_id,
                    "--tolerance", str(args.tolerance)
                ]
                try:
                    validate_colbert_gguf_inference.main()
                    print(f"Validation: SUCCESS ({g_file.name} matches reference model outputs!)")
                except SystemExit as se:
                    if se.code != 0:
                        print(f"Error: GGUF model validation failed for {g_file.name}! Aborting publication.", file=sys.stderr)
                        sys.exit(se.code)
                except Exception as e:
                    print(f"Error during GGUF model validation for {g_file.name}: {e}", file=sys.stderr)
                    sys.exit(1)
                finally:
                    sys.argv = old_argv

            # 1.c. Generate golden token-plan JSON
            golden_plan_path = Path(str(g_file) + ".token_plan_golden.json")
            try:
                cmd_golden = [
                    sys.executable,
                    str(Path(__file__).parent / "create_colbert_profile_golden.py"),
                    "--model-name-or-path", args.model_id,
                    "--texts-file", str(texts_file_path),
                    "--role", "query",
                    "--outfile", str(golden_plan_path)
                ]
                subprocess.run(cmd_golden, check=True, capture_output=True, text=True)
            except Exception as e:
                print(f"Warning: Failed to generate golden token plans for {g_file.name}: {e}", file=sys.stderr)

            # 1.d. Generate parity report JSON
            parity_report_path = Path(str(g_file) + ".parity_report.json")
            try:
                cmd_parity = [
                    sys.executable,
                    str(Path(__file__).parent / "verify_pylate_parity.py"),
                    "--model-name-or-path", args.model_id,
                    "--texts-file", str(texts_file_path),
                    "--role", "query",
                    "--outfile", str(parity_report_path)
                ]
                if not is_llama_cpp:
                    cmd_parity.extend(["--gguf", str(g_file)])
                    
                sidecar_path = Path(str(g_file) + ".colbert_profile.json")
                if sidecar_path.exists():
                    cmd_parity.extend(["--profile", str(sidecar_path)])
                    
                res_parity = subprocess.run(cmd_parity, capture_output=True, text=True)
                
                # Load parity report
                if parity_report_path.exists():
                    with open(parity_report_path, "r", encoding="utf-8") as f:
                        report_data = json.load(f)
                    
                    t_val = report_data.get("token_plan_valid")
                    v_avail = report_data.get("vector_golden_available", False)
                    v_val = report_data.get("vector_parity_valid")
                    
                    if token_plan_parity_passed is None or not is_llama_cpp:
                        token_plan_parity_passed = t_val
                    if not vector_parity_checked or not is_llama_cpp:
                        vector_parity_checked = v_avail
                    if vector_parity_passed is None or not is_llama_cpp:
                        vector_parity_passed = v_val
            except Exception as e:
                print(f"Warning: Failed to run parity verification for {g_file.name}: {e}", file=sys.stderr)

        # 2. Write model card (README.md)
        readme_path = tmp_path / "README.md"
        print(f"\nStep 2: Generating GGUF Model Card at {readme_path}...")
        
        cli_command = f"python tools/convert_colbert_hf_to_gguf.py --model-id {args.model_id} --outfile {model_name}.{args.outtype}.gguf --outtype {args.outtype} --target-runtime {args.target_runtime}"
        card_content = generate_model_card(
            model_id=args.model_id,
            target_repo_id=args.target_repo_id,
            outtype=args.outtype,
            target_runtime=args.target_runtime,
            token_plan_parity_passed=token_plan_parity_passed,
            vector_parity_checked=vector_parity_checked,
            vector_parity_passed=vector_parity_passed,
            cli_command=cli_command
        )
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

        # 4. Upload GGUF models and all sidecars
        for g_file in gguf_files:
            if not g_file.exists():
                continue
                
            # Upload GGUF
            g_filename = g_file.name
            print(f"\nUploading GGUF file: {g_filename}...")
            try:
                api.upload_file(
                    path_or_fileobj=str(g_file),
                    path_in_repo=g_filename,
                    repo_id=args.target_repo_id,
                    repo_type="model"
                )
            except Exception as e:
                print(f"Failed to upload GGUF model: {e}", file=sys.stderr)
                sys.exit(1)

            # Upload sidecar colbert_profile.json
            profile_path = Path(str(g_file) + ".colbert_profile.json")
            if profile_path.exists():
                print(f"Uploading profile sidecar: {profile_path.name}...")
                try:
                    api.upload_file(
                        path_or_fileobj=str(profile_path),
                        path_in_repo=profile_path.name,
                        repo_id=args.target_repo_id,
                        repo_type="model"
                    )
                except Exception as e:
                    print(f"Warning: Failed to upload profile sidecar: {e}", file=sys.stderr)

            # Upload token_plan_golden.json
            golden_plan_path = Path(str(g_file) + ".token_plan_golden.json")
            if golden_plan_path.exists():
                print(f"Uploading token-plan golden JSON: {golden_plan_path.name}...")
                try:
                    api.upload_file(
                        path_or_fileobj=str(golden_plan_path),
                        path_in_repo=golden_plan_path.name,
                        repo_id=args.target_repo_id,
                        repo_type="model"
                    )
                except Exception as e:
                    print(f"Warning: Failed to upload golden token plan: {e}", file=sys.stderr)

            # Upload parity_report.json
            parity_report_path = Path(str(g_file) + ".parity_report.json")
            if parity_report_path.exists():
                print(f"Uploading parity report JSON: {parity_report_path.name}...")
                try:
                    api.upload_file(
                        path_or_fileobj=str(parity_report_path),
                        path_in_repo=parity_report_path.name,
                        repo_id=args.target_repo_id,
                        repo_type="model"
                    )
                except Exception as e:
                    print(f"Warning: Failed to upload parity report: {e}", file=sys.stderr)

        # 5. Upload README.md model card
        print("\nUploading Model Card (README.md) to Hugging Face Hub...")
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
