#!/usr/bin/env python3
"""
Verify parity of a converted ColBERT GGUF model and/or profile JSON sidecar
against the original Hugging Face model and PyLate reference embeddings.
"""

import argparse
import json
import string
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add parent directory and tools directory to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from colbert_profile import ColbertProfile, validate_profile
from validate_colbert_gguf_inference import decode_gguf_field, rebuild_tokenizer

try:
    from gguf import GGUFReader
except ImportError:
    GGUFReader = None

try:
    from transformers import AutoTokenizer, AutoConfig, AutoModel
    import torch
    import torch.nn.functional as F
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from pylate import models
    PYLATE_AVAILABLE = True
except ImportError:
    PYLATE_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify ColBERT GGUF/profile parity with Hugging Face/PyLate.")
    parser.add_argument("--model-name-or-path", type=str, required=True,
                        help="Hugging Face model ID or path to local model directory")
    parser.add_argument("--gguf", type=str, default=None,
                        help="Path to the converted GGUF model file")
    parser.add_argument("--profile", type=str, default=None,
                        help="Path to the profile sidecar JSON file")
    parser.add_argument("--texts-file", type=str, required=True,
                        help="Path to a text file containing one text per line")
    parser.add_argument("--role", type=str, choices=["query", "doc"], required=True,
                        help="The role of the texts: query or doc")
    parser.add_argument("--outfile", type=str, required=True,
                        help="Path to write the output parity report JSON")
    return parser.parse_args()


def load_sentence_transformers_config(model_name_or_path: str) -> Dict[str, Any]:
    """Load config_sentence_transformers.json if available locally."""
    local_path = Path(model_name_or_path)
    if local_path.exists() and local_path.is_dir():
        st_config_file = local_path / "config_sentence_transformers.json"
        if st_config_file.exists():
            try:
                with open(st_config_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    else:
        # Try importing huggingface_hub to download config if remote repo
        try:
            from huggingface_hub import hf_hub_download
            config_path = hf_hub_download(repo_id=model_name_or_path, filename="config_sentence_transformers.json")
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def main() -> None:
    args = parse_args()
    
    profile_valid = False
    token_plan_valid = False
    vector_golden_available = False
    known_limitations = [
        "This tool verifies converter and profile correctness using reference tools, "
        "not the Postgres runtime C++/GGML execution correctness."
    ]
    texts_report = []
    
    # 1. Load and validate profile
    profile_data: Optional[ColbertProfile] = None
    profile_dict: Optional[Dict[str, Any]] = None
    
    if args.profile:
        profile_path = Path(args.profile)
        if not profile_path.exists():
            print(f"Error: Specified profile file does not exist: {profile_path}", file=sys.stderr)
            sys.exit(1)
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                profile_dict = json.load(f)
                profile_data = ColbertProfile.from_dict(profile_dict)
                validate_profile(profile_data)
                profile_valid = True
        except Exception as e:
            known_limitations.append(f"Failed to load or validate sidecar profile: {e}")
            profile_valid = False
            
    elif args.gguf:
        gguf_path = Path(args.gguf)
        if gguf_path.exists() and GGUFReader is not None:
            try:
                reader = GGUFReader(gguf_path)
                profile_json_field = reader.fields.get("pg_colbert.profile_json")
                if profile_json_field is not None:
                    profile_json = decode_gguf_field(profile_json_field)
                    profile_dict = json.loads(profile_json)
                    profile_data = ColbertProfile.from_dict(profile_dict)
                    validate_profile(profile_data)
                    profile_valid = True
                else:
                    known_limitations.append("GGUF model does not contain embedded profile ('pg_colbert.profile_json').")
            except Exception as e:
                known_limitations.append(f"Failed to extract or validate GGUF embedded profile: {e}")
                profile_valid = False
        else:
            known_limitations.append("GGUF file specified but does not exist or GGUFReader not available.")
            profile_valid = False
            
    else:
        known_limitations.append("No profile sidecar or GGUF model supplied. Profile validation skipped.")
        profile_valid = False

    # 2. Check consistency with GGUF metadata
    if args.gguf and GGUFReader is not None:
        gguf_path = Path(args.gguf)
        if gguf_path.exists():
            try:
                reader = GGUFReader(gguf_path)
                schema = decode_gguf_field(reader.fields.get("pg_colbert.gguf_schema"))
                if schema != "pg_colbert_v1":
                    known_limitations.append(f"GGUF schema '{schema}' is not 'pg_colbert_v1'.")
                    profile_valid = False
                
                # Check output dimension against GGUF projection weights shape
                proj_weight_tensor = next((t for t in reader.tensors if t.name == "colbert.proj.weight"), None)
                if proj_weight_tensor is not None:
                    g_out_dim = proj_weight_tensor.data.shape[0]
                    if profile_data and profile_data.output_dim != g_out_dim:
                        known_limitations.append(
                            f"Output dimension mismatch: GGUF tensor has {g_out_dim}, profile has {profile_data.output_dim}."
                        )
                        profile_valid = False
                
                # Verify prefixes and lengths
                g_q_prefix = decode_gguf_field(reader.fields.get("colbert.query_prefix"))
                g_d_prefix = decode_gguf_field(reader.fields.get("colbert.document_prefix"))
                g_q_len = decode_gguf_field(reader.fields.get("colbert.query_length"))
                g_d_len = decode_gguf_field(reader.fields.get("colbert.document_length"))
                
                if profile_data:
                    if g_q_prefix is not None and g_q_prefix != profile_data.query.prefix:
                        known_limitations.append(f"Query prefix mismatch: GGUF={g_q_prefix}, Profile={profile_data.query.prefix}")
                        profile_valid = False
                    if g_d_prefix is not None and g_d_prefix != profile_data.document.prefix:
                        known_limitations.append(f"Doc prefix mismatch: GGUF={g_d_prefix}, Profile={profile_data.document.prefix}")
                        profile_valid = False
                    if g_q_len is not None and int(g_q_len) != profile_data.query.max_length:
                        known_limitations.append(f"Query max length mismatch: GGUF={g_q_len}, Profile={profile_data.query.max_length}")
                        profile_valid = False
                    if g_d_len is not None and int(g_d_len) != profile_data.document.max_length:
                        known_limitations.append(f"Doc max length mismatch: GGUF={g_d_len}, Profile={profile_data.document.max_length}")
                        profile_valid = False
            except Exception as e:
                known_limitations.append(f"Failed during GGUF metadata consistency verification: {e}")
                profile_valid = False

    # 3. Read texts
    texts_path = Path(args.texts_file)
    if not texts_path.exists():
        print(f"Error: Texts file does not exist: {texts_path}", file=sys.stderr)
        sys.exit(1)
    with open(texts_path, "r", encoding="utf-8") as f:
        texts = [line.rstrip("\r\n") for line in f if line.strip()]

    # 4. Generate Token Plans and run Parity checks
    if TRANSFORMERS_AVAILABLE:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
            
            # Load config parameters
            st_config = load_sentence_transformers_config(args.model_name_or_path)
            
            query_prefix = st_config.get("query_prefix") or "[Q] "
            doc_prefix = st_config.get("document_prefix") or "[D] "
            query_length = int(st_config.get("query_length") or 32)
            doc_length = int(st_config.get("document_length") or 300)
            do_query_expansion = bool(st_config.get("do_query_expansion", True))
            attend_to_expansion = bool(st_config.get("attend_to_expansion_tokens", True))
            
            skiplist_words = st_config.get("skiplist_words")
            if skiplist_words is None:
                skiplist_words = list(string.punctuation)
            else:
                skiplist_words = list(skiplist_words)
                
            # Resolve skiplist token IDs
            skiplist_token_ids = []
            for word in skiplist_words:
                ids = tokenizer.encode(word, add_special_tokens=False)
                if len(ids) == 1:
                    if ids[0] != tokenizer.unk_token_id:
                        skiplist_token_ids.append(ids[0])
            skiplist_token_ids = sorted(list(set(skiplist_token_ids)))
            
            special_ids = {
                tokenizer.cls_token_id,
                tokenizer.sep_token_id,
                tokenizer.pad_token_id,
                tokenizer.mask_token_id,
                tokenizer.unk_token_id
            }
            for marker in ["[Q]", "[D]"]:
                tid = tokenizer.convert_tokens_to_ids(marker)
                if tid != tokenizer.unk_token_id:
                    special_ids.add(tid)
            special_ids = {i for i in special_ids if i is not None}
            
            # Reconstruct GGUF tokenizer if GGUF provided
            gguf_tokenizer = None
            if args.gguf and GGUFReader is not None:
                try:
                    reader = GGUFReader(Path(args.gguf))
                    gguf_tokenizer, _ = rebuild_tokenizer(reader)
                except Exception as e:
                    known_limitations.append(f"Could not reconstruct GGUF tokenizer: {e}")
            
            token_plan_valid = True
            
            for text in texts:
                # Generate HF token plan
                if args.role == "query":
                    prefix = query_prefix
                    full_text = prefix + text
                    
                    token_ids_before = tokenizer.encode(full_text, add_special_tokens=True, truncation=True, max_length=query_length)
                    
                    # Swap pad token for expansion
                    orig_pad_token = tokenizer.pad_token
                    orig_pad_token_id = tokenizer.pad_token_id
                    if do_query_expansion and tokenizer.mask_token_id is not None:
                        tokenizer.pad_token = tokenizer.mask_token
                        tokenizer.pad_token_id = tokenizer.mask_token_id
                        
                    inputs = tokenizer(
                        full_text,
                        padding="max_length",
                        max_length=query_length,
                        truncation=True
                    )
                    
                    tokenizer.pad_token = orig_pad_token
                    tokenizer.pad_token_id = orig_pad_token_id
                    
                    token_ids_after = inputs["input_ids"]
                    raw_attention_mask = inputs["attention_mask"]
                    attention_mask = [1] * len(token_ids_after) if attend_to_expansion else raw_attention_mask
                    token_type_ids = inputs.get("token_type_ids")
                    token_pieces = tokenizer.convert_ids_to_tokens(token_ids_after)
                    
                    retain_mask = [1] * len(token_ids_after)
                    retain_reasons = []
                    for i, tid in enumerate(token_ids_after):
                        if raw_attention_mask[i] == 0:
                            retain_reasons.append("query_expansion")
                        else:
                            retain_reasons.append("query_token")
                            
                    plan = {
                        "input_text": text,
                        "role": "query",
                        "token_ids_before_padding": token_ids_before,
                        "token_ids_after_padding_truncation": token_ids_after,
                        "token_pieces": token_pieces,
                        "attention_mask": attention_mask,
                        "token_type_ids": token_type_ids,
                        "retain_mask": retain_mask,
                        "retain_reasons": retain_reasons,
                        "skiplist_token_ids": [],
                        "final_vector_count": len(token_ids_after),
                        "token_plan_source": "hf_tokenizer_profile_rules"
                    }
                    
                else:  # doc
                    prefix = doc_prefix
                    full_text = prefix + text
                    
                    inputs = tokenizer(
                        full_text,
                        padding=False,
                        max_length=doc_length,
                        truncation=True
                    )
                    
                    token_ids_before = inputs["input_ids"]
                    token_ids_after = inputs["input_ids"]
                    attention_mask = inputs["attention_mask"]
                    token_type_ids = inputs.get("token_type_ids")
                    token_pieces = tokenizer.convert_ids_to_tokens(token_ids_after)
                    
                    retain_mask = []
                    retain_reasons = []
                    for i, tid in enumerate(token_ids_after):
                        if tid in skiplist_token_ids:
                            retain_mask.append(0)
                            retain_reasons.append("skipped_skiplist")
                        elif attention_mask[i] == 0:
                            retain_mask.append(0)
                            retain_reasons.append("padded")
                        else:
                            retain_mask.append(1)
                            if tid in special_ids:
                                retain_reasons.append("special_token")
                            else:
                                retain_reasons.append("retained_content")
                                
                    plan = {
                        "input_text": text,
                        "role": "doc",
                        "token_ids_before_padding": token_ids_before,
                        "token_ids_after_padding_truncation": token_ids_after,
                        "token_pieces": token_pieces,
                        "attention_mask": attention_mask,
                        "token_type_ids": token_type_ids,
                        "retain_mask": retain_mask,
                        "retain_reasons": retain_reasons,
                        "skiplist_token_ids": skiplist_token_ids,
                        "final_vector_count": sum(retain_mask),
                        "token_plan_source": "hf_tokenizer_profile_rules"
                    }
                
                # Check GGUF tokenizer parity if available
                if gguf_tokenizer is not None:
                    # Tokenize via GGUF tokenizer
                    if args.role == "query":
                        # Adjust pad token for query expansion in GGUF tokenizer
                        orig_g_pad = gguf_tokenizer.pad_token
                        orig_g_pad_id = gguf_tokenizer.pad_token_id
                        if do_query_expansion and gguf_tokenizer.mask_token_id is not None:
                            gguf_tokenizer.pad_token = gguf_tokenizer.mask_token
                            gguf_tokenizer.pad_token_id = gguf_tokenizer.mask_token_id
                            
                        g_inputs = gguf_tokenizer(
                            full_text,
                            padding="max_length",
                            max_length=query_length,
                            truncation=True
                        )
                        
                        gguf_tokenizer.pad_token = orig_g_pad
                        gguf_tokenizer.pad_token_id = orig_g_pad_id
                    else:
                        g_inputs = gguf_tokenizer(
                            full_text,
                            padding=False,
                            max_length=doc_length,
                            truncation=True
                        )
                        
                    g_ids = g_inputs["input_ids"]
                    if g_ids != token_ids_after:
                        token_plan_valid = False
                        known_limitations.append(
                            f"Tokenizer ID mismatch for text '{text[:20]}...': GGUF={g_ids} vs HF={token_ids_after}"
                        )
                        
                texts_report.append({
                    "text": text,
                    "token_plan": plan,
                    "vector_golden": None
                })
                
        except Exception as e:
            known_limitations.append(f"Failed during tokenizer / token plan parity check: {e}")
            token_plan_valid = False
    else:
        known_limitations.append("transformers library not available; skipped token plan generation.")
        token_plan_valid = False

    # 5. Generate Vector Goldens
    if PYLATE_AVAILABLE:
        try:
            print(f"Loading PyLate reference model: {args.model_name_or_path}")
            pylate_model = models.ColBERT(model_name_or_path=args.model_name_or_path, device="cpu")
            pylate_model.eval()
            
            # Encode texts
            is_query_flag = (args.role == "query")
            embeddings = pylate_model.encode(texts, is_query=is_query_flag, batch_size=1)
            
            for i, emb in enumerate(embeddings):
                # emb is a numpy array
                texts_report[i]["vector_golden"] = emb.tolist()
                
            vector_golden_available = True
        except Exception as e:
            known_limitations.append(f"Failed to generate PyLate reference vectors: {e}")
            vector_golden_available = False
    else:
        known_limitations.append("PyLate library is not installed or failed to import; reference vector generation skipped.")
        vector_golden_available = False

    # 6. Build and write Parity Report
    report = {
        "profile_valid": profile_valid,
        "token_plan_valid": token_plan_valid,
        "vector_golden_available": vector_golden_available,
        "known_limitations": known_limitations,
        "texts": texts_report
    }
    
    outfile_path = Path(args.outfile)
    outfile_path.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print(f"\nParity Report successfully written to: {outfile_path.resolve()}")
    print(f"  profile_valid:           {profile_valid}")
    print(f"  token_plan_valid:        {token_plan_valid}")
    print(f"  vector_golden_available: {vector_golden_available}")
    
    # 7. Exit codes
    # If the user supplied a profile/GGUF but it failed validation, exit with status 1.
    # If the token plan mismatches, exit with status 1.
    if (args.profile or args.gguf) and not profile_valid:
        print("Verification FAILED: Profile or GGUF validation failed.", file=sys.stderr)
        sys.exit(1)
    if args.gguf and not token_plan_valid:
        print("Verification FAILED: GGUF tokenization plan mismatch.", file=sys.stderr)
        sys.exit(1)
        
    print("Verification SUCCESS.")
    sys.exit(0)


if __name__ == "__main__":
    main()
