#!/usr/bin/env python3
"""
Generate a ColBERT token-plan golden file using Hugging Face tokenizers and profile rules.
This plan can be used by pg_colbert_llama to compare tokenization, query expansion,
retention, and skiplist behavior.
"""

import argparse
import json
import string
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Error: transformers package is not installed. Run 'pip install transformers'.", file=sys.stderr)
    sys.exit(1)

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a ColBERT token-plan golden file.")
    parser.add_argument("--model-name-or-path", type=str, required=True,
                        help="Hugging Face model ID or path to local model directory")
    parser.add_argument("--texts-file", type=str, required=True,
                        help="Path to a text file containing one text per line")
    parser.add_argument("--role", type=str, choices=["query", "doc"], required=True,
                        help="The role of the texts: query or doc")
    parser.add_argument("--outfile", type=str, required=True,
                        help="Path to write the golden output JSON file")
    return parser.parse_args()


def load_sentence_transformers_config(model_name_or_path: str) -> Dict[str, Any]:
    """Load config_sentence_transformers.json if available locally or from Hugging Face Hub."""
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
        try:
            config_path = hf_hub_download(repo_id=model_name_or_path, filename="config_sentence_transformers.json")
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def main() -> None:
    args = parse_args()
    
    # 1. Load tokenizer
    print(f"Loading tokenizer for: {args.model_name_or_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    except Exception as e:
        print(f"Error loading tokenizer: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Load config and extract profile parameters
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

    # Gather special token IDs
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

    # 3. Read input texts
    texts_path = Path(args.texts_file)
    if not texts_path.exists():
        print(f"Error: Texts file does not exist: {texts_path}", file=sys.stderr)
        sys.exit(1)
        
    with open(texts_path, "r", encoding="utf-8") as f:
        texts = [line.rstrip("\r\n") for line in f]

    print(f"Generating plans for {len(texts)} texts with role={args.role}...")

    # 4. Generate plans
    plans = []
    for text in texts:
        if args.role == "query":
            prefix = query_prefix
            full_text = prefix + text
            
            # Token ids before padding
            token_ids_before = tokenizer.encode(full_text, add_special_tokens=True, truncation=True, max_length=query_length)
            
            # Swap pad token to mask token for query expansion if active
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
            
            # Restore pad tokens
            tokenizer.pad_token = orig_pad_token
            tokenizer.pad_token_id = orig_pad_token_id
            
            token_ids_after = inputs["input_ids"]
            raw_attention_mask = inputs["attention_mask"]
            
            if attend_to_expansion:
                attention_mask = [1] * len(token_ids_after)
            else:
                attention_mask = raw_attention_mask
                
            token_type_ids = inputs.get("token_type_ids")
            token_pieces = tokenizer.convert_ids_to_tokens(token_ids_after)
            
            # For query, all are retained
            retain_mask = [1] * len(token_ids_after)
            retain_reasons = []
            for i, tid in enumerate(token_ids_after):
                if raw_attention_mask[i] == 0:
                    retain_reasons.append("query_expansion")
                else:
                    retain_reasons.append("query_token")
            
            plans.append({
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
            })
            
        else: # doc
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
            
            # Document retention & skiplist behavior
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
                        
            plans.append({
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
            })

    # 5. Output wrapper JSON
    wrapper = {
        "token_plan_source": "hf_tokenizer_profile_rules",
        "model_name_or_path": args.model_name_or_path,
        "role": args.role,
        "plans": plans
    }
    
    outfile_path = Path(args.outfile)
    outfile_path.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile_path, "w", encoding="utf-8") as f:
        json.dump(wrapper, f, indent=2)
        
    print(f"Successfully wrote golden token plans to: {outfile_path.resolve()}")


if __name__ == "__main__":
    main()
