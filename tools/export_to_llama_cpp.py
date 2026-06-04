#!/usr/bin/env python3
"""
Convert a pg_colbert_v1 GGUF file to a standard llama.cpp-compliant embedding GGUF model.
This maps custom prefixed tensors (e.g. hf.embeddings.*) to standard llama.cpp names (e.g. token_embd.*)
and drops pg_colbert-specific metadata/projection tensors, allowing direct validation in llama.cpp binaries.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

try:
    from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType
except ImportError:
    print("Error: gguf package is not installed. Run 'pip install gguf'.", file=sys.stderr)
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert pg_colbert GGUF to standard llama.cpp GGUF.")
    parser.add_argument("infile", type=str, help="Path to input pg_colbert_v1 GGUF file")
    parser.add_argument("outfile", type=str, help="Path to output llama.cpp GGUF file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose log details")
    return parser.parse_args()


def decode_field(field: Any) -> Any:
    if field is None:
        return None
    types = field.types
    parts = field.parts
    data = field.data
    if not types or not data:
        return None
    main_type = types[0]
    type_name = getattr(main_type, "name", str(main_type))
    if type_name == "ARRAY" or main_type == 9:
        sub_type = types[1] if len(types) > 1 else None
        sub_type_name = getattr(sub_type, "name", str(sub_type)) if sub_type else ""
        arr_val = []
        for idx in data:
            part = parts[idx]
            if sub_type_name == "STRING" or sub_type == 8:
                arr_val.append(bytes(part).decode("utf-8"))
            else:
                arr_val.append(part.item() if hasattr(part, "item") else part[0])
        return arr_val
    elif type_name == "STRING" or main_type == 8:
        return bytes(parts[data[0]]).decode("utf-8")
    else:
        part = parts[data[0]]
        return part.item() if hasattr(part, "item") else part[0]


def get_llama_tensor_map(arch: str) -> Dict[str, str]:
    """Map pg_colbert hf-prefixed tensor suffix to standard llama.cpp names."""
    if arch == "modernbert":
        return {
            "embeddings.tok_embeddings.weight": "token_embd.weight",
            "embeddings.norm.weight": "token_embd_norm.weight",  # ModernBERT input LayerNorm
            "embeddings.norm.bias": "token_embd_norm.bias",
            
            # Layer templates (will be processed per block index)
            "attn.Wqkv.weight": "attn_qkv.weight",
            "attn.Wqkv.bias": "attn_qkv.bias",
            "attn.out_proj.weight": "attn_out.weight",
            "attn.out_proj.bias": "attn_out.bias",
            "attn.norm.weight": "attn_norm.weight",
            "attn.norm.bias": "attn_norm.bias",
            
            "mlp.wi_0.weight": "ffn_gate.weight",  # wi_0 is gate projection
            "mlp.wi_0.bias": "ffn_gate.bias",
            "mlp.wi_1.weight": "ffn_up.weight",    # wi_1 is up projection
            "mlp.wi_1.bias": "ffn_up.bias",
            "mlp.wo.weight": "ffn_down.weight",    # wo is down projection
            "mlp.wo.bias": "ffn_down.bias",
            "mlp.norm.weight": "ffn_norm.weight",
            "mlp.norm.bias": "ffn_norm.bias",
            
            "norm.weight": "output_norm.weight",   # Final output norm
            "norm.bias": "output_norm.bias"
        }
    elif arch == "bert":
        return {
            "embeddings.word_embeddings.weight": "token_embd.weight",
            "embeddings.position_embeddings.weight": "position_embd.weight",
            "embeddings.token_type_embeddings.weight": "token_types.weight",
            "embeddings.LayerNorm.weight": "token_embd_norm.weight",
            "embeddings.LayerNorm.bias": "token_embd_norm.bias",
            
            # Layer templates
            "attention.self.query.weight": "attn_q.weight",
            "attention.self.query.bias": "attn_q.bias",
            "attention.self.key.weight": "attn_k.weight",
            "attention.self.key.bias": "attn_k.bias",
            "attention.self.value.weight": "attn_v.weight",
            "attention.self.value.bias": "attn_v.bias",
            "attention.output.dense.weight": "attn_out.weight",
            "attention.output.dense.bias": "attn_out.bias",
            "attention.output.LayerNorm.weight": "attn_norm.weight",
            "attention.output.LayerNorm.bias": "attn_norm.bias",
            
            "intermediate.dense.weight": "ffn_up.weight",
            "intermediate.dense.bias": "ffn_up.bias",
            "output.dense.weight": "ffn_down.weight",
            "output.dense.bias": "ffn_down.bias",
            "output.LayerNorm.weight": "ffn_norm.weight",
            "output.LayerNorm.bias": "ffn_norm.bias"
        }
    return {}


def main() -> None:
    args = parse_args()
    infile_path = Path(args.infile)
    outfile_path = Path(args.outfile)
    
    if not infile_path.exists():
        print(f"Error: Input GGUF file not found: {infile_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading pg_colbert model: {infile_path}")
    reader = GGUFReader(infile_path)
    
    # Check schema
    schema = decode_field(reader.fields.get("pg_colbert.gguf_schema"))
    if schema != "pg_colbert_v1":
        print(f"Warning: Model schema is '{schema}', expected 'pg_colbert_v1'. Converting anyway.")

    arch = decode_field(reader.fields.get("general.architecture"))
    if not arch:
        print("Error: Missing architecture metadata key in source GGUF.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Model architecture identified as: {arch}")
    
    # Map dictionary
    t_map = get_llama_tensor_map(arch)
    if not t_map:
        print(f"Error: Unsupported architecture for standard llama.cpp conversion: {arch}", file=sys.stderr)
        sys.exit(1)

    print(f"Writing llama.cpp-compliant model: {outfile_path}")
    writer = GGUFWriter(str(outfile_path), arch=arch)
    
    # Copy metadata fields, skipping pg_colbert specific ones
    skipped_prefixes = ["pg_colbert.", "colbert."]
    for key, field in reader.fields.items():
        # Skip colbert configs and pg_colbert tags
        if any(key.startswith(p) for p in skipped_prefixes):
            if args.verbose:
                print(f"Skipping metadata key: {key}")
            continue
            
        val = decode_field(field)
        
        # Write to GGUFWriter based on type
        val_type = field.types[0]
        type_name = getattr(val_type, "name", str(val_type))
        
        if type_name == "STRING":
            writer.add_string(key, val)
        elif type_name == "UINT32":
            writer.add_uint32(key, val)
        elif type_name == "INT32":
            writer.add_int32(key, val)
        elif type_name == "FLOAT32":
            writer.add_float32(key, val)
        elif type_name == "BOOL":
            writer.add_bool(key, val)
        elif type_name == "ARRAY":
            writer.add_array(key, val)
        else:
            if args.verbose:
                print(f"Skipping metadata key {key} due to unmapped type {type_name}")

    # Copy and rename tensors
    mapped_count = 0
    skipped_count = 0
    
    for t in reader.tensors:
        if t.name.startswith("hf."):
            # Strip "hf." prefix
            hf_raw_name = t.name[3:]
            
            # Find matching standard llama.cpp name
            llama_name = None
            
            # A. Layer specific match: e.g. model.layers.0.attn.Wqkv.weight -> layers.0.attn.Wqkv.weight
            # Standard llama.cpp expects: blk.0.attn_qkv.weight
            # Let's extract layer indices dynamically
            import re
            layer_match = re.search(r"layers?\.(\d+)\.(.+)", hf_raw_name)
            if layer_match:
                layer_idx = int(layer_match.group(1))
                suffix = layer_match.group(2)
                if suffix in t_map:
                    llama_name = f"blk.{layer_idx}.{t_map[suffix]}"
            else:
                # B. Non-layer specific match (embeddings, norm): e.g. model.norm.weight -> output_norm.weight
                # Strip model wrapper prefix if present
                clean_name = hf_raw_name
                if clean_name.startswith("model."):
                    clean_name = clean_name[6:]
                elif clean_name.startswith("bert."):
                    clean_name = clean_name[5:]
                    
                if clean_name in t_map:
                    llama_name = t_map[clean_name]
                    
            if llama_name:
                if args.verbose:
                    print(f"Mapping tensor: {t.name} -> {llama_name} (Shape: {list(t.shape)})")
                # Retrieve raw data using GGUFReader
                # t.data is a numpy array
                writer.add_tensor(llama_name, t.data)
                mapped_count += 1
            else:
                print(f"Warning: Could not map tensor: {t.name}")
                skipped_count += 1
        elif t.name.startswith("colbert.proj."):
            if args.verbose:
                print(f"Dropping runtime-only projection tensor: {t.name}")
            skipped_count += 1
        else:
            print(f"Warning: Skipping unmapped custom tensor: {t.name}")
            skipped_count += 1

    # Finalize GGUF
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    print(f"\nSuccess! Standard llama.cpp model written to: {outfile_path}")
    print(f"Tensors mapped: {mapped_count}, Tensors skipped/dropped: {skipped_count}")
    print("This file can now be loaded directly in llama.cpp binaries (e.g. llama-embedding).")


if __name__ == "__main__":
    main()
