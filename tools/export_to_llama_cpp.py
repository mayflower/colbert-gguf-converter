#!/usr/bin/env python3
"""
Convert a pg_colbert_v1 GGUF file to a standard llama.cpp-compliant embedding GGUF model.
This maps custom prefixed tensors (e.g. hf.embeddings.*) to standard llama.cpp names (e.g. token_embd.*)
and drops pg_colbert-specific metadata/projection tensors, allowing direct validation in llama.cpp binaries.
"""

import argparse
import struct
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

sys.path.append(str(Path(__file__).parent.resolve()))
from colbert_profile import (
    get_llama_tensor_map,
    get_llama_kv_canonical_map,
    build_ggml_bert_tokenizer,
)

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
    parser.add_argument("--projection-sidecar", dest="projection_sidecar", action="store_true", default=True,
                        help="Write the ColBERT projection sidecar (default: true)")
    parser.add_argument("--no-projection-sidecar", dest="projection_sidecar", action="store_false",
                        help="Do not write the ColBERT projection sidecar")
    parser.add_argument("--dense-in-gguf", dest="dense_in_gguf", action="store_true", default=True,
                        help="Embed the projection as dense_2.* tensors in the GGUF (default: true)")
    parser.add_argument("--no-dense-in-gguf", dest="dense_in_gguf", action="store_false",
                        help="Backbone-only GGUF that stock llama.cpp can load; projection only via sidecar")
    return parser.parse_args()


# ColBERT projection sidecar format (little-endian), consumed by the ollama
# serving layer. The backbone GGUF drops colbert.proj so it loads in stock
# llama.cpp; the projection travels alongside it here.
#   magic:        b"OLPROJ01" (8 bytes)
#   out_features: uint32
#   in_features:  uint32
#   has_bias:     uint32 (0/1)
#   weight:       out_features * in_features float32, row-major [out][in]
#                 (projected[j] = sum_k weight[j*in + k] * hidden[k])
#   bias:         out_features float32 (present iff has_bias == 1)
PROJ_SIDECAR_MAGIC = b"OLPROJ01"


def write_projection_sidecar(path: Path, weight: "np.ndarray", bias, out_features: int, in_features: int) -> None:
    w = np.ascontiguousarray(weight.reshape(out_features, in_features), dtype=np.float32)
    out = bytearray(PROJ_SIDECAR_MAGIC)
    out += struct.pack("<III", out_features, in_features, 1 if bias is not None else 0)
    out += w.tobytes()
    if bias is not None:
        out += np.ascontiguousarray(bias.reshape(out_features), dtype=np.float32).tobytes()
    path.write_bytes(out)


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
    
    # Copy metadata fields, skipping pg_colbert specific, internal GGUF headers, and duplicate keys.
    # The raw HF tokenizer blobs are replaced below by a llama.cpp ggml tokenizer.
    skipped_prefixes = ["pg_colbert.", "colbert.", "GGUF."]
    skipped_keys = [
        "general.architecture",
        "tokenizer.huggingface.json",
        "tokenizer.config.json",
        "tokenizer.special_tokens_map.json",
    ]
    # Rename the converter's HF-style hyperparameter keys to the canonical
    # llama.cpp keys (e.g. bert.hidden_size -> bert.embedding_length) so the
    # exported model loads in upstream llama.cpp / ollama.
    kv_canonical = get_llama_kv_canonical_map(arch)
    for key, field in reader.fields.items():
        # Keep pg_colbert.profile_json so it can be loaded directly from the GGUF by pg_colbert_llama
        if key == "pg_colbert.profile_json":
            pass
        # Skip colbert configs, pg_colbert tags, internal header fields, and duplicates
        elif key in skipped_keys or any(key.startswith(p) for p in skipped_prefixes):
            if args.verbose:
                print(f"Skipping metadata key: {key}")
            continue

        out_key = kv_canonical.get(key, key)
        if args.verbose and out_key != key:
            print(f"Canonicalizing metadata key: {key} -> {out_key}")

        val = decode_field(field)

        # Write to GGUFWriter based on type
        val_type = field.types[0]
        type_name = getattr(val_type, "name", str(val_type))

        if type_name == "STRING":
            writer.add_string(out_key, val)
        elif type_name == "UINT32":
            writer.add_uint32(out_key, val)
        elif type_name == "INT32":
            writer.add_int32(out_key, val)
        elif type_name == "FLOAT32":
            writer.add_float32(out_key, val)
        elif type_name == "BOOL":
            writer.add_bool(out_key, val)
        elif type_name == "ARRAY":
            writer.add_array(out_key, val)
        else:
            if args.verbose:
                print(f"Skipping metadata key {key} due to unmapped type {type_name}")

    # llama.cpp / ollama derive the embedding capability and the per-token
    # (multivector) output mode from {arch}.pooling_type. ColBERT serving needs
    # pooling "none" (0): one embedding row per token, pooled by the scorer.
    pooling_key = f"{arch}.pooling_type"
    if reader.fields.get(pooling_key) is None:
        writer.add_uint32(pooling_key, 0)
        print(f"Adding {pooling_key} = 0 (none) for per-token embeddings")

    # Build a llama.cpp ggml tokenizer from the embedded HF tokenizer.json. The
    # pg_colbert GGUF stores the raw tokenizer (tokenizer.huggingface.json) for its
    # own runtime; llama.cpp / ollama instead need tokenizer.ggml.* arrays.
    hf_tok = decode_field(reader.fields.get("tokenizer.huggingface.json"))
    if hf_tok:
        tokens, token_types = build_ggml_bert_tokenizer(hf_tok)
        writer.add_string("tokenizer.ggml.model", "bert")
        writer.add_array("tokenizer.ggml.tokens", tokens)
        writer.add_array("tokenizer.ggml.token_type", token_types)

        def _special_id(key: str, default: int) -> int:
            v = decode_field(reader.fields.get(key))
            return int(v) if v is not None else default

        # [CLS]=bos, [SEP]=separator, [PAD]=padding, [UNK]=100, [MASK]=103 (standard BERT).
        writer.add_uint32("tokenizer.ggml.bos_token_id", _special_id("colbert.cls_token_id", 101))
        writer.add_uint32("tokenizer.ggml.seperator_token_id", _special_id("colbert.sep_token_id", 102))
        writer.add_uint32("tokenizer.ggml.padding_token_id", _special_id("colbert.pad_token_id", 0))
        writer.add_uint32("tokenizer.ggml.unknown_token_id", 100)
        writer.add_uint32("tokenizer.ggml.mask_token_id", 103)
        writer.add_bool("tokenizer.ggml.add_bos_token", True)
        writer.add_bool("tokenizer.ggml.add_eos_token", False)
        writer.add_bool("tokenizer.ggml.add_sep_token", True)
        if args.verbose:
            print(f"Built ggml tokenizer with {len(tokens)} tokens")
    else:
        print("Warning: input GGUF has no tokenizer.huggingface.json; "
              "output may lack a llama.cpp-loadable tokenizer.", file=sys.stderr)

    # Copy and rename tensors
    mapped_count = 0
    skipped_count = 0
    proj_weight = None
    proj_bias = None

    for t in reader.tensors:
        if t.name.startswith("hf."):
            # Strip "hf." prefix
            hf_raw_name = t.name[3:]

            # Upcast to F32. llama.cpp's BERT graph evaluates LayerNorm in F32 and
            # its CPU binary ops reject mixed F32/F16 operands ("binary_op: unsupported
            # types: dst: f32, src0: f32, src1: f16"), so an all-F16 export aborts during
            # warmup. F32 also matches ggml's convention of keeping norm/embedding
            # weights in full precision. Quantize separately if a smaller file is needed.
            tdata = np.asarray(t.data, dtype=np.float32)
            
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
                
                # Handle concatenated Wi gate/up projections
                if suffix == "mlp.Wi.weight":
                    mid = tdata.shape[-1] // 2
                    gate_data = tdata[..., :mid]
                    up_data = tdata[..., mid:]
                    writer.add_tensor(f"blk.{layer_idx}.ffn_gate.weight", gate_data)
                    writer.add_tensor(f"blk.{layer_idx}.ffn_up.weight", up_data)
                    mapped_count += 2
                    continue
                elif suffix == "mlp.Wi.bias":
                    mid = tdata.shape[0] // 2
                    gate_data = tdata[:mid]
                    up_data = tdata[mid:]
                    writer.add_tensor(f"blk.{layer_idx}.ffn_gate.bias", gate_data)
                    writer.add_tensor(f"blk.{layer_idx}.ffn_up.bias", up_data)
                    mapped_count += 2
                    continue
                
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
                writer.add_tensor(llama_name, tdata)
                mapped_count += 1
            else:
                print(f"Warning: Could not map tensor: {t.name}")
                skipped_count += 1
        elif t.name.startswith("colbert.proj."):
            # The ColBERT projection is not part of the stock llama.cpp BERT
            # graph under this name. Capture it here; after the loop it is
            # re-emitted as the sentence-transformers dense module
            # (dense_2.* + {arch}.embedding_length_out, applied in-graph by
            # llama.cpp builds that support dense modules for BERT) and/or as
            # the .colbert_proj sidecar for serving layers that project on CPU.
            if t.name == "colbert.proj.weight":
                proj_weight = np.asarray(t.data, dtype=np.float32)
            elif t.name == "colbert.proj.bias":
                proj_bias = np.asarray(t.data, dtype=np.float32)
            if args.verbose:
                print(f"Captured projection tensor: {t.name}")
            skipped_count += 1
        else:
            print(f"Warning: Skipping unmapped custom tensor: {t.name}")
            skipped_count += 1

    # Determine the projection dimensions once for both emission paths.
    out_features = in_features = 0
    if proj_weight is not None:
        out_features = int(decode_field(reader.fields.get("colbert.projection.out_features")) or 0)
        in_features = int(decode_field(reader.fields.get("colbert.projection.in_features")) or 0)
        if (out_features <= 0 or in_features <= 0) and proj_weight.ndim == 2:
            # Fall back to the tensor shape; the projected dim (out) is the smaller one.
            a, b = proj_weight.shape
            out_features, in_features = (a, b) if a <= b else (b, a)
        if out_features <= 0 or in_features <= 0:
            raise SystemExit("cannot determine projection dimensions; colbert.projection.{out,in}_features missing")
        if proj_weight.size != out_features * in_features:
            raise SystemExit(
                f"projection size {proj_weight.size} != out*in ({out_features}*{in_features})")

    # Emit the projection as a sentence-transformers dense module inside the
    # GGUF: dense_2.weight/.bias plus {arch}.embedding_length_out. llama.cpp
    # applies it in-graph after pooling (for pooling none: per token), and
    # llama_model_n_embd_out() reports the projected width.
    #
    # Note: stock llama.cpp b9509 does not yet declare dense_2.* for the
    # bert/modernbert arch, so a file with these tensors needs a build that
    # does (e.g. ollama's). Use --no-dense-in-gguf for a backbone-only file
    # plus sidecar that stock llama.cpp can load.
    if args.dense_in_gguf and proj_weight is not None:
        writer.add_uint32(f"{arch}.embedding_length_out", out_features)
        writer.add_tensor("dense_2.weight", proj_weight.reshape(out_features, in_features))
        mapped_count += 1
        if proj_bias is not None:
            writer.add_tensor("dense_2.bias", proj_bias.reshape(out_features))
            mapped_count += 1
        print(f"Embedded dense projection: dense_2 ({in_features}->{out_features}, "
              f"bias={proj_bias is not None}), {arch}.embedding_length_out={out_features}")

    # Finalize GGUF
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"\nSuccess! Standard llama.cpp model written to: {outfile_path}")
    print(f"Tensors mapped: {mapped_count}, Tensors skipped/dropped: {skipped_count}")
    if args.dense_in_gguf and proj_weight is not None:
        print("The embedded dense_2 projection requires a llama.cpp build with BERT dense-module support (e.g. ollama).")
    else:
        print("This file can now be loaded directly in llama.cpp binaries (e.g. llama-embedding).")

    # Emit the projection sidecar (for serving layers that project on CPU).
    if args.projection_sidecar:
        if proj_weight is None:
            print("Warning: no colbert.proj.weight in input; skipping projection sidecar.", file=sys.stderr)
        else:
            sidecar_path = Path(str(outfile_path) + ".colbert_proj")
            write_projection_sidecar(sidecar_path, proj_weight, proj_bias, out_features, in_features)
            print(f"Wrote projection sidecar: {sidecar_path} (out={out_features}, in={in_features}, bias={proj_bias is not None})")


if __name__ == "__main__":
    main()
