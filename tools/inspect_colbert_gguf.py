#!/usr/bin/env python3
"""
Tool to inspect a custom ColBERT GGUF file (pg_colbert_v1 schema).
Reads the GGUF file, prints all metadata keys/values, and lists all tensors.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List

try:
    from gguf import GGUFReader, GGMLQuantizationType
except ImportError:
    print("Error: gguf package is not installed. Run 'pip install gguf'.", file=sys.stderr)
    sys.exit(1)

# Ensure parent directory is in path so tools.colbert_profile is importable
sys.path.append(str(Path(__file__).parent.parent))
from tools.colbert_profile import ColbertProfile, validate_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and validate a pg_colbert_v1 GGUF file.")
    parser.add_argument("gguf_path", type=str, help="Path to the GGUF file")
    parser.add_argument("--verbose", action="store_true", help="Print all tensors and detailed tokenizer info")
    return parser.parse_args()


def decode_gguf_field(field: Any) -> Any:
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


def format_field_value(field: Any) -> str:
    """Format GGUF field value for display."""
    val = decode_gguf_field(field)
    if val is None:
        return "None"
    if isinstance(val, list):
        if len(val) > 10:
            return "[" + ", ".join(map(str, val[:5])) + ", ..., " + ", ".join(map(str, val[-5:])) + f"] (length {len(val)})"
        return str(val)
    return str(val)


def main() -> None:
    args = parse_args()
    gguf_path = Path(args.gguf_path)
    if not gguf_path.exists():
        print(f"Error: GGUF file does not exist at {gguf_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading GGUF file: {gguf_path.resolve()}\n")
    try:
        reader = GGUFReader(gguf_path)
    except Exception as e:
        print(f"Error loading GGUF file: {e}", file=sys.stderr)
        sys.exit(1)

    # 1. Print all metadata fields
    print("=== METADATA ===")
    fields = sorted(reader.fields.keys())
    for f_name in fields:
        field = reader.fields[f_name]
        val_str = format_field_value(field)
        # Avoid printing massive tokenizer/profile JSON in non-verbose mode
        if ("tokenizer" in f_name or "profile_json" in f_name) and not args.verbose:
            val_str = f"[{len(val_str)} characters of JSON string]"
        print(f"{f_name}: {val_str}")

    print("\n=== VALIDATION STATUS ===")
    
    # Validation checks
    errors: List[str] = []
    
    # Check schema
    schema_field = reader.fields.get("pg_colbert.gguf_schema")
    if schema_field is None:
        errors.append("Missing required schema key: 'pg_colbert.gguf_schema'")
    else:
        schema_val = format_field_value(schema_field)
        if schema_val != "pg_colbert_v1":
            errors.append(f"Invalid schema: expected 'pg_colbert_v1', got '{schema_val}'")
            
    # Check architecture
    arch_field = reader.fields.get("general.architecture")
    if arch_field is None:
        errors.append("Missing 'general.architecture' metadata key")
    
    # Check ColBERT critical parameters
    colbert_keys = [
        "colbert.model_type",
        "colbert.backbone_model_type",
        "colbert.embedding_dim",
        "colbert.projection.out_features",
        "colbert.query_prefix",
        "colbert.document_prefix",
        "tokenizer.huggingface.json"
    ]
    for k in colbert_keys:
        if k not in reader.fields:
            errors.append(f"Missing ColBERT config key: '{k}'")

    # Check and validate ColBERT profile JSON
    profile = None
    profile_field = reader.fields.get("pg_colbert.profile_json")
    if profile_field is None:
        errors.append("Missing required ColBERT profile key: 'pg_colbert.profile_json'")
    else:
        profile_json_val = decode_gguf_field(profile_field)
        try:
            profile_dict = json.loads(profile_json_val)
            profile = ColbertProfile.from_dict(profile_dict)
            validate_profile(profile)
        except Exception as e:
            errors.append(f"Failed to parse or validate ColBERT profile: {e}")

    # Check tensors
    tensors = reader.tensors
    print(f"Total tensor count: {len(tensors)}")

    # Check projection weights
    proj_weight_present = False
    for t in tensors:
        if t.name == "colbert.proj.weight":
            proj_weight_present = True
            # Print details of the projection tensor
            print(f"Projection tensor 'colbert.proj.weight' details:")
            print(f"  Shape: {list(t.shape)}")
            # GGMLQuantizationType is integer code
            print(f"  Dtype / QType: {t.tensor_type}")
            break
            
    if not proj_weight_present:
        errors.append("Missing required ColBERT projection weight tensor: 'colbert.proj.weight'")

    if errors:
        print("Validation: FAILED")
        for err in errors:
            print(f"  - [ERROR] {err}")
    else:
        print("Validation: SUCCESS (All required keys and tensors are present and valid!)")

    if profile is not None:
        print("\n=== COLBERT PROFILE SUMMARY ===")
        print(f"schema: {profile.schema}")
        print(f"output_dim: {profile.output_dim}")
        print(f"query prefix/length/pad_to: prefix={json.dumps(profile.query.prefix)}, length={profile.query.max_length}, pad_to={profile.query.pad_to}")
        print(f"document prefix/length: prefix={json.dumps(profile.document.prefix)}, length={profile.document.max_length}")
        print(f"skiplist token count: {len(profile.document.skiplist_token_ids)}")
        
        if profile.projection.kind == "identity":
            print("projection kind/modules: kind=identity")
        else:
            modules_desc = []
            for m in profile.projection.modules:
                modules_desc.append(f"{m.type}({m.in_features} -> {m.out_features}, bias={m.bias})")
            print(f"projection kind/modules: kind={profile.projection.kind}, modules=[{', '.join(modules_desc)}]")
            
        comp_str = (
            f"llama_cpp_loadable={profile.compatibility.llama_cpp_loadable}, "
            f"requires_profile={profile.compatibility.requires_profile}, "
            f"strict_pylate_profile={profile.compatibility.strict_pylate_profile}"
        )
        if profile.compatibility.known_limitations:
            comp_str += f", limitations={profile.compatibility.known_limitations}"
        print(f"compatibility flags: {comp_str}")


    # 2. Print all tensors list
    if args.verbose:
        print("\n=== TENSORS ===")
        for i, t in enumerate(tensors):
            # Convert dtype using GGMLQuantizationType enum if possible
            try:
                dtype_name = GGMLQuantizationType(t.tensor_type).name
            except Exception:
                dtype_name = f"unknown ({t.tensor_type})"
            print(f"Tensor #{i+1:03d}: name={t.name}, shape={list(t.shape)}, dtype={dtype_name}")

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
