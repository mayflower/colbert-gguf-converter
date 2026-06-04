#!/usr/bin/env python3
"""
Convert a ColBERT Hugging Face / SentenceTransformers model repository to pg_colbert GGUF format.
Supports both ModernBERT and BERT backbones.
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    print("Error: gguf package is not installed. Run 'pip install gguf'.", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("convert_colbert_hf")

CONVERTER_VERSION = "1.0.0"


@dataclass
class DenseConfig:
    in_features: int
    out_features: int
    bias: bool
    activation_function: str


@dataclass
class BackboneConfig:
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    max_position_embeddings: int
    hidden_activation: str
    layer_norm_eps: float
    raw_config: Dict[str, Any] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a Hugging Face ColBERT model to pg_colbert GGUF.")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-id", type=str, help="Hugging Face model repository ID")
    group.add_argument("--model-dir", type=str, help="Path to local Hugging Face repository directory")
    
    parser.add_argument("--outfile", type=str, required=True, help="Path to write the GGUF file")
    parser.add_argument("--outtype", type=str, choices=["f32", "f16"], default="f16",
                        help="Data type for GGUF output tensors (default: f16)")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory for Hugging Face downloads")
    parser.add_argument("--revision", type=str, default="main", help="Git revision/branch name")
    parser.add_argument("--schema", type=str, default="pg_colbert_v1", help="Target GGUF schema (default: pg_colbert_v1)")
    parser.add_argument("--dry-run", action="store_true", help="Parse configs and validate shapes, do not write GGUF")
    parser.add_argument("--dump-tensors", action="store_true", help="Dump list of tensors to be written")
    parser.add_argument("--no-download", action="store_true", help="Do not download model if model-id is not cached")
    parser.add_argument("--trust-remote-code", type=str, default="false",
                        help="Trust remote code (true or false, default false)")
    parser.add_argument("--allow-shape-mismatch", action="store_true",
                        help="Allow dense projection dimension mismatch with backbone hidden_size")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    return parser.parse_args()


def load_modules_json(model_path: Path) -> List[Dict[str, Any]]:
    path = model_path / "modules.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing modules.json in {model_path}")
    with open(path, "r", encoding="utf-8") as f:
        modules = json.load(f)
    return modules


def parse_dense_config(model_path: Path, modules: List[Dict[str, Any]]) -> Tuple[DenseConfig, Path]:
    dense_module = next((m for m in modules if "Dense" in m.get("type", "")), None)
    if dense_module is None:
        raise ValueError(
            f"modules.json in {model_path} does not contain a pylate.models.Dense.Dense or Dense module."
        )
    
    dense_dir_name = dense_module.get("path", "1_Dense")
    dense_path = model_path / dense_dir_name
    dense_cfg_path = dense_path / "config.json"
    
    if not dense_cfg_path.exists():
        raise FileNotFoundError(f"Missing Dense projection config at {dense_cfg_path}")
        
    with open(dense_cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
        
    dense_cfg = DenseConfig(
        in_features=cfg["in_features"],
        out_features=cfg["out_features"],
        bias=cfg.get("bias", False),
        activation_function=cfg.get("activation_function", "torch.nn.modules.linear.Identity")
    )
    return dense_cfg, dense_path


def parse_backbone_config(model_path: Path) -> BackboneConfig:
    cfg_path = model_path / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing backbone config at {cfg_path}")
        
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
        
    model_type = cfg.get("model_type", "").lower()
    if model_type not in ["modernbert", "bert"]:
        raise ValueError(
            f"Unsupported model_type: '{model_type}'. Only 'modernbert' and 'bert' are supported."
        )
        
    # Extract common keys, handle potential variation in naming
    hidden_size = cfg.get("hidden_size") or cfg.get("d_model")
    if hidden_size is None:
        raise ValueError("Could not find hidden_size or d_model in config.json")
        
    intermediate_size = cfg.get("intermediate_size") or cfg.get("mlp_dim")
    if intermediate_size is None:
        raise ValueError("Could not find intermediate_size or mlp_dim in config.json")
        
    num_hidden_layers = cfg.get("num_hidden_layers") or cfg.get("num_layers")
    if num_hidden_layers is None:
        raise ValueError("Could not find num_hidden_layers or num_layers in config.json")
        
    num_attention_heads = cfg.get("num_attention_heads") or cfg.get("num_heads")
    if num_attention_heads is None:
        raise ValueError("Could not find num_attention_heads or num_heads in config.json")
        
    max_position_embeddings = cfg.get("max_position_embeddings")
    if max_position_embeddings is None:
        raise ValueError("Could not find max_position_embeddings in config.json")
        
    hidden_activation = cfg.get("hidden_act") or cfg.get("hidden_activation") or "gelu"
    layer_norm_eps = cfg.get("layer_norm_eps") or cfg.get("norm_eps") or 1e-12

    backbone_cfg = BackboneConfig(
        model_type=model_type,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        max_position_embeddings=max_position_embeddings,
        hidden_activation=hidden_activation,
        layer_norm_eps=layer_norm_eps,
        raw_config=cfg
    )
    return backbone_cfg


def load_tokenizer_info(model_path: Path) -> Dict[str, Any]:
    # We resolve tokenizer info and special tokens
    tokenizer_json_path = model_path / "tokenizer.json"
    tokenizer_config_path = model_path / "tokenizer_config.json"
    special_tokens_path = model_path / "special_tokens_map.json"
    
    if not tokenizer_json_path.exists():
        raise FileNotFoundError(f"tokenizer.json not found in {model_path}")
        
    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        tok_json_str = f.read()
        
    tok_cfg_str = ""
    if tokenizer_config_path.exists():
        with open(tokenizer_config_path, "r", encoding="utf-8") as f:
            tok_cfg_str = f.read()
            
    spec_tokens_str = ""
    if special_tokens_path.exists():
        with open(special_tokens_path, "r", encoding="utf-8") as f:
            spec_tokens_str = f.read()

    # Load via transformers to resolve tokens
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    except Exception as e:
        logger.warning(f"Could not load AutoTokenizer from path: {e}. Tokenizer details might be incomplete.")
        tokenizer = None

    pad_token_id = tokenizer.pad_token_id if (tokenizer and tokenizer.pad_token_id is not None) else 0
    cls_token_id = tokenizer.cls_token_id if (tokenizer and tokenizer.cls_token_id is not None) else 101
    sep_token_id = tokenizer.sep_token_id if (tokenizer and tokenizer.sep_token_id is not None) else 102
    bos_token_id = tokenizer.bos_token_id if (tokenizer and tokenizer.bos_token_id is not None) else None
    eos_token_id = tokenizer.eos_token_id if (tokenizer and tokenizer.eos_token_id is not None) else None

    # Defaults or configuration values for ColBERT prefixes
    query_prefix = "[Q] "
    doc_prefix = "[D] "
    
    # Try reading from config_sentence_transformers.json first
    st_cfg_path = model_path / "config_sentence_transformers.json"
    if st_cfg_path.exists():
        try:
            with open(st_cfg_path, "r", encoding="utf-8") as f:
                st_cfg = json.load(f)
                if isinstance(st_cfg, dict):
                    if "query_prefix" in st_cfg:
                        query_prefix = st_cfg["query_prefix"]
                    if "document_prefix" in st_cfg:
                        doc_prefix = st_cfg["document_prefix"]
        except Exception:
            pass

    if tokenizer and hasattr(tokenizer, "query_prefix") and tokenizer.query_prefix:
        query_prefix = tokenizer.query_prefix
    if tokenizer and hasattr(tokenizer, "doc_prefix") and tokenizer.doc_prefix:
        doc_prefix = tokenizer.doc_prefix

    # Get query/document prefix token IDs
    query_prefix_ids = []
    doc_prefix_ids = []
    if tokenizer:
        query_prefix_ids = tokenizer.encode(query_prefix, add_special_tokens=False)
        doc_prefix_ids = tokenizer.encode(doc_prefix, add_special_tokens=False)

    # q_token_id / d_token_id resolution
    # Try resolving literal "[Q]" and "[D]" or fall back to prefix IDs
    q_token_id = None
    d_token_id = None
    if tokenizer:
        q_id = tokenizer.convert_tokens_to_ids("[Q]")
        if q_id != tokenizer.unk_token_id:
            q_token_id = q_id
        elif query_prefix_ids:
            q_token_id = query_prefix_ids[0]
            
        d_id = tokenizer.convert_tokens_to_ids("[D]")
        if d_id != tokenizer.unk_token_id:
            d_token_id = d_id
        elif doc_prefix_ids:
            d_token_id = doc_prefix_ids[0]

    return {
        "tokenizer_json": tok_json_str,
        "tokenizer_config": tok_cfg_str,
        "special_tokens_map": spec_tokens_str,
        "pad_token_id": pad_token_id,
        "cls_token_id": cls_token_id,
        "sep_token_id": sep_token_id,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "query_prefix": query_prefix,
        "document_prefix": doc_prefix,
        "query_prefix_ids": query_prefix_ids,
        "document_prefix_ids": doc_prefix_ids,
        "q_token_id": q_token_id,
        "d_token_id": d_token_id
    }


def generate_tensor_map(backbone_model_type: str, safetensors_keys: List[str], num_layers: int) -> Dict[str, Any]:
    """
    Generate canonical mapping JSON of tensors for pg_colbert GGUF runtime.
    Maps canonical name (e.g. embeddings.word_embeddings.weight) to the actual stored name in GGUF.
    Stored names are prefixed with 'hf.' for backbone weights and 'colbert.proj.' for projection weights.
    """
    def find_stored_name(pattern: str) -> Optional[str]:
        for prefix in ["", "model.", "bert."]:
            full_pat = prefix + pattern
            for k in safetensors_keys:
                if k == full_pat:
                    return f"hf.{k}"
        return None

    def find_stored_layer_name(layer_idx: int, pattern: str) -> Optional[str]:
        for prefix in ["", "model.", "bert."]:
            pats = [
                f"layers.{layer_idx}.{pattern}",
                f"layer.{layer_idx}.{pattern}",
                f"encoder.layer.{layer_idx}.{pattern}"
            ]
            for pat in pats:
                full_pat = prefix + pat
                for k in safetensors_keys:
                    if k == full_pat:
                        return f"hf.{k}"
        return None

    backbone_map = {}
    
    if backbone_model_type == "modernbert":
        # Embeddings
        backbone_map["embeddings"] = {
            "word_embeddings": find_stored_name("embeddings.tok_embeddings.weight"),
            "norm_weight": find_stored_name("embeddings.norm.weight"),
            "norm_bias": find_stored_name("embeddings.norm.bias")
        }
        # Layers
        layers_map = []
        for i in range(num_layers):
            layers_map.append({
                "attn_qkv_weight": find_stored_layer_name(i, "attn.Wqkv.weight"),
                "attn_qkv_bias": find_stored_layer_name(i, "attn.Wqkv.bias"),
                "attn_out_weight": find_stored_layer_name(i, "attn.out_proj.weight"),
                "attn_out_bias": find_stored_layer_name(i, "attn.out_proj.bias"),
                "attn_norm_weight": find_stored_layer_name(i, "attn.norm.weight"),
                "attn_norm_bias": find_stored_layer_name(i, "attn.norm.bias"),
                "mlp_wi_0_weight": find_stored_layer_name(i, "mlp.wi_0.weight"),
                "mlp_wi_0_bias": find_stored_layer_name(i, "mlp.wi_0.bias"),
                "mlp_wi_1_weight": find_stored_layer_name(i, "mlp.wi_1.weight"),
                "mlp_wi_1_bias": find_stored_layer_name(i, "mlp.wi_1.bias"),
                "mlp_wo_weight": find_stored_layer_name(i, "mlp.wo.weight"),
                "mlp_wo_bias": find_stored_layer_name(i, "mlp.wo.bias"),
                "mlp_norm_weight": find_stored_layer_name(i, "mlp.norm.weight"),
                "mlp_norm_bias": find_stored_layer_name(i, "mlp.norm.bias")
            })
        backbone_map["layers"] = layers_map
        # Final Norm
        backbone_map["final_norm"] = {
            "weight": find_stored_name("model.norm.weight") or find_stored_name("norm.weight"),
            "bias": find_stored_name("model.norm.bias") or find_stored_name("norm.bias")
        }
        
    elif backbone_model_type == "bert":
        # Embeddings
        backbone_map["embeddings"] = {
            "word_embeddings": find_stored_name("embeddings.word_embeddings.weight"),
            "position_embeddings": find_stored_name("embeddings.position_embeddings.weight"),
            "token_type_embeddings": find_stored_name("embeddings.token_type_embeddings.weight"),
            "norm_weight": find_stored_name("embeddings.LayerNorm.weight"),
            "norm_bias": find_stored_name("embeddings.LayerNorm.bias")
        }
        # Layers
        layers_map = []
        for i in range(num_layers):
            layers_map.append({
                "attn_query_weight": find_stored_layer_name(i, "attention.self.query.weight"),
                "attn_query_bias": find_stored_layer_name(i, "attention.self.query.bias"),
                "attn_key_weight": find_stored_layer_name(i, "attention.self.key.weight"),
                "attn_key_bias": find_stored_layer_name(i, "attention.self.key.bias"),
                "attn_value_weight": find_stored_layer_name(i, "attention.self.value.weight"),
                "attn_value_bias": find_stored_layer_name(i, "attention.self.value.bias"),
                "attn_out_weight": find_stored_layer_name(i, "attention.output.dense.weight"),
                "attn_out_bias": find_stored_layer_name(i, "attention.output.dense.bias"),
                "attn_norm_weight": find_stored_layer_name(i, "attention.output.LayerNorm.weight"),
                "attn_norm_bias": find_stored_layer_name(i, "attention.output.LayerNorm.bias"),
                "mlp_intermediate_weight": find_stored_layer_name(i, "intermediate.dense.weight"),
                "mlp_intermediate_bias": find_stored_layer_name(i, "intermediate.dense.bias"),
                "mlp_output_weight": find_stored_layer_name(i, "output.dense.weight"),
                "mlp_output_bias": find_stored_layer_name(i, "output.dense.bias"),
                "mlp_norm_weight": find_stored_layer_name(i, "output.LayerNorm.weight"),
                "mlp_norm_bias": find_stored_layer_name(i, "output.LayerNorm.bias")
            })
        backbone_map["layers"] = layers_map

    # Clean up None values from map so we only include present tensors
    def prune_none(d: Any) -> Any:
        if isinstance(d, dict):
            return {k: prune_none(v) for k, v in d.items() if v is not None}
        elif isinstance(d, list):
            return [prune_none(x) for x in d]
        return d

    tensor_map = {
        "schema": "pg_colbert_v1",
        "stored_name_style": "hf_original",
        "backbone": prune_none(backbone_map),
        "projection": {
            "weight": "colbert.proj.weight"
        }
    }
    
    # If projection bias exists, add to map
    for k in safetensors_keys:
        if "1_Dense" in k or "proj" in k: # will be verified when checking dense projection
            pass

    return tensor_map


def main() -> None:
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # 1. Resolve repository directory
    model_path: Optional[Path] = None
    if args.model_dir:
        model_path = Path(args.model_dir)
        if not model_path.exists() or not model_path.is_dir():
            logger.error(f"Local model directory does not exist: {model_path}")
            sys.exit(1)
        logger.info(f"Using local Hugging Face repo: {model_path.resolve()}")
    else:
        if args.no_download:
            logger.error("Cannot resolve --model-id when --no-download is specified without local path.")
            sys.exit(1)
        
        logger.info(f"Downloading snapshot for model ID '{args.model_id}' (revision: {args.revision})...")
        try:
            from huggingface_hub import snapshot_download
            download_dir = snapshot_download(
                repo_id=args.model_id,
                revision=args.revision,
                cache_dir=args.cache_dir,
                ignore_patterns=["*.bin", "*.h5", "*.ot", "*.msgpack"]  # download safetensors only
            )
            model_path = Path(download_dir)
            logger.info(f"Resolved model to snapshot cache: {model_path.resolve()}")
        except Exception as e:
            logger.error(f"Failed to snapshot model from Hub: {e}")
            sys.exit(1)

    assert model_path is not None

    # 2. Parse configurations
    try:
        modules = load_modules_json(model_path)
        dense_cfg, dense_path = parse_dense_config(model_path, modules)
        backbone_cfg = parse_backbone_config(model_path)
        tokenizer_info = load_tokenizer_info(model_path)
    except Exception as e:
        logger.error(f"Config parsing failed: {e}")
        sys.exit(1)

    # Check validation requirements
    if dense_cfg.bias:
        logger.warning(
            f"Dense projection config specifies bias = {dense_cfg.bias}. Custom runtime expects bias == false."
        )
        
    if dense_cfg.out_features != 128:
        logger.error(
            f"Dense projection out_features is {dense_cfg.out_features}, but ColBERT specification demands exactly 128."
        )
        sys.exit(1)

    # Read backbone safetensors to check key list and weights
    backbone_safetensors = sorted(model_path.glob("model*.safetensors"))
    if not backbone_safetensors:
        logger.error(f"No backbone safetensors ('model*.safetensors') found in {model_path}")
        sys.exit(1)
        
    # Read dense safetensors
    dense_safetensors = list(dense_path.glob("*.safetensors"))
    if not dense_safetensors:
        logger.error(f"No dense projection safetensors found in {dense_path}")
        sys.exit(1)

    # Gather all tensor info and shapes
    backbone_tensor_meta = {}
    for sf in backbone_safetensors:
        with safe_open(sf, framework="pt", device="cpu") as f:
            for key in f.keys():
                backbone_tensor_meta[key] = f.get_slice(key).get_shape()

    dense_tensor_meta = {}
    for sf in dense_safetensors:
        with safe_open(sf, framework="pt", device="cpu") as f:
            for key in f.keys():
                dense_tensor_meta[key] = f.get_slice(key).get_shape()

    # Find the projection weight and shape validation
    proj_weight_key = next((k for k in dense_tensor_meta.keys() if "weight" in k), None)
    if proj_weight_key is None:
        logger.error(f"Could not find projection weight tensor in {dense_path}")
        sys.exit(1)
        
    proj_weight_shape = dense_tensor_meta[proj_weight_key]
    proj_in_features = proj_weight_shape[1] if len(proj_weight_shape) > 1 else proj_weight_shape[0]
    
    # Validation of shape
    if proj_in_features != backbone_cfg.hidden_size:
        msg = f"Dense projection in_features ({proj_in_features}) does not match backbone hidden_size ({backbone_cfg.hidden_size})."
        if not args.allow_shape_mismatch:
            logger.error(f"Validation FAILED: {msg} Pass --allow-shape-mismatch to bypass.")
            sys.exit(1)
        else:
            logger.warning(f"Validation WARNING: {msg} (Proceeding because --allow-shape-mismatch is set)")

    # Read license if available in model directory (provenance)
    license_str = "unknown"
    readme_path = model_path / "README.md"
    if readme_path.exists():
        try:
            with open(readme_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Simple heuristic search for license
                for line in content.split("\n"):
                    if "license:" in line:
                        license_str = line.split("license:")[-1].strip()
                        break
        except Exception:
            pass

    # Read similarity function name and other params from config_sentence_transformers.json if available
    similarity_fn = "cosine"
    query_prefix = tokenizer_info["query_prefix"]
    doc_prefix = tokenizer_info["document_prefix"]
    query_length = 32
    doc_length = 256
    attend_to_expansion = True
    do_query_expansion = True
    skiplist_words = []

    st_cfg_path = model_path / "config_sentence_transformers.json"
    if st_cfg_path.exists():
        try:
            with open(st_cfg_path, "r", encoding="utf-8") as f:
                st_cfg = json.load(f)
                if isinstance(st_cfg, dict):
                    fn = st_cfg.get("similarity_fn_name") or st_cfg.get("similarity")
                    if fn:
                        similarity_fn = fn
                    
                    q_prefix = st_cfg.get("query_prefix")
                    if q_prefix is not None:
                        query_prefix = q_prefix
                        
                    d_prefix = st_cfg.get("document_prefix")
                    if d_prefix is not None:
                        doc_prefix = d_prefix
                        
                    q_len = st_cfg.get("query_length")
                    if q_len is not None:
                        query_length = int(q_len)
                        
                    d_len = st_cfg.get("document_length")
                    if d_len is not None:
                        doc_length = int(d_len)
                        
                    att_to_exp = st_cfg.get("attend_to_expansion_tokens")
                    if att_to_exp is not None:
                        attend_to_expansion = bool(att_to_exp)
                        
                    do_q_exp = st_cfg.get("do_query_expansion")
                    if do_q_exp is not None:
                        do_query_expansion = bool(do_q_exp)
                        
                    skip_words = st_cfg.get("skiplist_words")
                    if skip_words is not None:
                        skiplist_words = list(skip_words)
        except Exception as e:
            logger.warning(f"Failed to parse config_sentence_transformers.json: {e}")

    # Build the canonical tensor map JSON
    all_backbone_keys = list(backbone_tensor_meta.keys())
    tensor_map = generate_tensor_map(backbone_cfg.model_type, all_backbone_keys, backbone_cfg.num_hidden_layers)
    if any(k for k in dense_tensor_meta.keys() if "bias" in k):
        tensor_map["projection"]["bias"] = "colbert.proj.bias"
    tensor_map_json_str = json.dumps(tensor_map, indent=2)

    logger.info("Config Validation:")
    logger.info(f"  Backbone model type: {backbone_cfg.model_type}")
    logger.info(f"  Hidden size:         {backbone_cfg.hidden_size}")
    logger.info(f"  Embedding dimension: {dense_cfg.out_features}")
    logger.info(f"  Similarity metric:   {similarity_fn}")
    logger.info(f"  Query Prefix:        '{tokenizer_info['query_prefix']}' -> {tokenizer_info['query_prefix_ids']}")
    logger.info(f"  Doc Prefix:          '{tokenizer_info['document_prefix']}' -> {tokenizer_info['document_prefix_ids']}")

    if args.dry_run or args.dump_tensors:
        if args.dump_tensors:
            print("\n=== Backbone Tensors ===")
            for k, shape in backbone_tensor_meta.items():
                print(f"  hf.{k}: shape={shape}")
            print("\n=== Projection Tensors ===")
            for k, shape in dense_tensor_meta.items():
                target_name = "colbert.proj.weight" if "weight" in k else "colbert.proj.bias"
                print(f"  {target_name}: shape={shape}")
        
        logger.info("Dry-run complete. Exiting without writing GGUF.")
        return

    # 3. Write GGUF File
    outfile_path = Path(args.outfile)
    outfile_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating GGUF model: {outfile_path}")
    # Initialize GGUFWriter with architecture type
    writer = GGUFWriter(path=str(outfile_path), arch=backbone_cfg.model_type)

    # Schema & general metadata
    writer.add_string("pg_colbert.gguf_schema", args.schema)
    writer.add_string("general.name", backbone_cfg.raw_config.get("name") or args.model_id or model_path.name)
    writer.add_string("general.basename", args.model_id or model_path.name)
    writer.add_string("general.architecture", backbone_cfg.model_type)
    
    ft_code = 1 if args.outtype == "f16" else 0  # 1 for F16, 0 for F32
    writer.add_uint32("general.file_type", ft_code)
    
    if args.model_id:
        writer.add_string("general.source.huggingface.repo_id", args.model_id)
    writer.add_string("general.source.huggingface.revision", args.revision)
    writer.add_string("general.license", license_str)
    
    # Converter provenance
    writer.add_string("pg_colbert.converter.version", CONVERTER_VERSION)
    writer.add_string("pg_colbert.converter.command", " ".join(sys.argv))
    writer.add_string("pg_colbert.tensor_map_json", tensor_map_json_str)

    # ColBERT Config Metadata
    writer.add_string("colbert.model_type", backbone_cfg.model_type)
    writer.add_string("colbert.backbone_model_type", backbone_cfg.model_type)
    writer.add_uint32("colbert.embedding_dim", dense_cfg.out_features)
    writer.add_uint32("colbert.projection.in_features", proj_in_features)
    writer.add_uint32("colbert.projection.out_features", dense_cfg.out_features)
    writer.add_bool("colbert.projection.bias", dense_cfg.bias)
    writer.add_string("colbert.similarity_fn_name", similarity_fn)
    writer.add_string("colbert.query_prefix", query_prefix)
    writer.add_string("colbert.document_prefix", doc_prefix)
    writer.add_uint32("colbert.query_length", query_length)
    writer.add_uint32("colbert.document_length", doc_length)
    writer.add_bool("colbert.attend_to_expansion_tokens", attend_to_expansion)
    writer.add_bool("colbert.do_query_expansion", do_query_expansion)
    writer.add_array("colbert.skiplist_words", skiplist_words)

    # resolved special token IDs
    if tokenizer_info["q_token_id"] is not None:
        writer.add_uint32("colbert.q_token_id", tokenizer_info["q_token_id"])
    if tokenizer_info["d_token_id"] is not None:
        writer.add_uint32("colbert.d_token_id", tokenizer_info["d_token_id"])
    if tokenizer_info["pad_token_id"] is not None:
        writer.add_uint32("colbert.pad_token_id", tokenizer_info["pad_token_id"])
    if tokenizer_info["cls_token_id"] is not None:
        writer.add_uint32("colbert.cls_token_id", tokenizer_info["cls_token_id"])
    if tokenizer_info["sep_token_id"] is not None:
        writer.add_uint32("colbert.sep_token_id", tokenizer_info["sep_token_id"])
    if tokenizer_info["bos_token_id"] is not None:
        writer.add_uint32("colbert.bos_token_id", tokenizer_info["bos_token_id"])
    if tokenizer_info["eos_token_id"] is not None:
        writer.add_uint32("colbert.eos_token_id", tokenizer_info["eos_token_id"])
    
    if tokenizer_info["query_prefix_ids"]:
        writer.add_array("colbert.query_prefix_token_ids", tokenizer_info["query_prefix_ids"])
    if tokenizer_info["document_prefix_ids"]:
        writer.add_array("colbert.document_prefix_token_ids", tokenizer_info["document_prefix_ids"])

    # Backbone specific metadata
    arch_prefix = f"{backbone_cfg.model_type}."
    
    if backbone_cfg.model_type == "modernbert":
        # Add ModernBERT variables
        writer.add_uint32(f"{arch_prefix}hidden_size", backbone_cfg.hidden_size)
        writer.add_uint32(f"{arch_prefix}intermediate_size", backbone_cfg.intermediate_size)
        writer.add_uint32(f"{arch_prefix}num_hidden_layers", backbone_cfg.num_hidden_layers)
        writer.add_uint32(f"{arch_prefix}num_attention_heads", backbone_cfg.num_attention_heads)
        writer.add_uint32(f"{arch_prefix}max_position_embeddings", backbone_cfg.max_position_embeddings)
        
        # RoPE, local attn, etc.
        local_attn = backbone_cfg.raw_config.get("local_attention") or 128
        writer.add_uint32(f"{arch_prefix}local_attention", local_attn)
        global_attn_every_n = backbone_cfg.raw_config.get("global_attn_every_n_layers") or 3
        writer.add_uint32(f"{arch_prefix}global_attn_every_n_layers", global_attn_every_n)
        
        local_rope_theta = float(backbone_cfg.raw_config.get("local_rope_theta") or 10000.0)
        global_rope_theta = float(backbone_cfg.raw_config.get("global_rope_theta") or 160000.0)
        writer.add_float32(f"{arch_prefix}local_rope_theta", local_rope_theta)
        writer.add_float32(f"{arch_prefix}global_rope_theta", global_rope_theta)
        
        writer.add_string(f"{arch_prefix}hidden_activation", backbone_cfg.hidden_activation)
        writer.add_float32(f"{arch_prefix}layer_norm_eps", backbone_cfg.layer_norm_eps)
        writer.add_float32(f"{arch_prefix}norm_eps", backbone_cfg.layer_norm_eps)
        
        writer.add_bool(f"{arch_prefix}attention_bias", backbone_cfg.raw_config.get("attention_bias", False))
        writer.add_bool(f"{arch_prefix}mlp_bias", backbone_cfg.raw_config.get("mlp_bias", False))
        writer.add_bool(f"{arch_prefix}norm_bias", backbone_cfg.raw_config.get("norm_bias", False))
        
    elif backbone_cfg.model_type == "bert":
        # Add BERT variables
        writer.add_uint32(f"{arch_prefix}hidden_size", backbone_cfg.hidden_size)
        writer.add_uint32(f"{arch_prefix}intermediate_size", backbone_cfg.intermediate_size)
        writer.add_uint32(f"{arch_prefix}num_hidden_layers", backbone_cfg.num_hidden_layers)
        writer.add_uint32(f"{arch_prefix}num_attention_heads", backbone_cfg.num_attention_heads)
        writer.add_uint32(f"{arch_prefix}max_position_embeddings", backbone_cfg.max_position_embeddings)
        writer.add_string(f"{arch_prefix}hidden_activation", backbone_cfg.hidden_activation)
        writer.add_float32(f"{arch_prefix}layer_norm_eps", backbone_cfg.layer_norm_eps)
        writer.add_uint32(f"{arch_prefix}type_vocab_size", backbone_cfg.raw_config.get("type_vocab_size", 2))

    # Tokenizer files embedded as string metadata
    writer.add_string("tokenizer.huggingface.json", tokenizer_info["tokenizer_json"])
    if tokenizer_info["tokenizer_config"]:
        writer.add_string("tokenizer.config.json", tokenizer_info["tokenizer_config"])
    if tokenizer_info["special_tokens_map"]:
        writer.add_string("tokenizer.special_tokens_map.json", tokenizer_info["special_tokens_map"])

    # Define a float-conversion helper to support float16 outtype
    def process_tensor(tensor: torch.Tensor, name: str) -> np.ndarray:
        # Convert to numpy array, handling BF16 safely
        if tensor.dtype == torch.bfloat16:
            # numpy doesn't support bfloat16 directly, cast to f32 first
            arr = tensor.to(torch.float32).detach().cpu().numpy()
        else:
            arr = tensor.detach().cpu().numpy()
            
        # Downcast floats to f16 if specified
        if args.outtype == "f16" and arr.dtype in (np.float32, np.float64):
            arr = arr.astype(np.float16)
        elif args.outtype == "f32" and arr.dtype in (np.float32, np.float64, np.float16):
            arr = arr.astype(np.float32)
            
        return arr

    # 4. Write Tensors
    # A. Write Backbone Tensors
    for sf in backbone_safetensors:
        logger.info(f"Writing backbone tensors from {sf.name}...")
        with safe_open(sf, framework="pt", device="cpu") as f:
            for key in sorted(f.keys()):
                tensor = f.get_tensor(key)
                arr = process_tensor(tensor, key)
                stored_name = f"hf.{key}"
                writer.add_tensor(stored_name, arr)

    # B. Write Dense Projection Tensors
    for sf in dense_safetensors:
        logger.info(f"Writing projection tensors from {sf.name}...")
        with safe_open(sf, framework="pt", device="cpu") as f:
            for key in sorted(f.keys()):
                tensor = f.get_tensor(key)
                arr = process_tensor(tensor, key)
                if "weight" in key:
                    stored_name = "colbert.proj.weight"
                elif "bias" in key:
                    stored_name = "colbert.proj.bias"
                else:
                    stored_name = f"colbert.proj.{key}"
                writer.add_tensor(stored_name, arr)

    # 5. Finalize file
    logger.info("Writing metadata headers and serializing GGUF data...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    logger.info(f"Successfully converted model to: {outfile_path.resolve()}")


if __name__ == "__main__":
    main()
