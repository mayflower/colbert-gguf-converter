#!/usr/bin/env python3
"""
Tool to validate pg_colbert_v1 GGUF models against original PyTorch/HF models
or PyLate golden vector fixtures. It reconstructs the PyTorch model structure
and loads weights directly from GGUF tensor data to verify numeric equivalence.
"""

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from gguf import GGUFReader
from transformers import AutoConfig, AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("validate_colbert_gguf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate pg_colbert_v1 GGUF model numerical inference.")
    parser.add_argument("gguf_path", type=str, help="Path to the converted GGUF model file")
    parser.add_argument("--hf-model", type=str, default=None,
                        help="Optional path or HF Repo ID of original model for direct cross-validation")
    parser.add_argument("--fixtures-dir", type=str, default="tests/fixtures",
                        help="Path to directory containing pylate_golden.json and pylate_golden.npz")
    parser.add_argument("--tolerance", type=float, default=1e-3,
                        help="Maximum mean absolute error tolerance (default: 1e-3)")
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


def rebuild_tokenizer(reader: GGUFReader) -> Tuple[AutoTokenizer, Dict[str, Any]]:
    """Rebuild tokenizer from tokenizer JSONs embedded in GGUF metadata."""
    tok_json_field = reader.fields.get("tokenizer.huggingface.json")
    if tok_json_field is None:
        raise ValueError("Missing 'tokenizer.huggingface.json' in GGUF metadata.")
    
    tokenizer_json = decode_gguf_field(tok_json_field)
    
    # We will write these to a temp dir and load via AutoTokenizer
    tmp_dir = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp_dir.name)
    
    with open(tmp_path / "tokenizer.json", "w", encoding="utf-8") as f:
        f.write(tokenizer_json)
        
    tok_cfg_field = reader.fields.get("tokenizer.config.json")
    if tok_cfg_field is not None:
        with open(tmp_path / "tokenizer_config.json", "w", encoding="utf-8") as f:
            f.write(decode_gguf_field(tok_cfg_field))
            
    spec_tokens_field = reader.fields.get("tokenizer.special_tokens_map.json")
    if spec_tokens_field is not None:
        with open(tmp_path / "special_tokens_map.json", "w", encoding="utf-8") as f:
            f.write(decode_gguf_field(spec_tokens_field))
            
    tokenizer = AutoTokenizer.from_pretrained(str(tmp_path), local_files_only=True)
    
    # Extract prefix configurations
    attend_val = reader.fields.get("colbert.attend_to_expansion_tokens")
    attend_to_expansion = decode_gguf_field(attend_val) if attend_val is not None else True
    
    skiplist_field = reader.fields.get("colbert.skiplist_words")
    skiplist_words = decode_gguf_field(skiplist_field) if skiplist_field is not None else []
    
    tokenizer_info = {
        "query_prefix": decode_gguf_field(reader.fields.get("colbert.query_prefix")) or "[Q] ",
        "document_prefix": decode_gguf_field(reader.fields.get("colbert.document_prefix")) or "[D] ",
        "query_length": int(decode_gguf_field(reader.fields.get("colbert.query_length")) or 32),
        "document_length": int(decode_gguf_field(reader.fields.get("colbert.document_length")) or 300),
        "attend_to_expansion_tokens": bool(attend_to_expansion),
        "skiplist_words": skiplist_words,
    }
    
    # Keep reference to tmp_dir so it doesn't get cleaned up too early
    tokenizer._temp_dir_ref = tmp_dir
    
    return tokenizer, tokenizer_info


def rebuild_pytorch_model(reader: GGUFReader, arch: str) -> Tuple[torch.nn.Module, torch.Tensor, Optional[torch.Tensor]]:
    """Construct empty PyTorch model and load GGUF tensors into its state_dict."""
    config_dict = {}
    
    # Parse generic config keys
    config_dict["hidden_size"] = int(decode_gguf_field(reader.fields.get(f"{arch}.hidden_size")))
    config_dict["intermediate_size"] = int(decode_gguf_field(reader.fields.get(f"{arch}.intermediate_size")))
    config_dict["num_hidden_layers"] = int(decode_gguf_field(reader.fields.get(f"{arch}.num_hidden_layers")))
    config_dict["num_attention_heads"] = int(decode_gguf_field(reader.fields.get(f"{arch}.num_attention_heads")))
    config_dict["max_position_embeddings"] = int(decode_gguf_field(reader.fields.get(f"{arch}.max_position_embeddings")))
    config_dict["hidden_act"] = decode_gguf_field(reader.fields.get(f"{arch}.hidden_activation"))
    
    if arch == "modernbert":
        config_dict["local_attention"] = int(decode_gguf_field(reader.fields.get("modernbert.local_attention")))
        config_dict["global_attn_every_n_layers"] = int(decode_gguf_field(reader.fields.get("modernbert.global_attn_every_n_layers")))
        config_dict["local_rope_theta"] = float(decode_gguf_field(reader.fields.get("modernbert.local_rope_theta")) or 10000.0)
        config_dict["global_rope_theta"] = float(decode_gguf_field(reader.fields.get("modernbert.global_rope_theta")) or 160000.0)
        config_dict["layer_norm_eps"] = float(decode_gguf_field(reader.fields.get("modernbert.layer_norm_eps")) or 1e-5)
        config_dict["attention_bias"] = bool(decode_gguf_field(reader.fields.get("modernbert.attention_bias")) or False)
        config_dict["mlp_bias"] = bool(decode_gguf_field(reader.fields.get("modernbert.mlp_bias")) or False)
        config_dict["norm_bias"] = bool(decode_gguf_field(reader.fields.get("modernbert.norm_bias")) or False)
    elif arch == "bert":
        config_dict["layer_norm_eps"] = float(decode_gguf_field(reader.fields.get("bert.layer_norm_eps")) or 1e-12)
        config_dict["type_vocab_size"] = int(decode_gguf_field(reader.fields.get("bert.type_vocab_size")) or 2)
        
    # Copy special token IDs from GGUF metadata if present
    pad_val = reader.fields.get("colbert.pad_token_id")
    if pad_val is not None:
        config_dict["pad_token_id"] = int(decode_gguf_field(pad_val))
    bos_val = reader.fields.get("colbert.bos_token_id")
    if bos_val is not None:
        config_dict["bos_token_id"] = int(decode_gguf_field(bos_val))
    eos_val = reader.fields.get("colbert.eos_token_id")
    if eos_val is not None:
        config_dict["eos_token_id"] = int(decode_gguf_field(eos_val))
        
    # Set vocab_size dynamically based on GGUF embedding tensor shape
    vocab_size = None
    for t in reader.tensors:
        clean_name = t.name
        if clean_name.startswith("hf."):
            clean_name = clean_name[3:]
        for prefix in ["model.", "bert."]:
            if clean_name.startswith(prefix):
                clean_name = clean_name[len(prefix):]
        if clean_name in ["embeddings.tok_embeddings.weight", "embeddings.word_embeddings.weight"]:
            vocab_size = t.data.shape[0]
            break
            
    if vocab_size is not None:
        config_dict["vocab_size"] = vocab_size
        
    config = AutoConfig.for_model(arch, **config_dict)
    # Instantiate uninitialized architecture
    model = AutoModel.from_config(config)
    
    # Map GGUF tensors to PyTorch state_dict
    gguf_tensors = {}
    proj_weight = None
    proj_bias = None
    
    for t in reader.tensors:
        if t.name.startswith("hf."):
            clean_name = t.name[3:]
            gguf_tensors[clean_name] = torch.from_numpy(t.data.copy())
        elif t.name == "colbert.proj.weight":
            proj_weight = torch.from_numpy(t.data.copy())
        elif t.name == "colbert.proj.bias":
            proj_bias = torch.from_numpy(t.data.copy())
            
    if proj_weight is None:
        raise ValueError("Missing 'colbert.proj.weight' tensor in GGUF file.")
        
    # Build aligned state dict
    state_dict = {}
    
    if arch == "modernbert":
        # Helper to lookup GGUF tensors with optional model. prefix
        def get_tensor(key: str) -> Optional[torch.Tensor]:
            for prefix in ["", "model."]:
                full = prefix + key
                if full in gguf_tensors:
                    return gguf_tensors[full]
            return None
            
        def copy_if_present(gguf_k: str, model_k: str):
            val = get_tensor(gguf_k)
            if val is not None:
                state_dict[model_k] = val

        copy_if_present("embeddings.tok_embeddings.weight", "embeddings.tok_embeddings.weight")
        copy_if_present("embeddings.norm.weight", "embeddings.norm.weight")
        copy_if_present("embeddings.norm.bias", "embeddings.norm.bias")
        
        # Layers
        num_layers = config_dict["num_hidden_layers"]
        for i in range(num_layers):
            copy_if_present(f"layers.{i}.attn.Wqkv.weight", f"layers.{i}.attn.Wqkv.weight")
            copy_if_present(f"layers.{i}.attn.Wqkv.bias", f"layers.{i}.attn.Wqkv.bias")
            
            # attn.Wo (official HF name) or attn.out_proj
            copy_if_present(f"layers.{i}.attn.Wo.weight", f"layers.{i}.attn.Wo.weight")
            copy_if_present(f"layers.{i}.attn.Wo.bias", f"layers.{i}.attn.Wo.bias")
            copy_if_present(f"layers.{i}.attn.out_proj.weight", f"layers.{i}.attn.Wo.weight")
            copy_if_present(f"layers.{i}.attn.out_proj.bias", f"layers.{i}.attn.Wo.bias")
            
            # Layer 0 has no attn_norm in standard ModernBertModel
            if i > 0:
                copy_if_present(f"layers.{i}.attn.norm.weight", f"layers.{i}.attn_norm.weight")
                copy_if_present(f"layers.{i}.attn.norm.bias", f"layers.{i}.attn_norm.bias")
                copy_if_present(f"layers.{i}.attn_norm.weight", f"layers.{i}.attn_norm.weight")
                copy_if_present(f"layers.{i}.attn_norm.bias", f"layers.{i}.attn_norm.bias")
                
            copy_if_present(f"layers.{i}.mlp.norm.weight", f"layers.{i}.mlp_norm.weight")
            copy_if_present(f"layers.{i}.mlp.norm.bias", f"layers.{i}.mlp_norm.bias")
            copy_if_present(f"layers.{i}.mlp_norm.weight", f"layers.{i}.mlp_norm.weight")
            copy_if_present(f"layers.{i}.mlp_norm.bias", f"layers.{i}.mlp_norm.bias")
            
            # Try loading mlp.Wi directly (official HF name) or concatenate split GLU projections (wi_0 and wi_1)
            copy_if_present(f"layers.{i}.mlp.Wi.weight", f"layers.{i}.mlp.Wi.weight")
            copy_if_present(f"layers.{i}.mlp.Wi.bias", f"layers.{i}.mlp.Wi.bias")
            
            if f"layers.{i}.mlp.Wi.weight" not in state_dict:
                wi_0_w = get_tensor(f"layers.{i}.mlp.wi_0.weight")
                wi_1_w = get_tensor(f"layers.{i}.mlp.wi_1.weight")
                if wi_0_w is not None and wi_1_w is not None:
                    state_dict[f"layers.{i}.mlp.Wi.weight"] = torch.cat([wi_0_w, wi_1_w], dim=0)
                    
            if f"layers.{i}.mlp.Wi.bias" not in state_dict:
                wi_0_b = get_tensor(f"layers.{i}.mlp.wi_0.bias")
                wi_1_b = get_tensor(f"layers.{i}.mlp.wi_1.bias")
                if wi_0_b is not None and wi_1_b is not None:
                    state_dict[f"layers.{i}.mlp.Wi.bias"] = torch.cat([wi_0_b, wi_1_b], dim=0)
                
            # mlp.Wo (official HF name) or mlp.wo
            copy_if_present(f"layers.{i}.mlp.Wo.weight", f"layers.{i}.mlp.Wo.weight")
            copy_if_present(f"layers.{i}.mlp.Wo.bias", f"layers.{i}.mlp.Wo.bias")
            copy_if_present(f"layers.{i}.mlp.wo.weight", f"layers.{i}.mlp.Wo.weight")
            copy_if_present(f"layers.{i}.mlp.wo.bias", f"layers.{i}.mlp.Wo.bias")
            
        # Final Norm
        copy_if_present("norm.weight", "final_norm.weight")
        copy_if_present("norm.bias", "final_norm.bias")
        copy_if_present("final_norm.weight", "final_norm.weight")
        copy_if_present("final_norm.bias", "final_norm.bias")
        
    else:
        # Standard prefix stripping for BERT and generic models
        model_state = model.state_dict()
        def clean_key(k: str) -> str:
            for prefix in ["model.", "bert."]:
                if k.startswith(prefix):
                    return k[len(prefix):]
            return k
            
        gguf_clean_lookup = {clean_key(k): v for k, v in gguf_tensors.items()}
        for m_key in model_state.keys():
            m_key_clean = clean_key(m_key)
            if m_key_clean in gguf_clean_lookup:
                state_dict[m_key] = gguf_clean_lookup[m_key_clean]

    # Load state dict
    msg = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Loaded state dict into model. Missing keys: {len(msg.missing_keys)} keys, Unexpected keys: {len(msg.unexpected_keys)} keys")
    
    # Cast projection tensors to match model parameter precision (avoid float vs Half mismatch)
    if proj_weight is not None:
        proj_weight = proj_weight.to(dtype=next(model.parameters()).dtype)
    if proj_bias is not None:
        proj_bias = proj_bias.to(dtype=next(model.parameters()).dtype)
        
    model.eval()
    return model, proj_weight, proj_bias


def run_inference(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    tok_info: Dict[str, Any],
    text: str,
    is_query: bool,
    proj_weight: torch.Tensor,
    proj_bias: Optional[torch.Tensor]
) -> np.ndarray:
    """Run model inference to generate query/document token-level ColBERT embeddings."""
    prefix = tok_info["query_prefix"] if is_query else tok_info["document_prefix"]
    full_text = prefix + text
    
    # Encode with HuggingFace tokenizer
    if is_query and tok_info.get("attend_to_expansion_tokens", True):
        # ColBERT queries are padded to query_length
        inputs = tokenizer(
            full_text,
            padding="max_length",
            max_length=tok_info["query_length"],
            truncation=True,
            return_tensors="pt"
        )
    else:
        # Documents are padded normally or dynamically chunked,
        # and queries if attend_to_expansion_tokens is False
        inputs = tokenizer(
            full_text,
            padding=False,
            max_length=tok_info["document_length"] if not is_query else tok_info["query_length"],
            truncation=True,
            return_tensors="pt"
        )
        
    with torch.no_grad():
        outputs = model(**inputs)
        # last_hidden_state shape: [batch_size=1, seq_len, hidden_size]
        last_hidden = outputs.last_hidden_state
        
        # Apply projection: projected = last_hidden @ W.T + bias
        # proj_weight shape in GGUF is: [out_features, in_features]
        projected = F.linear(last_hidden, proj_weight, proj_bias)
        
        # Apply L2-normalization on final dimension
        embeddings = F.normalize(projected, p=2, dim=-1).squeeze(0)
        
        # Apply skiplist mask for documents (removing punctuation)
        if not is_query:
            skiplist_ids = [tokenizer.convert_tokens_to_ids(w) for w in tok_info.get("skiplist_words", [])]
            # Strip out unk tokens or None from resolving
            skiplist_ids = [t for t in skiplist_ids if t is not None and t != tokenizer.unk_token_id]
            
            input_ids = inputs["input_ids"].squeeze(0)
            mask = torch.ones_like(input_ids, dtype=torch.bool)
            for token_id in skiplist_ids:
                mask = mask & (input_ids != token_id)
            if "attention_mask" in inputs:
                mask = mask & inputs["attention_mask"].squeeze(0).bool()
                
            embeddings = embeddings[mask]
        
    return embeddings.cpu().numpy()


def validate_against_golden(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    tok_info: Dict[str, Any],
    proj_weight: torch.Tensor,
    proj_bias: Optional[torch.Tensor],
    fixtures_dir: Path,
    tolerance: float
) -> bool:
    """Validate GGUF-based PyTorch inference outputs against pylate_golden files."""
    json_path = fixtures_dir / "pylate_golden.json"
    npz_path = fixtures_dir / "pylate_golden.npz"
    
    if not json_path.exists() or not npz_path.exists():
        logger.error(f"Golden fixtures not found in '{fixtures_dir}'. Skipping golden validation.")
        return False
        
    with open(json_path, "r") as f:
        metadata = json.load(f)
        
    golden_data = np.load(npz_path)
    success = True
    
    logger.info(f"Comparing GGUF inference outputs against golden vectors ({metadata.get('model_name')})")
    
    # 1. Validate Queries
    for q_item in metadata.get("queries", []):
        text = q_item["text"]
        tensor_key = q_item["tensor_key"]
        golden_emb = golden_data[tensor_key]
        
        # Run GGUF model inference
        gguf_emb = run_inference(model, tokenizer, tok_info, text, is_query=True, proj_weight=proj_weight, proj_bias=proj_bias)
        
        # Calculate error metrics
        mae = np.mean(np.abs(gguf_emb - golden_emb))
        max_ae = np.max(np.abs(gguf_emb - golden_emb))
        
        status = "PASSED" if mae < tolerance else "FAILED"
        logger.info(f"Query index {q_item['index']}: '{text[:30]}...' -> Shape: {gguf_emb.shape} vs Golden: {golden_emb.shape}")
        logger.info(f"  MAE: {mae:.6f}, Max AE: {max_ae:.6f} -> Status: {status}")
        
        if mae >= tolerance:
            success = False
            
    # 2. Validate Documents
    for d_item in metadata.get("documents", []):
        text = d_item["text"]
        tensor_key = d_item["tensor_key"]
        golden_emb = golden_data[tensor_key]
        
        # Run GGUF model inference
        gguf_emb = run_inference(model, tokenizer, tok_info, text, is_query=False, proj_weight=proj_weight, proj_bias=proj_bias)
        
        # Calculate error metrics
        mae = np.mean(np.abs(gguf_emb - golden_emb))
        max_ae = np.max(np.abs(gguf_emb - golden_emb))
        
        status = "PASSED" if mae < tolerance else "FAILED"
        logger.info(f"Doc index {d_item['index']}: '{text[:30]}...' -> Shape: {gguf_emb.shape} vs Golden: {golden_emb.shape}")
        logger.info(f"  MAE: {mae:.6f}, Max AE: {max_ae:.6f} -> Status: {status}")
        
        if mae >= tolerance:
            success = False
            
    return success


def validate_against_hf_direct(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    tok_info: Dict[str, Any],
    proj_weight: torch.Tensor,
    proj_bias: Optional[torch.Tensor],
    hf_model_path: str,
    tolerance: float
) -> bool:
    """Validate GGUF-based model inference outputs directly against loaded HF model inference outputs."""
    logger.info(f"Loading reference Hugging Face model from: '{hf_model_path}'")
    
    hf_proj_w = None
    hf_proj_b = None
    
    try:
        from pylate import models
        hf_pylate = models.ColBERT(model_name_or_path=hf_model_path, device="cpu")
    except Exception as e:
        logger.error(f"Failed to load PyLate reference model: {e}")
        logger.info("Attempting to load raw Transformers model instead...")
        try:
            hf_pylate = AutoModel.from_pretrained(hf_model_path)
            hf_pylate.eval()
            
            # Map the local safetensors keys if they are split (for fake models or SentenceTransformers models)
            hf_model_dir = Path(hf_model_path)
            if hf_model_dir.exists():
                backbone_files = list(hf_model_dir.glob("model*.safetensors"))
                if backbone_files:
                    from safetensors.torch import load_file
                    orig_backbone = load_file(backbone_files[0])
                    
                    state_dict = {}
                    
                    def get_tensor(key: str) -> Optional[torch.Tensor]:
                        for prefix in ["", "model."]:
                            full = prefix + key
                            if full in orig_backbone:
                                return orig_backbone[full]
                        return None
                        
                    def copy_if_present(gguf_k: str, model_k: str):
                        val = get_tensor(gguf_k)
                        if val is not None:
                            state_dict[model_k] = val

                    copy_if_present("embeddings.tok_embeddings.weight", "embeddings.tok_embeddings.weight")
                    copy_if_present("embeddings.norm.weight", "embeddings.norm.weight")
                    copy_if_present("embeddings.norm.bias", "embeddings.norm.bias")
                    
                    num_layers = hf_pylate.config.num_hidden_layers
                    for i in range(num_layers):
                        copy_if_present(f"layers.{i}.attn.Wqkv.weight", f"layers.{i}.attn.Wqkv.weight")
                        copy_if_present(f"layers.{i}.attn.Wqkv.bias", f"layers.{i}.attn.Wqkv.bias")
                        copy_if_present(f"layers.{i}.attn.out_proj.weight", f"layers.{i}.attn.Wo.weight")
                        copy_if_present(f"layers.{i}.attn.out_proj.bias", f"layers.{i}.attn.Wo.bias")
                        
                        if i > 0:
                            copy_if_present(f"layers.{i}.attn.norm.weight", f"layers.{i}.attn_norm.weight")
                            copy_if_present(f"layers.{i}.attn.norm.bias", f"layers.{i}.attn_norm.bias")
                            
                        copy_if_present(f"layers.{i}.mlp.norm.weight", f"layers.{i}.mlp_norm.weight")
                        copy_if_present(f"layers.{i}.mlp.norm.bias", f"layers.{i}.mlp_norm.bias")
                        
                        wi_0_w = get_tensor(f"layers.{i}.mlp.wi_0.weight")
                        wi_1_w = get_tensor(f"layers.{i}.mlp.wi_1.weight")
                        if wi_0_w is not None and wi_1_w is not None:
                            state_dict[f"layers.{i}.mlp.Wi.weight"] = torch.cat([wi_0_w, wi_1_w], dim=0)
                            
                        wi_0_b = get_tensor(f"layers.{i}.mlp.wi_0.bias")
                        wi_1_b = get_tensor(f"layers.{i}.mlp.wi_1.bias")
                        if wi_0_b is not None and wi_1_b is not None:
                            state_dict[f"layers.{i}.mlp.Wi.bias"] = torch.cat([wi_0_b, wi_1_b], dim=0)
                            
                        copy_if_present(f"layers.{i}.mlp.wo.weight", f"layers.{i}.mlp.Wo.weight")
                        copy_if_present(f"layers.{i}.mlp.wo.bias", f"layers.{i}.mlp.Wo.bias")
                        
                    copy_if_present("norm.weight", "final_norm.weight")
                    copy_if_present("norm.bias", "final_norm.bias")
                    
                    hf_pylate.load_state_dict(state_dict, strict=False)
                    logger.info("Aligned and loaded split safetensors weights into raw AutoModel fallback")

            # Load projection layer from the local Dense directory if it exists
            if hf_model_dir.exists():
                dense_dirs = list(hf_model_dir.glob("*Dense")) or list(hf_model_dir.glob("1_Dense"))
                if dense_dirs:
                    dense_dir = dense_dirs[0]
                    dense_files = list(dense_dir.glob("*.safetensors"))
                    if dense_files:
                        from safetensors.torch import load_file
                        dense_weights = load_file(dense_files[0])
                        hf_proj_w = next((v for k, v in dense_weights.items() if "weight" in k), None)
                        hf_proj_b = next((v for k, v in dense_weights.items() if "bias" in k), None)
                        logger.info(f"Loaded reference projection layer from {dense_files[0].name}")
            
            if hf_proj_w is None:
                # Fallback: search model modules if no local Dense directory matches
                linear_layers = [m for m in hf_pylate.modules() if isinstance(m, torch.nn.Linear)]
                if linear_layers:
                    hf_proj_w = linear_layers[-1].weight
                    hf_proj_b = linear_layers[-1].bias
                    
            if hf_proj_w is None:
                logger.error("Could not find any projection weights for reference model.")
                return False
        except Exception as e2:
            logger.error(f"Failed to load standard AutoModel: {e2}")
            return False
            
    # Test cases
    test_queries = [
        "What is the average speed of a swallow?",
        "How do you convert PyTorch to GGUF format?",
        "PostgreSQL pg_colbert extension validation checks."
    ]
    test_docs = [
        "The average speed of an unladen European swallow is roughly 11 meters per second or 24 miles per hour.",
        "To convert model weights to GGUF, read the safetensors file and use gguf.GGUFWriter to pack weights and metadata.",
        "pg_colbert is an extension for PostgreSQL that enables efficient late-interaction vector search utilizing GGML."
    ]
    
    success = True
    
    # Helper to run HF reference
    def run_hf_inference(text: str, is_query: bool) -> np.ndarray:
        if hasattr(hf_pylate, "encode") and not isinstance(hf_pylate, AutoModel.__class__):
            # Use PyLate encode directly
            emb_list = hf_pylate.encode([text], is_query=is_query, batch_size=1)
            return emb_list[0]
        else:
            # Fall back to raw PyTorch evaluation matching the GGUF forward pass
            hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
            prefix = tok_info["query_prefix"] if is_query else tok_info["document_prefix"]
            full_text = prefix + text
            
            pad_query = is_query and tok_info.get("attend_to_expansion_tokens", True)
            
            inputs = hf_tokenizer(
                full_text,
                padding="max_length" if pad_query else False,
                max_length=tok_info["query_length"] if is_query else tok_info["document_length"],
                truncation=True,
                return_tensors="pt"
            )
            with torch.no_grad():
                backbone_out = hf_pylate(**inputs).last_hidden_state
                proj_out = F.linear(backbone_out, hf_proj_w, hf_proj_b)
                emb = F.normalize(proj_out, p=2, dim=-1).squeeze(0)
                
            # Filter fallback reference using the same mask
            if not is_query:
                skiplist_ids = [tokenizer.convert_tokens_to_ids(w) for w in tok_info.get("skiplist_words", [])]
                skiplist_ids = [t for t in skiplist_ids if t is not None and t != tokenizer.unk_token_id]
                
                input_ids = inputs["input_ids"].squeeze(0)
                mask = torch.ones_like(input_ids, dtype=torch.bool)
                for token_id in skiplist_ids:
                    mask = mask & (input_ids != token_id)
                if "attention_mask" in inputs:
                    mask = mask & inputs["attention_mask"].squeeze(0).bool()
                    
                emb = emb[mask]
                
            return emb.cpu().numpy()

    logger.info("Running cross-validation queries...")
    for text in test_queries:
        hf_emb = run_hf_inference(text, is_query=True)
        gguf_emb = run_inference(model, tokenizer, tok_info, text, is_query=True, proj_weight=proj_weight, proj_bias=proj_bias)
        
        mae = np.mean(np.abs(gguf_emb - hf_emb))
        max_ae = np.max(np.abs(gguf_emb - hf_emb))
        
        status = "PASSED" if mae < tolerance else "FAILED"
        logger.info(f"Query '{text[:30]}...' -> Shape: {gguf_emb.shape} vs HF: {hf_emb.shape}")
        logger.info(f"  MAE: {mae:.6f}, Max AE: {max_ae:.6f} -> Status: {status}")
        
        if mae >= tolerance:
            success = False
            
    logger.info("Running cross-validation documents...")
    for text in test_docs:
        hf_emb = run_hf_inference(text, is_query=False)
        gguf_emb = run_inference(model, tokenizer, tok_info, text, is_query=False, proj_weight=proj_weight, proj_bias=proj_bias)
        
        mae = np.mean(np.abs(gguf_emb - hf_emb))
        max_ae = np.max(np.abs(gguf_emb - hf_emb))
        
        status = "PASSED" if mae < tolerance else "FAILED"
        logger.info(f"Doc '{text[:30]}...' -> Shape: {gguf_emb.shape} vs HF: {hf_emb.shape}")
        logger.info(f"  MAE: {mae:.6f}, Max AE: {max_ae:.6f} -> Status: {status}")
        
        if mae >= tolerance:
            success = False
            
    return success


def main() -> None:
    args = parse_args()
    gguf_path = Path(args.gguf_path)
    
    if not gguf_path.exists():
        logger.error(f"GGUF file not found: {gguf_path}")
        sys.exit(1)
        
    logger.info(f"Initializing GGUFReader for: {gguf_path.resolve()}")
    try:
        reader = GGUFReader(gguf_path)
    except Exception as e:
        logger.error(f"Failed to read GGUF file: {e}")
        sys.exit(1)
        
    # Check schema
    schema = decode_gguf_field(reader.fields.get("pg_colbert.gguf_schema"))
    if schema != "pg_colbert_v1":
        logger.error(f"Unsupported schema version: '{schema}'. Only 'pg_colbert_v1' is supported.")
        sys.exit(1)
        
    arch = decode_gguf_field(reader.fields.get("general.architecture"))
    if not arch:
        logger.error("Missing architecture key 'general.architecture' in GGUF metadata.")
        sys.exit(1)
    logger.info(f"Model architecture: '{arch}'")
    
    # 1. Rebuild tokenizer
    logger.info("Rebuilding tokenizer from GGUF metadata...")
    try:
        tokenizer, tok_info = rebuild_tokenizer(reader)
    except Exception as e:
        logger.error(f"Failed to rebuild tokenizer from GGUF metadata: {e}")
        sys.exit(1)
    logger.info(f"Tokenizer loaded successfully. Query prefix: '{tok_info['query_prefix']}', Doc prefix: '{tok_info['document_prefix']}'")
    
    # 2. Rebuild PyTorch model and load GGUF weights
    logger.info("Rebuilding PyTorch model from GGUF metadata and loading weights...")
    try:
        model, proj_weight, proj_bias = rebuild_pytorch_model(reader, arch)
    except Exception as e:
        logger.error(f"Failed to rebuild PyTorch model or load weights: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    # 3. Perform validations
    validation_status = True
    
    # Option A: Directly validate against HF model if path/repo provided
    if args.hf_model:
        validation_status = validate_against_hf_direct(
            model, tokenizer, tok_info, proj_weight, proj_bias, args.hf_model, args.tolerance
        )
    # Option B: Otherwise, validate against local pylate_golden fixtures
    else:
        fixtures_dir = Path(args.fixtures_dir)
        validation_status = validate_against_golden(
            model, tokenizer, tok_info, proj_weight, proj_bias, fixtures_dir, args.tolerance
        )
        
    if validation_status:
        print("\n=======================================================")
        print("GGUF MODEL VALIDATION STATUS: SUCCESS")
        print("Embedding outputs are numerically equivalent to original.")
        print("=======================================================")
        sys.exit(0)
    else:
        print("\n=======================================================")
        print("GGUF MODEL VALIDATION STATUS: FAILED")
        print("Embedding outputs differ from reference (exceeded tolerance).")
        print("=======================================================")
        sys.exit(1)


if __name__ == "__main__":
    main()
