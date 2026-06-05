#!/usr/bin/env python3
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

@dataclass
class SpecialTokensProfile:
    cls_token_id: Optional[int]
    sep_token_id: Optional[int]
    pad_token_id: Optional[int]
    mask_token_id: Optional[int]
    q_token_id: Optional[int]
    d_token_id: Optional[int]

@dataclass
class PrefixTokenIdsProfile:
    query: List[int]
    document: List[int]

@dataclass
class TokenizerProfile:
    source: str  # "llama" | "hf_json" | "canonical_ggml"
    tokenizer_model: str
    tokenizer_json_sha256: Optional[str]
    special_tokens: SpecialTokensProfile
    prefix_token_ids: PrefixTokenIdsProfile

@dataclass
class QueryProfile:
    prefix: str
    max_length: int
    pad_to: Optional[int]
    pad_token_id: Optional[int]
    pad_token: Optional[str]
    attend_to_expansion_tokens: bool
    retain_policy: str
    output_policy: str
    token_type_id: Optional[int]

@dataclass
class DocumentProfile:
    prefix: str
    max_length: int
    pad_to: Optional[int]
    retain_policy: str
    skiplist_words: List[str]
    skiplist_token_ids: List[int]
    token_type_id: Optional[int]

@dataclass
class ProjectionModule:
    type: str
    in_features: int
    out_features: int
    bias: bool

@dataclass
class ProjectionProfile:
    kind: str  # "identity" | "dense" | "module_chain"
    input_dim: int
    output_dim: int
    modules: List[ProjectionModule]
    normalize_after: bool

@dataclass
class CompatibilityProfile:
    llama_cpp_loadable: bool
    requires_profile: bool
    strict_pylate_profile: bool
    known_limitations: List[str]

@dataclass
class ColbertProfile:
    schema: str  # must be "pg_colbert_profile_v1"
    source_model_id: str
    source_revision: str
    converter_version: str
    backbone_family: str
    colbert_family: str
    similarity: str
    output_dim: int
    normalize: bool
    tokenizer: TokenizerProfile
    query: QueryProfile
    document: DocumentProfile
    projection: ProjectionProfile
    compatibility: CompatibilityProfile

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ColbertProfile":
        st = SpecialTokensProfile(**d["tokenizer"]["special_tokens"])
        prefix = PrefixTokenIdsProfile(**d["tokenizer"]["prefix_token_ids"])
        tokenizer = TokenizerProfile(
            source=d["tokenizer"]["source"],
            tokenizer_model=d["tokenizer"]["tokenizer_model"],
            tokenizer_json_sha256=d["tokenizer"].get("tokenizer_json_sha256"),
            special_tokens=st,
            prefix_token_ids=prefix
        )
        query = QueryProfile(**d["query"])
        doc = DocumentProfile(**d["document"])
        
        proj_modules = [ProjectionModule(**m) for m in d["projection"].get("modules", [])]
        projection = ProjectionProfile(
            kind=d["projection"]["kind"],
            input_dim=d["projection"]["input_dim"],
            output_dim=d["projection"]["output_dim"],
            modules=proj_modules,
            normalize_after=d["projection"].get("normalize_after", False)
        )
        
        comp = CompatibilityProfile(**d["compatibility"])
        
        return cls(
            schema=d["schema"],
            source_model_id=d["source_model_id"],
            source_revision=d["source_revision"],
            converter_version=d["converter_version"],
            backbone_family=d["backbone_family"],
            colbert_family=d["colbert_family"],
            similarity=d["similarity"],
            output_dim=d["output_dim"],
            normalize=d["normalize"],
            tokenizer=tokenizer,
            query=query,
            document=doc,
            projection=projection,
            compatibility=comp
        )


def validate_profile(profile: ColbertProfile) -> None:
    """Validate ColBERT profile parameters against the specification schema."""
    if profile.schema != "pg_colbert_profile_v1":
        raise ValueError(f"Invalid schema: expected 'pg_colbert_profile_v1', got '{profile.schema}'")
        
    if profile.output_dim <= 0:
        raise ValueError(f"output_dim must be positive, got {profile.output_dim}")
        
    if profile.query.max_length <= 0:
        raise ValueError(f"query.max_length must be positive, got {profile.query.max_length}")
        
    if profile.document.max_length <= 0:
        raise ValueError(f"document.max_length must be positive, got {profile.document.max_length}")
        
    if profile.projection.kind != "identity":
        if profile.projection.output_dim != profile.output_dim:
            raise ValueError(
                f"projection.output_dim ({profile.projection.output_dim}) must match top-level output_dim ({profile.output_dim}) "
                f"when projection.kind is '{profile.projection.kind}'"
            )
            
    # Check special token IDs
    st = profile.tokenizer.special_tokens
    for key in ['cls_token_id', 'sep_token_id', 'pad_token_id', 'mask_token_id', 'q_token_id', 'd_token_id']:
        val = getattr(st, key)
        if val is not None and not isinstance(val, int):
            raise TypeError(f"Special token ID '{key}' must be integer or None, got {type(val)}")
            
    # Check skiplist_token_ids
    for idx, token_id in enumerate(profile.document.skiplist_token_ids):
        if not isinstance(token_id, int):
            raise TypeError(f"skiplist_token_ids element at index {idx} must be integer, got {type(token_id)}")
        if token_id < 0:
            raise ValueError(f"skiplist_token_ids element at index {idx} must be non-negative, got {token_id}")


def write_profile_sidecar(profile: ColbertProfile, gguf_path: Union[str, Path]) -> Path:
    """Validate profile and write it to <model>.gguf.colbert_profile.json sidecar file."""
    validate_profile(profile)
    sidecar_path = Path(str(gguf_path) + ".colbert_profile.json")
    with open(sidecar_path, "w", encoding="utf-8") as f:
        f.write(profile.to_json())
    return sidecar_path


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
            "attn.Wo.weight": "attn_out.weight",
            "attn.Wo.bias": "attn_out.bias",
            "attn.norm.weight": "attn_norm.weight",
            "attn.norm.bias": "attn_norm.bias",
            "attn_norm.weight": "attn_norm.weight",
            "attn_norm.bias": "attn_norm.bias",
            
            "mlp.wi_0.weight": "ffn_gate.weight",  # wi_0 is gate projection
            "mlp.wi_0.bias": "ffn_gate.bias",
            "mlp.wi_1.weight": "ffn_up.weight",    # wi_1 is up projection
            "mlp.wi_1.bias": "ffn_up.bias",
            "mlp.wo.weight": "ffn_down.weight",    # wo is down projection
            "mlp.wo.bias": "ffn_down.bias",
            "mlp.Wo.weight": "ffn_down.weight",    # Wo is down projection
            "mlp.Wo.bias": "ffn_down.bias",
            "mlp.norm.weight": "ffn_norm.weight",
            "mlp.norm.bias": "ffn_norm.bias",
            "mlp_norm.weight": "ffn_norm.weight",
            "mlp_norm.bias": "ffn_norm.bias",
            
            "norm.weight": "output_norm.weight",   # Final output norm
            "norm.bias": "output_norm.bias",
            "final_norm.weight": "output_norm.weight",
            "final_norm.bias": "output_norm.bias",
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


def canonicalize_tensor_name(hf_name: str, arch: str) -> Optional[str]:
    """
    Map a Hugging Face tensor name (with or without 'hf.' prefix)
    to a standard llama.cpp name.
    """
    clean_name = hf_name
    if clean_name.startswith("hf."):
        clean_name = clean_name[3:]
    if clean_name.startswith("model."):
        clean_name = clean_name[6:]
    elif clean_name.startswith("bert."):
        clean_name = clean_name[5:]

    import re
    layer_match = re.search(r"layers?\.(\d+)\.(.+)", clean_name)
    t_map = get_llama_tensor_map(arch)
    
    if layer_match:
        layer_idx = int(layer_match.group(1))
        suffix = layer_match.group(2)
        if suffix in t_map:
            return f"blk.{layer_idx}.{t_map[suffix]}"
    else:
        if clean_name in t_map:
            return t_map[clean_name]
            
    return None

