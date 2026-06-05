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
