import json
import tempfile
from pathlib import Path
import pytest

from tools.colbert_profile import (
    ColbertProfile,
    TokenizerProfile,
    SpecialTokensProfile,
    PrefixTokenIdsProfile,
    QueryProfile,
    DocumentProfile,
    ProjectionProfile,
    ProjectionModule,
    CompatibilityProfile,
    validate_profile,
    write_profile_sidecar
)

def get_valid_minimal_profile() -> ColbertProfile:
    return ColbertProfile(
        schema="pg_colbert_profile_v1",
        source_model_id="test/model",
        source_revision="main",
        converter_version="1.0.0",
        backbone_family="bert",
        colbert_family="pylate",
        similarity="cosine",
        output_dim=128,
        normalize=True,
        tokenizer=TokenizerProfile(
            source="hf_json",
            tokenizer_model="bert",
            tokenizer_json_sha256=None,
            special_tokens=SpecialTokensProfile(
                cls_token_id=101,
                sep_token_id=102,
                pad_token_id=0,
                mask_token_id=103,
                q_token_id=30522,
                d_token_id=30523
            ),
            prefix_token_ids=PrefixTokenIdsProfile(
                query=[30522],
                document=[30523]
            )
        ),
        query=QueryProfile(
            prefix="[Q] ",
            max_length=32,
            pad_to=32,
            pad_token_id=103,
            pad_token="[MASK]",
            attend_to_expansion_tokens=False,
            retain_policy="all",
            output_policy="all",
            token_type_id=0
        ),
        document=DocumentProfile(
            prefix="[D] ",
            max_length=300,
            pad_to=None,
            retain_policy="mask_and_skiplist",
            skiplist_words=["!"],
            skiplist_token_ids=[106],
            token_type_id=0
        ),
        projection=ProjectionProfile(
            kind="dense",
            input_dim=288,
            output_dim=128,
            modules=[
                ProjectionModule(
                    type="linear",
                    in_features=288,
                    out_features=128,
                    bias=False
                )
            ],
            normalize_after=True
        ),
        compatibility=CompatibilityProfile(
            llama_cpp_loadable=True,
            requires_profile=True,
            strict_pylate_profile=True,
            known_limitations=[]
        )
    )

def test_minimal_valid_profile_serializes():
    profile = get_valid_minimal_profile()
    # Should validate without raising
    validate_profile(profile)
    
    # Serialize to JSON and deserialize back
    json_str = profile.to_json()
    data = json.loads(json_str)
    
    assert data["schema"] == "pg_colbert_profile_v1"
    assert data["tokenizer"]["special_tokens"]["cls_token_id"] == 101

def test_invalid_output_dim_fails():
    profile = get_valid_minimal_profile()
    profile.output_dim = 0
    with pytest.raises(ValueError, match="output_dim must be positive"):
        validate_profile(profile)
        
    profile.output_dim = -5
    with pytest.raises(ValueError, match="output_dim must be positive"):
        validate_profile(profile)

def test_invalid_query_max_length_fails():
    profile = get_valid_minimal_profile()
    profile.query.max_length = 0
    with pytest.raises(ValueError, match="query.max_length must be positive"):
        validate_profile(profile)

def test_invalid_document_max_length_fails():
    profile = get_valid_minimal_profile()
    profile.document.max_length = -10
    with pytest.raises(ValueError, match="document.max_length must be positive"):
        validate_profile(profile)

def test_projection_dim_mismatch_fails():
    profile = get_valid_minimal_profile()
    profile.projection.output_dim = 64  # top-level output_dim is 128
    with pytest.raises(ValueError, match="must match top-level output_dim"):
        validate_profile(profile)

def test_projection_identity_ignores_dim_mismatch():
    profile = get_valid_minimal_profile()
    profile.projection.kind = "identity"
    profile.projection.output_dim = 64  # top-level is 128
    # Should validate because kind is identity
    validate_profile(profile)

def test_skiplist_token_ids_validate():
    profile = get_valid_minimal_profile()
    profile.document.skiplist_token_ids = [-1]
    with pytest.raises(ValueError, match="must be non-negative"):
        validate_profile(profile)
        
    profile.document.skiplist_token_ids = ["not-an-int"]
    with pytest.raises(TypeError, match="must be integer"):
        validate_profile(profile)

def test_special_tokens_type_validate():
    profile = get_valid_minimal_profile()
    profile.tokenizer.special_tokens.cls_token_id = "invalid-type"
    with pytest.raises(TypeError, match="must be integer or None"):
        validate_profile(profile)

def test_write_profile_sidecar():
    profile = get_valid_minimal_profile()
    with tempfile.TemporaryDirectory() as tmpdir:
        gguf_path = Path(tmpdir) / "model.gguf"
        sidecar_path = write_profile_sidecar(profile, gguf_path)
        
        assert sidecar_path.exists()
        assert sidecar_path.name == "model.gguf.colbert_profile.json"
        
        # Load and verify content
        with open(sidecar_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["schema"] == "pg_colbert_profile_v1"
