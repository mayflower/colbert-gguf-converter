import json
from pathlib import Path
import pytest

import sys
sys.path.append(str(Path(__file__).parent.parent))
from tools.convert_colbert_hf_to_gguf import generate_tensor_map


def test_modernbert_tensor_map():
    """Test generating a tensor map for a ModernBERT model."""
    safetensors_keys = [
        "embeddings.tok_embeddings.weight",
        "embeddings.norm.weight",
        "embeddings.norm.bias",
        "layers.0.attn.Wqkv.weight",
        "layers.0.attn.Wqkv.bias",
        "layers.0.attn.out_proj.weight",
        "layers.0.attn.norm.weight",
        "layers.0.mlp.wi_0.weight",
        "layers.0.mlp.wi_1.weight",
        "layers.0.mlp.wo.weight",
        "layers.0.mlp.norm.weight",
        "layers.1.attn.Wqkv.weight",
        "layers.1.attn.out_proj.weight",
        "layers.1.attn.norm.weight",
        "layers.1.mlp.wi_0.weight",
        "layers.1.mlp.wi_1.weight",
        "layers.1.mlp.wo.weight",
        "layers.1.mlp.norm.weight",
        "norm.weight"
    ]
    
    num_layers = 2
    tensor_map = generate_tensor_map("modernbert", safetensors_keys, num_layers)
    
    assert tensor_map["schema"] == "pg_colbert_v1"
    assert tensor_map["stored_name_style"] == "hf_original"
    
    # Check projection weights mapping is preset
    assert tensor_map["projection"]["weight"] == "colbert.proj.weight"
    
    # Check embeddings
    assert tensor_map["backbone"]["embeddings"]["word_embeddings"] == "hf.embeddings.tok_embeddings.weight"
    assert tensor_map["backbone"]["embeddings"]["norm_weight"] == "hf.embeddings.norm.weight"
    
    # Check layers
    assert len(tensor_map["backbone"]["layers"]) == num_layers
    assert tensor_map["backbone"]["layers"][0]["attn_qkv_weight"] == "hf.layers.0.attn.Wqkv.weight"
    assert tensor_map["backbone"]["layers"][0]["attn_qkv_bias"] == "hf.layers.0.attn.Wqkv.bias"
    assert tensor_map["backbone"]["layers"][0]["attn_out_weight"] == "hf.layers.0.attn.out_proj.weight"
    # Layer 0 has attn_qkv_bias but Layer 1 does not in this sample keys list
    assert "attn_qkv_bias" not in tensor_map["backbone"]["layers"][1]
    
    # Check final norm
    assert tensor_map["backbone"]["final_norm"]["weight"] == "hf.norm.weight"


def test_bert_tensor_map():
    """Test generating a tensor map for a BERT model."""
    safetensors_keys = [
        "bert.embeddings.word_embeddings.weight",
        "bert.embeddings.position_embeddings.weight",
        "bert.embeddings.token_type_embeddings.weight",
        "bert.embeddings.LayerNorm.weight",
        "bert.embeddings.LayerNorm.bias",
        "bert.encoder.layer.0.attention.self.query.weight",
        "bert.encoder.layer.0.attention.self.query.bias",
        "bert.encoder.layer.0.attention.self.key.weight",
        "bert.encoder.layer.0.attention.self.key.bias",
        "bert.encoder.layer.0.attention.self.value.weight",
        "bert.encoder.layer.0.attention.self.value.bias",
        "bert.encoder.layer.0.attention.output.dense.weight",
        "bert.encoder.layer.0.attention.output.dense.bias",
        "bert.encoder.layer.0.attention.output.LayerNorm.weight",
        "bert.encoder.layer.0.attention.output.LayerNorm.bias",
        "bert.encoder.layer.0.intermediate.dense.weight",
        "bert.encoder.layer.0.intermediate.dense.bias",
        "bert.encoder.layer.0.output.dense.weight",
        "bert.encoder.layer.0.output.dense.bias",
        "bert.encoder.layer.0.output.LayerNorm.weight",
        "bert.encoder.layer.0.output.LayerNorm.bias",
    ]
    
    num_layers = 1
    tensor_map = generate_tensor_map("bert", safetensors_keys, num_layers)
    
    assert tensor_map["schema"] == "pg_colbert_v1"
    
    # Check embeddings
    assert tensor_map["backbone"]["embeddings"]["word_embeddings"] == "hf.bert.embeddings.word_embeddings.weight"
    assert tensor_map["backbone"]["embeddings"]["position_embeddings"] == "hf.bert.embeddings.position_embeddings.weight"
    
    # Check layers
    assert len(tensor_map["backbone"]["layers"]) == num_layers
    assert tensor_map["backbone"]["layers"][0]["attn_query_weight"] == "hf.bert.encoder.layer.0.attention.self.query.weight"
    assert tensor_map["backbone"]["layers"][0]["mlp_intermediate_weight"] == "hf.bert.encoder.layer.0.intermediate.dense.weight"
