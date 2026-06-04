import json
import tempfile
from pathlib import Path
import pytest
import torch
from safetensors.torch import save_file
from gguf import GGUFReader

# Import functions/classes from the converter script to test them directly
import sys
sys.path.append(str(Path(__file__).parent.parent))
from tools.convert_colbert_hf_to_gguf import (
    load_modules_json,
    parse_dense_config,
    parse_backbone_config,
    load_tokenizer_info,
    generate_tensor_map
)


@pytest.fixture
def fake_colbert_model_dir():
    """Create a temporary directory structure for a small fake ColBERT model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # 1. config.json (backbone)
        config = {
            "model_type": "modernbert",
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "max_position_embeddings": 512,
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-5,
            "local_attention": 64,
            "global_attn_every_n_layers": 1
        }
        with open(tmp_path / "config.json", "w") as f:
            json.dump(config, f)
            
        # 2. modules.json
        modules = [
            {
                "type": "sentence_transformers.models.Transformer",
                "path": "",
                "kwargs": {"max_seq_length": 512}
            },
            {
                "type": "pylate.models.Dense.Dense",
                "path": "1_Dense",
                "kwargs": {
                    "in_features": 256,
                    "out_features": 128,
                    "bias": False
                }
            }
        ]
        with open(tmp_path / "modules.json", "w") as f:
            json.dump(modules, f)
            
        # 2.b. config_sentence_transformers.json
        st_config = {
            "similarity_fn_name": "MaxSim",
            "query_prefix": "[Q] ",
            "document_prefix": "[D] ",
            "query_length": 32,
            "document_length": 300,
            "attend_to_expansion_tokens": False,
            "skiplist_words": [",", ".", "!"]
        }
        with open(tmp_path / "config_sentence_transformers.json", "w") as f:
            json.dump(st_config, f)
            
        # 3. 1_Dense/config.json
        dense_dir = tmp_path / "1_Dense"
        dense_dir.mkdir()
        dense_config = {
            "in_features": 256,
            "out_features": 128,
            "bias": False,
            "activation_function": "torch.nn.modules.linear.Identity"
        }
        with open(dense_dir / "config.json", "w") as f:
            json.dump(dense_config, f)
            
        # 4. Tokenizer files
        tokenizer_data = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [
                {"id": 30000, "content": "[Q]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
                {"id": 30001, "content": "[D]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}
            ],
            "model": {
                "type": "WordPiece",
                "vocab": {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[Q]": 30000, "[D]": 30001, "mars": 4},
                "unk_token": "[UNK]",
                "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 100
            }
        }
        with open(tmp_path / "tokenizer.json", "w") as f:
            json.dump(tokenizer_data, f)
            
        tokenizer_config = {
            "tokenizer_class": "ModernBertTokenizer",
            "model_max_length": 512,
            "clean_up_tokenization_spaces": True,
            "query_prefix": "[Q] ",
            "doc_prefix": "[D] "
        }
        with open(tmp_path / "tokenizer_config.json", "w") as f:
            json.dump(tokenizer_config, f)
            
        special_tokens = {
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]"
        }
        with open(tmp_path / "special_tokens_map.json", "w") as f:
            json.dump(special_tokens, f)

        # 5. safetensors (backbone and projection weights)
        # Small weights for testing
        backbone_tensors = {
            "embeddings.tok_embeddings.weight": torch.randn(10, 256),
            "embeddings.norm.weight": torch.randn(256),
            "layers.0.attn.Wqkv.weight": torch.randn(768, 256),
            "layers.0.attn.out_proj.weight": torch.randn(256, 256),
            "layers.0.attn.norm.weight": torch.randn(256),
            "layers.0.mlp.wi_0.weight": torch.randn(512, 256),
            "layers.0.mlp.wi_1.weight": torch.randn(512, 256),
            "layers.0.mlp.wo.weight": torch.randn(256, 512),
            "layers.0.mlp.norm.weight": torch.randn(256),
            "layers.1.attn.Wqkv.weight": torch.randn(768, 256),
            "layers.1.attn.out_proj.weight": torch.randn(256, 256),
            "layers.1.attn.norm.weight": torch.randn(256),
            "layers.1.mlp.wi_0.weight": torch.randn(512, 256),
            "layers.1.mlp.wi_1.weight": torch.randn(512, 256),
            "layers.1.mlp.wo.weight": torch.randn(256, 512),
            "layers.1.mlp.norm.weight": torch.randn(256),
            "norm.weight": torch.randn(256)
        }
        save_file(backbone_tensors, tmp_path / "model.safetensors")

        dense_tensors = {
            "linear.weight": torch.randn(128, 256)
        }
        save_file(dense_tensors, dense_dir / "model.safetensors")

        yield tmp_path


def test_config_parsing(fake_colbert_model_dir):
    """Test that all configuration files are parsed and validated correctly."""
    modules = load_modules_json(fake_colbert_model_dir)
    assert len(modules) == 2
    assert "Transformer" in modules[0]["type"]
    
    dense_cfg, dense_path = parse_dense_config(fake_colbert_model_dir, modules)
    assert dense_cfg.in_features == 256
    assert dense_cfg.out_features == 128
    assert dense_cfg.bias is False
    assert dense_path == fake_colbert_model_dir / "1_Dense"
    
    backbone_cfg = parse_backbone_config(fake_colbert_model_dir)
    assert backbone_cfg.model_type == "modernbert"
    assert backbone_cfg.hidden_size == 256
    assert backbone_cfg.intermediate_size == 512
    assert backbone_cfg.num_hidden_layers == 2
    
    tok_info = load_tokenizer_info(fake_colbert_model_dir)
    assert tok_info["pad_token_id"] == 0
    assert tok_info["cls_token_id"] == 2
    assert tok_info["sep_token_id"] == 3
    # Check prefixes
    assert tok_info["query_prefix"] == "[Q] "
    assert tok_info["document_prefix"] == "[D] "
    # tokenizer encode results should not crash
    assert isinstance(tok_info["query_prefix_ids"], list)


def test_gguf_writing(fake_colbert_model_dir):
    """Test full conversion to GGUF using fake model and verify with GGUFReader."""
    import subprocess
    import sys
    
    outfile = fake_colbert_model_dir / "model.gguf"
    
    # Run the converter script directly as a subprocess to test execution integration
    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "tools/convert_colbert_hf_to_gguf.py"),
        "--model-dir", str(fake_colbert_model_dir),
        "--outfile", str(outfile),
        "--outtype", "f32",
        "--verbose"
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Converter failed: {res.stderr}\nOutput: {res.stdout}"
    
    # Verify the generated GGUF file
    assert outfile.exists()
    reader = GGUFReader(outfile)
    
    def decode_field(field):
        if field is None:
            return None
        main_type = field.types[0]
        type_name = getattr(main_type, "name", str(main_type))
        if type_name == "ARRAY" or main_type == 9:
            sub_type = field.types[1] if len(field.types) > 1 else None
            sub_type_name = getattr(sub_type, "name", str(sub_type)) if sub_type else ""
            arr_val = []
            for idx in field.data:
                part = field.parts[idx]
                if sub_type_name == "STRING" or sub_type == 8:
                    arr_val.append(bytes(part).decode("utf-8"))
                else:
                    arr_val.append(part.item() if hasattr(part, "item") else part[0])
            return arr_val
        elif type_name == "STRING" or main_type == 8:
            return bytes(field.parts[field.data[0]]).decode("utf-8")
        else:
            part = field.parts[field.data[0]]
            return part.item() if hasattr(part, "item") else part[0]

    # Verify key metadata
    assert decode_field(reader.fields.get("pg_colbert.gguf_schema")) == "pg_colbert_v1"
    assert decode_field(reader.fields.get("general.architecture")) == "modernbert"
    assert decode_field(reader.fields.get("colbert.embedding_dim")) == 128
    assert decode_field(reader.fields.get("colbert.projection.in_features")) == 256
    assert decode_field(reader.fields.get("colbert.projection.out_features")) == 128
    assert decode_field(reader.fields.get("colbert.query_prefix")) == "[Q] "
    assert decode_field(reader.fields.get("colbert.document_prefix")) == "[D] "
    assert decode_field(reader.fields.get("colbert.similarity_fn_name")) == "MaxSim"
    assert decode_field(reader.fields.get("colbert.query_length")) == 32
    assert decode_field(reader.fields.get("colbert.document_length")) == 300
    assert decode_field(reader.fields.get("colbert.attend_to_expansion_tokens")) is False
    assert decode_field(reader.fields.get("colbert.skiplist_words")) == [",", ".", "!"]
    
    # Verify tokenizer is embedded
    assert "tokenizer.huggingface.json" in reader.fields
    tok_json_val = decode_field(reader.fields["tokenizer.huggingface.json"])
    assert tok_json_val is not None
    assert len(tok_json_val) > 0
    
    # Verify projection tensor
    tensor_names = [t.name for t in reader.tensors]
    assert "colbert.proj.weight" in tensor_names
    
    proj_tensor = next(t for t in reader.tensors if t.name == "colbert.proj.weight")
    # GGUF tensor shape is stored in reverse order or standard order. Let's check dimensions.
    assert list(proj_tensor.shape) == [256, 128] # gguf-py shapes are stored transposed (column-major style: width, height)
