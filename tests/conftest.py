import json
import tempfile
from pathlib import Path
import pytest
import torch
from safetensors.torch import save_file

@pytest.fixture
def fake_colbert_model_dir():
    """Create a temporary directory structure for a small fake ColBERT model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # 1. config.json (backbone)
        config = {
            "model_type": "modernbert",
            "vocab_size": 10,
            "pad_token_id": 0,
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
                {"id": 5, "content": "[Q]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
                {"id": 6, "content": "[D]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}
            ],
            "model": {
                "type": "WordPiece",
                "vocab": {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[Q]": 5, "[D]": 6, "mars": 4},
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
