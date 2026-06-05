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
    load_sentence_transformers_config,
    load_tokenizer_profile,
    load_query_document_profile,
    load_projection_profile,
    build_colbert_profile,
    generate_tensor_map
)





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
    
    st_config = load_sentence_transformers_config(fake_colbert_model_dir)
    assert st_config.get("query_prefix") == "[Q] "
    
    tokenizer_prof, tokenizer, tok_info = load_tokenizer_profile(fake_colbert_model_dir, st_config)
    assert tok_info["pad_token_id"] == 0
    assert tok_info["cls_token_id"] == 2
    assert tok_info["sep_token_id"] == 3
    # Check prefixes
    assert tok_info["query_prefix"] == "[Q] "
    assert tok_info["document_prefix"] == "[D] "
    assert isinstance(tok_info["query_prefix_ids"], list)

    query_prof, doc_prof, limitations = load_query_document_profile(fake_colbert_model_dir, tokenizer, st_config)
    assert query_prof.prefix == "[Q] "
    assert query_prof.max_length == 32
    assert doc_prof.prefix == "[D] "
    assert doc_prof.max_length == 300
    assert len(doc_prof.skiplist_words) == 3
    assert len(doc_prof.skiplist_token_ids) == 3

    proj_prof, dense_cfg, dense_path = load_projection_profile(fake_colbert_model_dir, modules)
    assert proj_prof.input_dim == 256
    assert proj_prof.output_dim == 128
    
    profile = build_colbert_profile(
        source_model_id="test-model",
        source_revision="main",
        backbone_family="modernbert",
        similarity_fn="cosine",
        tokenizer_profile=tokenizer_prof,
        query_profile=query_prof,
        doc_profile=doc_prof,
        projection_profile=proj_prof,
        known_limitations=limitations
    )
    from tools.colbert_profile import validate_profile
    validate_profile(profile)


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

    # Verify profile is embedded in GGUF
    profile_json_val = decode_field(reader.fields.get("pg_colbert.profile_json"))
    assert profile_json_val is not None
    profile_data = json.loads(profile_json_val)
    assert profile_data["schema"] == "pg_colbert_profile_v1"
    assert profile_data["output_dim"] == 128
    
    # Verify sidecar profile file was created
    sidecar_file = fake_colbert_model_dir / "model.gguf.colbert_profile.json"
    assert sidecar_file.exists()
    with open(sidecar_file, "r", encoding="utf-8") as f:
        sidecar_data = json.load(f)
    assert sidecar_data["schema"] == "pg_colbert_profile_v1"
    assert sidecar_data["projection"]["input_dim"] == 256

    # Run tools/inspect_colbert_gguf.py on the converted GGUF file
    inspect_cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "tools/inspect_colbert_gguf.py"),
        str(outfile)
    ]
    inspect_res = subprocess.run(inspect_cmd, capture_output=True, text=True)
    assert inspect_res.returncode == 0, f"Inspector failed: {inspect_res.stderr}\nOutput: {inspect_res.stdout}"
    
    # Verify inspector validates and prints profile summary
    output = inspect_res.stdout
    assert "Validation: SUCCESS" in output
    assert "=== COLBERT PROFILE SUMMARY ===" in output
    assert "schema: pg_colbert_profile_v1" in output
    assert "output_dim: 128" in output
    assert 'query prefix/length/pad_to: prefix="[Q] ", length=32, pad_to=32' in output
    assert 'document prefix/length: prefix="[D] ", length=300' in output
    assert "skiplist token count: 3" in output
    assert "projection kind/modules: kind=dense, modules=[linear(256 -> 128, bias=False)]" in output
    assert "compatibility flags: llama_cpp_loadable=True, requires_profile=True, strict_pylate_profile=True" in output


def test_gguf_writing_llama_cpp(fake_colbert_model_dir):
    """Test target-runtime llama_cpp conversion to GGUF and verify metadata and tensors."""
    import subprocess
    import sys
    
    outfile = fake_colbert_model_dir / "model_llama.gguf"
    
    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "tools/convert_colbert_hf_to_gguf.py"),
        "--model-dir", str(fake_colbert_model_dir),
        "--outfile", str(outfile),
        "--outtype", "f32",
        "--target-runtime", "llama_cpp",
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

    # Verify standard llama.cpp metadata is present
    assert decode_field(reader.fields.get("general.architecture")) == "modernbert"
    
    # Verify tokenizer.ggml.* metadata
    assert decode_field(reader.fields.get("tokenizer.ggml.model")) == "bert"
    tokens = decode_field(reader.fields.get("tokenizer.ggml.tokens"))
    assert tokens is not None
    assert len(tokens) > 0
    assert "[PAD]" in tokens
    assert "[Q]" in tokens
    assert "[D]" in tokens
    
    token_types = decode_field(reader.fields.get("tokenizer.ggml.token_type"))
    assert token_types is not None
    assert len(token_types) == len(tokens)
    
    # Verify pg_colbert.profile_json is embedded
    profile_json_val = decode_field(reader.fields.get("pg_colbert.profile_json"))
    assert profile_json_val is not None
    profile_data = json.loads(profile_json_val)
    assert profile_data["schema"] == "pg_colbert_profile_v1"
    
    # Verify tensor names are canonical llama.cpp names
    tensor_names = [t.name for t in reader.tensors]
    assert "token_embd.weight" in tensor_names
    assert "blk.0.attn_qkv.weight" in tensor_names
    assert "colbert.proj.weight" in tensor_names
    
    # Verify sidecar profile file was created
    sidecar_file = fake_colbert_model_dir / "model_llama.gguf.colbert_profile.json"
    assert sidecar_file.exists()


def test_gguf_writing_both(fake_colbert_model_dir):
    """Test target-runtime both conversion to GGUF and verify both output files exist."""
    import subprocess
    import sys
    
    outfile = fake_colbert_model_dir / "model_both.gguf"
    
    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "tools/convert_colbert_hf_to_gguf.py"),
        "--model-dir", str(fake_colbert_model_dir),
        "--outfile", str(outfile),
        "--outtype", "f32",
        "--target-runtime", "both",
        "--verbose"
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Converter failed: {res.stderr}\nOutput: {res.stdout}"
    
    # Verify both output GGUF files and sidecars exist
    pg_colbert_file = fake_colbert_model_dir / "model_both.pg_colbert.gguf"
    llama_file = fake_colbert_model_dir / "model_both.llama.gguf"
    
    assert pg_colbert_file.exists()
    assert llama_file.exists()
    
    pg_sidecar = fake_colbert_model_dir / "model_both.pg_colbert.gguf.colbert_profile.json"
    llama_sidecar = fake_colbert_model_dir / "model_both.llama.gguf.colbert_profile.json"
    
    assert pg_sidecar.exists()
    assert llama_sidecar.exists()


