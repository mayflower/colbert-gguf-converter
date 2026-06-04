import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from gguf import GGUFReader
from safetensors.torch import load_file

# Add parent directory to path to import tools
sys.path.append(str(Path(__file__).parent.parent))
from tools.validate_colbert_gguf_inference import (
    rebuild_tokenizer,
    rebuild_pytorch_model,
    run_inference
)


def test_gguf_inference_equivalence(fake_colbert_model_dir):
    """Test that GGUF weights produce identical embeddings to original Safetensors."""
    outfile = fake_colbert_model_dir / "model.gguf"
    
    # 1. Run converter
    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "tools/convert_colbert_hf_to_gguf.py"),
        "--model-dir", str(fake_colbert_model_dir),
        "--outfile", str(outfile),
        "--outtype", "f32"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Converter failed: {res.stderr}"
    
    # 2. Reconstruct tokenizer and model from GGUF
    reader = GGUFReader(outfile)
    tokenizer, tok_info = rebuild_tokenizer(reader)
    model, proj_weight, proj_bias = rebuild_pytorch_model(reader, "modernbert")
    
    # 3. Load original weights directly to compare
    orig_backbone = load_file(fake_colbert_model_dir / "model.safetensors")
    orig_proj = load_file(fake_colbert_model_dir / "1_Dense" / "model.safetensors")
    
    # Verify weights are identical by comparing specific layers
    # 1. Embeddings
    np.testing.assert_allclose(
        model.embeddings.tok_embeddings.weight.detach().cpu().numpy(),
        orig_backbone["embeddings.tok_embeddings.weight"].cpu().numpy(),
        rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        model.embeddings.norm.weight.detach().cpu().numpy(),
        orig_backbone["embeddings.norm.weight"].cpu().numpy(),
        rtol=1e-5, atol=1e-5
    )
    
    # 2. Attention query/key/value projection
    np.testing.assert_allclose(
        model.layers[0].attn.Wqkv.weight.detach().cpu().numpy(),
        orig_backbone["layers.0.attn.Wqkv.weight"].cpu().numpy(),
        rtol=1e-5, atol=1e-5
    )
    
    # 3. Attention output projection (mapped to Wo)
    np.testing.assert_allclose(
        model.layers[0].attn.Wo.weight.detach().cpu().numpy(),
        orig_backbone["layers.0.attn.out_proj.weight"].cpu().numpy(),
        rtol=1e-5, atol=1e-5
    )
    
    # 4. MLP (wi_0 and wi_1 concatenated into Wi)
    np.testing.assert_allclose(
        model.layers[0].mlp.Wi.weight[:512].detach().cpu().numpy(),
        orig_backbone["layers.0.mlp.wi_0.weight"].cpu().numpy(),
        rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        model.layers[0].mlp.Wi.weight[512:].detach().cpu().numpy(),
        orig_backbone["layers.0.mlp.wi_1.weight"].cpu().numpy(),
        rtol=1e-5, atol=1e-5
    )
    
    # 5. MLP Output (mapped to Wo)
    np.testing.assert_allclose(
        model.layers[0].mlp.Wo.weight.detach().cpu().numpy(),
        orig_backbone["layers.0.mlp.wo.weight"].cpu().numpy(),
        rtol=1e-5, atol=1e-5
    )
    
    # 6. Final Norm (mapped to final_norm)
    np.testing.assert_allclose(
        model.final_norm.weight.detach().cpu().numpy(),
        orig_backbone["norm.weight"].cpu().numpy(),
        rtol=1e-5, atol=1e-5
    )
    
    # 7. Dense Projection weight (mapped to proj_weight)
    np.testing.assert_allclose(
        proj_weight.cpu().numpy(),
        orig_proj["linear.weight"].cpu().numpy(),
        rtol=1e-5, atol=1e-5
    )
    
    # 4. Compare inference outputs
    test_text = "Mars is the Red Planet."
    
    # Run GGUF inference
    gguf_emb = run_inference(
        model, tokenizer, tok_info, test_text, is_query=False,
        proj_weight=proj_weight, proj_bias=proj_bias
    )
    
    # Run raw manual PyTorch reference inference using original weights
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    orig_cfg = AutoConfig.from_pretrained(fake_colbert_model_dir)
    orig_tokenizer = AutoTokenizer.from_pretrained(fake_colbert_model_dir)
    orig_model = AutoModel.from_config(orig_cfg)
    
    # Load state dict with correct ModernBERT mapping
    from typing import Optional
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
    
    num_layers = orig_cfg.num_hidden_layers
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
    
    orig_model.load_state_dict(state_dict)
    orig_model.eval()
    
    inputs = orig_tokenizer("[D] " + test_text, return_tensors="pt")
    with torch.no_grad():
        last_hidden = orig_model(**inputs).last_hidden_state
        proj = F.linear(last_hidden, orig_proj["linear.weight"])
        orig_emb = F.normalize(proj, p=2, dim=-1).squeeze(0).cpu().numpy()
        
    # Reconstructed GGUF inference must be numerically identical to original PyTorch reference
    np.testing.assert_allclose(gguf_emb, orig_emb, rtol=1e-5, atol=1e-5)


def test_gguf_validation_script(fake_colbert_model_dir):
    """Test that the validate_colbert_gguf_inference.py CLI script runs successfully."""
    outfile = fake_colbert_model_dir / "model.gguf"
    
    # 1. Run converter
    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "tools/convert_colbert_hf_to_gguf.py"),
        "--model-dir", str(fake_colbert_model_dir),
        "--outfile", str(outfile),
        "--outtype", "f32"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    
    # 2. Run validator tool (since we don't have golden vector matching this fake model, we cross-validate with the HF folder)
    cmd_val = [
        sys.executable,
        str(Path(__file__).parent.parent / "tools/validate_colbert_gguf_inference.py"),
        str(outfile),
        "--hf-model", str(fake_colbert_model_dir),
        "--tolerance", "1e-5"
    ]
    res_val = subprocess.run(cmd_val, capture_output=True, text=True)
    assert res_val.returncode == 0, f"Validator script failed: {res_val.stderr}\n{res_val.stdout}"
    assert "GGUF MODEL VALIDATION STATUS: SUCCESS" in res_val.stdout


def test_llama_cpp_export(fake_colbert_model_dir):
    """Test converting pg_colbert_v1 GGUF to standard llama.cpp-compliant format."""
    infile = fake_colbert_model_dir / "model.gguf"
    outfile = fake_colbert_model_dir / "llama_cpp_model.gguf"
    
    # 1. Convert to pg_colbert GGUF first
    subprocess.run([
        sys.executable,
        str(Path(__file__).parent.parent / "tools/convert_colbert_hf_to_gguf.py"),
        "--model-dir", str(fake_colbert_model_dir),
        "--outfile", str(infile),
        "--outtype", "f32"
    ], check=True)
    
    # 2. Export to llama.cpp format
    cmd_export = [
        sys.executable,
        str(Path(__file__).parent.parent / "tools/export_to_llama_cpp.py"),
        str(infile),
        str(outfile),
        "--verbose"
    ]
    res_export = subprocess.run(cmd_export, capture_output=True, text=True)
    assert res_export.returncode == 0, f"Export script failed: {res_export.stderr}"
    
    # 3. Read llama.cpp GGUF and check structure
    assert outfile.exists()
    reader = GGUFReader(outfile)
    
    # Standard GGUF keys only
    assert "general.architecture" in reader.fields
    assert "pg_colbert.gguf_schema" not in reader.fields
    assert "colbert.model_type" not in reader.fields
    
    # Check that tensors are successfully mapped to llama.cpp standard names
    tensor_names = [t.name for t in reader.tensors]
    
    # Should contain standard blk prefix
    assert "token_embd.weight" in tensor_names
    assert "token_embd_norm.weight" in tensor_names
    assert "blk.0.attn_qkv.weight" in tensor_names
    assert "blk.0.ffn_gate.weight" in tensor_names
    assert "blk.1.attn_out.weight" in tensor_names
    assert "output_norm.weight" in tensor_names
    
    # Should NOT contain pg_colbert prefixes
    assert not any(t.name.startswith("hf.") for t in reader.tensors)
    assert not any("colbert" in t.name for t in reader.tensors)
    
    # Validate numerical correctness of exported weights
    pg_reader = GGUFReader(infile)
    
    # Map token embeddings weight
    pg_emb = next(t for t in pg_reader.tensors if t.name == "hf.embeddings.tok_embeddings.weight")
    cpp_emb = next(t for t in reader.tensors if t.name == "token_embd.weight")
    np.testing.assert_allclose(cpp_emb.data, pg_emb.data)


def test_publisher_with_validation(fake_colbert_model_dir, monkeypatch):
    """Test the complete download -> convert -> validate -> upload publisher workflow with mocked HF Hub uploads."""
    mock_calls = []
    
    class MockHfApi:
        def __init__(self, token=None):
            mock_calls.append("init")
            
        def whoami(self):
            return {"name": "mock-user"}
            
        def create_repo(self, repo_id, repo_type, private=False, exist_ok=True):
            mock_calls.append(("create_repo", repo_id))
            
        def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type):
            mock_calls.append(("upload_file", path_in_repo, repo_id))
            
    # Mock Hugging Face API HfApi
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", MockHfApi)
    
    # Run publish main
    import sys
    old_argv = sys.argv
    sys.argv = [
        "publish_colbert_gguf.py",
        "--model-id", str(fake_colbert_model_dir),
        "--target-repo-id", "mock-user/test-model-gguf",
        "--outtype", "f32"
    ]
    
    from tools.publish_colbert_gguf import main as publish_main
    try:
        publish_main()
    finally:
        sys.argv = old_argv
        
    # Verify sequence of conversion, validation, repository creation, and file uploads
    expected_gguf_name = f"{str(fake_colbert_model_dir).replace('/', '_').replace('-', '_')}.f32.gguf"
    assert "init" in mock_calls
    assert ("create_repo", "mock-user/test-model-gguf") in mock_calls
    assert ("upload_file", expected_gguf_name, "mock-user/test-model-gguf") in mock_calls
    assert ("upload_file", "README.md", "mock-user/test-model-gguf") in mock_calls

