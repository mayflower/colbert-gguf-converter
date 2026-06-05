import json
import subprocess
import sys
from pathlib import Path
import pytest

# Add parent directory to path to import tools
sys.path.append(str(Path(__file__).parent.parent))

def test_verify_parity_success(fake_colbert_model_dir):
    # 1. Create a texts file
    texts_file = fake_colbert_model_dir / "texts.txt"
    with open(texts_file, "w", encoding="utf-8") as f:
        f.write("red planet\n")
        
    outfile = fake_colbert_model_dir / "parity_report.json"
    
    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "tools/verify_pylate_parity.py"),
        "--model-name-or-path", str(fake_colbert_model_dir),
        "--texts-file", str(texts_file),
        "--role", "query",
        "--outfile", str(outfile)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Script failed: {res.stderr}\n{res.stdout}"
    
    assert outfile.exists()
    with open(outfile, "r") as f:
        report = json.load(f)
        
    assert report["profile_valid"] is False
    assert "texts" in report
    assert len(report["texts"]) == 1
    assert report["texts"][0]["text"] == "red planet"
    assert "token_plan" in report["texts"][0]


def test_verify_parity_invalid_profile(fake_colbert_model_dir):
    # 1. Create a texts file
    texts_file = fake_colbert_model_dir / "texts.txt"
    with open(texts_file, "w", encoding="utf-8") as f:
        f.write("red planet\n")
        
    # 2. Create an invalid profile JSON
    invalid_profile_path = fake_colbert_model_dir / "invalid_profile.json"
    invalid_profile = {
        "schema": "invalid_schema_name", # invalid
        "source_model_id": "test",
        "source_revision": "main",
        "converter_version": "1.0",
        "backbone_family": "modernbert",
        "colbert_family": "pylate",
        "similarity": "l2",
        "output_dim": -10, # invalid
        "normalize": True,
        "tokenizer": {
            "source": "hf_json",
            "tokenizer_model": "test",
            "special_tokens": {
                "cls_token_id": 1,
                "sep_token_id": 2,
                "pad_token_id": 3,
                "mask_token_id": 4,
                "q_token_id": 5,
                "d_token_id": 6
            },
            "prefix_token_ids": {
                "query": [1],
                "document": [2]
            }
        },
        "query": {
            "prefix": "[Q] ",
            "max_length": 32,
            "pad_to": 32,
            "pad_token_id": 0,
            "pad_token": "[PAD]",
            "attend_to_expansion_tokens": True,
            "retain_policy": "all",
            "output_policy": "all",
            "token_type_id": 0
        },
        "document": {
            "prefix": "[D] ",
            "max_length": 300,
            "pad_to": 0,
            "retain_policy": "all",
            "skiplist_words": [],
            "skiplist_token_ids": [],
            "token_type_id": 0
        },
        "projection": {
            "kind": "dense",
            "input_dim": 256,
            "output_dim": 128,
            "modules": [],
            "normalize_after": True
        },
        "compatibility": {
            "llama_cpp_loadable": True,
            "requires_profile": True,
            "strict_pylate_profile": True,
            "known_limitations": []
        }
    }
    with open(invalid_profile_path, "w") as f:
        json.dump(invalid_profile, f)
        
    outfile = fake_colbert_model_dir / "parity_report_invalid.json"
    
    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "tools/verify_pylate_parity.py"),
        "--model-name-or-path", str(fake_colbert_model_dir),
        "--profile", str(invalid_profile_path),
        "--texts-file", str(texts_file),
        "--role", "query",
        "--outfile", str(outfile)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 1
    
    assert outfile.exists()
    with open(outfile, "r") as f:
        report = json.load(f)
        
    assert report["profile_valid"] is False


def test_verify_parity_missing_pylate(fake_colbert_model_dir, monkeypatch):
    import tools.verify_pylate_parity as verify_tool
    
    # Mock PYLATE_AVAILABLE to False
    monkeypatch.setattr(verify_tool, "PYLATE_AVAILABLE", False)
    
    texts_file = fake_colbert_model_dir / "texts.txt"
    with open(texts_file, "w", encoding="utf-8") as f:
        f.write("red planet\n")
        
    outfile = fake_colbert_model_dir / "parity_report_nopylate.json"
    
    old_argv = sys.argv
    sys.argv = [
        "verify_pylate_parity.py",
        "--model-name-or-path", str(fake_colbert_model_dir),
        "--texts-file", str(texts_file),
        "--role", "query",
        "--outfile", str(outfile)
    ]
    
    try:
        verify_tool.main()
    except SystemExit as se:
        assert se.code == 0
    finally:
        sys.argv = old_argv
        
    assert outfile.exists()
    with open(outfile, "r") as f:
        report = json.load(f)
        
    assert report["vector_golden_available"] is False
    assert any("PyLate library is not installed" in limit for limit in report["known_limitations"])
