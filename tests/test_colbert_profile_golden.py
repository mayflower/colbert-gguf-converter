import json
import subprocess
import sys
from pathlib import Path

def test_create_colbert_profile_golden_query(fake_colbert_model_dir):
    # 1. Create a texts file with queries
    texts_file = fake_colbert_model_dir / "queries.txt"
    with open(texts_file, "w", encoding="utf-8") as f:
        f.write("red planet\n")
        
    outfile = fake_colbert_model_dir / "golden_queries.json"
    
    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "tools/create_colbert_profile_golden.py"),
        "--model-name-or-path", str(fake_colbert_model_dir),
        "--texts-file", str(texts_file),
        "--role", "query",
        "--outfile", str(outfile)
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Golden generator failed: {res.stderr}\nOutput: {res.stdout}"
    
    # 2. Verify golden output structure and queries plan
    assert outfile.exists()
    with open(outfile, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["token_plan_source"] == "hf_tokenizer_profile_rules"
    assert data["model_name_or_path"] == str(fake_colbert_model_dir)
    assert data["role"] == "query"
    
    assert len(data["plans"]) == 1
    plan = data["plans"][0]
    
    assert plan["input_text"] == "red planet"
    assert plan["role"] == "query"
    
    # query "red planet" must include mask padding to query length (32)
    assert len(plan["token_ids_after_padding_truncation"]) == 32
    assert plan["final_vector_count"] == 32
    
    # check that it contains mask padding reasons
    assert "query_expansion" in plan["retain_reasons"]
    assert len(plan["retain_mask"]) == 32
    assert len(plan["attention_mask"]) == 32
    # Since attend_to_expansion is False in the mock config, the attention mask should have 1s for valid tokens and 0s for padding
    assert plan["attention_mask"][0] == 1
    assert plan["attention_mask"][4] == 1
    assert plan["attention_mask"][5] == 0



def test_create_colbert_profile_golden_doc(fake_colbert_model_dir):
    # 1. Create a texts file with documents
    texts_file = fake_colbert_model_dir / "docs.txt"
    with open(texts_file, "w", encoding="utf-8") as f:
        f.write("red planet.\n")
        
    outfile = fake_colbert_model_dir / "golden_docs.json"
    
    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "tools/create_colbert_profile_golden.py"),
        "--model-name-or-path", str(fake_colbert_model_dir),
        "--texts-file", str(texts_file),
        "--role", "doc",
        "--outfile", str(outfile)
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Golden generator failed: {res.stderr}\nOutput: {res.stdout}"
    
    # 2. Verify golden output structure and doc plan
    assert outfile.exists()
    with open(outfile, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["role"] == "doc"
    assert len(data["plans"]) == 1
    plan = data["plans"][0]
    
    assert plan["input_text"] == "red planet."
    assert plan["role"] == "doc"
    
    # check skiplist token ids
    assert len(plan["skiplist_token_ids"]) > 0
    
    # document "red planet." marks punctuation as skipped when skiplist contains "."
    # Let's find "." in token pieces.
    dot_index = -1
    for idx, piece in enumerate(plan["token_pieces"]):
        if piece == ".":
            dot_index = idx
            break
            
    assert dot_index != -1, f"Could not find '.' in token pieces: {plan['token_pieces']}"
    assert plan["retain_mask"][dot_index] == 0
    assert plan["retain_reasons"][dot_index] == "skipped_skiplist"
    
    # CLS/SEP should be special tokens
    assert plan["retain_mask"][0] == 1
    assert plan["retain_reasons"][0] == "special_token"
