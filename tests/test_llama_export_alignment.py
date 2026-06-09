"""Tests for the llama.cpp / ollama (b9509) naming and KV alignment used by
tools/export_to_llama_cpp.py."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from tools.colbert_profile import (
    get_llama_tensor_map,
    get_llama_kv_canonical_map,
    canonicalize_tensor_name,
)


def test_bert_tensor_names_match_b9509():
    """BERT post-LN naming must match upstream llama.cpp / ollama (b9509)."""
    m = get_llama_tensor_map("bert")
    # attention output projection
    assert m["attention.output.dense.weight"] == "attn_output.weight"
    assert m["attention.output.dense.bias"] == "attn_output.bias"
    # post-attention LayerNorm -> attn_output_norm
    assert m["attention.output.LayerNorm.weight"] == "attn_output_norm.weight"
    assert m["attention.output.LayerNorm.bias"] == "attn_output_norm.bias"
    # post-FFN LayerNorm -> layer_output_norm
    assert m["output.LayerNorm.weight"] == "layer_output_norm.weight"
    assert m["output.LayerNorm.bias"] == "layer_output_norm.bias"
    # unchanged, already canonical
    assert m["intermediate.dense.weight"] == "ffn_up.weight"
    assert m["output.dense.weight"] == "ffn_down.weight"
    assert m["attention.self.query.weight"] == "attn_q.weight"
    # no legacy names should leak through
    assert "attn_out.weight" not in m.values()
    assert "attn_norm.weight" not in m.values()
    assert "ffn_norm.weight" not in m.values()


def test_modernbert_tensor_names_match_b9509():
    """ModernBERT is pre-norm: attn_norm/ffn_norm/output_norm stay; only the
    attention output projection renames to attn_output."""
    m = get_llama_tensor_map("modernbert")
    assert m["attn.out_proj.weight"] == "attn_output.weight"
    assert m["attn.Wo.weight"] == "attn_output.weight"
    # pre-norms are already the canonical b9509 names for ModernBERT
    assert m["attn.norm.weight"] == "attn_norm.weight"
    assert m["mlp.norm.weight"] == "ffn_norm.weight"
    assert m["norm.weight"] == "output_norm.weight"
    assert m["attn.Wqkv.weight"] == "attn_qkv.weight"
    assert m["mlp.wi_0.weight"] == "ffn_gate.weight"
    assert m["mlp.wi_1.weight"] == "ffn_up.weight"
    assert m["mlp.wo.weight"] == "ffn_down.weight"
    assert "attn_out.weight" not in m.values()


def test_unknown_arch_returns_empty():
    assert get_llama_tensor_map("gpt2") == {}


def test_kv_canonical_map_bert():
    c = get_llama_kv_canonical_map("bert")
    assert c["bert.hidden_size"] == "bert.embedding_length"
    assert c["bert.intermediate_size"] == "bert.feed_forward_length"
    assert c["bert.num_hidden_layers"] == "bert.block_count"
    assert c["bert.num_attention_heads"] == "bert.attention.head_count"
    assert c["bert.max_position_embeddings"] == "bert.context_length"
    assert c["bert.layer_norm_eps"] == "bert.attention.layer_norm_epsilon"
    assert c["bert.type_vocab_size"] == "tokenizer.ggml.token_type_count"


def test_kv_canonical_map_is_arch_prefixed():
    c = get_llama_kv_canonical_map("modernbert")
    assert c["modernbert.hidden_size"] == "modernbert.embedding_length"
    assert c["modernbert.max_position_embeddings"] == "modernbert.context_length"
    # token_type_count is a shared tokenizer key, not arch-prefixed
    assert c["modernbert.type_vocab_size"] == "tokenizer.ggml.token_type_count"


def test_canonicalize_tensor_name_layer_and_global():
    # layer-specific
    assert (
        canonicalize_tensor_name("hf.bert.encoder.layer.3.attention.output.dense.weight", "bert")
        == "blk.3.attn_output.weight"
    )
    assert (
        canonicalize_tensor_name("hf.layers.5.attn.out_proj.weight", "modernbert")
        == "blk.5.attn_output.weight"
    )
    # global / embedding
    assert (
        canonicalize_tensor_name("hf.bert.embeddings.word_embeddings.weight", "bert")
        == "token_embd.weight"
    )
