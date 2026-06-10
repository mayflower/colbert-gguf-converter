# Changes: align llama.cpp export to upstream/ollama (b9509)

Branch: `align-b9509-llama-cpp-naming`

## Summary

`tools/export_to_llama_cpp.py` now produces a GGUF that loads and runs **directly
in upstream llama.cpp and ollama** (validated against llama.cpp tag **b9509**).
Previously the exported `.llama.gguf` used an older llama.cpp BERT convention and
omitted a ggml tokenizer, so it failed to load in current llama.cpp / ollama.

The backbone forward in ollama now reproduces PyLate token embeddings essentially
exactly (**~8e-7 cosine** on content tokens vs the shipped golden vectors).

## Motivation

ollama embeds llama.cpp (tag b9509). The converter's llama.cpp export had drifted
from what b9509 expects, so dropping the published GGUFs into ollama failed at
three layers, in order:

1. `key not found in model: bert.context_length` — HF-style hparam KV keys.
2. `check_tensor_dims … attn_k … expected 384, 0` / `missing tensor blk.0.attn_output.weight` — tensor naming.
3. `binary_op: unsupported types: dst: f32, src0: f32, src1: f16` — all-F16 tensors in BERT LayerNorm.
4. `key not found: tokenizer.ggml.model` — the input only carries `tokenizer.huggingface.json`.

## What changed

### `tools/colbert_profile.py`
- **`get_llama_tensor_map`** now emits upstream llama.cpp tensor names:
  - BERT (post-LN): `attention.output.dense → attn_output`,
    `attention.output.LayerNorm → attn_output_norm`,
    `output.LayerNorm → layer_output_norm`.
  - ModernBERT (pre-norm): `attn.out_proj` / `attn.Wo → attn_output`;
    `attn_norm` / `ffn_norm` / `output_norm` were already correct and are unchanged.
- **`get_llama_kv_canonical_map(arch)`** (new): maps the converter's HF-style
  hyperparameter keys to canonical llama.cpp keys, e.g.
  `{arch}.hidden_size → {arch}.embedding_length`,
  `{arch}.max_position_embeddings → {arch}.context_length`,
  `{arch}.layer_norm_eps → {arch}.attention.layer_norm_epsilon`,
  `{arch}.type_vocab_size → tokenizer.ggml.token_type_count`.
- **`build_ggml_bert_tokenizer(tokenizer_json_str)`** (new): builds
  `tokenizer.ggml.tokens` / `token_type` from a raw HF `tokenizer.json`
  (WordPiece rendering: word-start tokens prefixed with U+2581, `##`
  continuations stripped, added/special tokens verbatim).

### `tools/export_to_llama_cpp.py`
- Canonicalizes hyperparameter KV keys on copy via `get_llama_kv_canonical_map`.
- Builds and writes a `tokenizer.ggml.*` BERT tokenizer from the embedded
  `tokenizer.huggingface.json` (tokens, token_type, model, special-token ids,
  `add_bos/eos/sep_token`); skips the raw HF tokenizer blobs.
- Retains the `pg_colbert.profile_json` metadata key in the exported GGUF (an
  unknown KV is ignored by llama.cpp, so it is harmless there, while letting the
  serving layer read the ColBERT runtime profile directly from the file).
- Drops the `colbert.proj.*` tensors from the GGUF (an unknown tensor trips
  llama.cpp's tensor-count check, so the model would not load) and instead writes
  them to a **`<outfile>.colbert_proj` sidecar** (magic `OLPROJ01`; `out_features`,
  `in_features`, `has_bias`, then row-major `[out][in]` float32 weight + optional
  bias). The serving layer loads the sidecar and applies the 384→128 projection.
  Disable with `--no-projection-sidecar`.
- Upcasts all exported tensors to **F32** (BERT LayerNorm in b9509 is F32 and its
  CPU binary ops reject mixed F32/F16). Quantize separately if a smaller file is
  needed.
- Emits **`{arch}.pooling_type = 0`** (none) when the source GGUF does not carry
  the key. llama.cpp / ollama derive the embedding capability and the per-token
  (multivector) output mode from this key; without it the exported model loads
  but is not detected as an embedding model.
- Embeds the ColBERT projection **inside the GGUF** as a sentence-transformers
  dense module by default: `dense_2.weight` (+ `dense_2.bias`) plus
  `{arch}.embedding_length_out = out_features`. llama.cpp builds that declare
  the dense module for BERT/ModernBERT (e.g. ollama) apply it in-graph — per
  token under pooling none — and `llama_model_n_embd_out()` reports the
  projected width, so the serving layer needs no sidecar. Stock llama.cpp
  b9509 does not yet declare `dense_2.*` for these archs and will refuse the
  file; pass `--no-dense-in-gguf` for a backbone-only export it can load (the
  `.colbert_proj` sidecar then carries the projection).

### `tests/test_llama_export_alignment.py` (new)
- Asserts the b9509 tensor names for BERT and ModernBERT.
- Asserts the KV canonicalization map.
- Asserts `canonicalize_tensor_name` produces `blk.{i}.*` names.
- Asserts the ggml tokenizer builder (WPM rendering, contiguity check).

### `docs/COLBERT_GGUF_SPEC.md`
- New "llama.cpp / ollama Export" section: tensor-name table, KV-key table, F32
  rationale, tokenizer construction, and the parity note.

## Validation

Ran the real `export_to_llama_cpp.py` on the published
`VAGOsolutions_SauerkrautLM_Multi_ColBERT_33m.f16.pg_colbert.gguf`, then loaded the
output in ollama's b9509 `llama-server` (`--embedding --pooling none`):

- Model loads and serves embeddings (all four previous load errors resolved).
- Per-token forward, projected (`colbert.proj`, 384→128) and L2-normalized,
  matches the shipped PyLate golden vectors to **~8e-7 cosine** on content tokens.
- `tests/test_llama_export_alignment.py`: 8 passed.

## Notes / limitations

- **Backbone-only export.** The ColBERT projection and profile are dropped (the
  projection is runtime-only; it is not part of the llama.cpp graph). The serving
  layer must apply the token plan, the 384→128 projection, and L2 normalization.
- **Query expansion attention.** Full query parity additionally requires
  `attend_to_expansion_tokens=false` (content tokens not attending to the `[MASK]`
  expansion). Standard llama.cpp `/embeddings` uses full attention; this is the
  same approximation the pg_colbert runtime documents (`vector_parity_valid:
  false`). Content-token parity is unaffected (~1e-6).
- **Pre-existing test failures** in this repo (HF-config parsing under
  `transformers` 5.10.2, `rope_parameters` strict validation) are unrelated to
  these changes and reproduce on `master`.
