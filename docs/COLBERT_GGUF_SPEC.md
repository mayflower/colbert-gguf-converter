# ColBERT GGUF Specification (`pg_colbert_v1`)

This document defines the custom GGUF layout, metadata keys, and tensor mapping used by our GGML-based ColBERT runtime and PostgreSQL extension (`pg_colbert`).

## GGUF Schema Version

All compliant ColBERT GGUF files must define the schema version key:
* `pg_colbert.gguf_schema` = `"pg_colbert_v1"`

For comprehensive runtime configuration, models should also embed a formal ColBERT runtime profile under `pg_colbert.profile_json` or provide it as a sidecar JSON file (see [COLBERT_PROFILE_SPEC.md](COLBERT_PROFILE_SPEC.md)).

## ColBERT Runtime Profile

To support complete self-contained inference, the GGUF file embeds a structured model runtime profile:
- **GGUF Metadata Key**: `pg_colbert.profile_json`
- **Type**: `string`
- **Format**: UTF-8 encoded JSON string matching the `pg_colbert_profile_v1` schema defined in [COLBERT_PROFILE_SPEC.md](COLBERT_PROFILE_SPEC.md).

### Sidecar Profile JSON

By default, every conversion also writes a standalone JSON file alongside the GGUF file:
- **Filename Convention**: `<outfile>.colbert_profile.json` (where `<outfile>` is the path to the written GGUF file).
- **Format**: Identical UTF-8 JSON matching the `pg_colbert_profile_v1` schema.
- **Control**: Generation of the sidecar can be disabled by passing the `--no-profile-sidecar` flag to the converter.
- **Purpose**: Enables rapid profiling, inspection, and tooling consumption without requiring the caller to parse the main GGUF header.

## General & Provenance Metadata

Standard llama.cpp / GGUF metadata keys for model provenance:

| Key | Type | Description |
|---|---|---|
| `general.name` | `string` | Human-readable name of the model. |
| `general.basename` | `string` | The base name or slug of the model. |
| `general.architecture` | `string` | The backbone architecture, e.g. `"modernbert"` or `"bert"`. |
| `general.file_type` | `uint32` | GGUF file type code (typically `1` for F16, `0` for F32). |
| `general.source.huggingface.repo_id` | `string` | Hugging Face repository ID. |
| `general.source.huggingface.revision` | `string` | Git revision/branch. |
| `general.source.huggingface.commit_hash` | `string` | Full git commit SHA of the source snapshot. |
| `general.license` | `string` | Model license (if declared). |

## Converter Metadata

Information about how the GGUF file was generated:

| Key | Type | Description |
|---|---|---|
| `pg_colbert.converter.version` | `string` | Version of the converter script. |
| `pg_colbert.converter.command` | `string` | The exact CLI command used to trigger conversion. |
| `pg_colbert.tensor_map_json` | `string` | JSON string mapping canonical runtime tensors to stored GGUF tensor names. |
| `pg_colbert.profile_json` | `string` | UTF-8 encoded JSON string containing the formal model runtime profile matching `pg_colbert_profile_v1` (see [COLBERT_PROFILE_SPEC.md](COLBERT_PROFILE_SPEC.md)). |

## ColBERT Metadata

Metadata keys specific to ColBERT retrieval parameters:

| Key | Type | Description |
|---|---|---|
| `colbert.model_type` | `string` | Either `"modernbert"` or `"bert"`. |
| `colbert.backbone_model_type` | `string` | The HuggingFace backbone model type (e.g. `"modernbert"` or `"bert"`). |
| `colbert.embedding_dim` | `uint32` | Projected embedding dimension (typically `128`). |
| `colbert.projection.in_features` | `uint32` | Backbone hidden size input to the projection (e.g. `768`). |
| `colbert.projection.out_features` | `uint32` | Projected dimensions (typically `128`). |
| `colbert.projection.bias` | `bool` | True if the linear projection uses bias, otherwise false. |
| `colbert.similarity_fn_name` | `string` | Similarity metric name: `"l2"` (for L2-squared/Euclidean distance) or `"cosine"`. |
| `colbert.query_prefix` | `string` | The prefix added to queries (e.g., `"[Q] "` or `"query: "`). |
| `colbert.document_prefix` | `string` | The prefix added to documents (e.g., `"[D] "` or `"passage: "`). |
| `colbert.query_length` | `uint32` | Maximum token length for queries (e.g., `32`). |
| `colbert.document_length` | `uint32` | Maximum token length for documents (e.g., `256`). |
| `colbert.attend_to_expansion_tokens` | `bool` | Whether query token padding is attended to. |
| `colbert.skiplist_words` | `array of string` | List of words/punctuation to skip when indexing/searching. |
| `colbert.q_token_id` | `uint32` | Token ID for the query marker (e.g. `[Q]`). |
| `colbert.d_token_id` | `uint32` | Token ID for the document marker (e.g. `[D]`). |
| `colbert.pad_token_id` | `uint32` | Padding token ID. |
| `colbert.cls_token_id` | `uint32` | CLS token ID. |
| `colbert.sep_token_id` | `uint32` | SEP token ID. |
| `colbert.bos_token_id` | `uint32` | BOS token ID. |
| `colbert.eos_token_id` | `uint32` | EOS token ID. |
| `colbert.query_prefix_token_ids` | `array of uint32` | Token ID sequence for the query prefix (e.g. `[Q] `). |
| `colbert.document_prefix_token_ids` | `array of uint32` | Token ID sequence for the document prefix (e.g. `[D] `). |

## Backbone Architecture Metadata

Depending on `colbert.backbone_model_type`, the keys are prefixed by the architecture name.

### ModernBERT Config Keys
* `modernbert.hidden_size` (`uint32`)
* `modernbert.intermediate_size` (`uint32`)
* `modernbert.num_hidden_layers` (`uint32`)
* `modernbert.num_attention_heads` (`uint32`)
* `modernbert.max_position_embeddings` (`uint32`)
* `modernbert.local_attention` (`uint32`)
* `modernbert.global_attn_every_n_layers` (`uint32`)
* `modernbert.local_rope_theta` (`float32`)
* `modernbert.global_rope_theta` (`float32`)
* `modernbert.hidden_activation` (`string`)
* `modernbert.layer_norm_eps` (`float32`)
* `modernbert.norm_eps` (`float32`)
* `modernbert.attention_bias` (`bool`)
* `modernbert.mlp_bias` (`bool`)
* `modernbert.norm_bias` (`bool`)

### BERT Config Keys
* `bert.hidden_size` (`uint32`)
* `bert.intermediate_size` (`uint32`)
* `bert.num_hidden_layers` (`uint32`)
* `bert.num_attention_heads` (`uint32`)
* `bert.max_position_embeddings` (`uint32`)
* `bert.hidden_activation` (`string`)
* `bert.layer_norm_eps` (`float32`)
* `bert.type_vocab_size` (`uint32`)

## Tokenizer Metadata

To enable self-contained tokenization in the database extension:

| Key | Type | Description |
|---|---|---|
| `tokenizer.huggingface.json` | `string` | The serialized `tokenizer.json` content as a UTF-8 string. |
| `tokenizer.config.json` | `string` | The serialization of `tokenizer_config.json`. |
| `tokenizer.special_tokens_map.json` | `string` | The serialization of `special_tokens_map.json`. |

---

## Tensors Layout & Naming

Tensors are written using the following conventions:

1. **Backbone Tensors**: Stored under their original Hugging Face safetensors names prefixed with `hf.`. For example, `hf.model.embeddings.tok_embeddings.weight`.
2. **ColBERT Projection Tensors**: The projection weight is explicitly named and stored as:
   * `colbert.proj.weight` (Shape: `[out_features, in_features]`, usually `[128, hidden_size]`).
   * `colbert.proj.bias` (if present, usually absent in modern ColBERT models).

### Canonical Runtime Tensor Mapping

The `pg_colbert.tensor_map_json` key contains a JSON map mapping the canonical runtime tensor names to the actual stored GGUF tensor names.

#### ModernBERT Canonical Tensors

| Canonical Name | Stored Name Example (`hf_original` style) |
|---|---|
| `embeddings.tok_embeddings.weight` | `hf.model.embeddings.tok_embeddings.weight` |
| `embeddings.norm.weight` | `hf.model.embeddings.norm.weight` |
| `embeddings.norm.bias` | `hf.model.embeddings.norm.bias` |
| `layers.{i}.attn.Wqkv.weight` | `hf.model.layers.{i}.attn.Wqkv.weight` |
| `layers.{i}.attn.Wqkv.bias` | `hf.model.layers.{i}.attn.Wqkv.bias` |
| `layers.{i}.attn.out_proj.weight` | `hf.model.layers.{i}.attn.out_proj.weight` |
| `layers.{i}.attn.out_proj.bias` | `hf.model.layers.{i}.attn.out_proj.bias` |
| `layers.{i}.attn.norm.weight` | `hf.model.layers.{i}.attn.norm.weight` |
| `layers.{i}.attn.norm.bias` | `hf.model.layers.{i}.attn.norm.bias` |
| `layers.{i}.mlp.wi_0.weight` | `hf.model.layers.{i}.mlp.wi_0.weight` |
| `layers.{i}.mlp.wi_1.weight` | `hf.model.layers.{i}.mlp.wi_1.weight` |
| `layers.{i}.mlp.wo.weight` | `hf.model.layers.{i}.mlp.wo.weight` |
| `layers.{i}.mlp.norm.weight` | `hf.model.layers.{i}.mlp.norm.weight` |
| `layers.{i}.mlp.norm.bias` | `hf.model.layers.{i}.mlp.norm.bias` |
| `final_norm.weight` | `hf.model.norm.weight` |
| `final_norm.bias` | `hf.model.norm.bias` |
| `projection.weight` | `colbert.proj.weight` |
| `projection.bias` | `colbert.proj.bias` (if present) |

#### BERT Canonical Tensors

| Canonical Name | Stored Name Example (`hf_original` style) |
|---|---|
| `embeddings.word_embeddings.weight` | `hf.bert.embeddings.word_embeddings.weight` |
| `embeddings.position_embeddings.weight` | `hf.bert.embeddings.position_embeddings.weight` |
| `embeddings.token_type_embeddings.weight` | `hf.bert.embeddings.token_type_embeddings.weight` |
| `embeddings.LayerNorm.weight` | `hf.bert.embeddings.LayerNorm.weight` |
| `embeddings.LayerNorm.bias` | `hf.bert.embeddings.LayerNorm.bias` |
| `layers.{i}.attention.self.query.weight` | `hf.bert.encoder.layer.{i}.attention.self.query.weight` |
| `layers.{i}.attention.self.query.bias` | `hf.bert.encoder.layer.{i}.attention.self.query.bias` |
| `layers.{i}.attention.self.key.weight` | `hf.bert.encoder.layer.{i}.attention.self.key.weight` |
| `layers.{i}.attention.self.key.bias` | `hf.bert.encoder.layer.{i}.attention.self.key.bias` |
| `layers.{i}.attention.self.value.weight` | `hf.bert.encoder.layer.{i}.attention.self.value.weight` |
| `layers.{i}.attention.self.value.bias` | `hf.bert.encoder.layer.{i}.attention.self.value.bias` |
| `layers.{i}.attention.output.dense.weight` | `hf.bert.encoder.layer.{i}.attention.output.dense.weight` |
| `layers.{i}.attention.output.dense.bias` | `hf.bert.encoder.layer.{i}.attention.output.dense.bias` |
| `layers.{i}.attention.output.LayerNorm.weight` | `hf.bert.encoder.layer.{i}.attention.output.LayerNorm.weight` |
| `layers.{i}.attention.output.LayerNorm.bias` | `hf.bert.encoder.layer.{i}.attention.output.LayerNorm.bias` |
| `layers.{i}.intermediate.dense.weight` | `hf.bert.encoder.layer.{i}.intermediate.dense.weight` |
| `layers.{i}.intermediate.dense.bias` | `hf.bert.encoder.layer.{i}.intermediate.dense.bias` |
| `layers.{i}.output.dense.weight` | `hf.bert.encoder.layer.{i}.output.dense.weight` |
| `layers.{i}.output.dense.bias` | `hf.bert.encoder.layer.{i}.output.dense.bias` |
| `layers.{i}.output.LayerNorm.weight` | `hf.bert.encoder.layer.{i}.output.LayerNorm.weight` |
| `layers.{i}.output.LayerNorm.bias` | `hf.bert.encoder.layer.{i}.output.LayerNorm.bias` |
| `projection.weight` | `colbert.proj.weight` |
| `projection.bias` | `colbert.proj.bias` (if present) |
