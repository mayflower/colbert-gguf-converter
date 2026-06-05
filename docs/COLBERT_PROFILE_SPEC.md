# ColBERT Runtime Profile Specification (`pg_colbert_profile_v1`)

This document defines the formal runtime profile contract for ColBERT models. The profile encapsulates the entire inference environment, preprocessing logic, similarity functions, and dense projection layer properties required to successfully load and execute a model within a late-interaction runtime (such as `pg_colbert_llama` in `mayflower/pgturbohybrid`).

This profile schema version is: **`pg_colbert_profile_v1`**

---

## 1. Distribution Formats

The ColBERT profile can be distributed in two forms:
1. **Embedded in GGUF**: Under the GGUF metadata key `pg_colbert.profile_json` as a UTF-8 encoded serialized JSON string.
2. **As a Sidecar File**: Alongside the model GGUF file, using the naming convention:
   `<model_name>.gguf.colbert_profile.json`

---

## 2. Profile Schema Reference

### 2.1. Top-Level Structure

| Field | Type | Description |
|---|---|---|
| `schema` | `string` | Must be exactly `"pg_colbert_profile_v1"`. |
| `source_model_id` | `string` | The HuggingFace repository ID of the source model (e.g. `"VAGOsolutions/SauerkrautLM-Multi-ColBERT-15m"`). |
| `source_revision` | `string` | The git revision/branch name of the source snapshot (e.g., `"main"`). |
| `converter_version` | `string` | Version of the colbert-gguf-converter tool that processed this model. |
| `backbone_family` | `string` | The model architecture family of the encoder (e.g., `"bert"`, `"modernbert"`). |
| `colbert_family` | `string` | The library or training framework context (e.g., `"pylate"`, `"colbert-v2"`). |
| `similarity` | `string` | Similarity metric used during late-interaction MaxSim. Supported: `"cosine"`, `"l2"`, `"dot"`. |
| `output_dim` | `integer` | The final late-interaction token embedding dimension (typically `128`). |
| `normalize` | `boolean` | Whether token embeddings must be L2-normalized on the final dimension after projection. |
| `tokenizer` | `object` | Specifies the tokenization parameters and special token configurations. |
| `query` | `object` | Preprocessing rules, length constraints, and query-expansion settings. |
| `document` | `object` | Preprocessing rules, length constraints, and punctuation filtering parameters. |
| `projection` | `object` | Properties of the dense projection layers mapped from the PyTorch model. |
| `compatibility` | `object` | Runtime compatibility, hardware constraints, and loader limitations. |

---

### 2.2. The `tokenizer` Object

| Field | Type | Description |
|---|---|---|
| `source` | `string` | The tokenizer configuration format. One of `"llama"`, `"hf_json"`, `"canonical_ggml"`. |
| `tokenizer_model` | `string` | The class/type of tokenizer (e.g., `"bert"`, `"modernbert"`). |
| `tokenizer_json_sha256` | `string` | SHA256 checksum of the original `tokenizer.json` file for validation. |
| `special_tokens` | `object` | Mapping of special token IDs. |
| `prefix_token_ids` | `object` | Token ID sequences prepended to inputs. |

#### `special_tokens` fields:
* `cls_token_id` (`integer`): ID for the sequence start marker (e.g., `101`).
* `sep_token_id` (`integer`): ID for the sequence boundary marker (e.g., `102`).
* `pad_token_id` (`integer`): Default padding token ID (e.g., `0`).
* `mask_token_id` (`integer` or `null`): ID for the mask token (e.g., `103`), often used for query expansion.
* `q_token_id` (`integer` or `null`): ID corresponding to the literal query marker token (e.g., `[Q]`).
* `d_token_id` (`integer` or `null`): ID corresponding to the literal document marker token (e.g., `[D]`).

#### `prefix_token_ids` fields:
* `query` (`integer[]`): Token ID sequence prepended to query texts (e.g. `[30522]`).
* `document` (`integer[]`): Token ID sequence prepended to document texts (e.g. `[30523]`).

---

### 2.3. The `query` Object

| Field | Type | Description |
|---|---|---|
| `prefix` | `string` | Plain-text prefix prepended to raw query strings (e.g., `"[Q] "`). |
| `max_length` | `integer` | Maximum allowed tokens for query representation. |
| `pad_to` | `integer` or `null` | Exact token length to pad queries to (query expansion target size, e.g. `32`). |
| `pad_token_id` | `integer` | Token ID used to fill query sequences during padding (e.g., `103` for MASK). |
| `pad_token` | `string` | Plain-text string of the padding token (e.g. `"[MASK]"`). |
| `attend_to_expansion_tokens` | `boolean` | If true, attention mask for padding tokens is set to 1; if false, it remains 0. |
| `retain_policy` | `string` | Policy for filtering query tokens (typically `"all"`). |
| `output_policy` | `string` | Defines which token states are projected (typically `"all"`). |
| `token_type_id` | `integer` or `null` | Segment type ID assigned to query tokens (typically `0` or `null`). |

---

### 2.4. The `document` Object

| Field | Type | Description |
|---|---|---|
| `prefix` | `string` | Plain-text prefix prepended to raw document strings (e.g., `"[D] "`). |
| `max_length` | `integer` | Maximum allowed tokens for document representation. |
| `pad_to` | `integer` or `null` | Exact token length to pad documents to (typically `null` for dynamic sizing). |
| `retain_policy` | `string` | Policy for filtering document tokens. E.g., `"mask_and_skiplist"` to omit padding and punctuation. |
| `skiplist_words` | `string[]` | Words or characters filtered out of the document token representation. |
| `skiplist_token_ids` | `integer[]` | Resolved token IDs corresponding to the `skiplist_words` array. |
| `token_type_id` | `integer` or `null` | Segment type ID assigned to document tokens (typically `0` or `null`). |

---

### 2.5. The `projection` Object

| Field | Type | Description |
|---|---|---|
| `kind` | `string` | The projection architecture. One of `"identity"`, `"dense"`, `"module_chain"`. |
| `input_dim` | `integer` | Input dimension from the backbone model's hidden representation (e.g., `288`, `768`). |
| `output_dim` | `integer` | Output embedding dimension (e.g., `128`). |
| `modules` | `object[]` | List of layer details matching the projection. |
| `normalize_after` | `boolean` | True if final vectors should undergo L2-normalization. |

#### `modules` list entries:
* `type` (`string`): e.g., `"linear"`.
* `in_features` (`integer`): Input dimension.
* `out_features` (`integer`): Output dimension.
* `bias` (`boolean`): Whether the linear layer contains a bias tensor.

---

### 2.6. The `compatibility` Object

| Field | Type | Description |
|---|---|---|
| `llama_cpp_loadable` | `boolean` | True if the weights can be loaded into llama.cpp as a standard embedding model. |
| `requires_profile` | `boolean` | True if runtime execution *must* use this JSON config. |
| `strict_pylate_profile` | `boolean` | True if this config strictly mimics PyLate encoder parameters. |
| `known_limitations` | `string[]` | Human-readable notes or limitations in the runtime engine (e.g., missing attention heads). |

---

## 3. Example Profile: `VAGOsolutions/SauerkrautLM-Multi-ColBERT-15m`

Below is the complete profile for the 15-million parameter SauerkrautLM Multilingual ColBERT model:

```json
{
  "schema": "pg_colbert_profile_v1",
  "source_model_id": "VAGOsolutions/SauerkrautLM-Multi-ColBERT-15m",
  "source_revision": "fbe6bc16d0cd4099fb4ac770e556b5df65058bbf",
  "converter_version": "1.0.0",
  "backbone_family": "bert",
  "colbert_family": "pylate",
  "similarity": "cosine",
  "output_dim": 128,
  "normalize": true,
  "tokenizer": {
    "source": "hf_json",
    "tokenizer_model": "bert",
    "tokenizer_json_sha256": "8f3750eb2617f1a30282b0e6ab4f923c5e8c1b2f0a8d6e9d6d02bb144b6f1234",
    "special_tokens": {
      "cls_token_id": 101,
      "sep_token_id": 102,
      "pad_token_id": 0,
      "mask_token_id": 103,
      "q_token_id": 30522,
      "d_token_id": 30523
    },
    "prefix_token_ids": {
      "query": [30522],
      "document": [30523]
    }
  },
  "query": {
    "prefix": "[Q] ",
    "max_length": 32,
    "pad_to": 32,
    "pad_token_id": 103,
    "pad_token": "[MASK]",
    "attend_to_expansion_tokens": false,
    "retain_policy": "all",
    "output_policy": "all",
    "token_type_id": 0
  },
  "document": {
    "prefix": "[D] ",
    "max_length": 300,
    "pad_to": null,
    "retain_policy": "mask_and_skiplist",
    "skiplist_words": [
      "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", ":", ";", "<", "=", ">", "?", "@", "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~"
    ],
    "skiplist_token_ids": [
      106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137
    ],
    "token_type_id": 0
  },
  "projection": {
    "kind": "dense",
    "input_dim": 288,
    "output_dim": 128,
    "modules": [
      {
        "type": "linear",
        "in_features": 288,
        "out_features": 128,
        "bias": false
      }
    ],
    "normalize_after": true
  },
  "compatibility": {
    "llama_cpp_loadable": true,
    "requires_profile": true,
    "strict_pylate_profile": true,
    "known_limitations": []
  }
}
```
