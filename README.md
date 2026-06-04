# ColBERT Hugging Face-to-GGUF Converter

This repository provides a Python-based converter and suite of inspection tools for producing custom ColBERT GGUF files (`pg_colbert_v1` schema) consumed by our GGML-based ColBERT C++ runtime. 

Unlike generic llama.cpp LLM decoder converters, this tool preserves the backbone transformer structures and embeds the ColBERT late-interaction dense projection layers, similarity metrics, tokenizer configurations, and query/document metadata.

## Included Tools & Structure

* **[docs/COLBERT_GGUF_SPEC.md](docs/COLBERT_GGUF_SPEC.md)**: Specifications detailing metadata keys, tensor naming conventions, and schema requirements.
* **[tools/convert_colbert_hf_to_gguf.py](tools/convert_colbert_hf_to_gguf.py)**: Primary converter script for downloading or reading local models and exporting GGUF files.
* **[tools/publish_colbert_gguf.py](tools/publish_colbert_gguf.py)**: Helper pipeline that converts a model, writes a custom GGUF model card, and uploads them to the Hugging Face Hub.
* **[tools/inspect_colbert_gguf.py](tools/inspect_colbert_gguf.py)**: Inspection utility to print GGUF metadata values, tensor tables, and validate schema compliance.
* **[tools/inspect_colbert_hf.py](tools/inspect_colbert_hf.py)**: Diagnostic tool to list safetensors tensor shapes, dtypes, and configuration parameters of source Hugging Face repositories.
* **[tools/create_pylate_golden.py](tools/create_pylate_golden.py)**: Reference vector generator that uses the original `pylate` package to create query/document embedding test benchmarks.
* **[tests/](tests/)**: Automated unit test suites validating configuration loading, GGUF metadata writing, and tensor maps using mock fixtures.
* **[examples/convert_sauerkraut_moderncolbert.sh](examples/convert_sauerkraut_moderncolbert.sh)**: End-to-end local conversion and validation smoke test script.

---

## Installation & Setup

Ensure you are running Python >= 3.10 and install the dependencies:

```bash
pip install numpy gguf safetensors huggingface_hub pytest tokenizers transformers torch pylate
```

---

## Usage Guide

### 1. Local GGUF Conversion

Convert a Hugging Face model repository (either from the Hugging Face Hub or a local directory) into an `F16` GGUF file:

```bash
python tools/convert_colbert_hf_to_gguf.py \
  --model-id VAGOsolutions/SauerkrautLM-Multi-ModernColBERT \
  --outfile /tmp/sauerkraut-multi-moderncolbert.f16.gguf \
  --outtype f16 \
  --verbose
```

**Common Options**:
* `--model-dir /path/to/local/model`: Use a local directory instead of downloading from the Hub.
* `--outtype f32|f16`: Export weights in Float32 or Float16 precision (default is `f16`).
* `--dry-run`: Parse configs and validate shape compatibility without writing GGUF weights.
* `--allow-shape-mismatch`: Bypass verification checks matching the dense projection features with the backbone hidden size.

### 2. Inspecting the GGUF Model

Verify compliance and check metadata fields and tensor shapes inside the generated GGUF file:

```bash
python tools/inspect_colbert_gguf.py /tmp/sauerkraut-multi-moderncolbert.f16.gguf
```

### 3. Publishing to the Hugging Face Hub

Convert a source model, generate a custom GGUF usage README model card, and push everything directly to your Hugging Face repository:

```bash
python tools/publish_colbert_gguf.py \
  --model-id VAGOsolutions/SauerkrautLM-Multi-ModernColBERT \
  --target-repo-id <your-hf-username>/SauerkrautLM-Multi-ModernColBERT-GGUF \
  --outtype f16 \
  --token <your-hf-write-token-or-use-HF_TOKEN-env> \
  --verbose
```

---

## Running Verification & Tests

To execute the unit test suites (which dynamically construct lightweight mock safetensors models for speed and offline compatibility):

```bash
pytest -v
```
