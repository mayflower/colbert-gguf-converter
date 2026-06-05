# Changelog

All notable changes to `colbert-gguf-converter` will be documented in this file.

## [1.0.0] - 2026-06-05

### Added
- **GGUF Converter**: High-performance Python converter for exporting Hugging Face models into specialized ColBERT GGUF layouts.
- **Backbone Support**: Native support for ModernBERT and BERT encoder backbones.
- **Double Target Runtimes**: CLI option to output either `pg_colbert` (custom layout with schema & JSON tensor maps) or `llama_cpp` (standard llama.cpp compliant embedding layouts).
- **ColBERT Runtime Profile (`pg_colbert_profile_v1`)**: Dataclasses and validators for serializing configurations, prefixes, lengths, projection layers, and compatibility flags.
- **Reference Vector Parity**: Detailed verify scripts to run PyLate reference vector comparison and calculate MAE tolerance checks.
- **Token-Plan Goldens**: Helper to output tokenization plans (pre/post-padding, masks, retention flags, and skiplists) for downstream C++ test suites.
- **Publisher Utility**: Programmatic converter, model card generator, and uploader script to push GGUF assets and sidecar configs directly to Hugging Face Hub.
- **Quantization Support**: Post-conversion quantization using llama.cpp's `quantize` binary (supporting formats like `Q8_0`, `Q4_K_M`, etc.).
- **Automated CI Workflow**: GitHub Actions integration to run tests on push and pull request.
