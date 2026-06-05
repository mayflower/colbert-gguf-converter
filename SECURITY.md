# Security Notes

`colbert-gguf-converter` is a conversion toolset for exporting Hugging Face models into GGML/GGUF format. While it runs as local utility scripts, it downloads models programmatically and executes Python/PyTorch logic.

## Reporting Vulnerabilities

Report security issues privately to the maintainers before opening a public issue. If this code is published under the Mayflower organization, use the repository's GitHub security advisory workflow or email security@mayflower.de.

## General Guidelines

- Avoid running the converter on untrusted Model IDs or local folders containing unverified weights.
- The utility uses standard PyTorch and Hugging Face libraries (`transformers`, `safetensors`, `torch`) to load model parameters. Be aware that loading untrusted PyTorch checkpoints (which utilize standard Python pickle deserialization) carries standard operational risks. Prefer using models published with `safetensors` format files where possible.
