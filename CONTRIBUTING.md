# Contributing

This file helps you set up a local development environment, run the expected checks, and contribute changes to `colbert-gguf-converter`.

## Local Setup

Ensure you are using Python >= 3.10 and install all dependencies:

```sh
pip install -r requirements.txt
```

To run the automated unit test suite locally, execute:

```sh
pytest -v
```

All 27+ test cases must pass successfully before submitting code changes.

## Development & Test Guidelines

- **Code Style**: Maintain clean, readable Python code following PEP 8.
- **Tests**: Any new features, tensor mappings, or model backbones must include unit tests under `tests/` (utilizing lightweight mock state dict fixtures to keep testing fast and dependency-free).
- **Parity Verification**: If your change modifies conversion logic, you must generate a verification parity report and confirm that it passes vector equivalence:
  ```sh
  python tools/verify_pylate_parity.py \
    --model-name-or-path VAGOsolutions/SauerkrautLM-Multi-ModernColBERT \
    --gguf /path/to/converted.gguf \
    --profile /path/to/converted.gguf.colbert_profile.json \
    --texts-file tests/fixtures/validation_texts.txt \
    --role query \
    --outfile parity_report.json
  ```
- **Documentation**: Update specification files under `docs/` if modifying GGUF metadata keys, tensor names, or schema versions.

## Pull Requests

- Push your changes to a branch and open a Pull Request targeting `master`.
- Provide a clear explanation of your change, describe the verification steps taken, and attach test logs or parity reports if relevant.
- Ensure the automated GitHub Actions CI workflow passes successfully on your PR.
