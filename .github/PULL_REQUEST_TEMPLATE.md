## Description

Please include a summary of the changes and the technical rationale. List any dependencies that are required for this change.

Fixes # (issue number)

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Verification & Parity Checks

Describe the verification steps taken to confirm correctness.

- **Unit Tests**: Did you run `pytest`?
- **Parity Verification**: Paste terminal output or contents of `parity_report.json` if modifying conversion/vector output math.
- **inspect GGUF**: Run `python tools/inspect_colbert_gguf.py <model.gguf>` and paste the printed profile summary.
