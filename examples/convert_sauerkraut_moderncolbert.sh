#!/usr/bin/env bash
# Real-model smoke test script for pg_colbert GGUF conversion.
# Performs a environment check, a dry-run validation, full conversion,
# output inspection, and reference PyLate golden generation.

set -e

echo "=== 1. Checking Environment & PyTorch/CUDA Availability ==="
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA Available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Device:     {torch.cuda.get_device_name(0)}')
"

MODEL_ID="VAGOsolutions/SauerkrautLM-Multi-ModernColBERT"
OUT_GGUF="/tmp/sauerkraut-multi-moderncolbert.f16.gguf"

echo ""
echo "=== 2. Running Converter in Dry-Run Mode ==="
python3 tools/convert_colbert_hf_to_gguf.py \
  --model-id "$MODEL_ID" \
  --outfile "$OUT_GGUF" \
  --outtype f16 \
  --dry-run \
  --verbose

echo ""
echo "=== 3. Executing Full Conversion to GGUF F16 ==="
python3 tools/convert_colbert_hf_to_gguf.py \
  --model-id "$MODEL_ID" \
  --outfile "$OUT_GGUF" \
  --outtype f16 \
  --verbose

echo ""
echo "=== 4. Inspecting the Generated GGUF File ==="
python3 tools/inspect_colbert_gguf.py "$OUT_GGUF"

echo ""
echo "=== 5. Running PyLate Reference Golden Generation ==="
python3 tools/create_pylate_golden.py \
  --model-name "$MODEL_ID" \
  --outdir tests/fixtures

echo ""
echo "=== Conversion Smoke Test Completed Successfully! ==="
