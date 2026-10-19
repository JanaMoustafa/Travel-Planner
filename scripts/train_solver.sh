#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/../.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
"${PYTHON_BIN}" -m travel_planner.training.build_sft_dataset --out "${ROOT}/artifacts/data/sft_train.jsonl"
head -n 20 "${ROOT}/artifacts/data/sft_train.jsonl" > "${ROOT}/artifacts/sft_train_smoke.jsonl"
"${PYTHON_BIN}" -m travel_planner.training.train_qlora_unsloth \
  --backend transformers \
  --train_file "${ROOT}/artifacts/sft_train_smoke.jsonl" \
  --output_dir "${ROOT}/artifacts/models/solver-lora" \
  --model_name unsloth/Llama-3.2-3B-bnb-4bit \
  --max_seq_length 1024 \
  --epochs 1 \
  --max_steps 20 \
  --batch_size 1 \
  --grad_accum 4
