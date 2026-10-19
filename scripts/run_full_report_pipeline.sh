#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/../.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export PREFER_OPENAI="${PREFER_OPENAI:-0}"

TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
TRAIN_GRAD_ACCUM="${TRAIN_GRAD_ACCUM:-4}"
TRAIN_MAX_SEQ_LEN="${TRAIN_MAX_SEQ_LEN:-1024}"
TRAIN_MODEL_NAME="${TRAIN_MODEL_NAME:-unsloth/Llama-3.2-3B-bnb-4bit}"

cd "${ROOT}"

mkdir -p artifacts/data artifacts/models artifacts/runs artifacts/eval artifacts/logs
rm -f artifacts/runs/*.jsonl artifacts/eval/*.json artifacts/eval/*.csv
mkdir -p artifacts/experiments
: > artifacts/experiments/manifest.jsonl

echo "Building full SFT dataset..."
"${PYTHON_BIN}" -m travel_planner.training.build_sft_dataset --out artifacts/data/sft_train.jsonl

echo "Training LoRA adapter on full train split..."
"${PYTHON_BIN}" -m travel_planner.training.train_qlora_unsloth \
  --backend transformers \
  --train_file artifacts/data/sft_train.jsonl \
  --output_dir artifacts/models/solver-lora \
  --model_name "${TRAIN_MODEL_NAME}" \
  --max_seq_length "${TRAIN_MAX_SEQ_LEN}" \
  --epochs "${TRAIN_EPOCHS}" \
  --batch_size "${TRAIN_BATCH_SIZE}" \
  --grad_accum "${TRAIN_GRAD_ACCUM}"

echo "Running baseline API validation..."
"${PYTHON_BIN}" scripts/run_agent_validation.py \
  --split validation \
  --use-full-split \
  --experiment-name baseline \
  --set-baseline \
  --temperature 0.4 \
  --out artifacts/runs/baseline_runs.jsonl

echo "Running enhanced local-model validation..."
"${PYTHON_BIN}" scripts/run_agent_validation.py \
  --split validation \
  --use-full-split \
  --experiment-name enhanced \
  --temperature 0.4 \
  --use-local-model \
  --lora-path artifacts/models/solver-lora \
  --out artifacts/runs/enhanced_runs.jsonl

echo "Running temperature sweep..."
"${PYTHON_BIN}" scripts/run_temperature_experiments.py \
  --split validation \
  --use-full-split \
  --out-dir artifacts/runs \
  --output-prefix temp \
  --temperatures 0.2 0.7

cp artifacts/runs/baseline_runs.jsonl artifacts/runs/temp_0.4_runs.jsonl
cp artifacts/runs/temp_0.2_runs.jsonl artifacts/runs/conservative_runs.jsonl
cp artifacts/runs/temp_0.7_runs.jsonl artifacts/runs/creative_runs.jsonl

echo "Running decoding-variation experiments..."
"${PYTHON_BIN}" scripts/run_agent_validation.py \
  --split validation \
  --use-full-split \
  --experiment-name top_p_0.8 \
  --temperature 0.4 \
  --top-p 0.8 \
  --out artifacts/runs/top_p_0.8_runs.jsonl

"${PYTHON_BIN}" scripts/run_agent_validation.py \
  --split validation \
  --use-full-split \
  --experiment-name max_tokens_384 \
  --temperature 0.4 \
  --max-tokens 384 \
  --out artifacts/runs/max_tokens_384_runs.jsonl

"${PYTHON_BIN}" scripts/run_agent_validation.py \
  --split validation \
  --use-full-split \
  --experiment-name freq_penalty_0.3 \
  --temperature 0.4 \
  --freq-penalty 0.3 \
  --out artifacts/runs/freq_penalty_0.3_runs.jsonl

echo "Saving aggregate evaluation JSON..."
"${PYTHON_BIN}" -m travel_planner.eval.run_eval artifacts/runs/*.jsonl --use-reference > artifacts/eval/full_report_evaluation.json

echo "Pipeline complete. Key outputs:"
echo "  - artifacts/models/solver-lora"
echo "  - artifacts/runs/*.jsonl"
echo "  - artifacts/eval/full_report_evaluation.json"
