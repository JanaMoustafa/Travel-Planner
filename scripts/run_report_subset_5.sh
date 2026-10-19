#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/../.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export PREFER_OPENAI="${PREFER_OPENAI:-0}"
export MAX_AUDIT_ITERATIONS="${MAX_AUDIT_ITERATIONS:-1}"
export ROW_COOLDOWN_SECONDS="${ROW_COOLDOWN_SECONDS:-8}"

cd "${ROOT}"

mkdir -p artifacts/runs artifacts/eval artifacts/logs artifacts/experiments
rm -f artifacts/runs/*.jsonl artifacts/eval/*.json artifacts/eval/*.csv
: > artifacts/experiments/manifest.jsonl

LIMIT="${LIMIT:-5}"
LORA_PATH="${LORA_PATH:-artifacts/models/solver-lora}"

echo "Running report subset pipeline with LIMIT=${LIMIT}"
echo "Low-rate-limit mode: MAX_AUDIT_ITERATIONS=${MAX_AUDIT_ITERATIONS}, ROW_COOLDOWN_SECONDS=${ROW_COOLDOWN_SECONDS}"

echo "Baseline API validation..."
"${PYTHON_BIN}" scripts/run_agent_validation.py \
  --split validation \
  --limit "${LIMIT}" \
  --experiment-name baseline \
  --set-baseline \
  --temperature 0.4 \
  --out artifacts/runs/baseline_runs.jsonl

echo "Enhanced local-model validation..."
"${PYTHON_BIN}" scripts/run_agent_validation.py \
  --split validation \
  --limit "${LIMIT}" \
  --experiment-name enhanced \
  --temperature 0.4 \
  --use-local-model \
  --lora-path "${LORA_PATH}" \
  --out artifacts/runs/enhanced_runs.jsonl

echo "Temperature sweep without duplicate equivalent runs..."
"${PYTHON_BIN}" scripts/run_temperature_experiments.py \
  --split validation \
  --limit "${LIMIT}" \
  --out-dir artifacts/runs \
  --output-prefix temp \
  --temperatures 0.2 0.7

echo "Decoding variations..."
"${PYTHON_BIN}" scripts/run_agent_validation.py \
  --split validation \
  --limit "${LIMIT}" \
  --experiment-name top_p_0.8 \
  --temperature 0.4 \
  --top-p 0.8 \
  --out artifacts/runs/top_p_0.8_runs.jsonl

"${PYTHON_BIN}" scripts/run_agent_validation.py \
  --split validation \
  --limit "${LIMIT}" \
  --experiment-name max_tokens_384 \
  --temperature 0.4 \
  --max-tokens 384 \
  --out artifacts/runs/max_tokens_384_runs.jsonl

"${PYTHON_BIN}" scripts/run_agent_validation.py \
  --split validation \
  --limit "${LIMIT}" \
  --experiment-name freq_penalty_0.3 \
  --temperature 0.4 \
  --freq-penalty 0.3 \
  --out artifacts/runs/freq_penalty_0.3_runs.jsonl

echo "Saving aggregate evaluation JSON..."
"${PYTHON_BIN}" -m travel_planner.eval.run_eval artifacts/runs/*.jsonl --use-reference > artifacts/eval/report_subset_5_evaluation.json

echo "Subset pipeline complete."
echo "Outputs:"
echo "  - artifacts/runs/*.jsonl"
echo "  - artifacts/eval/report_subset_5_evaluation.json"
