#!/usr/bin/env bash
set -euo pipefail

# Full rebuild script for TravelPlanner project
# Usage: GROQ_API_KEY=xxx ./scripts/full_rebuild.sh  (optional OPENAI_API_KEY fallback)

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "Cleaning artifacts..."
rm -rf artifacts/*
mkdir -p artifacts

echo "Building SFT dataset..."
PYTHONPATH=src "/media/hazem/Windows-SSD/Desktop/BUE/Year 3/Semester 2/NLP/Phase 2/phase2/.venv/bin/python" -m travel_planner.training.build_sft_dataset --out artifacts/data/sft_train.jsonl

echo "Creating smoke-train subset..."
head -n 20 artifacts/data/sft_train.jsonl > artifacts/sft_train_smoke.jsonl

echo "Training LoRA adapter (short run)..."
# Adjust backend/model/epochs as needed for larger training
USE_LOCAL_SOLVER_MODEL=1 PYTHONPATH=src "/media/hazem/Windows-SSD/Desktop/BUE/Year 3/Semester 2/NLP/Phase 2/phase2/.venv/bin/python" -m travel_planner.training.train_qlora_unsloth --backend transformers --train_file artifacts/sft_train_smoke.jsonl --output_dir artifacts/models/solver-lora --model_name unsloth/Llama-3.2-3B-bnb-4bit --max_seq_length 1024 --epochs 1 --batch_size 1 --grad_accum 4

echo "Running baseline and enhanced experiments (full validation split)..."
PYTHONPATH=src "/media/hazem/Windows-SSD/Desktop/BUE/Year 3/Semester 2/NLP/Phase 2/phase2/.venv/bin/python" scripts/run_agent_validation.py --split validation --experiment-name baseline --set-baseline --temperature 0.4 --use-local-model --lora-path artifacts/models/solver-lora --out artifacts/runs/baseline_runs.jsonl
PYTHONPATH=src "/media/hazem/Windows-SSD/Desktop/BUE/Year 3/Semester 2/NLP/Phase 2/phase2/.venv/bin/python" scripts/run_agent_validation.py --split validation --experiment-name enhanced --temperature 0.4 --use-local-model --lora-path artifacts/models/solver-lora --out artifacts/runs/enhanced_runs.jsonl

echo "Temperature sweep (0.2,0.7,1.0)..."
PYTHONPATH=src "/media/hazem/Windows-SSD/Desktop/BUE/Year 3/Semester 2/NLP/Phase 2/phase2/.venv/bin/python" scripts/run_temperature_experiments.py --split validation --temperatures 0.2 0.7 1.0 --use-local-model --lora-path artifacts/models/solver-lora

echo "Conservative & creative runs..."
PYTHONPATH=src "/media/hazem/Windows-SSD/Desktop/BUE/Year 3/Semester 2/NLP/Phase 2/phase2/.venv/bin/python" scripts/run_agent_validation.py --split validation --experiment-name conservative --temperature 0.2 --use-local-model --lora-path artifacts/models/solver-lora --out artifacts/runs/conservative_runs.jsonl
PYTHONPATH=src "/media/hazem/Windows-SSD/Desktop/BUE/Year 3/Semester 2/NLP/Phase 2/phase2/.venv/bin/python" scripts/run_agent_validation.py --split validation --experiment-name creative --temperature 0.7 --use-local-model --lora-path artifacts/models/solver-lora --out artifacts/runs/creative_runs.jsonl

echo "Running aggregated evaluation..."
PYTHONPATH=src "/media/hazem/Windows-SSD/Desktop/BUE/Year 3/Semester 2/NLP/Phase 2/phase2/.venv/bin/python" -m travel_planner.eval.run_eval artifacts/*.jsonl --use-reference

echo "Full rebuild complete. Artifacts are in $REPO_ROOT/artifacts"
if [ -z "${GROQ_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "Note: Set GROQ_API_KEY (or OPENAI_API_KEY) for full constraint extraction / audit with Groq or OpenAI."
fi
