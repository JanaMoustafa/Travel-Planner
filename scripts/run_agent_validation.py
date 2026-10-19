#!/usr/bin/env python3
"""Run the multi-agent system on TravelPlanner validation rows; write JSONL results."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from travel_planner.config import HF_TOKEN
from travel_planner.data.load_travelplanner import load_split
from travel_planner.experiment import ExperimentRun, ExperimentTracker
from travel_planner.graph.model_loader import should_use_local_model
from travel_planner.graph.workflow import run_travel_planner

logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run multi-agent validation with experiment tracking."
    )
    p.add_argument("--out", type=Path, default=Path("artifacts/validation_runs.jsonl"))
    p.add_argument("--limit", type=int, default=5, help="Number of validation rows to run")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument(
        "--experiment-name",
        default="unnamed",
        help="Name for this experiment run",
    )
    p.add_argument(
        "--split",
        default="validation",
        choices=["train", "validation", "test"],
        help="Dataset split to run",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Sampling temperature for solver inference",
    )
    p.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Optional nucleus sampling top_p for solver inference",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional max_tokens cap for solver inference",
    )
    p.add_argument(
        "--freq-penalty",
        type=float,
        default=None,
        dest="frequency_penalty",
        help="Optional frequency_penalty for solver inference",
    )
    p.add_argument(
        "--use-local-model",
        action="store_true",
        help="Use the local LoRA solver instead of the API",
    )
    p.add_argument(
        "--use-full-split",
        action="store_true",
        help="Process the entire dataset split and ignore --limit",
    )
    p.add_argument(
        "--lora-path",
        type=Path,
        default=None,
        help="Path to the LoRA adapter to load when using the local model",
    )
    p.add_argument(
        "--set-baseline",
        action="store_true",
        help="Mark this run as baseline for future comparisons",
    )
    p.add_argument("--token", default=HF_TOKEN)
    p.add_argument(
        "--row-cooldown",
        type=float,
        default=float(os.environ.get("ROW_COOLDOWN_SECONDS", "0") or "0"),
        help="Seconds to sleep between rows to reduce API rate-limit pressure",
    )
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.use_local_model:
        os.environ["USE_LOCAL_SOLVER_MODEL"] = "1"
        if args.lora_path is not None:
            os.environ["SOLVER_LORA_PATH"] = str(args.lora_path)
    else:
        os.environ["USE_LOCAL_SOLVER_MODEL"] = "0"

    ds = load_split(split=args.split, token=args.token)
    if args.use_full_split:
        n = max(0, len(ds) - args.offset)
    else:
        n = min(args.limit, max(0, len(ds) - args.offset))

    print("\n" + "=" * 80)
    print("MULTI-AGENT TRAVEL PLANNER")
    print("=" * 80)
    print("System: Message-based communication with async coordination")
    print("Features: Parallel constraint extraction and research")
    print("=" * 80 + "\n")

    if should_use_local_model():
        logger.info("Using local LoRA model for inference")
    else:
        logger.info("Using OpenAI API for inference")

    with args.out.open("w", encoding="utf-8") as f:
        for i in range(args.offset, args.offset + n):
            row = ds[i]
            meta = {k: row[k] for k in row.keys() if k != "query"}
            
            logger.info(f"Processing row {i + 1}/{n}")
            st = run_travel_planner(
                row["query"],
                row_metadata=meta,
                row_id=i,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                frequency_penalty=args.frequency_penalty,
            )
            
            rec = {
                "row_index": i,
                "split": args.split,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "frequency_penalty": args.frequency_penalty,
                "plan": st.get("draft_plan", ""),
                "constraints": st.get("constraints", {}),
                "audit_passed": st.get("audit_passed"),
                "audit_feedback": st.get("audit_feedback"),
                "solver_calls": st.get("solver_calls"),
                "error": st.get("error"),
                "message_history": st.get("message_history", ""),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if args.row_cooldown > 0 and i < (args.offset + n - 1):
                logger.info("Cooling down for %.1f seconds before next row", args.row_cooldown)
                time.sleep(args.row_cooldown)

    print(f"Wrote {n} rows to {args.out}")

    logger.info("Running evaluation...")
    
    eval_args = [
        sys.executable,
        "-m",
        "travel_planner.eval.run_eval",
        str(args.out),
        "--use-reference",
    ]
    
    eval_env = dict(os.environ)
    eval_env["PYTHONPATH"] = str(ROOT / "src") + (
        os.pathsep + eval_env["PYTHONPATH"] if eval_env.get("PYTHONPATH") else ""
    )
    result = subprocess.run(
        eval_args,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=eval_env,
    )
    
    if result.returncode != 0:
        logger.error(f"Evaluation failed: {result.stderr}")
        print("Evaluation failed. Run manually with:")
        print(f"  python -m travel_planner.eval.run_eval {args.out} --use-reference")
        return

    try:
        eval_output = json.loads(result.stdout)
        agg = eval_output.get("aggregate", {})
        
        tracker = ExperimentTracker()
        run = ExperimentRun(
            name=args.experiment_name,
            model_type="lora" if should_use_local_model() else "openai",
            model_name=(
                "unsloth/Llama-3.1-8B"
                if should_use_local_model()
                else "gpt-4o-mini"
            ),
            delivery_rate=agg.get("delivery_rate", 0.0),
            commonsense_rate=agg.get("commonsense_rate", 0.0),
            hard_constraint_rate=agg.get("hard_constraint_rate", 0.0),
            avg_rouge_l=agg.get("avg_rouge_l"),
            avg_constraint_adherence=agg.get("avg_constraint_adherence"),
            num_examples=int(agg.get("n", 0)),
            hyperparams={
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "frequency_penalty": args.frequency_penalty,
                "use_local_model": args.use_local_model,
            },
            is_baseline=args.set_baseline,
        )
        tracker.save_run(run)
        
        tracker.print_history()
        tracker.print_comparison(run)
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse evaluation output: {e}")
        print("Failed to parse evaluation. Raw output:")
        print(result.stdout)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
