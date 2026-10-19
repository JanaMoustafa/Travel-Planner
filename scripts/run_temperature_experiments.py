#!/usr/bin/env python3
"""Run temperature variation experiments with tool usage tracking."""

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
from travel_planner.experiment import ExperimentTracker
from travel_planner.graph.model_loader import should_use_local_model
from travel_planner.graph.workflow import run_travel_planner

logger = logging.getLogger(__name__)


def run_temperature_experiments() -> None:
    """Run validation experiments with multiple temperatures."""
    
    p = argparse.ArgumentParser(description="Run temperature variation experiments with tools.")
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/runs"))
    p.add_argument(
        "--output-prefix",
        default="temp",
        help="Prefix used when writing run files, e.g. temp_0.4_runs.jsonl",
    )
    p.add_argument("--limit", type=int, default=3, help="Number of validation rows per temperature")
    p.add_argument(
        "--use-full-split",
        action="store_true",
        help="Process the entire dataset split and ignore --limit",
    )
    p.add_argument(
        "--temperatures",
        nargs="+",
        type=float,
        default=[0.2, 0.7, 1.0],
        help="Temperature values to test (default 0.2 0.7 1.0)",
    )
    p.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Optional nucleus sampling top_p for Groq/OpenAI solver path",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional max_tokens cap for solver completions",
    )
    p.add_argument(
        "--freq-penalty",
        type=float,
        default=None,
        dest="frequency_penalty",
        help="Optional frequency_penalty for solver completions",
    )
    p.add_argument("--use-local-model", action="store_true", help="Use local LoRA model")
    p.add_argument("--lora-path", type=Path, default=None, help="Path to local LoRA adapter")
    p.add_argument(
        "--split",
        default="validation",
        choices=["train", "validation", "test"],
        help="Dataset split to run",
    )
    p.add_argument("--token", default=HF_TOKEN)
    p.add_argument(
        "--row-cooldown",
        type=float,
        default=float(os.environ.get("ROW_COOLDOWN_SECONDS", "0") or "0"),
        help="Seconds to sleep between rows to reduce API rate-limit pressure",
    )
    args = p.parse_args()
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = args.out_dir.parent / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    if args.use_local_model:
        os.environ["USE_LOCAL_SOLVER_MODEL"] = "1"
        if args.lora_path is not None:
            os.environ["SOLVER_LORA_PATH"] = str(args.lora_path)
    else:
        os.environ["USE_LOCAL_SOLVER_MODEL"] = "0"
    
    print("\n" + "=" * 80)
    print("TEMPERATURE EXPERIMENTS WITH TOOL USAGE")
    print("=" * 80)
    print(f"Temperatures: {args.temperatures}")
    print(f"Queries per temperature: {args.limit}")
    print(f"top_p={args.top_p} max_tokens={args.max_tokens} frequency_penalty={args.frequency_penalty}")
    print("Tools: Search, RAG, Amadeus (flights/hotels), Open-Meteo, Frankfurter")
    print("=" * 80 + "\n")
    
    ds = load_split(split=args.split, token=args.token)
    run_count = len(ds) if args.use_full_split else min(args.limit, len(ds))
    
    if should_use_local_model():
        logger.info("Using local LoRA model for inference")
    else:
        logger.info("Using Groq/OpenAI chat API for inference")
    
    all_results = []
    temperature_metrics = {}
    
    for temp in args.temperatures:
        print(f"\n{'='*80}")
        print(f"TEMPERATURE: {temp}")
        print(f"{'='*80}\n")
        
        out_file = args.out_dir / f"{args.output_prefix}_{temp}_runs.jsonl"
        temp_results = []
        
        with out_file.open("w", encoding="utf-8") as f:
            for i in range(run_count):
                row = ds[i]
                meta = {k: row[k] for k in row.keys() if k != "query"}
                
                print(f"  Query {i+1}/{run_count}: {row['query'][:60]}...")
                
                try:
                    # Run with specific temperature
                    st = run_travel_planner(
                        row["query"],
                        row_metadata=meta,
                        row_id=i,
                        temperature=temp,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        frequency_penalty=args.frequency_penalty,
                    )

                    rec = {
                        "row_index": i,
                        "split": args.split,
                        "temperature": temp,
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
                    
                    temp_results.append(rec)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    
                    # Log tool usage from message history
                    if rec.get("message_history"):
                        if "[TOOL]" in rec["message_history"]:
                            tool_lines = [line for line in rec["message_history"].split("\n") 
                                        if "[TOOL]" in line]
                            if tool_lines:
                                print(f"    Tools used: {len(tool_lines)}")
                                for tool_line in tool_lines[:2]:  # Show first 2
                                    print(f"       - {tool_line[:70]}")
                    
                    if rec.get("audit_passed"):
                        print(f"    Audit PASSED")
                    else:
                        print(f"    Audit FAILED")
                        
                except Exception as e:
                    logger.error(f"Query {i} failed: {e}", exc_info=False)
                    rec = {
                        "row_index": i,
                        "temperature": temp,
                        "error": str(e),
                    }
                    temp_results.append(rec)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"    Error: {str(e)[:60]}")
                if args.row_cooldown > 0 and i < (run_count - 1):
                    logger.info("Cooling down for %.1f seconds before next row", args.row_cooldown)
                    time.sleep(args.row_cooldown)
        
        print(f"\nWrote {len(temp_results)} results to {out_file}")
        
        # Track temperature-specific metrics
        passed = sum(1 for r in temp_results if r.get("audit_passed"))
        temperature_metrics[temp] = {
            "total": len(temp_results),
            "passed": passed,
            "pass_rate": passed / len(temp_results) if temp_results else 0.0,
            "artifact": str(out_file),
        }
        
        all_results.extend(temp_results)
        
        # Run evaluation for this temperature
        logger.info(f"Running evaluation for temperature {temp}...")
        
        eval_args = [
            sys.executable,
            "-m",
            "travel_planner.eval.run_eval",
            str(out_file),
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
        
        if result.returncode == 0:
            try:
                eval_output = json.loads(result.stdout)
                agg = eval_output.get("aggregate", {})
                temperature_metrics[temp].update({
                    "delivery_rate": agg.get("delivery_rate", 0.0),
                    "hard_constraint_rate": agg.get("hard_constraint_rate", 0.0),
                    "avg_rouge_l": agg.get("avg_rouge_l"),
                    "avg_bleu": agg.get("avg_bleu"),
                })
            except Exception as e:
                logger.warning(f"Could not parse eval output: {e}")
        else:
            logger.warning(f"Evaluation failed for temperature {temp}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("TEMPERATURE EXPERIMENT SUMMARY")
    print(f"{'='*80}\n")
    
    for temp in args.temperatures:
        metrics = temperature_metrics.get(temp, {})
        print(f"Temperature {temp}:")
        print(f"  Pass Rate: {metrics.get('pass_rate', 0)*100:.1f}% ({metrics.get('passed', 0)}/{metrics.get('total', 0)})")
        print(f"  Delivery Rate: {metrics.get('delivery_rate', 0)*100:.1f}%")
        print(f"  Hard Constraint Rate: {metrics.get('hard_constraint_rate', 0)*100:.1f}%")
        if metrics.get('avg_rouge_l') is not None:
            print(f"  Avg ROUGE-L: {metrics.get('avg_rouge_l', 0):.3f}")
        if metrics.get('avg_bleu') is not None:
            print(f"  Avg BLEU: {metrics.get('avg_bleu', 0):.4f}")
        print()
    
    # Summary file
    summary_file = eval_dir / f"{args.output_prefix}_experiment_summary.json"
    with summary_file.open("w") as f:
        json.dump({
            "output_prefix": args.output_prefix,
            "temperatures": args.temperatures,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "frequency_penalty": args.frequency_penalty,
            "metrics": temperature_metrics,
            "total_examples": len(all_results),
        }, f, indent=2)
    
    print(f"Summary saved to {summary_file}\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    run_temperature_experiments()
