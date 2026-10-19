"""Experiment tracking for iterative improvement.

This module was added to fix the critical missing requirement:
iterative improvement with baseline vs improved comparison.

Previously, there was NO way to track experiments, compare models, or
prove that changes actually improved performance.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExperimentRun:
    """Single experiment run record.
    
    Captures:
    - Model configuration (API vs LoRA)
    - Evaluation metrics
    - Improvement vs previous run
    """
    
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    name: str = ""
    description: str = ""
    
    # Model config
    model_type: str = "openai"  # "openai" or "lora"
    model_name: str = "gpt-4o-mini"
    
    # Eval metrics
    delivery_rate: float = 0.0
    commonsense_rate: float = 0.0
    hard_constraint_rate: float = 0.0
    avg_rouge_l: float | None = None
    avg_constraint_adherence: float | None = None
    
    # Metadata
    num_examples: int = 0
    notes: str = ""
    hyperparams: dict[str, Any] = field(default_factory=dict)
    
    # Improvement tracking
    improvement_over_previous: float | None = None  # % improvement in avg score
    is_baseline: bool = False


class ExperimentTracker:
    """Track and compare experiments across runs.
    
    This was added to solve the MISSING improvement/iteration requirement.
    Allows:
    - Saving baseline metrics
    - Comparing new runs against baseline
    - Tracking improvement over time
    """
    
    def __init__(self, log_dir: Path = Path("artifacts/experiments")):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.log_dir / "manifest.jsonl"
        logger.info(f"ExperimentTracker initialized at {self.log_dir}")
    
    def save_run(self, run: ExperimentRun) -> None:
        """Save experiment run to manifest.
        
        Args:
            run: ExperimentRun to save
        """
        with self.manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(run), default=str) + "\n")
        logger.info(f"Saved run {run.run_id}: {run.name}")
    
    def get_runs(self) -> list[ExperimentRun]:
        """Load all runs from manifest.
        
        Returns:
            List of all ExperimentRun objects, in chronological order
        """
        if not self.manifest_path.exists():
            return []
        
        runs = []
        with self.manifest_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        runs.append(ExperimentRun(**data))
                    except Exception as e:
                        logger.warning(f"Failed to parse run: {e}")
        return runs
    
    def get_baseline(self) -> ExperimentRun | None:
        """Get baseline run (if marked).
        
        Returns:
            The run marked as baseline, or None
        """
        for run in self.get_runs():
            if run.is_baseline:
                logger.info(f"Found baseline: {run.run_id} ({run.name})")
                return run
        logger.warning("No baseline run found")
        return None
    
    def get_latest(self) -> ExperimentRun | None:
        """Get most recent run.
        
        Returns:
            Most recent ExperimentRun, or None
        """
        runs = self.get_runs()
        return runs[-1] if runs else None
    
    def compare_to_baseline(self, run: ExperimentRun) -> dict[str, Any]:
        """Compare run against baseline.
        
        Args:
            run: ExperimentRun to compare
            
        Returns:
            Comparison dict with improvements
        """
        baseline = self.get_baseline()
        if not baseline:
            logger.warning("No baseline to compare against")
            return {}
        
        # Calculate average score (unweighted average of all metrics)
        baseline_avg = (baseline.delivery_rate + baseline.hard_constraint_rate) / 2
        run_avg = (run.delivery_rate + run.hard_constraint_rate) / 2
        
        improvement = ((run_avg - baseline_avg) / max(baseline_avg, 0.01)) * 100
        
        return {
            "baseline_id": baseline.run_id,
            "current_id": run.run_id,
            "baseline_avg_score": baseline_avg,
            "current_avg_score": run_avg,
            "improvement_percent": improvement,
            "delivery_delta": (run.delivery_rate - baseline.delivery_rate) * 100,
            "constraint_delta": (run.hard_constraint_rate - baseline.hard_constraint_rate) * 100,
            "rouge_delta": (
                (run.avg_rouge_l - baseline.avg_rouge_l) * 100
                if run.avg_rouge_l and baseline.avg_rouge_l else None
            ),
        }
    
    def compare_runs(self, run_a_id: str, run_b_id: str) -> dict[str, Any]:
        """Compare two runs side-by-side.
        
        Args:
            run_a_id: First run ID
            run_b_id: Second run ID
            
        Returns:
            Comparison dict
        """
        runs = {run.run_id: run for run in self.get_runs()}
        run_a = runs.get(run_a_id)
        run_b = runs.get(run_b_id)
        
        if not run_a or not run_b:
            raise ValueError(f"Run not found: {run_a_id} or {run_b_id}")
        
        return {
            "run_a": asdict(run_a),
            "run_b": asdict(run_b),
            "delivery_improvement": (run_b.delivery_rate - run_a.delivery_rate) * 100,
            "constraint_improvement": (run_b.hard_constraint_rate - run_a.hard_constraint_rate) * 100,
            "rouge_improvement": (
                (run_b.avg_rouge_l - run_a.avg_rouge_l) * 100
                if run_a.avg_rouge_l and run_b.avg_rouge_l else None
            ),
        }
    
    def print_history(self) -> None:
        """Print formatted experiment history."""
        runs = self.get_runs()
        if not runs:
            print("No experiments recorded yet")
            return
        
        print("\n" + "=" * 120)
        print("EXPERIMENT HISTORY")
        print("=" * 120)
        print(f"{'Date':<12} {'ID':<8} {'Name':<20} {'Model':<20} {'Delivery':<12} {'Constraint':<12} {'ROUGE':<12} {'Baseline'}")
        print("-" * 120)
        
        for run in runs:
            marker = "BASELINE" if run.is_baseline else ""
            rouge_str = f"{run.avg_rouge_l:.2%}" if run.avg_rouge_l else "N/A"
            print(
                f"{run.timestamp[:10]:<12} {run.run_id:<8} {run.name[:20]:<20} "
                f"{run.model_type[:20]:<20} {run.delivery_rate:<12.2%} "
                f"{run.hard_constraint_rate:<12.2%} {rouge_str:<12} {marker}"
            )
        
        print("=" * 120 + "\n")
    
    def print_comparison(self, run: ExperimentRun) -> None:
        """Print comparison to baseline."""
        comparison = self.compare_to_baseline(run)
        if not comparison:
            return
        
        print("\n" + "=" * 80)
        print("COMPARISON TO BASELINE")
        print("=" * 80)
        print(f"Baseline: {comparison['baseline_id']} (Avg Score: {comparison['baseline_avg_score']:.2%})")
        print(f"Current:  {comparison['current_id']} (Avg Score: {comparison['current_avg_score']:.2%})")
        print("-" * 80)
        
        improvement = comparison['improvement_percent']
        print(f"Overall Improvement: {improvement:+.1f}%")
        
        delivery_delta = comparison['delivery_delta']
        print(f"Delivery Rate: {delivery_delta:+.1f}%")
        
        constraint_delta = comparison['constraint_delta']
        print(f"Constraint Rate: {constraint_delta:+.1f}%")
        
        if comparison['rouge_delta'] is not None:
            rouge_delta = comparison['rouge_delta']
            print(f"ROUGE-L: {rouge_delta:+.1f}%")
        
        print("=" * 80 + "\n")
