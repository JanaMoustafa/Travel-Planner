"""CLI: evaluate JSONL outputs against HuggingFace validation rows.

IMPROVEMENT: Now supports ground truth comparison with --use-reference flag.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from collections import defaultdict

from travel_planner.config import HF_TOKEN
from travel_planner.data.load_travelplanner import load_split

logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate generated travel plans against validation set."
    )
    p.add_argument(
        "outputs_jsonl",
        nargs="+",
        type=Path,
        help="One or more JSONL files with keys: row_index (optional), plan (or output), constraints (optional JSON)",
    )
    p.add_argument("--split", default="validation", choices=["validation", "train", "test"])
    p.add_argument(
        "--use-reference",
        action="store_true",
        help="Compare against annotated_plan (rouge_score + evaluate BLEU). "
        "Activates improved metrics: ROUGE-L, BLEU, constraint adherence.",
    )
    p.add_argument("--token", default=HF_TOKEN)
    args = p.parse_args()

    ds = load_split(split=args.split, token=args.token)
    all_records: list[dict[str, object]] = []
    file_summaries: list[dict[str, object]] = []

    for outputs_path in args.outputs_jsonl:
        rows = []
        with outputs_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))

        if rows and any("instruction" in row and "row_index" not in row for row in rows):
            logger.info("Skipping non-experiment JSONL file: %s", outputs_path)
            continue

        file_results = []
    
        if args.use_reference:
            try:
                from travel_planner.eval.metrics_improved import (
                    aggregate_improved,
                    compute_generalization_score,
                    evaluate_record_with_reference,
                )
            except ImportError:
                logger.error("metrics_improved module not found")
                raise

            logger.info("Using improved metrics with ground truth comparison")
            for i, rec in enumerate(rows):
                idx = rec.get("row_index", i)
                row = ds[int(idx)]
                plan = rec.get("plan") or rec.get("output") or ""
                cons = rec.get("constraints")
                if isinstance(cons, str):
                    try:
                        cons = json.loads(cons)
                    except Exception:
                        cons = None

                reference = row.get("annotated_plan") or row.get("reference_information") or ""
                if not reference:
                    logger.warning("Skipping unlabeled row %s from %s", idx, outputs_path)
                    continue

                er = evaluate_record_with_reference(
                    plan, reference, row=row, constraints=cons
                )
                er["row_index"] = idx
                er["source_file"] = str(outputs_path)
                er["generated_plan"] = plan
                if rec.get("temperature") is not None:
                    er["temperature"] = rec.get("temperature")
                file_results.append(er)
                all_records.append(er)

            file_generalization = None
            if len(rows) > 0:
                grouped: dict[int, list[str]] = defaultdict(list)
                for rec in rows:
                    idx = int(rec.get("row_index", 0))
                    plan = rec.get("plan") or rec.get("output") or ""
                    if plan:
                        grouped[idx].append(plan)
                pair_scores = []
                for plans in grouped.values():
                    score = compute_generalization_score(plans)
                    if score is not None:
                        pair_scores.append(score)
                if pair_scores:
                    file_generalization = sum(pair_scores) / len(pair_scores)

            file_agg = aggregate_improved(file_results, generalization_score=file_generalization)
        else:
            from travel_planner.eval.metrics import aggregate, evaluate_record

            logger.info("Using basic heuristic metrics (no ground truth)")

            for i, rec in enumerate(rows):
                idx = rec.get("row_index", i)
                row = ds[int(idx)]
                plan = rec.get("plan") or rec.get("output") or ""
                cons = rec.get("constraints")
                if isinstance(cons, str):
                    try:
                        cons = json.loads(cons)
                    except Exception:
                        cons = None
                er = evaluate_record(plan, row=row, constraints=cons)
                er["row_index"] = idx
                er["source_file"] = str(outputs_path)
                er["generated_plan"] = plan
                if rec.get("temperature") is not None:
                    er["temperature"] = rec.get("temperature")
                file_results.append(er)
                all_records.append(er)

            file_agg = aggregate(file_results)

        file_summaries.append({"artifact": str(outputs_path), "aggregate": file_agg})

    if args.use_reference:
        from travel_planner.eval.metrics_improved import aggregate_improved, compute_generalization_score

        overall_generalization = None
        temp_groups: dict[int, list[str]] = defaultdict(list)
        for rec in all_records:
            idx = int(rec.get("row_index", 0))
            source_plan = rec.get("generated_plan") or rec.get("plan") or rec.get("output") or ""
            if source_plan:
                temp_groups[idx].append(source_plan)
        scores = [s for s in (compute_generalization_score(plans) for plans in temp_groups.values()) if s is not None]
        if scores:
            overall_generalization = sum(scores) / len(scores)
        agg = aggregate_improved(all_records, generalization_score=overall_generalization)
    else:
        from travel_planner.eval.metrics import aggregate
        agg = aggregate(all_records)

    output = {"aggregate": agg, "per_row": all_records, "files": file_summaries}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

