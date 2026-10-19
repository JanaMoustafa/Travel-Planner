from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


_RE_CONTRADICTORY_DAY = re.compile(r"(\d+)\s*days?.*?(\d+)\s*days?", re.IGNORECASE)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def contradictory_feedback_count(rows: list[dict]) -> int:
    count = 0
    for row in rows:
        feedback = str(row.get("audit_feedback") or "")
        lowered = feedback.lower()
        if "matches (" in lowered or "is correct" in lowered:
            count += 1
            continue
        day_match = _RE_CONTRADICTORY_DAY.search(feedback)
        if day_match and day_match.group(1) == day_match.group(2):
            if any(token in lowered for token in ("mismatch", "insufficient", "requested duration", "<", "vs")):
                count += 1
                continue
        if "!=" in feedback:
            pieces = [piece.strip().lower() for piece in feedback.split("!=")[:2]]
            if len(pieces) == 2 and pieces[0] == pieces[1]:
                count += 1
    return count


def summarize_runs(runs_dir: Path) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for path in sorted(runs_dir.glob("*_runs.jsonl")):
        rows = load_jsonl(path)
        audit_passes = sum(1 for row in rows if row.get("audit_passed"))
        avg_solver_calls = (sum(float(row.get("solver_calls") or 0) for row in rows) / len(rows)) if rows else 0.0
        avg_plan_chars = (sum(len(str(row.get("plan") or "")) for row in rows) / len(rows)) if rows else 0.0
        summary[path.name] = {
            "rows": len(rows),
            "audit_pass_rate": (audit_passes / len(rows)) if rows else 0.0,
            "avg_solver_calls": avg_solver_calls,
            "avg_plan_chars": avg_plan_chars,
            "contradictory_audit_count": contradictory_feedback_count(rows),
        }
    return summary


def file_metric_map(eval_payload: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for item in eval_payload.get("files", []):
        out[Path(item["artifact"]).name] = item.get("aggregate", {})
    return out


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_float(value: float) -> str:
    return f"{value:.3f}"


def build_report(old_eval: dict, new_eval: dict, old_runs: dict, new_runs: dict) -> tuple[str, dict]:
    old_files = file_metric_map(old_eval)
    new_files = file_metric_map(new_eval)
    names = sorted(set(old_files) | set(new_files) | set(old_runs) | set(new_runs))

    overall = {
        "old": old_eval.get("aggregate", {}),
        "new": new_eval.get("aggregate", {}),
    }

    detail_rows: list[dict] = []
    for name in names:
        old_metrics = old_files.get(name, {})
        new_metrics = new_files.get(name, {})
        old_run = old_runs.get(name, {})
        new_run = new_runs.get(name, {})
        detail_rows.append(
            {
                "experiment": name,
                "old": {
                    "delivery_rate": old_metrics.get("delivery_rate", 0.0),
                    "hard_constraint_rate": old_metrics.get("hard_constraint_rate", 0.0),
                    "avg_rouge1": old_metrics.get("avg_rouge1", 0.0),
                    "avg_rougeL": old_metrics.get("avg_rougeL", 0.0),
                    "avg_bleu": old_metrics.get("avg_bleu", 0.0),
                    "audit_pass_rate": old_run.get("audit_pass_rate", 0.0),
                    "avg_solver_calls": old_run.get("avg_solver_calls", 0.0),
                    "avg_plan_chars": old_run.get("avg_plan_chars", 0.0),
                    "contradictory_audit_count": old_run.get("contradictory_audit_count", 0),
                },
                "new": {
                    "delivery_rate": new_metrics.get("delivery_rate", 0.0),
                    "hard_constraint_rate": new_metrics.get("hard_constraint_rate", 0.0),
                    "avg_rouge1": new_metrics.get("avg_rouge1", 0.0),
                    "avg_rougeL": new_metrics.get("avg_rougeL", 0.0),
                    "avg_bleu": new_metrics.get("avg_bleu", 0.0),
                    "audit_pass_rate": new_run.get("audit_pass_rate", 0.0),
                    "avg_solver_calls": new_run.get("avg_solver_calls", 0.0),
                    "avg_plan_chars": new_run.get("avg_plan_chars", 0.0),
                    "contradictory_audit_count": new_run.get("contradictory_audit_count", 0),
                },
            }
        )

    best_rouge = max(
        detail_rows,
        key=lambda item: item["new"]["avg_rouge1"] - item["old"]["avg_rouge1"],
        default=None,
    )
    best_constraints = max(
        detail_rows,
        key=lambda item: item["new"]["hard_constraint_rate"] - item["old"]["hard_constraint_rate"],
        default=None,
    )

    lines: list[str] = []
    lines.append("# Final Pipeline Comparison Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "This report compares the finished old 5-row reduced pipeline against the improved 5-row Groq-first rerun "
        "with a stricter solver prompt, contradiction-filtered auditing, and up to 2 audit iterations."
    )
    lines.append("")
    lines.append("| Metric | Old | New | Delta |")
    lines.append("| --- | ---: | ---: | ---: |")
    for key in ("delivery_rate", "commonsense_rate", "hard_constraint_rate", "avg_rouge1", "avg_rougeL", "avg_bleu", "avg_constraint_score"):
        old_value = float(overall["old"].get(key, 0.0))
        new_value = float(overall["new"].get(key, 0.0))
        lines.append(f"| {key} | {format_float(old_value)} | {format_float(new_value)} | {format_float(new_value - old_value)} |")
    lines.append("")

    lines.append("## What Changed")
    lines.append("")
    lines.append("- Auditor logic now filters contradictory mismatch/day/budget claims instead of trusting them blindly.")
    lines.append("- Groq remains the main solver path for the final report runs; the LoRA model is kept as the comparison condition only.")
    lines.append("- Final subset rerun uses up to 2 audit iterations, allowing one revision pass when time permits.")
    lines.append("- Solver prompt now explicitly prioritizes origin/destination correctness, exact day coverage, and budget accounting.")
    lines.append("")

    lines.append("## Experiment Details")
    lines.append("")
    lines.append("| Experiment | Audit Pass Old | Audit Pass New | Hard Constraint Old | Hard Constraint New | ROUGE-1 Old | ROUGE-1 New | Avg Solver Calls Old | Avg Solver Calls New | Contradictory Audits Old | Contradictory Audits New |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in detail_rows:
        old_block = row["old"]
        new_block = row["new"]
        lines.append(
            f"| {row['experiment']} | "
            f"{format_pct(old_block['audit_pass_rate'])} | {format_pct(new_block['audit_pass_rate'])} | "
            f"{format_float(old_block['hard_constraint_rate'])} | {format_float(new_block['hard_constraint_rate'])} | "
            f"{format_float(old_block['avg_rouge1'])} | {format_float(new_block['avg_rouge1'])} | "
            f"{format_float(old_block['avg_solver_calls'])} | {format_float(new_block['avg_solver_calls'])} | "
            f"{int(old_block['contradictory_audit_count'])} | {int(new_block['contradictory_audit_count'])} |"
        )
    lines.append("")

    lines.append("## Highlights")
    lines.append("")
    if best_rouge:
        delta = best_rouge["new"]["avg_rouge1"] - best_rouge["old"]["avg_rouge1"]
        lines.append(f"- Largest ROUGE-1 gain: `{best_rouge['experiment']}` with a delta of {delta:.3f}.")
    if best_constraints:
        delta = best_constraints["new"]["hard_constraint_rate"] - best_constraints["old"]["hard_constraint_rate"]
        lines.append(f"- Largest hard-constraint gain: `{best_constraints['experiment']}` with a delta of {delta:.3f}.")

    old_contradictions = sum(item["old"]["contradictory_audit_count"] for item in detail_rows)
    new_contradictions = sum(item["new"]["contradictory_audit_count"] for item in detail_rows)
    lines.append(
        f"- Contradictory audit messages dropped from {old_contradictions} in the old run to {new_contradictions} in the improved run."
    )
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The new pipeline should be read as a fairer, better-instrumented evaluation rather than a guaranteed raw-quality jump on every metric. "
        "Its main value is that failures are now more trustworthy: a failed audit is far less likely to be caused by obviously self-contradictory feedback."
    )
    lines.append("")

    payload = {
        "overall": overall,
        "per_experiment": detail_rows,
    }
    return "\n".join(lines) + "\n", payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare old and new pipeline evaluation outputs.")
    parser.add_argument("--old-eval", type=Path, required=True)
    parser.add_argument("--new-eval", type=Path, required=True)
    parser.add_argument("--old-runs-dir", type=Path, required=True)
    parser.add_argument("--new-runs-dir", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    old_eval = load_json(args.old_eval)
    new_eval = load_json(args.new_eval)
    old_runs = summarize_runs(args.old_runs_dir)
    new_runs = summarize_runs(args.new_runs_dir)
    markdown, payload = build_report(old_eval, new_eval, old_runs, new_runs)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(markdown, encoding="utf-8")
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
