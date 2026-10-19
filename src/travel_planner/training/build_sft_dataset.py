"""Build instruction-tuning JSON from TravelPlanner train split (query -> annotated_plan)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from travel_planner.config import HF_TOKEN
from travel_planner.data.context import row_to_prompt_context
from travel_planner.data.load_travelplanner import load_split


def build_example(row: dict) -> dict:
    ctx = row_to_prompt_context(row)
    instruction = (
        "You are a travel planner. Write a detailed travel itinerary that satisfies the constraints.\n\n"
        f"{ctx}"
    )
    assistant = row.get("annotated_plan") or row.get("reference_information") or ""
    return {
        "instruction": instruction,
        "output": assistant,
        "metadata": {
            "org": row.get("org"),
            "dest": row.get("dest"),
            "days": row.get("days"),
            "level": row.get("level"),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("artifacts/sft_train.jsonl"))
    p.add_argument("--token", default=HF_TOKEN)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    ds = load_split(split="train", token=args.token)
    with args.out.open("w", encoding="utf-8") as f:
        for row in ds:
            ex = build_example(row)
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {ds.num_rows} examples to {args.out}")


if __name__ == "__main__":
    main()
