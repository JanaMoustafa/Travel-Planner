#!/usr/bin/env python3
"""Build Chroma index from TravelPlanner train + validation reference_information."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without install: repo root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from travel_planner.config import CHROMA_PERSIST_DIR, HF_TOKEN
from travel_planner.data.load_travelplanner import load_split
from travel_planner.rag.ingest import ingest_dataset_to_chroma


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--persist", default=str(CHROMA_PERSIST_DIR))
    p.add_argument("--reset", action="store_true")
    p.add_argument("--token", default=HF_TOKEN)
    args = p.parse_args()

    rows: list[dict] = []
    for split in ("train", "validation"):
        ds = load_split(split=split, token=args.token)
        for i, row in enumerate(ds):
            d = dict(row)
            d["_split"] = split
            d["_row_in_split"] = i
            rows.append(d)

    ingest_dataset_to_chroma(rows, persist_dir=args.persist, reset=args.reset)
    print(f"Ingested {len(rows)} rows into Chroma at {args.persist}")


if __name__ == "__main__":
    main()
