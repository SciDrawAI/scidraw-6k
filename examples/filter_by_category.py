"""Subset SciDraw-6K by release category.

Reads ``metadata.parquet`` directly from the Hugging Face Hub (no local
clone required) and prints the first ``--limit`` rows of the requested
category together with their English prompt.

Run::

    python examples/filter_by_category.py --category biomedical --limit 5
"""

from __future__ import annotations

import argparse

import pandas as pd

PARQUET_URL = "hf://datasets/SciDrawAI/SciDraw-6K/metadata.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter SciDraw-6K by release category.")
    parser.add_argument(
        "--category",
        required=True,
        choices=[
            "ai_system",
            "biomedical",
            "chemistry",
            "electronics",
            "environment",
            "materials",
            "other",
            "physics",
        ],
    )
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    df = pd.read_parquet(PARQUET_URL)
    subset = df[df["release_category"] == args.category].head(args.limit)
    for _, row in subset.iterrows():
        print(f"[{row['id']}] {row['image']}")
        print("  en:", row["prompts"]["en"][:200])
        print()


if __name__ == "__main__":
    main()
