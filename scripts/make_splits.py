"""Reproduce the train/val/test split file from the published metadata.

Rows are grouped by their normalized English prompt. Groups (not
individual rows) are then assigned to ``train`` / ``val`` / ``test`` in an
8/1/1 cycle within each release category. This mitigates evaluation
leakage from near-duplicate prompts that differ only in minor phrasing.

Example::

    python scripts/make_splits.py \\
        --metadata ./scidraw6k/metadata.parquet \\
        --out ./splits.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

WHITESPACE_RE = re.compile(r"\s+")
SPLIT_NAMES = ("test", "val", "train")


def _load_rows(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _normalize(text: str | None) -> str:
    if not text:
        return ""
    return WHITESPACE_RE.sub(" ", text.strip()).lower()


def _assign_split(index: int) -> str:
    mod = index % 10
    if mod == 0:
        return "test"
    if mod == 1:
        return "val"
    return "train"


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate splits.json.")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("./splits.json"))
    args = parser.parse_args()

    df = _load_rows(args.metadata)
    print(f"loaded {len(df)} rows")

    groups: dict[str, dict] = {}
    for row in df.to_dict(orient="records"):
        prompt_en = _normalize(row["prompts"].get("en"))
        digest = hashlib.sha1(prompt_en.encode("utf-8")).hexdigest()
        release_category = row.get("release_category") or row.get("category") or "other"
        entry = groups.setdefault(
            digest,
            {
                "release_category": release_category,
                "ids": [],
                "prompt_length": len(prompt_en),
            },
        )
        entry["ids"].append(row["id"])

    ordered = sorted(
        groups.items(),
        key=lambda kv: (
            kv[1]["release_category"],
            -kv[1]["prompt_length"],
            kv[0],
        ),
    )

    out: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    counters: dict[str, int] = {}
    for _, entry in ordered:
        cat = entry["release_category"]
        idx = counters.get(cat, 0)
        out[_assign_split(idx)].extend(entry["ids"])
        counters[cat] = idx + 1

    payload = {
        "strategy": (
            "Group by normalized English prompt, then assign groups to "
            "train/val/test in an 8/1/1 cycle within each release category."
        ),
        "counts": {
            "train": len(out["train"]),
            "val": len(out["val"]),
            "test": len(out["test"]),
            "prompt_groups": len(ordered),
        },
        "splits": out,
    }

    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    print(json.dumps(payload["counts"], indent=2))


if __name__ == "__main__":
    main()
