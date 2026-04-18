"""Verify that downloaded images match the published SHA-256 hashes.

Re-hashes every image referenced by the ``image`` column of
``metadata.jsonl`` / ``metadata.parquet`` and compares against the
``image_sha256`` column. Any mismatch or missing file is reported.

Example::

    python scripts/verify_integrity.py --root ./scidraw6k
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def _load(meta_path: Path) -> pd.DataFrame:
    if meta_path.suffix.lower() == ".parquet":
        return pd.read_parquet(meta_path)
    rows: list[dict] = []
    with meta_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify SHA-256 hashes of downloaded images.")
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Local dataset root (the folder passed to scripts/download.py).",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Path to metadata file (defaults to <root>/metadata.parquet or metadata.jsonl).",
    )
    args = parser.parse_args()

    meta_path = args.metadata
    if meta_path is None:
        for candidate in (args.root / "metadata.parquet", args.root / "metadata.jsonl"):
            if candidate.exists():
                meta_path = candidate
                break
    if meta_path is None or not meta_path.exists():
        raise SystemExit(f"metadata file not found under {args.root}")

    df = _load(meta_path)
    if "image_sha256" not in df.columns:
        raise SystemExit("metadata has no image_sha256 column; nothing to verify")

    missing: list[str] = []
    mismatched: list[str] = []
    ok = 0

    for row in tqdm(df.to_dict(orient="records"), desc="verifying"):
        rel = row.get("image")
        if not rel:
            continue
        path = args.root / rel
        if not path.exists():
            missing.append(row["id"])
            continue
        want = row.get("image_sha256")
        if not want:
            continue
        got = _sha256(path)
        if got != want:
            mismatched.append(row["id"])
        else:
            ok += 1

    print(f"\nok={ok} mismatched={len(mismatched)} missing={len(missing)}")
    if mismatched:
        print("first 10 mismatched ids:", mismatched[:10])
    if missing:
        print("first 10 missing ids:", missing[:10])

    if mismatched or missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
