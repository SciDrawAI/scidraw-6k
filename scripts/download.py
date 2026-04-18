"""Download the SciDraw-6K dataset from the Hugging Face Hub.

This script uses :mod:`huggingface_hub` to snapshot the dataset repo into a
local directory. By default it pulls every file; the ``--metadata-only``
flag pulls only the schema / metadata / split / README files (~50 MB) and
skips the image bytes (~19 GB).

Example::

    python scripts/download.py --output ./scidraw6k
    python scripts/download.py --output ./scidraw6k --metadata-only
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "SciDrawAI/SciDraw-6K"
REPO_TYPE = "dataset"

METADATA_PATTERNS = [
    "README.md",
    "metadata.jsonl",
    "metadata.parquet",
    "metadata.validation.json",
    "splits.json",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SciDraw-6K from the Hugging Face Hub.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./scidraw6k"),
        help="Local directory to snapshot into (default: ./scidraw6k).",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Skip the images/ tree and only pull metadata files.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Git revision (branch, tag, or commit) to download (default: main).",
    )
    args = parser.parse_args()

    allow_patterns = METADATA_PATTERNS if args.metadata_only else None

    local_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=str(args.output),
        revision=args.revision,
        allow_patterns=allow_patterns,
    )
    print(f"downloaded to {local_dir}")


if __name__ == "__main__":
    main()
