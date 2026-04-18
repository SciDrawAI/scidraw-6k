"""Minimal loading example.

Uses the ``datasets`` library to stream SciDraw-6K metadata directly
from the Hugging Face Hub without downloading image bytes.

Run::

    python examples/load_dataset.py
"""

from __future__ import annotations

from datasets import load_dataset


def main() -> None:
    ds = load_dataset("SciDrawAI/SciDraw-6K", split="train")
    print(ds)
    sample = ds[0]
    print("id             :", sample["id"])
    print("release_category:", sample["release_category"])
    print("gemini_model   :", sample.get("gemini_model"))
    print("english prompt :", sample["prompts"]["en"])


if __name__ == "__main__":
    main()
