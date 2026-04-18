"""Recompute the descriptive statistics used in the SciDraw-6K paper.

Reads the published metadata (JSONL or Parquet) and writes five PNG
figures plus a ``summary.json`` into the output directory.

Example::

    python scripts/compute_stats.py \\
        --metadata ./scidraw6k/metadata.parquet \\
        --out ./stats
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LANGS = ["en", "zh", "ja", "ko", "de", "fr", "es", "pt_br", "zh_tw", "it", "ru"]


def _load_metadata(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _fig_category(df: pd.DataFrame, out_dir: Path) -> Path:
    counts = df["release_category"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(counts.index, counts.values, color="#3b82f6")
    ax.set_ylabel("# images")
    ax.set_title("Images per release category")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    path = out_dir / "category_dist.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _fig_lang(df: pd.DataFrame, out_dir: Path) -> Path:
    coverage = [
        100.0 * df["prompts"].apply(lambda p: bool(p.get(lang))).sum() / len(df)
        for lang in LANGS
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(LANGS, coverage, color="#10b981")
    ax.set_ylim(0, 105)
    ax.set_ylabel("non-null rate (%)")
    ax.set_title("Multilingual prompt coverage (11 languages)")
    plt.tight_layout()
    path = out_dir / "lang_dist.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _fig_prompt_length(df: pd.DataFrame, out_dir: Path) -> Path:
    lengths = df["prompts"].apply(lambda p: len(p.get("en") or ""))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(lengths, bins=50, color="#f59e0b")
    ax.set_xlabel("English prompt length (chars)")
    ax.set_ylabel("# images")
    ax.set_title("Prompt length distribution")
    plt.tight_layout()
    path = out_dir / "prompt_length.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _fig_time(df: pd.DataFrame, out_dir: Path) -> Path:
    ts = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    monthly = ts.dt.to_period("M").value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(monthly.index.astype(str), monthly.values, color="#8b5cf6")
    ax.set_ylabel("# images")
    ax.set_title("Images generated per month")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    path = out_dir / "time_dist.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _fig_model(df: pd.DataFrame, out_dir: Path) -> Path:
    counts = Counter(df.get("gemini_model", pd.Series([])).fillna("<unknown>"))
    items = sorted(counts.items(), key=lambda x: -x[1])
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color="#ef4444")
    ax.set_ylabel("# images")
    ax.set_title("Source-model distribution")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path = out_dir / "model_dist.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _summary(df: pd.DataFrame) -> dict:
    prompt_en = df["prompts"].apply(lambda p: p.get("en") or "")
    prompt_original = df["prompts"].apply(lambda p: p.get("original") or "")
    return {
        "rows": int(len(df)),
        "release_category_counts": df["release_category"].value_counts().to_dict(),
        "raw_category_counts": df["raw_category"].value_counts().to_dict(),
        "gemini_model_counts": df["gemini_model"].fillna("<unknown>").value_counts().to_dict(),
        "generation_type_counts": df["generation_type"].fillna("<unknown>").value_counts().to_dict(),
        "lang_non_null_counts": {
            lang: int(df["prompts"].apply(lambda p: bool(p.get(lang))).sum())
            for lang in LANGS
        },
        "english_prompt_length": {
            "min": int(prompt_en.str.len().min()),
            "median": float(prompt_en.str.len().median()),
            "mean": float(prompt_en.str.len().mean()),
            "max": int(prompt_en.str.len().max()),
        },
        "original_prompt_length": {
            "min": int(prompt_original.str.len().min()),
            "median": float(prompt_original.str.len().median()),
            "mean": float(prompt_original.str.len().mean()),
            "max": int(prompt_original.str.len().max()),
        },
        "duplicate_english_prompts_extra": int(prompt_en.duplicated().sum()),
        "duplicate_original_prompts_extra": int(prompt_original.duplicated().sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute dataset statistics.")
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Path to metadata.parquet or metadata.jsonl.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("./stats"),
        help="Output directory for figures and summary.json.",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    df = _load_metadata(args.metadata)
    print(f"loaded {len(df)} rows from {args.metadata.name}")

    for fig_fn in (_fig_category, _fig_lang, _fig_prompt_length, _fig_time, _fig_model):
        path = fig_fn(df, args.out)
        print("wrote", path)

    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps(_summary(df), indent=2), encoding="utf-8")
    print("wrote", summary_path)


if __name__ == "__main__":
    main()
