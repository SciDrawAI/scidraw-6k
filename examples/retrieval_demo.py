"""Few-shot prompt retrieval demo.

Given a short user query, retrieve the top-k most similar English
prompts from SciDraw-6K by sentence-embedding cosine similarity. This
mirrors the few-shot prompt-rewriting pattern used by sci-draw.com
(see Section 5 of the SciDraw-6K paper).

Run::

    pip install sentence-transformers
    python examples/retrieval_demo.py --query "mechanism of CRISPR gene editing" --k 3
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

PARQUET_URL = "hf://datasets/SciDrawAI/SciDraw-6K/metadata.parquet"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _build_index(model: SentenceTransformer, df: pd.DataFrame) -> np.ndarray:
    prompts = df["prompts"].apply(lambda p: p.get("en") or "").tolist()
    return model.encode(
        prompts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot retrieval demo over SciDraw-6K prompts.")
    parser.add_argument("--query", required=True, help="User query (natural language).")
    parser.add_argument("--k", type=int, default=3, help="Number of neighbours to show.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="sentence-transformers model id.")
    parser.add_argument(
        "--category",
        default=None,
        help="Optional release_category filter (e.g. biomedical).",
    )
    args = parser.parse_args()

    df = pd.read_parquet(PARQUET_URL)
    if args.category is not None:
        df = df[df["release_category"] == args.category].reset_index(drop=True)
    if df.empty:
        raise SystemExit("no rows to index after filtering")

    print(f"indexing {len(df)} prompts with {args.model}")
    model = SentenceTransformer(args.model)
    corpus = _build_index(model, df)
    query_vec = model.encode([args.query], normalize_embeddings=True, convert_to_numpy=True)[0]

    scores = corpus @ query_vec
    top = np.argsort(-scores)[: args.k]

    print(f"\nquery: {args.query}\n")
    for rank, idx in enumerate(top, start=1):
        row = df.iloc[int(idx)]
        print(f"#{rank}  cosine={scores[idx]:.3f}  [{row['release_category']}] id={row['id']}")
        en = row["prompts"].get("en") or ""
        print(f"     {en[:240]}{'...' if len(en) > 240 else ''}")
        print()


if __name__ == "__main__":
    main()
