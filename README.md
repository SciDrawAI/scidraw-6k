# SciDraw-6K

[![Dataset on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/SciDrawAI/SciDraw-6K)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19642870.svg)](https://doi.org/10.5281/zenodo.19642870)
[![Website](https://img.shields.io/badge/Service-sci--draw.com-blue)](https://sci-draw.com)
[![License](https://img.shields.io/badge/Data-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Code License](https://img.shields.io/badge/Code-MIT-green.svg)](LICENSE)

A curated dataset of **6,291 scientific illustrations** synthesized by
Google Gemini, with aligned prompts in **11 languages**, powering
[sci-draw.com](https://sci-draw.com) — a public scientific drawing service.

This repository contains the **code and documentation** that accompany
the dataset. The dataset itself (metadata + images) is hosted on the
Hugging Face Hub and archived on Zenodo; this repository does not
redistribute any data files.

---

## At a glance

| | |
|---|---|
| Rows | 6,291 |
| Categories | 8 (biomedical, chemistry, materials, electronics, environment, ai_system, physics, other) |
| Languages | 11 (en, zh, ja, ko, de, fr, es, pt_br, zh_tw, it, ru) |
| Source model | Google Gemini (`gemini-2.5-flash-image`, `gemini-3-pro-image-preview`, `gemini-3.1-flash-image-preview`) |
| Time span | 2026-01 &ndash; 2026-04 |
| Total size | ~19 GB images + 48 MB Parquet metadata |
| Data license | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Code license | [MIT](LICENSE) |

---

## Quickstart

Install dependencies:

```bash
pip install -r requirements.txt
```

Load the metadata (no image download) via the `datasets` library:

```python
from datasets import load_dataset

ds = load_dataset("SciDrawAI/SciDraw-6K", split="train")
print(ds)
print(ds[0]["prompts"]["en"])
```

Or stream directly from the Parquet file without pulling images:

```python
import pandas as pd

df = pd.read_parquet(
    "hf://datasets/SciDrawAI/SciDraw-6K/metadata.parquet"
)
print(df["release_category"].value_counts())
```

Download the full dataset (images included) to local disk:

```bash
python scripts/download.py --output ./scidraw6k
```

Regenerate the paper's statistical figures from metadata:

```bash
python scripts/compute_stats.py --metadata ./scidraw6k/metadata.parquet --out ./stats
```

Verify image integrity against published SHA-256 hashes:

```bash
python scripts/verify_integrity.py --root ./scidraw6k
```

---

## Repository layout

```
scidraw-6k/
├── README.md
├── LICENSE                       # MIT (code)
├── CITATION.cff                  # cite via Zenodo DOI
├── requirements.txt
├── schema/
│   └── metadata_schema.json      # JSON Schema for metadata rows
├── scripts/
│   ├── download.py               # fetch metadata + images from the HF Hub
│   ├── compute_stats.py          # reproduce category / language / length / time / model figures
│   ├── make_splits.py            # reproduce prompt-grouped train/val/test splits
│   └── verify_integrity.py       # re-hash images and check against metadata SHA-256 column
├── examples/
│   ├── load_dataset.py           # smallest possible load example
│   ├── filter_by_category.py     # category-wise subsetting
│   └── retrieval_demo.py         # few-shot prompt retrieval with sentence-transformers
└── docs/
    └── construction_pipeline.md  # extended version of paper Section 3
```

---

## Metadata schema

Each row in `metadata.parquet` / `metadata.jsonl` contains:

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique image identifier |
| `image` | string | Relative path to the image file (e.g. `images/biomedical/gal_xxx.png`) |
| `image_ext` | string | File extension (usually `png`) |
| `raw_category` | string | Original fine-grained category label |
| `release_category` | string | Normalized 8-class category |
| `category` | string | Alias of `release_category` |
| `prompts` | object | 11-language prompt object: `original`, `en`, `zh`, `ja`, `ko`, `de`, `fr`, `es`, `pt_br`, `zh_tw`, `it`, `ru` |
| `gemini_model` | string \| null | Gemini model identifier (null for ~7% of rows) |
| `generation_type` | string \| null | Generation type (e.g. `text_to_image`) |
| `created_at` | string | ISO 8601 timestamp |
| `image_sha256` | string | SHA-256 of the image bytes (used by `verify_integrity.py`) |

The canonical machine-readable schema is [`schema/metadata_schema.json`](schema/metadata_schema.json).

---

## Application: sci-draw.com

SciDraw-6K is the substrate of
[**sci-draw.com**](https://sci-draw.com), a public scientific drawing
service. The dataset powers three concrete functions of the live service:

- **Template seeding** — curated prompts serve as one-click templates,
  organized by category, with aligned 11-language prompts letting users
  start in their preferred language.
- **Few-shot prompt rewriting** — user requests are rewritten using
  nearest-neighbour prompts from SciDraw-6K as in-context exemplars
  before dispatch to Gemini.
- **Regression evaluation** — a held-out slice is used as a regression
  suite when the underlying image-generation model is upgraded.

See [`examples/retrieval_demo.py`](examples/retrieval_demo.py) for a
minimal implementation of the few-shot retrieval pattern.

Visit [sci-draw.com](https://sci-draw.com) to try the service.

---

## Citation

Please cite the Zenodo record when using SciDraw-6K:

```bibtex
@misc{chen2026scidraw6k,
  author       = {Chen, Davie},
  title        = {{SciDraw-6K}: A Multilingual Scientific Illustration
                  Dataset Generated by {Google Gemini}},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19642870},
  url          = {https://doi.org/10.5281/zenodo.19642870}
}
```

GitHub also exposes a "Cite this repository" button sourced from
[`CITATION.cff`](CITATION.cff).

---

## Related resources

- **Service**: [sci-draw.com](https://sci-draw.com) — public scientific drawing platform powered by this dataset
- **Dataset**: [SciDrawAI/SciDraw-6K on Hugging Face](https://huggingface.co/datasets/SciDrawAI/SciDraw-6K)
- **Archive / DOI**: [10.5281/zenodo.19642870](https://doi.org/10.5281/zenodo.19642870)

---

## License

- **Data** (images and prompts, hosted on Hugging Face and Zenodo):
  [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
- **Code** (this repository): [MIT License](LICENSE).

Downstream users are responsible for independently verifying current
licensing and redistribution constraints of Google Gemini's outputs
before republishing mirrors or derived artefacts.
