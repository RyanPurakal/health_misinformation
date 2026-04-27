# data/

Parquet dataset utilities. All scripts are run from the repo root and expect parquet files under `data/train/`.

## Files

| File | Role |
|---|---|
| `load_health_fact.py` | Loads all parquet files in `data/train/`, applies the label remap, and prints size + sample rows — useful for quick inspection |
| `label_check.py` | Same as above with slightly cleaner output; intended as a standalone sanity check on label distribution |

## Dataset schema

Columns present in every parquet file:

| Column | Type | Notes |
|---|---|---|
| `claim_id` | str | Unique identifier |
| `claim` | str | The health claim text |
| `main_text` | str | Full article body — may be empty for augmented rows |
| `explanation` | str | Human-written explanation of the verdict |
| `label` | int | Original 4-class label (see mapping below) |
| `date_published` | str | Optional publication date |
| `fact_checkers` | str | Source attribution |
| `subjects` | str | Topic tag |

## Label mapping (4-class → binary)

| Original | Meaning | Binary |
|---|---|---|
| 0 | false / misleading | 1 (MISINFORMATION) |
| 1 | true | 0 (RELIABLE) |
| 2 | mixture | 1 (MISINFORMATION) |
| 3 | unproven | 1 (MISINFORMATION) |
| -1 | invalid | dropped |

## Adding new training data

Drop additional `.parquet` files into `data/train/`. They must contain at minimum `claim`, `label`, `main_text`, and `explanation` columns. The training script picks up all `*.parquet` files in that folder automatically.
