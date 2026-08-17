# Dataset audit -- `calibration`

_Generated 2026-08-12 by `scripts/dataset_qa.py`._

- **Task**: `pose`  |  **classes**: 1
- **Resolved root**: `D:\SportsStrategyCoachAI\datasets\processed`
- **Source dirs**:
  - `D:\SportsStrategyCoachAI\datasets\processed\field_datasets\2`

- **Images**: 317  |  **labelled instances**: 317

**Verdict: PASS** (0 blocking issue(s), 0 cross-split duplicate group(s))

## Splits

| split | images | label files | instances | empty labels |
|---|---:|---:|---:|---:|
| `2/train` | 255 | 255 | 255 | 0 |
| `2/valid` | 34 | 34 | 34 | 0 |
| `2/test` | 28 | 28 | 28 | 0 |

## Class balance

| split | class | instances |
|---|---|---:|
| test | `pitch` | 28 |
| train | `pitch` | 255 |
| valid | `pitch` | 34 |

## Issues

None found.
## Policy

Nothing here was auto-deleted or auto-corrected. `INFO` rows are expected characteristics, not defects -- in particular `tiny` boxes in the ball dataset are the intended signal, not noise. Only `BLOCKING` rows and cross-split duplicates should gate training.
