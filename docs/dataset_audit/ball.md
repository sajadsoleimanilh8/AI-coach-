# Dataset audit -- `ball`

_Generated 2026-08-12 by `scripts/dataset_qa.py`._

- **Task**: `detect`  |  **classes**: 1
- **Resolved root**: `D:\SportsStrategyCoachAI\datasets\processed`
- **Source dirs**:
  - `D:\SportsStrategyCoachAI\datasets\processed\ball_datasets\1`
  - `D:\SportsStrategyCoachAI\datasets\processed\ball_datasets\2`
  - `D:\SportsStrategyCoachAI\datasets\processed\ball_datasets\3`
  - `D:\SportsStrategyCoachAI\datasets\processed\ball_datasets\4`
  - `D:\SportsStrategyCoachAI\datasets\processed\ball_datasets\5`
  - `D:\SportsStrategyCoachAI\datasets\processed\ball_datasets\6`

- **Images**: 1,912  |  **labelled instances**: 1,575

**Verdict: REVIEW REQUIRED** (0 blocking issue(s), 114 cross-split duplicate group(s))

## Splits

| split | images | label files | instances | empty labels |
|---|---:|---:|---:|---:|
| `1/train` | 159 | 54 | 152 | 0 |
| `1/valid` | 46 | 23 | 58 | 0 |
| `1/test` | 23 | 11 | 39 | 0 |
| `2/train` | 193 | 60 | 159 | 0 |
| `2/valid` | 56 | 24 | 66 | 0 |
| `2/test` | 28 | 11 | 39 | 0 |
| `3/train` | 175 | 42 | 85 | 0 |
| `3/valid` | 60 | 28 | 91 | 0 |
| `3/test` | 42 | 25 | 88 | 0 |
| `4/train` | 193 | 33 | 64 | 0 |
| `4/valid` | 54 | 39 | 142 | 0 |
| `4/test` | 28 | 23 | 58 | 0 |
| `5/train` | 297 | 66 | 66 | 0 |
| `5/valid` | 200 | 143 | 143 | 0 |
| `5/test` | 77 | 58 | 58 | 0 |
| `6/train` | 197 | 35 | 66 | 0 |
| `6/valid` | 55 | 40 | 143 | 0 |
| `6/test` | 29 | 23 | 58 | 0 |

## Class balance

| split | class | instances |
|---|---|---:|
| test | `ball` | 340 |
| train | `ball` | 592 |
| valid | `ball` | 643 |

## Issues

### `dup_images` -- 500 (REVIEW)

- `[LEAKAGE across splits] 2x identical: test:-1-_jpeg.rf.26986135a9cd833bc9a050941cf27fd5.jpg, train:-1-_jpeg.rf.26986135a9cd833bc9a050941cf27fd5.jpg`
- `[LEAKAGE across splits] 2x identical: test:0be4fb66e0cb1d13_jpg.rf.dc0c1fb2a22252c93938cf915d39eb81.jpg, train:0be4fb66e0cb1d13_jpg.rf.dc0c1fb2a22252c93938cf915d39eb81.jpg`
- `[LEAKAGE across splits] 2x identical: test:0cf27fab930477a2d968b93a47a86d97_jpg.rf.e61ad92459808d9de9c539c748acd466.jpg, train:0cf27fab930477a2d968b93a47a86d97_jpg.rf.e61ad92459808d9de9c539c748acd466.jpg`
- `[LEAKAGE across splits] 2x identical: test:11_jpg.rf.0d2c43849711c69626406227f74bdb85.jpg, valid:11_jpg.rf.0d2c43849711c69626406227f74bdb85.jpg`
- `[LEAKAGE across splits] 2x identical: test:12_jpg.rf.73594c0055858bf3089af4dac2d81e2f.jpg, valid:12_jpg.rf.73594c0055858bf3089af4dac2d81e2f.jpg`
- `[LEAKAGE across splits] 2x identical: test:13_jpg.rf.b524c128494d595c7861c62472075aae.jpg, valid:13_jpg.rf.b524c128494d595c7861c62472075aae.jpg`
- `[LEAKAGE across splits] 2x identical: test:14_jpg.rf.5907adb64e9aadb11450c8931e463a71.jpg, valid:14_jpg.rf.5907adb64e9aadb11450c8931e463a71.jpg`
- `[LEAKAGE across splits] 2x identical: test:15_jpg.rf.701a36efab6d1712aeba81681fb3f413.jpg, valid:15_jpg.rf.701a36efab6d1712aeba81681fb3f413.jpg`
- `[LEAKAGE across splits] 2x identical: test:16_jpg.rf.09a2b8c1e92af177f83a5b91a7bffe0e.jpg, valid:16_jpg.rf.09a2b8c1e92af177f83a5b91a7bffe0e.jpg`
- `[LEAKAGE across splits] 2x identical: test:17_jpg.rf.52d6b79620be5ff51f6a70f44047ecd5.jpg, valid:17_jpg.rf.52d6b79620be5ff51f6a70f44047ecd5.jpg`
- `[LEAKAGE across splits] 2x identical: test:17b97ffbc6bc32d8_jpg.rf.10130eb3de9d9bad45e4b44727b2347d.jpg, train:17b97ffbc6bc32d8_jpg.rf.10130eb3de9d9bad45e4b44727b2347d.jpg`
- `[LEAKAGE across splits] 2x identical: test:26_jpg.rf.2e543dd6c2ccd4fb51407ac9b0c07d74.jpg, valid:26_jpg.rf.2e543dd6c2ccd4fb51407ac9b0c07d74.jpg`
- `[LEAKAGE across splits] 2x identical: test:2916d352c9c849b4_jpg.rf.cbccac3a5b275c2dfd9c16d0daa142ec.jpg, train:2916d352c9c849b4_jpg.rf.cbccac3a5b275c2dfd9c16d0daa142ec.jpg`
- `[LEAKAGE across splits] 2x identical: test:2_jpg.rf.9317462c84c6c25b27931a8f9487a2e1.jpg, valid:2_jpg.rf.9317462c84c6c25b27931a8f9487a2e1.jpg`
- `[LEAKAGE across splits] 2x identical: test:30_jpg.rf.74f10b6fef528028d8317398d55b851a.jpg, valid:30_jpg.rf.74f10b6fef528028d8317398d55b851a.jpg`
- _... and 485 more_

### `orphans` -- 1174 (REVIEW)

- `image without label: D:\SportsStrategyCoachAI\datasets\processed\ball_datasets\1\train\images\-2-_jpeg.rf.a163989e49b0c03ca9bad35e9965d945.jpg`
- `image without label: D:\SportsStrategyCoachAI\datasets\processed\ball_datasets\1\train\images\-3-_jpeg.rf.18956ec97792b8467ce9dd4a7ffde096.jpg`
- `image without label: D:\SportsStrategyCoachAI\datasets\processed\ball_datasets\1\train\images\-_jpeg.rf.8a59234c3833c008267451d03cf075cc.jpg`
- `image without label: D:\SportsStrategyCoachAI\datasets\processed\ball_datasets\1\train\images\004130acea29204f_jpg.rf.28ab507cc8c0083710bbf76f24c3a3fe.jpg`
- `image without label: D:\SportsStrategyCoachAI\datasets\processed\ball_datasets\1\train\images\01ab4be3f275d44c_jpg.rf.6e2353871f25027314e469503f8fa430.jpg`
- `image without label: D:\SportsStrategyCoachAI\datasets\processed\ball_datasets\1\train\images\01ae2ee150f3c0e0_jpg.rf.e5f2b0468f52bbd673e5d749f9075748.jpg`
- `image without label: D:\SportsStrategyCoachAI\datasets\processed\ball_datasets\1\train\images\0a1e1ed43c0cbe88_jpg.rf.59212c6060d6fa7019d1972c2996335f.jpg`
- `image without label: D:\SportsStrategyCoachAI\datasets\processed\ball_datasets\1\train\images\0a56cf0058e7c510_jpg.rf.ee52762a075da21c8df0c2ba2985a57d.jpg`
- `image without label: D:\SportsStrategyCoachAI\datasets\processed\ball_datasets\1\train\images\0aa63e5298ff5224_jpg.rf.225d1b22e027bf6b8409ca77695bacfd.jpg`
- `image without label: D:\SportsStrategyCoachAI\datasets\processed\ball_datasets\1\train\images\0c87f3cbd6098703_jpg.rf.0b9939bbcd0b7b68c83e8aa033146107.jpg`
- `image without label: D:\SportsStrategyCoachAI\datasets\processed\ball_datasets\1\train\images\0c8d378bf5c82792_jpg.rf.ada5e473adb241f651df859de40e2776.jpg`
- `image without label: D:\SportsStrategyCoachAI\datasets\processed\ball_datasets\1\train\images\0d47a2468567ce59_jpg.rf.fc62ad7366df08de595c23f8cca89a93.jpg`
- `image without label: D:\SportsStrategyCoachAI\datasets\processed\ball_datasets\1\train\images\0d5f29db5a6cd05b_jpg.rf.7d5c9b2d976d285cfdf8e7d4c2ce3f40.jpg`
- `image without label: D:\SportsStrategyCoachAI\datasets\processed\ball_datasets\1\train\images\1f5fc52e3b0e98dc_jpg.rf.e01609298c3984e313d20ff33a5d53e6.jpg`
- `image without label: D:\SportsStrategyCoachAI\datasets\processed\ball_datasets\1\train\images\1f77c4827cd8469f_jpg.rf.cb9a679e49cf121bdddd719fc6a857a2.jpg`
- _... and 1159 more_

## Policy

Nothing here was auto-deleted or auto-corrected. `INFO` rows are expected characteristics, not defects -- in particular `tiny` boxes in the ball dataset are the intended signal, not noise. Only `BLOCKING` rows and cross-split duplicates should gate training.
