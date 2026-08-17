# Dataset audit -- `goalpost`

_Generated 2026-08-12 by `scripts/dataset_qa.py`._

- **Task**: `detect`  |  **classes**: 1
- **Resolved root**: `D:\SportsStrategyCoachAI\datasets\processed`
- **Source dirs**:
  - `D:\SportsStrategyCoachAI\datasets\processed\goalpost_datast\1`

- **Images**: 3,161  |  **labelled instances**: 3,167

**Verdict: REVIEW REQUIRED** (2 blocking issue(s), 0 cross-split duplicate group(s))

## Splits

| split | images | label files | instances | empty labels |
|---|---:|---:|---:|---:|
| `1/train` | 2,585 | 2,585 | 2,586 | 11 |
| `1/valid` | 359 | 359 | 363 | 0 |
| `1/test` | 217 | 217 | 218 | 1 |

## Class balance

| split | class | instances |
|---|---|---:|
| test | `goalpost` | 218 |
| train | `goalpost` | 2,586 |
| valid | `goalpost` | 363 |

## Issues

### `long_path` -- 2 (BLOCKING)

- `298 chars: D:\SportsStrategyCoachAI\datasets\processed\goalpost_datast\1\train\images\514Btuv4idL-_SR600-315_PIWhiteStrip-BottomLeft-0-35_PIStarRatingONEANDHALF-BottomLeft-360-6_SR600-315_ZA8-445-290-400-400-AmazonEmberBold-12-4-0-0-5_SCLZZZZZZZ_FMpng_BG255-255-255_png.rf.e21465ef25040df48d3202363f12bd6b.jpg`
- `298 chars: D:\SportsStrategyCoachAI\datasets\processed\goalpost_datast\1\train\labels\514Btuv4idL-_SR600-315_PIWhiteStrip-BottomLeft-0-35_PIStarRatingONEANDHALF-BottomLeft-360-6_SR600-315_ZA8-445-290-400-400-AmazonEmberBold-12-4-0-0-5_SCLZZZZZZZ_FMpng_BG255-255-255_png.rf.e21465ef25040df48d3202363f12bd6b.txt`

### `dup_images` -- 3 (REVIEW)

- `[within split] 2x identical: train:9758f1841a04ada1c9ad83d21332edeb_jpg.rf.b7c273222b86ad1cb65b327b1e218a52.jpg, train:Image_280_jpg.rf.77033265184ad3c502189d1db0e77f5c.jpg`
- `[within split] 2x identical: train:images-19-_jpeg.rf.0c6b6fbac3cba92749db6c99c66f824d.jpg, train:images-3-_jpg.rf.6ee6eafb0ff39b18320122e1485656ae.jpg`
- `[within split] 2x identical: train:images-29-_jpg.rf.88c4ad8c72d0e690c15a670aae0b72a9.jpg, train:images-4-_jpeg.rf.a6c70461f29860ca7f0c905b255d3d47.jpg`

### `mixed_format` -- 288 (INFO)

- `D:\SportsStrategyCoachAI\datasets\processed\goalpost_datast\1\train\labels\1676b247b114033e7555fbe1602e76ba_jpg.rf.779226cbafe25be59968eb17baadf63a.txt:1 polygon with 5 vertices (auto-converted to bbox by ultralytics)`
- `D:\SportsStrategyCoachAI\datasets\processed\goalpost_datast\1\train\labels\1676b247b114033e7555fbe1602e76ba_jpg.rf.8f2254b2a5377a92915b00614145b69d.txt:1 polygon with 5 vertices (auto-converted to bbox by ultralytics)`
- `D:\SportsStrategyCoachAI\datasets\processed\goalpost_datast\1\train\labels\240_F_314186512_ut0quULH6FpZw3OJJdOgEBke0kT2lUqV_jpg.rf.c4a06d2f0922573d6f9a034fb4961e34.txt:1 polygon with 5 vertices (auto-converted to bbox by ultralytics)`
- `D:\SportsStrategyCoachAI\datasets\processed\goalpost_datast\1\train\labels\240_F_314186512_ut0quULH6FpZw3OJJdOgEBke0kT2lUqV_jpg.rf.fc24ebba68dbdfb999817636dae1c33f.txt:1 polygon with 5 vertices (auto-converted to bbox by ultralytics)`
- `D:\SportsStrategyCoachAI\datasets\processed\goalpost_datast\1\train\labels\240_F_533308019_QvXWcIEp8pWa7GmIXbliFmGtL6gQ28P2_jpg.rf.69c0659e73a1c657e29d83146d5b54ce.txt:1 polygon with 6 vertices (auto-converted to bbox by ultralytics)`
- `D:\SportsStrategyCoachAI\datasets\processed\goalpost_datast\1\train\labels\514Btuv4idL-_SR600-315_PIWhiteStrip-BottomLeft-0-35_PIStarRatingONEANDHALF-BottomLeft-360-6_SR600-315_ZA8-445-290-400-400-AmazonEmberBold-12-4-0-0-5_SCLZZZZZZZ_FMpng_BG255-255-255_png.rf.e21465ef25040df48d3202363f12bd6b.txt:1 polygon with 5 vertices (auto-converted to bbox by ultralytics)`
- `D:\SportsStrategyCoachAI\datasets\processed\goalpost_datast\1\train\labels\61id-t8-8sS-_AC_UF894-1000_QL80__jpg.rf.e1572adc093b69f9a9de5fec16f37471.txt:1 polygon with 5 vertices (auto-converted to bbox by ultralytics)`
- `D:\SportsStrategyCoachAI\datasets\processed\goalpost_datast\1\train\labels\aerial-closeup-penalty-spot-on-260nw-2294079723_webp.rf.5a77822b3db9f599476ce03a8936f2b7.txt:1 polygon with 5 vertices (auto-converted to bbox by ultralytics)`
- `D:\SportsStrategyCoachAI\datasets\processed\goalpost_datast\1\train\labels\ancor00781_png.rf.6ac2f671643368c7a29194bbd1c85d40.txt:1 polygon with 6 vertices (auto-converted to bbox by ultralytics)`
- `D:\SportsStrategyCoachAI\datasets\processed\goalpost_datast\1\train\labels\ancor00781_png.rf.e727e31c1d6f1bcc8fc4a81522f50632.txt:1 polygon with 6 vertices (auto-converted to bbox by ultralytics)`
- `D:\SportsStrategyCoachAI\datasets\processed\goalpost_datast\1\train\labels\ba280bdfabf724ec850a8be902a61b37_jpg.rf.e5d7a9a4ed301961e1b94ff2d8847a48.txt:1 polygon with 5 vertices (auto-converted to bbox by ultralytics)`
- `D:\SportsStrategyCoachAI\datasets\processed\goalpost_datast\1\train\labels\Brazil_jpg.rf.527e50c5b9c1bf8587d78189874eb95a.txt:1 polygon with 5 vertices (auto-converted to bbox by ultralytics)`
- `D:\SportsStrategyCoachAI\datasets\processed\goalpost_datast\1\train\labels\football-260nw-74726185_webp.rf.266d506ff6bb620d2bace37db8d44e5e.txt:1 polygon with 5 vertices (auto-converted to bbox by ultralytics)`
- `D:\SportsStrategyCoachAI\datasets\processed\goalpost_datast\1\train\labels\football-gate-1836729_jpeg_jpg.rf.3e3399c1ca0efcc973b52a02c25d0893.txt:1 polygon with 5 vertices (auto-converted to bbox by ultralytics)`
- `D:\SportsStrategyCoachAI\datasets\processed\goalpost_datast\1\train\labels\football-gate-1836729_jpeg_jpg.rf.735062c9c18e0a42bbd27c5f7e77045c.txt:1 polygon with 5 vertices (auto-converted to bbox by ultralytics)`
- _... and 273 more_

## Policy

Nothing here was auto-deleted or auto-corrected. `INFO` rows are expected characteristics, not defects -- in particular `tiny` boxes in the ball dataset are the intended signal, not noise. Only `BLOCKING` rows and cross-split duplicates should gate training.
