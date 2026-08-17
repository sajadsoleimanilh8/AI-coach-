# Dataset audit -- `field`

_Generated 2026-08-12 by `scripts/dataset_qa.py`._

- **Task**: `segment`  |  **classes**: 1
- **Resolved root**: `D:\SportsStrategyCoachAI\datasets\processed`
- **Source dirs**:
  - `D:\SportsStrategyCoachAI\datasets\processed\..\derived\field_clipped`

- **Images**: 976  |  **labelled instances**: 976

**Verdict: PASS** (0 blocking issue(s), 0 cross-split duplicate group(s))

## Splits

| split | images | label files | instances | empty labels |
|---|---:|---:|---:|---:|
| `field_clipped/train` | 926 | 926 | 926 | 0 |
| `field_clipped/valid` | 35 | 35 | 35 | 0 |
| `field_clipped/test` | 15 | 15 | 15 | 0 |

## Class balance

| split | class | instances |
|---|---|---:|
| test | `pitch` | 15 |
| train | `pitch` | 926 |
| valid | `pitch` | 35 |

## Issues

### `dup_images` -- 46 (REVIEW)

- `[within split] 2x identical: train:0bfacc_0_png_jpg.rf.317f49ece289bf4c2b4ce4eccd7f2062.jpg, train:0bfacc_0_png_jpg.rf.4aeff27047e3d2e35eae2396c731e366.jpg`
- `[within split] 2x identical: train:0bfacc_4_png_jpg.rf.3ff4dc650035efd1951e7ffb5af06d2f.jpg, train:0bfacc_4_png_jpg.rf.5723c6043366e26ba427aa441945da7a.jpg`
- `[within split] 2x identical: train:2e57b9_4_png_jpg.rf.482b3952352aa95476a99a0306020ef9.jpg, train:2e57b9_4_png_jpg.rf.a3e8b634166bee2d08683dbad088955c.jpg`
- `[within split] 2x identical: train:2e57b9_4_png_jpg.rf.4983538dcc64583321133345cc4e2492.jpg, train:2e57b9_4_png_jpg.rf.ab7becd827d57384e72abca5c2eb5c2e.jpg`
- `[within split] 2x identical: train:54745b_0_png_jpg.rf.0308b8b2374d0f5cda89e1208b56adbb.jpg, train:54745b_0_png_jpg.rf.bb409f2124ccad3c06ed8bc549960cfd.jpg`
- `[within split] 2x identical: train:744b27_7_png_jpg.rf.2c01a1f915b40bd3c86eb59ba5c42c8d.jpg, train:744b27_7_png_jpg.rf.97bf62f327abd1dcec165e0ae7f2afd0.jpg`
- `[within split] 2x identical: train:798b45_9_png_jpg.rf.44c5f952e3fef46169801b277d125f11.jpg, train:798b45_9_png_jpg.rf.847393145e38cd46f2a6098d793c09a2.jpg`
- `[within split] 3x identical: train:08fd33_1_png_jpg.rf.9d15b2f1c61b3d6b245854ea7d402c24.jpg, train:08fd33_1_png_jpg.rf.c1835afd372737733f5e5a62aa32be15.jpg, train:08fd33_1_png_jpg.rf.e6516c0dda42baf2226057bd76be7c7f.jpg`
- `[within split] 3x identical: train:08fd33_3_png_jpg.rf.6d6c39fa20b6dda2ca92157f10859d4a.jpg, train:08fd33_3_png_jpg.rf.8c8fc684ed3c06fc9ff8f73fe6c59086.jpg, train:08fd33_3_png_jpg.rf.f3635a142ef18b2e32c7a63c6ea58d2d.jpg`
- `[within split] 3x identical: train:0a2d9b_4_png_jpg.rf.6af66a0847240370be6b868ef191240f.jpg, train:0a2d9b_4_png_jpg.rf.dc4023da967fc5dad2f6663090b675b4.jpg, train:0a2d9b_4_png_jpg.rf.f6edde9dcd2ebcfffd0ae18cbe45243c.jpg`
- `[within split] 3x identical: train:0a2d9b_8_png_jpg.rf.10b0691b2cf2b891febaedf7b041c8d9.jpg, train:0a2d9b_8_png_jpg.rf.923af42415f52252bd28d3621df102ae.jpg, train:0a2d9b_8_png_jpg.rf.f0edfd0b41fce6514cb9aae7d7aa01a1.jpg`
- `[within split] 3x identical: train:0a2d9b_9_png_jpg.rf.492bc44231e439c4b59acc7ebd5ee1e5.jpg, train:0a2d9b_9_png_jpg.rf.78ccf6eae50b55437b43608f07f610e3.jpg, train:0a2d9b_9_png_jpg.rf.a7064e58c3fd95e3beebe652d2864e8b.jpg`
- `[within split] 3x identical: train:0bfacc_0_png_jpg.rf.229e9da1bff2bcfb09dae8547ea69108.jpg, train:0bfacc_0_png_jpg.rf.35f95d98fe20b72915a9d4370de6a2f5.jpg, train:0bfacc_0_png_jpg.rf.5267474a31819ad6706b5f1a3146c11c.jpg`
- `[within split] 3x identical: train:0bfacc_7_png_jpg.rf.752766decd4d5740306342b4018ee9ae.jpg, train:0bfacc_7_png_jpg.rf.ce482dedbdee23b965c692be1a062954.jpg, train:0bfacc_7_png_jpg.rf.fb2a3482b2843ec96764e304e3102d42.jpg`
- `[within split] 3x identical: train:121364_4_png_jpg.rf.3dc51b6898844616f41fcd4d01384c23.jpg, train:121364_4_png_jpg.rf.5dd733fda55c7aaa1537d5165dfeae1f.jpg, train:121364_4_png_jpg.rf.ef1a6b5a256a3ad75dcc2f56f61cb824.jpg`
- _... and 31 more_

## Policy

Nothing here was auto-deleted or auto-corrected. `INFO` rows are expected characteristics, not defects -- in particular `tiny` boxes in the ball dataset are the intended signal, not noise. Only `BLOCKING` rows and cross-split duplicates should gate training.
