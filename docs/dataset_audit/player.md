# Dataset audit -- `player`

_Generated 2026-08-12 by `scripts/dataset_qa.py`._

- **Task**: `detect`  |  **classes**: 1
- **Resolved root**: `D:\SportsStrategyCoachAI\datasets\processed`
- **Source dirs**:
  - `D:\SportsStrategyCoachAI\datasets\processed\player_datasets\1`
  - `D:\SportsStrategyCoachAI\datasets\processed\player_datasets\2`

- **Images**: 13,480  |  **labelled instances**: 195,830

**Verdict: PASS** (0 blocking issue(s), 0 cross-split duplicate group(s))

## Splits

| split | images | label files | instances | empty labels |
|---|---:|---:|---:|---:|
| `1/train` | 5,436 | 5,430 | 83,421 | 0 |
| `1/valid` | 284 | 283 | 4,189 | 0 |
| `1/test` | 184 | 183 | 2,766 | 0 |
| `2/train` | 6,812 | 6,808 | 95,106 | 0 |
| `2/valid` | 664 | 664 | 9,012 | 0 |
| `2/test` | 100 | 100 | 1,336 | 0 |

## Class balance

| split | class | instances |
|---|---|---:|
| test | `player` | 4,102 |
| train | `player` | 178,527 |
| valid | `player` | 13,201 |

## Issues

### `orphans` -- 12 (REVIEW)

- `image without label: D:\SportsStrategyCoachAI\datasets\processed\player_datasets\1\train\images\video1-1-2105570_jpg.rf.50641538e869787477ab6af0a3ee02ba.jpg`
- `image without label: D:\SportsStrategyCoachAI\datasets\processed\player_datasets\1\train\images\video1-1-2105570_jpg.rf.83e86f61fa412c2cf89d38ca08732964.jpg`
- `image without label: D:\SportsStrategyCoachAI\datasets\processed\player_datasets\1\train\images\video1-1-2105570_jpg.rf.ff75d548115a02d24626667ed4e93930.jpg`
- `image without label: D:\SportsStrategyCoachAI\datasets\processed\player_datasets\1\train\images\video1-1-2110262_jpg.rf.4a5d92738c54b08c373048c5606d9d71.jpg`
- `image without label: D:\SportsStrategyCoachAI\datasets\processed\player_datasets\1\train\images\video1-1-2110262_jpg.rf.6124060983754ae069c1c06d9c2416de.jpg`
- `image without label: D:\SportsStrategyCoachAI\datasets\processed\player_datasets\1\train\images\video1-1-2110262_jpg.rf.f2829d56f375b7a1e863a11cab0b5789.jpg`
- `image without label: D:\SportsStrategyCoachAI\datasets\processed\player_datasets\1\valid\images\youtube-79_jpg.rf.a599b853f902afb03323ae76a251f089.jpg`
- `image without label: D:\SportsStrategyCoachAI\datasets\processed\player_datasets\1\test\images\video1-1-2107916_jpg.rf.6dbc14d3d522080bf1dce5cb352abc79.jpg`
- `image without label: D:\SportsStrategyCoachAI\datasets\processed\player_datasets\2\train\images\n_197_jpg.rf.1c7f291a09cd568bff2232e717c22d66.jpg`
- `image without label: D:\SportsStrategyCoachAI\datasets\processed\player_datasets\2\train\images\n_41_jpg.rf.f6dcf2df80f17b252879eee077499753.jpg`
- `image without label: D:\SportsStrategyCoachAI\datasets\processed\player_datasets\2\train\images\red-card_jpeg.rf.df03ff08959a091f77aea01a1f614a3b.jpg`
- `image without label: D:\SportsStrategyCoachAI\datasets\processed\player_datasets\2\train\images\yellow_card_set2_nvn_110_png.rf.2b6b05dc21ab17d490c169adb77a9f7f.jpg`

### `mixed_format` -- 3 (INFO)

- `D:\SportsStrategyCoachAI\datasets\processed\player_datasets\1\train\labels\youtube-29_jpg.rf.20421ec7d8818ff29c9af18a8118d712.txt:1 polygon with 36 vertices (auto-converted to bbox by ultralytics)`
- `D:\SportsStrategyCoachAI\datasets\processed\player_datasets\1\train\labels\youtube-29_jpg.rf.229250e3de11597984d3da9a46f27f22.txt:1 polygon with 36 vertices (auto-converted to bbox by ultralytics)`
- `D:\SportsStrategyCoachAI\datasets\processed\player_datasets\1\train\labels\youtube-29_jpg.rf.a4f658668db13996f09c1280a6447993.txt:1 polygon with 36 vertices (auto-converted to bbox by ultralytics)`

### `tiny` -- 14 (INFO)

- `D:\SportsStrategyCoachAI\datasets\processed\player_datasets\1\train\labels\CosmosGame126684_jpg.rf.a4f1a12ac2a882dd71a7e1c0d7c2ceaf.txt:12 area=0.000093`
- `D:\SportsStrategyCoachAI\datasets\processed\player_datasets\1\train\labels\CosmosGame126684_jpg.rf.ba91fc5e750e8ec77b05955a86a4f235.txt:12 area=0.000093`
- `D:\SportsStrategyCoachAI\datasets\processed\player_datasets\1\train\labels\CosmosGame126684_jpg.rf.d4dba4350c6661c45d2232c17365b403.txt:12 area=0.000093`
- `D:\SportsStrategyCoachAI\datasets\processed\player_datasets\1\train\labels\frame0-00-00-00_jpg.rf.65ba70b152e6a6164f37fb0fa25cb419.txt:6 area=0.000001`
- `D:\SportsStrategyCoachAI\datasets\processed\player_datasets\1\train\labels\frame0-00-00-00_jpg.rf.9adfc8e1647c95e1308428a3579f051c.txt:6 area=0.000001`
- `D:\SportsStrategyCoachAI\datasets\processed\player_datasets\1\train\labels\frame0-00-00-00_jpg.rf.b2ceb8aa5223ce0da5eafa3bf07ed594.txt:6 area=0.000001`
- `D:\SportsStrategyCoachAI\datasets\processed\player_datasets\2\train\labels\3320_png.rf.14b7366124b5e8d3e0a839a9e307d6a6.txt:2 area=0.000000`
- `D:\SportsStrategyCoachAI\datasets\processed\player_datasets\2\train\labels\3860_png.rf.29ca03c50125ade9c340f8985b3e5ada.txt:7 area=0.000000`
- `D:\SportsStrategyCoachAI\datasets\processed\player_datasets\2\train\labels\4967_png.rf.89546f4825a5fa5e765bf135c5e16b6f.txt:22 area=0.000087`
- `D:\SportsStrategyCoachAI\datasets\processed\player_datasets\2\train\labels\6783_png.rf.89782562c88fb6fabf8b3d473d38b35a.txt:2 area=0.000000`
- `D:\SportsStrategyCoachAI\datasets\processed\player_datasets\2\train\labels\7005_png.rf.3d8907f06669e5de776876b9c682af45.txt:10 area=0.000000`
- `D:\SportsStrategyCoachAI\datasets\processed\player_datasets\2\train\labels\7477_png.rf.48d1e4a7db245615f127da0af9f83065.txt:11 area=0.000069`
- `D:\SportsStrategyCoachAI\datasets\processed\player_datasets\2\train\labels\809_png.rf.11d8d408c42186c69d07b6c5f261cdd7.txt:8 area=0.000078`
- `D:\SportsStrategyCoachAI\datasets\processed\player_datasets\2\train\labels\8500_png.rf.ec4d10bd5d653898cf60b0867dd3c5cc.txt:3 area=0.000000`

## Policy

Nothing here was auto-deleted or auto-corrected. `INFO` rows are expected characteristics, not defects -- in particular `tiny` boxes in the ball dataset are the intended signal, not noise. Only `BLOCKING` rows and cross-split duplicates should gate training.
