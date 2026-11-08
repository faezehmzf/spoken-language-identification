# Data

The audio is not included in this repository. The clips were cut from public audiobooks and podcasts by the people who contributed them, and the recordings are not ours to redistribute.

## Expected layout

```text
data/raw/
├── German/
│   ├── female90/*.mp3
│   └── male90/*.mp3
├── Italian/{Female,Male}/*.mp3
├── Korean/{Female,Male}/*.mp3
└── Spanish/{Female,Male}/*.mp3
```

Labels are read from the two folder levels (language, then gender). Any folder name that starts with `female` (case-insensitive) counts as female, everything else as male. File names follow `<contributor-id>_<gender>_<language>_voiceNN.mp3`. Only the contributor ID is used, and it is replaced by an anonymous code (`S01` … `S78`) as soon as the index is built, so no original ID appears in any output.

The raw folder can live anywhere:

```bash
export SLID_DATA_DIR=/path/to/dataset
```

## Generated folders

| Folder | Contents | Size |
|---|---|---|
| `data/interim/cleaned/` | cleaned 16 kHz mono FLAC, one per clip | ~670 MB |
| `data/processed/` | `metadata.parquet`, `cleaning_report.parquet`, `segment_features.parquet` (15,683 rows: 3,920 original 10 s segments plus three augmented copies, × 381 features) | ~45 MB |
| `models/` | fitted pipelines from notebook 2 (`joblib`) | ~50 MB |

All three are git-ignored. Notebook 1 rebuilds the first two in about 30 minutes on a 10-core laptop; notebook 2 writes `models/`.

## Summary

| | German | Italian | Korean | Spanish |
|---|:-:|:-:|:-:|:-:|
| clips (female / male) | 90 / 90 | 90 / 89 | 90 / 90 | 90 / 90 |
| contributors | 28 | 18 | 14 | 18 |
| estimated narrators | 5 | 9 | 5 | 5 |
| minutes of audio | 187 | 179 | 184 | 183 |

One 25-minute Italian upload that contains the full source of a contributor's other clips is excluded, which leaves 719 of 720 files and 12.2 hours of audio.
