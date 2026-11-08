"""Project-wide paths and signal-processing constants."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Raw dataset location. Expected layout: <DATA_DIR>/<Language>/<Gender>/*.mp3
DATA_DIR = Path(os.environ.get("SLID_DATA_DIR", ROOT / "data" / "raw")).expanduser()

INTERIM_DIR = ROOT / "data" / "interim"      # cleaned audio (FLAC)
PROCESSED_DIR = ROOT / "data" / "processed"  # metadata, features, splits
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"

LANGUAGES = ["German", "Italian", "Korean", "Spanish"]

# Signal processing
SR = 16_000            # speech band is < 8 kHz; also standard for speech models
N_FFT = 512            # 32 ms
WIN_LENGTH = 400       # 25 ms analysis window
HOP_LENGTH = 160       # 10 ms hop -> 100 frames / s
N_MELS = 40
N_MFCC = 20
FMIN, FMAX = 20, 8_000

# Cleaning
HIGHPASS_HZ = 80       # removes rumble / DC drift below the voice range
SILENCE_TOP_DB = 35    # frames this far below the peak count as silence
MAX_PAUSE_SEC = 0.30   # long pauses are shortened to this, short ones are kept
TARGET_DBFS = -23.0    # RMS loudness target for active speech
MIN_DURATION_SEC = 20
MAX_DURATION_SEC = 180  # longer uploads are full-source files, not clips

# Segmentation
SEGMENT_SEC = 10.0
MIN_SEGMENT_SEC = 5.0

RANDOM_STATE = 42
TEST_SIZE = 0.20


def ensure_dirs() -> None:
    for d in (INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
