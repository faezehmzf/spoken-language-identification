"""Audio cleaning, segmentation and augmentation."""
from __future__ import annotations

import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt

from . import config as C


def load(path: str, sr: int = C.SR) -> np.ndarray:
    """Decode, down-mix to mono and resample."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


def _db(x: float) -> float:
    return float(20 * np.log10(max(x, 1e-10)))


def clean(y: np.ndarray, sr: int = C.SR) -> tuple[np.ndarray, dict]:
    """Clean one recording.

    1. remove DC offset and high-pass at 80 Hz (rumble, hum below the voice range)
    2. detect speech regions with an energy threshold
    3. shorten long pauses to 0.3 s but keep short ones (pauses carry rhythm)
    4. normalise loudness of active speech to -23 dBFS, then guard against clipping
    """
    info: dict = {"raw_duration": len(y) / sr}
    info["clipping_ratio"] = float(np.mean(np.abs(y) >= 0.999))

    y = y - y.mean()
    sos = butter(4, C.HIGHPASS_HZ, btype="highpass", fs=sr, output="sos")
    y = sosfiltfilt(sos, y).astype(np.float32)

    intervals = librosa.effects.split(y, top_db=C.SILENCE_TOP_DB, frame_length=1024, hop_length=256)
    speech = np.concatenate([y[s:e] for s, e in intervals]) if len(intervals) else y
    mask = np.zeros(len(y), bool)
    for s, e in intervals:
        mask[s:e] = True
    noise = y[~mask]
    speech_rms = np.sqrt(np.mean(speech**2)) + 1e-10
    noise_rms = np.sqrt(np.mean(noise**2)) + 1e-10 if noise.size > sr // 10 else np.nan
    info["speech_ratio"] = float(mask.mean())
    info["snr_db"] = _db(speech_rms) - _db(noise_rms) if np.isfinite(noise_rms) else np.nan
    pauses = (intervals[1:, 0] - intervals[:-1, 1]) / sr if len(intervals) > 1 else np.array([])
    info["n_long_pauses"] = int(np.sum(pauses > C.MAX_PAUSE_SEC))

    # rebuild the signal with long pauses capped
    max_gap = int(C.MAX_PAUSE_SEC * sr)
    parts = []
    for i, (s, e) in enumerate(intervals):
        parts.append(y[s:e])
        if i + 1 < len(intervals):
            gap_end = min(intervals[i + 1][0], e + max_gap)
            parts.append(y[e:gap_end])
    out = np.concatenate(parts) if parts else y

    gain = 10 ** ((C.TARGET_DBFS - _db(speech_rms)) / 20)
    out = out * gain
    peak = np.abs(out).max()
    if peak > 0.99:
        out = out * (0.99 / peak)
    info["clean_duration"] = len(out) / sr
    return out.astype(np.float32), info


def segment(y: np.ndarray, sr: int = C.SR, seg_sec: float = C.SEGMENT_SEC,
            min_sec: float = C.MIN_SEGMENT_SEC) -> list[np.ndarray]:
    """Cut into non-overlapping fixed-length segments; keep a short tail only if >= min_sec."""
    n, m = int(seg_sec * sr), int(min_sec * sr)
    segs = [y[i:i + n] for i in range(0, len(y), n)]
    return [s for s in segs if len(s) >= m]


# ---------------------------------------------------------------- augmentation

def add_noise(y: np.ndarray, rng: np.random.Generator, snr_db: tuple = (10, 20)) -> np.ndarray:
    """Add pink-ish noise at a random SNR (simulates low-quality recordings)."""
    white = rng.standard_normal(len(y)).astype(np.float32)
    spec = np.fft.rfft(white)
    spec /= np.sqrt(np.maximum(np.arange(len(spec)), 1))  # 1/f power
    noise = np.fft.irfft(spec, n=len(y)).astype(np.float32)
    snr = rng.uniform(*snr_db)
    p_sig, p_noise = np.mean(y**2), np.mean(noise**2) + 1e-12
    return y + noise * np.sqrt(p_sig / (p_noise * 10 ** (snr / 10)))


def speed_perturb(y: np.ndarray, rng: np.random.Generator, sr: int = C.SR) -> np.ndarray:
    """Speed perturbation (tempo + pitch together), as used in Kaldi recipes."""
    factor = rng.choice([0.9, 1.1])
    return librosa.resample(y, orig_sr=sr, target_sr=int(round(sr / factor)), res_type="soxr_hq")


def pitch_shift(y: np.ndarray, rng: np.random.Generator, sr: int = C.SR) -> np.ndarray:
    """Shift pitch by +-1..3 semitones without changing tempo (speaker variability)."""
    steps = rng.choice([-3, -2, -1, 1, 2, 3])
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=float(steps))


AUGMENTATIONS = {
    "noise": add_noise,
    "speed": speed_perturb,
    "pitch": pitch_shift,
}


def augment(y: np.ndarray, kind: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = AUGMENTATIONS[kind](y, rng)
    peak = np.abs(out).max()
    return (out * (0.99 / peak) if peak > 0.99 else out).astype(np.float32)
