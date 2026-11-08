"""Segment-level acoustic features for language identification.

Every feature group is summarised over time (mean / std, sometimes more), giving
one fixed-length vector per segment. Column names start with the group name so
groups can be selected or ablated later (e.g. all columns starting with "sdc_").
"""
from __future__ import annotations

import librosa
import numpy as np

from . import config as C

FEATURE_GROUPS = {
    "mfcc": "MFCC + delta + delta-delta (spectral envelope / phonetic content)",
    "sdc": "Shifted Delta Cepstra 7-1-3-7 (longer-range phonotactic dynamics)",
    "mel": "Log-mel band energies",
    "spec": "Spectral centroid, bandwidth, roll-off, flatness",
    "contrast": "Spectral contrast (peak vs valley per sub-band)",
    "chroma": "Chroma (pitch-class energy)",
    "zcr": "Zero-crossing rate",
    "rms": "Frame energy",
    "lpc": "Linear Predictive Coding coefficients (vocal-tract model)",
    "f0": "Fundamental frequency / intonation (speaker-normalised)",
    "rhythm": "Energy-envelope modulation spectrum, onset rate (speech rhythm)",
}


def _stats(name: str, x: np.ndarray, out: dict, funcs=("mean", "std")) -> None:
    """x: (n_features, n_frames)."""
    x = np.atleast_2d(x)
    for f in funcs:
        v = getattr(np, f)(x, axis=1)
        for i, val in enumerate(v):
            out[f"{name}_{f}_{i:02d}"] = float(val)


def shifted_delta_cepstra(c: np.ndarray, n: int = 7, d: int = 1, p: int = 3, k: int = 7) -> np.ndarray:
    """SDC with the usual N-d-P-k = 7-1-3-7 setting (Torres-Carrasquillo et al., 2002).

    c: (n_ceps, T). Returns (n * k, T).
    """
    c = c[:n]
    T = c.shape[1]
    pad = np.pad(c, ((0, 0), (d, (k - 1) * p + d)), mode="edge")
    blocks = []
    for i in range(k):
        plus = pad[:, d + i * p + d: d + i * p + d + T]
        minus = pad[:, d + i * p - d: d + i * p - d + T]
        blocks.append(plus - minus)
    return np.vstack(blocks)


def _f0_features(y: np.ndarray, sr: int, out: dict) -> None:
    hop = 320  # 20 ms is enough for intonation contours
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr, frame_length=1024, hop_length=hop)
    f0 = f0[voiced & np.isfinite(f0)]
    out["f0_voiced_ratio"] = float(voiced.mean())
    if f0.size < 10:
        for k in ("median_hz", "range_st", "std_st", "iqr_st", "slope_std_st", "skew_st"):
            out[f"f0_{k}"] = np.nan
        return
    st = 12 * np.log2(f0 / np.median(f0))  # semitones relative to the speaker's median
    out["f0_median_hz"] = float(np.median(f0))
    out["f0_std_st"] = float(st.std())
    out["f0_range_st"] = float(np.percentile(st, 95) - np.percentile(st, 5))
    out["f0_iqr_st"] = float(np.percentile(st, 75) - np.percentile(st, 25))
    out["f0_slope_std_st"] = float(np.diff(st).std())
    out["f0_skew_st"] = float(((st - st.mean()) ** 3).mean() / (st.std() ** 3 + 1e-9))


def _rhythm_features(y: np.ndarray, sr: int, rms: np.ndarray, out: dict) -> None:
    # modulation spectrum of the (log) energy envelope: syllable rate lives at ~2-8 Hz
    env = np.log(rms + 1e-6)
    env = env - env.mean()
    fs_env = sr / C.HOP_LENGTH
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env)))) ** 2
    freqs = np.fft.rfftfreq(len(env), 1 / fs_env)
    bands = [(0.5, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 8), (8, 12), (12, 16)]
    total = spec[(freqs >= 0.5) & (freqs < 16)].sum() + 1e-12
    for lo, hi in bands:
        out[f"rhythm_mod_{lo:g}_{hi:g}hz"] = float(spec[(freqs >= lo) & (freqs < hi)].sum() / total)
    sel = (freqs >= 1) & (freqs < 10)
    out["rhythm_mod_peak_hz"] = float(freqs[sel][np.argmax(spec[sel])])

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=C.HOP_LENGTH)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=C.HOP_LENGTH, units="time")
    out["rhythm_onset_rate"] = float(len(onsets) / (len(y) / sr))
    ioi = np.diff(onsets)
    if ioi.size > 2:
        out["rhythm_ioi_cv"] = float(ioi.std() / ioi.mean())
        out["rhythm_npvi"] = float(100 * np.mean(np.abs(np.diff(ioi)) / ((ioi[1:] + ioi[:-1]) / 2)))
    else:
        out["rhythm_ioi_cv"] = out["rhythm_npvi"] = np.nan
    thr = np.percentile(rms, 95) * 0.1
    out["rhythm_pause_ratio"] = float(np.mean(rms < thr))


def _lpc_features(y: np.ndarray, rms: np.ndarray, out: dict, order: int = 12) -> None:
    frames = librosa.util.frame(y, frame_length=C.WIN_LENGTH, hop_length=C.HOP_LENGTH)
    n = min(frames.shape[1], len(rms))
    frames, r = frames[:, :n], rms[:n]
    frames = frames[:, r > np.median(r)]  # voiced / active frames only
    frames = frames[:, :: 4] * np.hamming(C.WIN_LENGTH)[:, None]  # subsample for speed
    a = librosa.lpc(np.ascontiguousarray(frames.T), order=order, axis=-1)  # (n_frames, order + 1)
    a = a[np.all(np.isfinite(a), axis=1)]
    _stats("lpc", a[:, 1:].T, out, funcs=("mean",))


def extract(y: np.ndarray, sr: int = C.SR, with_f0: bool = True) -> dict:
    """Return a flat dict of features for one segment."""
    out: dict = {}
    S = np.abs(librosa.stft(y, n_fft=C.N_FFT, hop_length=C.HOP_LENGTH, win_length=C.WIN_LENGTH))
    P = S**2
    mel = librosa.feature.melspectrogram(S=P, sr=sr, n_mels=C.N_MELS, fmin=C.FMIN, fmax=C.FMAX)
    logmel = librosa.power_to_db(mel)
    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=C.N_MFCC)
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)

    _stats("mfcc", mfcc, out)
    _stats("mfcc_d1", d1, out)
    _stats("mfcc_d2", d2, out)
    # SDC on mean-normalised cepstra (cepstral mean normalisation removes channel offset)
    cmn = mfcc - mfcc.mean(axis=1, keepdims=True)
    _stats("sdc", shifted_delta_cepstra(cmn), out)
    _stats("mel", logmel, out)

    spec = np.vstack([
        librosa.feature.spectral_centroid(S=S, sr=sr),
        librosa.feature.spectral_bandwidth(S=S, sr=sr),
        librosa.feature.spectral_rolloff(S=S, sr=sr),
        librosa.feature.spectral_flatness(S=S),
    ])
    _stats("spec", spec, out)
    _stats("contrast", librosa.feature.spectral_contrast(S=S, sr=sr, n_bands=6, fmin=100), out)
    _stats("chroma", librosa.feature.chroma_stft(S=P, sr=sr), out)

    zcr = librosa.feature.zero_crossing_rate(y, frame_length=C.WIN_LENGTH, hop_length=C.HOP_LENGTH)
    _stats("zcr", zcr, out)
    rms = librosa.feature.rms(S=S, frame_length=C.N_FFT)[0]
    _stats("rms", librosa.amplitude_to_db(rms[None, :]), out)

    _lpc_features(y, rms, out)
    _rhythm_features(y, sr, rms, out)
    if with_f0:
        _f0_features(y, sr, out)
    return out


def feature_columns(columns, groups=None) -> list[str]:
    """Select feature columns, optionally restricted to some groups."""
    groups = groups or list(FEATURE_GROUPS)
    cols = []
    for c in columns:
        g = c.split("_")[0]
        if g in groups:
            cols.append(c)
    return cols
