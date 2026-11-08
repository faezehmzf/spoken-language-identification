"""Batch processing: clean every file, cut segments, extract features (+ augmented copies)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from . import cleaning, config as C, features

AUG_KINDS = ("noise", "speed", "pitch")


def _segment_rows(y: np.ndarray, file_id: str, aug: str) -> list[dict]:
    rows = []
    for k, seg in enumerate(cleaning.segment(y)):
        f = features.extract(seg)
        rows.append({"file_id": file_id, "segment": k, "aug": aug, "seg_duration": len(seg) / C.SR, **f})
    return rows


def process_file(path: str, file_id: str, seed: int, augment: bool = True,
                 save_clean: bool = True) -> tuple[dict, list[dict]]:
    warnings.filterwarnings("ignore")
    y = cleaning.load(path)
    y_clean, info = cleaning.clean(y)
    info["file_id"] = file_id
    if save_clean:
        out = C.INTERIM_DIR / "cleaned" / f"{file_id}.flac"
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, y_clean, C.SR, subtype="PCM_16")

    rows = _segment_rows(y_clean, file_id, "none")
    if augment:
        for j, kind in enumerate(AUG_KINDS):
            y_aug = cleaning.augment(y_clean, kind, seed=seed * 10 + j)
            rows += _segment_rows(y_aug, file_id, kind)
    return info, rows


def run(meta: pd.DataFrame, n_jobs: int = -1, augment: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process all files in `meta` in parallel. Returns (cleaning_report, segment_features)."""
    jobs = (
        delayed(process_file)(p, fid, seed, augment)
        for seed, (p, fid) in enumerate(zip(meta.path, meta.file_id))
    )
    results = Parallel(n_jobs=n_jobs, return_as="generator")(jobs)
    infos, rows = [], []
    for info, r in tqdm(results, total=len(meta), desc="files"):
        infos.append(info)
        rows.extend(r)
    return pd.DataFrame(infos), pd.DataFrame(rows)
