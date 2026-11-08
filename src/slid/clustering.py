"""Clustering helpers and metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score, silhouette_score)


def purity(y_true, y_pred) -> float:
    """Fraction of points that belong to the majority true class of their cluster (noise label -1 counts as wrong)."""
    ct = pd.crosstab(pd.Series(y_pred, name="c"), pd.Series(y_true, name="t"))
    if -1 in ct.index:
        ct = ct.drop(index=-1)
    return float(ct.max(axis=1).sum() / len(y_true))


def file_level_matrix(seg: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """One vector per file: the mean of its original-audio segment features."""
    return seg[seg.aug == "none"].groupby("file_id")[feats].mean()


def evaluate(X, labels, refs: dict[str, np.ndarray]) -> dict:
    """Internal (silhouette) and external (purity / ARI / NMI against each reference labelling) scores."""
    labels = np.asarray(labels)
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    out = {"n_clusters": n_clusters, "noise_frac": float((~mask).mean()),
           "silhouette": float(silhouette_score(X[mask], labels[mask])) if n_clusters > 1 else np.nan}
    for ref_name, y in refs.items():
        out[f"purity_{ref_name}"] = purity(y, labels)
        out[f"ARI_{ref_name}"] = adjusted_rand_score(y, labels)
        out[f"NMI_{ref_name}"] = normalized_mutual_info_score(y, labels)
    return out
