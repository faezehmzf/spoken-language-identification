"""Shared plotting style and helpers."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from . import config as C

LANG_COLORS = dict(zip(C.LANGUAGES, ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]))


def setup() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight"})


def save(fig, name: str) -> None:
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES_DIR / f"{name}.png")


def confusion(ax, y_true, y_pred, title: str, normalize: bool = True) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=range(len(C.LANGUAGES)))
    shown = cm / cm.sum(axis=1, keepdims=True) if normalize else cm
    annot = np.array([[f"{v:.0%}\n({n})" for v, n in zip(r1, r2)] for r1, r2 in zip(shown, cm)])
    sns.heatmap(shown, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1 if normalize else None,
                xticklabels=C.LANGUAGES, yticklabels=C.LANGUAGES, cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
