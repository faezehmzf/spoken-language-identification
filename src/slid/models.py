"""Classifier definitions, training helpers and metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config as C

RS = C.RANDOM_STATE


def make_pipeline(clf, reduce="none") -> Pipeline:
    reducers = {
        "none": "passthrough",
        "pca": PCA(n_components=0.95, random_state=RS),
        "lda": LinearDiscriminantAnalysis(n_components=len(C.LANGUAGES) - 1),
    }
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("reduce", reducers[reduce]),
        ("clf", clf),
    ])


def model_zoo() -> dict[str, tuple[Pipeline, dict]]:
    """name -> (pipeline, hyper-parameter grid)."""
    pca = PCA(n_components=0.95, random_state=RS)
    lda = LinearDiscriminantAnalysis(n_components=len(C.LANGUAGES) - 1)
    return {
        "Logistic Regression": (
            make_pipeline(LogisticRegression(max_iter=3000)),
            {"clf__C": [0.01, 0.1, 1.0]},
        ),
        "SVM (RBF)": (
            make_pipeline(SVC(kernel="rbf", probability=True, random_state=RS)),
            {"clf__C": [1, 10], "clf__gamma": ["scale", 0.001]},
        ),
        "KNN": (
            make_pipeline(KNeighborsClassifier()),
            {"reduce": [pca, lda], "clf__n_neighbors": [5, 15, 31], "clf__weights": ["distance"]},
        ),
        "Random Forest": (
            make_pipeline(RandomForestClassifier(n_estimators=400, n_jobs=1, random_state=RS)),
            {"clf__max_features": ["sqrt", 0.1], "clf__min_samples_leaf": [1, 3]},
        ),
        "MLP": (
            make_pipeline(MLPClassifier(hidden_layer_sizes=(256, 64), early_stopping=True,
                                        max_iter=300, random_state=RS)),
            {"clf__alpha": [1e-3, 1e-2, 1e-1]},
        ),
        "Gaussian NB": (
            make_pipeline(GaussianNB(), reduce="pca"),
            {"clf__var_smoothing": [1e-9, 1e-6]},
        ),
    }


def metrics(y_true, y_pred) -> dict:
    out = {"accuracy": accuracy_score(y_true, y_pred)}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        out.update({f"precision_{avg}": p, f"recall_{avg}": r, f"f1_{avg}": f})
    return out


def file_level(seg_pred: pd.DataFrame, proba_cols: list[str]) -> pd.DataFrame:
    """Aggregate segment probabilities into one prediction per file (mean of probabilities)."""
    agg = seg_pred.groupby(["model", "file_id", "y_true"], as_index=False)[proba_cols].mean()
    agg["y_pred"] = agg[proba_cols].to_numpy().argmax(axis=1)
    return agg


def fit_predict(pipe: Pipeline, X_tr, y_tr, X_te) -> np.ndarray:
    model = clone(pipe).fit(X_tr, y_tr)
    return model.predict_proba(X_te)


def file_group_cv(train_seg: pd.DataFrame, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """CV splits over training segments, grouped by file.

    Training indices include augmented copies; validation indices only original audio,
    so model selection is scored on the same kind of data as the test set.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    files = train_seg.drop_duplicates("file_id")
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RS)
    pos = np.arange(len(train_seg))
    is_orig = (train_seg.aug == "none").to_numpy()
    splits = []
    for tr_f, va_f in sgkf.split(files, files.label, groups=files.file_id):
        va_ids = set(files.file_id.iloc[va_f])
        in_va = train_seg.file_id.isin(va_ids).to_numpy()
        splits.append((pos[~in_va], pos[in_va & is_orig]))
    return splits


def slug(name: str) -> str:
    """'SVM (RBF)' -> 'svm_rbf'"""
    return "".join(ch for ch in name.lower().replace(" ", "_") if ch.isalnum() or ch == "_")
