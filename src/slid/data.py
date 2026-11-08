"""Dataset indexing, auditing and train/test splitting."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from . import config as C

_NAME_RE = re.compile(r"^(?P<sid>\d+)_(?P<gender>male|female)_", re.IGNORECASE)


def _gender_from_folder(folder: str) -> str:
    folder = folder.lower()
    return "female" if folder.startswith("female") else "male"


def build_index(data_dir: Path = C.DATA_DIR) -> pd.DataFrame:
    """Scan the dataset and return one row per audio file.

    Labels come from the folder structure (Language/Gender), which is more
    reliable than file names (some contain typos such as "koren").
    The contributor ID in the file name identifies the person who cut the clips;
    it is replaced by an anonymous code (S01, S02, ...). Several contributors may
    have cut clips from the same audiobook, see `estimate_narrators`.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_dir}. Set SLID_DATA_DIR or place the data in data/raw."
        )

    rows = []
    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() != ".mp3" or path.name.startswith("."):
            continue
        rel = path.relative_to(data_dir)
        language, gender_folder = rel.parts[0], rel.parts[1]
        m = _NAME_RE.match(path.name)
        rows.append(
            dict(
                path=str(path),
                language=language,
                gender=_gender_from_folder(gender_folder),
                contributor=m.group("sid") if m else "unknown",
                name_gender=m.group("gender").lower() if m else None,
            )
        )
    df = pd.DataFrame(rows)

    codes = {c: f"S{i + 1:02d}" for i, c in enumerate(sorted(df.contributor.unique()))}
    df["contributor"] = df.contributor.map(codes)
    df["file_id"] = [
        f"{lang[:3].upper()}_{g[0].upper()}_{spk}_{i:03d}"
        for i, (lang, g, spk) in enumerate(zip(df.language, df.gender, df.contributor))
    ]
    df["label"] = df.language.map({l: i for i, l in enumerate(C.LANGUAGES)})
    return df


def audit(df: pd.DataFrame) -> pd.DataFrame:
    """Add container-level info (sample rate, channels, duration, md5)."""
    info = []
    for p in df.path:
        try:
            i = sf.info(p)
            info.append((i.samplerate, i.channels, i.duration, None))
        except Exception as e:  # corrupt or unsupported file
            info.append((np.nan, np.nan, np.nan, str(e)))
    out = df.copy()
    out[["orig_sr", "channels", "duration", "read_error"]] = pd.DataFrame(info, index=df.index)
    out["md5"] = [hashlib.md5(Path(p).read_bytes()).hexdigest() for p in df.path]
    out["name_has_double_ext"] = [Path(p).name.lower().count(".mp3") > 1 for p in df.path]
    out["gender_name_mismatch"] = out.name_gender.notna() & (out.name_gender != out.gender)
    return out


def flag_exclusions(df: pd.DataFrame) -> pd.Series:
    """Return the reason a file is excluded, or '' if it is kept."""
    reason = pd.Series("", index=df.index)
    reason[df.read_error.notna()] = "unreadable"
    reason[df.duration < C.MIN_DURATION_SEC] = "too short"
    reason[df.duration > C.MAX_DURATION_SEC] = "full-source upload (overlaps other clips)"
    reason[df.md5.duplicated(keep="first") & (reason == "")] = "exact duplicate"
    return reason


def make_split(meta: pd.DataFrame) -> pd.Series:
    """Stratified 80/20 split at file level (stratified by language x gender).

    Splitting files (not segments) keeps all segments and augmented copies of
    one recording on the same side of the split.
    """
    strata = meta.language + "_" + meta.gender
    train_idx, test_idx = train_test_split(
        meta.index, test_size=C.TEST_SIZE, stratify=strata, random_state=C.RANDOM_STATE
    )
    split = pd.Series("train", index=meta.index)
    split[test_idx] = "test"
    return split


def group_folds(meta: pd.DataFrame, groups: str, n_splits: int = 5) -> pd.Series:
    """Assign each file to a fold so that no value of `groups` appears in two folds."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=C.RANDOM_STATE)
    fold = pd.Series(-1, index=meta.index)
    for k, (_, te) in enumerate(sgkf.split(meta, meta.label, groups=meta[groups])):
        fold.iloc[te] = k
    return fold


def contributor_distance_ratios(Z: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    """For every pair of contributors with the same language and gender:
    mean between-contributor file distance / geometric mean of their within-contributor distances.

    A ratio close to 1 means the two contributors' clips are as similar to each other as
    clips of one contributor are among themselves -> very likely the same narrator/audiobook.
    Z: standardised file-level features, rows aligned with `meta`.
    """
    from scipy.spatial.distance import cdist

    con = meta.contributor.to_numpy()
    rows = []
    for (lang, gender), d in meta.groupby(["language", "gender"]):
        names = sorted(d.contributor.unique())
        idx = {s: np.where(con == s)[0] for s in names}
        within = {}
        for s in names:
            D = cdist(Z[idx[s]], Z[idx[s]])
            within[s] = D[np.triu_indices(len(idx[s]), 1)].mean()
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                between = cdist(Z[idx[a]], Z[idx[b]]).mean()
                rows.append(dict(language=lang, gender=gender, a=a, b=b,
                                 ratio=between / np.sqrt(within[a] * within[b])))
    return pd.DataFrame(rows)


def estimate_narrators(ratios: pd.DataFrame, meta: pd.DataFrame, cut: float = 1.2) -> pd.Series:
    """Merge contributors into estimated narrator groups (average-linkage on ratio - 1).

    Merging only happens within the same language and gender. Returns contributor -> narrator code.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    r = ratios.set_index(["a", "b"]).ratio
    mapping = {}
    for (lang, gender), d in meta.groupby(["language", "gender"]):
        names = sorted(d.contributor.unique())
        k = len(names)
        if k == 1:
            mapping[names[0]] = (lang, gender, 1)
            continue
        D = np.zeros((k, k))
        for i in range(k):
            for j in range(i + 1, k):
                D[i, j] = D[j, i] = max(r[(names[i], names[j])] - 1, 0)
        labels = fcluster(linkage(squareform(D), "average"), cut - 1, "distance")
        for s, lab in zip(names, labels):
            mapping[s] = (lang, gender, lab)
    keys = sorted(set(mapping.values()))
    code = {k: f"N{i + 1:02d}" for i, k in enumerate(keys)}
    return pd.Series({s: code[v] for s, v in mapping.items()}, name="narrator")
