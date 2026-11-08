"""Unit tests on synthetic signals and toy metadata (no audio dataset needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from slid import cleaning, clustering, data, features, models
from slid import config as C

SR = C.SR


def voiced_tone(sec: float, f0: float = 150.0, seed: int = 0) -> np.ndarray:
    """Harmonic tone with a slow pitch wobble and syllable-like amplitude bursts."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sec * SR)) / SR
    pitch = f0 * (1 + 0.05 * np.sin(2 * np.pi * 0.7 * t))
    phase = 2 * np.pi * np.cumsum(pitch) / SR
    y = sum(np.sin(k * phase) / k for k in range(1, 6))
    y = y * (0.55 + 0.45 * np.sin(2 * np.pi * 4 * t))
    return (0.1 * y + 0.002 * rng.standard_normal(len(t))).astype(np.float32)


# ----------------------------------------------------------------- features

@pytest.fixture(scope="module")
def feats():
    return features.extract(voiced_tone(10.0))


def test_feature_vector_has_381_columns_in_documented_groups(feats):
    assert len(feats) == 381
    sizes = pd.Series([c.split("_")[0] for c in feats]).value_counts().to_dict()
    assert sizes == {"mfcc": 120, "sdc": 98, "mel": 80, "spec": 8, "contrast": 14, "chroma": 24,
                     "zcr": 2, "rms": 2, "lpc": 12, "f0": 7, "rhythm": 14}
    assert set(features.feature_columns(feats, ["f0"])) == {c for c in feats if c.startswith("f0_")}


def test_pitch_and_rhythm_features_recover_the_synthetic_signal(feats):
    assert feats["f0_median_hz"] == pytest.approx(150, rel=0.05)
    assert feats["f0_voiced_ratio"] > 0.8
    # 4 Hz amplitude bursts put most modulation energy in the 3-5 Hz bands
    assert feats["rhythm_mod_3_4hz"] + feats["rhythm_mod_4_5hz"] > 0.5


def test_pitch_features_are_speaker_normalised():
    lo = features.extract(voiced_tone(10.0, f0=110.0))
    hi = features.extract(voiced_tone(10.0, f0=220.0))
    assert hi["f0_median_hz"] == pytest.approx(2 * lo["f0_median_hz"], rel=0.05)
    assert hi["f0_range_st"] == pytest.approx(lo["f0_range_st"], abs=0.3)


def test_sdc_shape_and_values_on_a_linear_ramp():
    c = np.tile(np.arange(50.0), (20, 1))  # every cepstrum rises by 1 per frame
    sdc = features.shifted_delta_cepstra(c)
    assert sdc.shape == (7 * 7, 50)
    # c[t + d] - c[t - d] = 2 wherever the k shifted windows stay inside the signal
    assert np.allclose(sdc[:, 2:25], 2.0)


# ----------------------------------------------------------------- cleaning

def test_cleaning_caps_long_pauses_and_normalises_loudness():
    speech = voiced_tone(2.0)
    y = np.concatenate([speech, np.zeros(2 * SR, np.float32), speech])
    out, info = cleaning.clean(y)
    assert info["n_long_pauses"] == 1
    assert info["clean_duration"] < 4.5  # the 2 s pause is cut to about 0.3 s
    assert np.abs(out).max() <= 0.99 + 1e-6
    active = out[np.abs(out) > 1e-3]
    assert 20 * np.log10(np.sqrt(np.mean(active**2))) == pytest.approx(C.TARGET_DBFS, abs=2)


@pytest.mark.parametrize("sec, n_segments", [(25.0, 3), (24.0, 2), (4.0, 0)])
def test_segmentation_keeps_tails_of_at_least_five_seconds(sec, n_segments):
    segs = cleaning.segment(np.zeros(int(sec * SR), np.float32))
    assert len(segs) == n_segments
    assert all(len(s) <= C.SEGMENT_SEC * SR for s in segs)


def test_augmentations_respect_their_ranges():
    y = voiced_tone(3.0)
    noisy = cleaning.add_noise(y, np.random.default_rng(1))
    snr = 10 * np.log10(np.mean(y**2) / np.mean((noisy - y) ** 2))
    assert 10 - 0.01 <= snr <= 20 + 0.01
    ratio = len(cleaning.augment(y, "speed", seed=3)) / len(y)
    assert min(abs(ratio - 1 / 0.9), abs(ratio - 1 / 1.1)) < 0.01
    assert len(cleaning.augment(y, "pitch", seed=3)) == len(y)
    assert np.array_equal(cleaning.augment(y, "noise", seed=5), cleaning.augment(y, "noise", seed=5))


# ----------------------------------------------------------------- data and splits

def test_index_takes_labels_from_folders_and_anonymises_ids(tmp_path):
    names = {("German", "female90"): ["111111111_female_german_voice01.mp3",
                                      "111111111_female_german_voice02.mp3"],
             ("Korean", "Male"): ["222222222_male_koren_voice01.mp3.mp3"],
             ("Korean", "Female"): ["333333333_female_korean_voice01.mp3", ".hidden.mp3"]}
    for (lang, g), files in names.items():
        (tmp_path / lang / g).mkdir(parents=True)
        for f in files:
            (tmp_path / lang / g / f).touch()
    df = data.build_index(tmp_path)
    assert len(df) == 4
    assert sorted(df.contributor.unique()) == ["S01", "S02", "S03"]
    assert not df.drop(columns="path").astype(str).apply(lambda s: s.str.contains("111111111")).any().any()
    assert df.set_index("contributor").loc["S02", "language"] == "Korean"  # typo "koren" in the name is ignored
    assert df.file_id.is_unique


@pytest.fixture
def toy_meta():
    rows = []
    for li, lang in enumerate(C.LANGUAGES):
        for g in ("female", "male"):
            for s in range(4):
                for i in range(5):
                    rows.append(dict(language=lang, gender=g, label=li,
                                     contributor=f"{lang[:2]}{g[0]}{s}", file_id=f"{lang}{g}{s}_{i}"))
    return pd.DataFrame(rows)


def test_random_split_is_stratified_80_20(toy_meta):
    split = data.make_split(toy_meta)
    assert (split == "test").mean() == pytest.approx(0.2)
    per_cell = (split == "test").groupby([toy_meta.language, toy_meta.gender]).sum()
    assert per_cell.nunique() == 1


def test_group_folds_never_share_a_contributor(toy_meta):
    fold = data.group_folds(toy_meta, "contributor")
    assert (fold >= 0).all()
    assert toy_meta.groupby("contributor").apply(lambda d: fold[d.index].nunique()).eq(1).all()


def test_contributors_reading_the_same_recording_are_merged():
    rng = np.random.default_rng(0)
    meta = pd.DataFrame({"language": "German", "gender": "male",
                         "contributor": np.repeat(["A", "B", "C"], 8)})
    centre = {"A": 0.0, "B": 0.0, "C": 6.0}  # A and B share a narrator, C does not
    Z = np.stack([rng.normal(centre[c], 1.0, 10) for c in meta.contributor])
    ratios = data.contributor_distance_ratios(Z, meta)
    r = ratios.set_index(["a", "b"]).ratio
    assert r[("A", "B")] < 1.1 < r[("A", "C")]
    narr = data.estimate_narrators(ratios, meta)
    assert narr["A"] == narr["B"] != narr["C"]


# ----------------------------------------------------------------- models and clustering

def test_validation_folds_hold_out_whole_files_on_original_audio():
    rows = [dict(file_id=f"f{f}", label=f % 4, aug=a)
            for f in range(20) for a in ("none", "noise", "speed", "pitch") for _ in range(2)]
    seg = pd.DataFrame(rows)
    for tr, va in models.file_group_cv(seg):
        assert set(seg.aug.iloc[va]) == {"none"}
        assert not set(seg.file_id.iloc[tr]) & set(seg.file_id.iloc[va])


def test_file_level_decision_averages_segment_probabilities():
    seg = pd.DataFrame({"model": "m", "file_id": ["a", "a", "a"], "y_true": 0,
                        "p0": [0.9, 0.2, 0.6], "p1": [0.1, 0.8, 0.4]})
    out = models.file_level(seg, ["p0", "p1"])
    assert out.y_pred.tolist() == [0]
    assert out.p0.iloc[0] == pytest.approx(0.5667, abs=1e-3)


def test_purity_counts_noise_points_as_errors():
    y_true = np.array([0, 0, 1, 1, 1, 0])
    assert clustering.purity(y_true, [0, 0, 1, 1, 1, 1]) == pytest.approx(5 / 6)
    assert clustering.purity(y_true, [0, 0, 1, 1, -1, -1]) == pytest.approx(4 / 6)
    X = np.r_[np.zeros((3, 2)), np.ones((3, 2))] + np.random.default_rng(0).normal(0, 0.01, (6, 2))
    res = clustering.evaluate(X, [0, 0, 0, 1, 1, 1], {"language": y_true})
    assert res["n_clusters"] == 2 and res["silhouette"] > 0.9


def test_model_zoo_covers_the_six_classifiers():
    zoo = models.model_zoo()
    assert list(zoo) == ["Logistic Regression", "SVM (RBF)", "KNN", "Random Forest", "MLP", "Gaussian NB"]
    assert models.slug("SVM (RBF)") == "svm_rbf"
