"""Feature pipeline unit tests (synthetic data, no dataset required)."""

import numpy as np
import pandas as pd
import pytest

from src.data.features import _entropy_and_unique, build_windows
from src.data.split import split_boundaries


def test_split_boundaries_cover_everything():
    bounds = split_boundaries(1000, (0.6, 0.2, 0.2))
    assert bounds["train"] == (0, 600)
    assert bounds["val"] == (600, 800)
    assert bounds["test"] == (800, 1000)


def test_split_boundaries_are_time_ordered():
    bounds = split_boundaries(12345)
    assert bounds["train"][1] == bounds["val"][0]
    assert bounds["val"][1] == bounds["test"][0]
    assert bounds["test"][1] == 12345


def test_split_rejects_bad_fractions():
    with pytest.raises(ValueError):
        split_boundaries(100, (0.5, 0.2, 0.2))


def test_entropy_and_unique_matches_manual():
    codes = np.array([
        [1, 1, 1, 1],          # one id -> entropy 0, unique 1
        [1, 2, 1, 2],          # two ids 50/50 -> entropy 1 bit, unique 2
        [1, 2, 3, 4],          # four ids -> entropy 2 bits, unique 4
    ])
    entropy, unique = _entropy_and_unique(codes)
    assert entropy == pytest.approx([0.0, 1.0, 2.0], abs=1e-6)
    assert list(unique) == [1, 2, 4]


def _synthetic_segment(n: int = 200, attack_rows: slice = slice(100, 120)):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "timestamp": np.cumsum(rng.uniform(0.0004, 0.0006, n)),
        "can_id": rng.choice([0x130, 0x131, 0x316], n).astype(np.uint16),
        "dt_prev": np.full(n, 0.0005, dtype=np.float32),
        "dt_same_id": np.full(n, 0.0015, dtype=np.float32),
        "rel_dev": np.zeros(n, dtype=np.float32),
        "is_unknown": np.zeros(n, dtype=np.uint8),
        "label": np.zeros(n, dtype=np.uint8),
    })
    # iloc: end-exclusive slicing (loc slices would include the end row)
    df.iloc[attack_rows, df.columns.get_loc("label")] = 1
    return df


def test_window_attack_labeling():
    segment = _synthetic_segment()
    windows = build_windows(segment, "rpm", "test", global_offset=0,
                            window=64, stride=32)
    # Windows overlapping rows 100-119 must be labeled attack
    assert windows is not None
    overlapping = windows[(windows["start_row"] + 64 > 100)
                          & (windows["start_row"] < 120)]
    clean = windows[(windows["start_row"] + 64 <= 100)
                    | (windows["start_row"] >= 120)]
    assert (overlapping["label"] == 1).all()
    assert (clean["label"] == 0).all()
    assert set(overlapping["attack_type"]) == {"rpm"}
    assert set(clean["attack_type"]) == {"normal"}


def test_window_injected_ratio_exact():
    segment = _synthetic_segment(n=64, attack_rows=slice(0, 16))
    windows = build_windows(segment, "dos", "test", global_offset=0,
                            window=64, stride=32)
    assert len(windows) == 1
    assert windows.loc[0, "injected_ratio"] == pytest.approx(16 / 64)


def test_windows_never_cross_short_segment():
    segment = _synthetic_segment(n=40)
    assert build_windows(segment, "dos", "test", 0, window=64, stride=32) is None


def test_live_featurizer_matches_offline_semantics():
    """The streaming featurizer must agree with the offline definitions."""
    from src.data.features_live import MessageFeaturizer

    stats = {"period_by_id": {0x316: 0.010}, "global_median_period": 0.02,
             "known_ids": [0x316]}
    featurizer = MessageFeaturizer(stats)
    first = featurizer.feed(100.000, 0x316, 8, [0] * 8)
    # First occurrence of a known id: neutral deviation
    assert first.rel_dev == pytest.approx(0.0)
    second = featurizer.feed(100.005, 0x316, 8, [0] * 8)
    # 5 ms against a 10 ms period -> rel_dev = -0.5
    assert second.rel_dev == pytest.approx(-0.5)
    assert second.dt_prev == pytest.approx(0.005)
    unknown = featurizer.feed(100.006, 0x7FF, 8, [0] * 8)
    assert unknown.is_unknown == 1
    assert unknown.rel_dev == 10.0  # UNKNOWN_ID_DEV
