"""Parser unit tests on synthetic raw files."""

import numpy as np
import pytest

from src.data.parser import (
    detect_timestamp_gap,
    parse_attack_csv,
    parse_normal_txt,
)

ATTACK_CSV = """1478198376.389427,0316,8,05,21,68,09,21,21,00,6f,R
1478198376.389636,018f,8,fe,5b,00,00,00,3c,00,00,T
1478198376.390000,02b0,5,ff,7f,00,05,49,R
1478198376.390500,0002,2,aa,bb,T
1478198500.000000,0316,8,05,21,68,09,21,21,00,6f,R
"""

NORMAL_TXT = """Timestamp: 1479121434.850202        ID: 0350    000    DLC: 8    05 28 84 66 6d 00 00 a2
Timestamp: 1479121434.850423        ID: 02c0    000    DLC: 8    14 00 00 00 00 00 00 00
Timestamp: 1479121434.850977        ID: 05f0    000    DLC: 2    00 3a
"""


@pytest.fixture
def attack_file(tmp_path):
    path = tmp_path / "mini_attack.csv"
    path.write_text(ATTACK_CSV, encoding="ascii")
    return path


@pytest.fixture
def normal_file(tmp_path):
    path = tmp_path / "mini_normal.txt"
    path.write_text(NORMAL_TXT, encoding="ascii")
    return path


def test_flag_position_shifts_with_dlc(attack_file):
    df, _ = parse_attack_csv(attack_file, "dos")
    # DLC=5 row: flag R sits right after 5 payload bytes
    assert df.loc[2, "dlc"] == 5
    assert df.loc[2, "label"] == 0
    # DLC=2 row: flag T sits right after 2 payload bytes
    assert df.loc[3, "dlc"] == 2
    assert df.loc[3, "label"] == 1


def test_short_payloads_zero_filled(attack_file):
    df, _ = parse_attack_csv(attack_file, "dos")
    assert df.loc[3, "byte0"] == 0xAA
    assert df.loc[3, "byte1"] == 0xBB
    assert all(df.loc[3, f"byte{i}"] == 0 for i in range(2, 8))


def test_hex_conversion(attack_file):
    df, _ = parse_attack_csv(attack_file, "dos")
    assert df.loc[0, "can_id"] == 0x316
    assert df.loc[0, "byte7"] == 0x6F


def test_gap_trimming(attack_file):
    df, stats = parse_attack_csv(attack_file, "dos")
    # The 123.6 s jump before the last row must be trimmed away
    assert len(df) == 4
    assert stats["rows_before_trim"] == 5
    assert stats["gap_seconds"] > 100


def test_attack_type_assignment(attack_file):
    df, _ = parse_attack_csv(attack_file, "dos")
    assert list(df["attack_type"].astype(str)) == [
        "normal", "dos", "normal", "dos"]


def test_normal_txt_parsing(normal_file):
    df, stats = parse_normal_txt(normal_file)
    assert len(df) == 3
    assert df.loc[0, "can_id"] == 0x350
    assert df.loc[2, "dlc"] == 2
    assert df.loc[2, "byte1"] == 0x3A
    assert df.loc[2, "byte7"] == 0
    assert (df["label"] == 0).all()
    assert stats["malformed_lines"] == 0


def test_detect_timestamp_gap():
    timestamps = np.array([0.0, 0.1, 0.2, 100.0, 100.1])
    index, gap = detect_timestamp_gap(timestamps, threshold_s=10.0)
    assert index == 3
    assert gap == pytest.approx(99.8)
    index, _ = detect_timestamp_gap(np.array([0.0, 0.1, 0.2]), threshold_s=10.0)
    assert index is None
