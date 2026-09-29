import pandas as pd
from rainfall.data.gfs import accumulation_window, valid_date


def test_windows_cover_imd_day():
    for L in (1, 2, 3):
        f0, f1 = accumulation_window(L)
        assert f1 - f0 == 24 and f0 % 6 == 3          # 03Z -> 03Z next day
    assert accumulation_window(1) == (3, 27)


def test_valid_date():
    assert valid_date(pd.Timestamp("2025-08-01 00:00"), 1) == pd.Timestamp("2025-08-02")
